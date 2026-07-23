"""Trust-region BoTorch BO baselines — TuRBO-1 and SCBO (WP-E slate, §9.8/§12.3).

The two BoTorch families a reviewer of the M2 "RIG reaches spec ~2x cheaper than
BO" claim would demand, added FAITHFULLY and steelmanned alongside the existing
continuous-acquisition :class:`~rig.baselines.botorch_bo.BoTorchBO`:

- :class:`TuRBOBaseline` — **TuRBO-1** trust-region BO (Eriksson et al., NeurIPS
  2019, *"Scalable Global Optimization via Local Bayesian Optimization"*,
  arXiv:1910.01739). Implements the canonical BoTorch tutorial's ``TurboState``
  machine verbatim: a hyperrectangular trust region centred on the incumbent,
  Thompson sampling (``MaxPosteriorSampling``) over a lengthscale-shaped Sobol
  candidate set, success/failure counters that expand/shrink the region, and a
  restart when the region collapses below ``length_min``.
- :class:`SCBOBaseline` — **SCBO** scalable *constrained* BO (Eriksson & Poloczek,
  AISTATS 2021, *"Scalable Constrained Bayesian Optimization"*, arXiv:2002.08526).
  The spec box is expressed as ``2m`` outcome constraints ``c(x) <= 0`` (one GP
  per constraint, SCBO's defining feature), and candidates are drawn with
  ``ConstrainedMaxPosteriorSampling`` (constrained Thompson sampling: prefer
  posterior-feasible candidates, fall back to minimum-violation), inside the same
  TuRBO trust-region machine driven by *feasible* improvement.

Both are DROP-IN :class:`~rig.eval.m2_sweep` ``MethodFactory`` objects
(``run() -> Trajectory``) and share **every fairness knob** with the RIG loop and
the two existing BO arms (this is the point of the slate — a RIG win must be
unimpeachable, a RIG loss reported plainly):

- **Identical warm start** — the SAME scrambled-Sobol DoE over the expert box,
  from the SAME ``RecipeTransform`` + campaign seed, so the seed batch is
  bit-identical to ``WarmStartedBO``/``BoTorchBO`` (checked in tests).
- **Identical objective / hit rule / budget / cost** — the SAME tolerance-
  normalized box scalarization ``g(x)=||relu(L-y, y-U)/w||_2`` (``w`` = per-output
  box half-width); BoTorch maximizes, so the objective GP models ``f = -g``. Any
  recipe whose machine outcome is ``in_spec`` counts as a hit; only machine
  queries count against ``budget`` (all GP/acquisition/sampling work is free).
- **Same interior search domain** — ``optimize``/Thompson candidates live in the
  SAME box-sigmoid interior ``x = lo + (hi-lo)*sigma(u)``, ``u in [-u_bound,
  u_bound]``, that the reference arms reach, never the exact box edge.
- **Same GP tier** — ``SingleTaskGP`` + input ``Normalize`` + outcome
  ``Standardize`` + the §20.5 Hvarfner sqrt(D) dim-scaled Matern-5/2 prior. No
  RIG-only trick (the ``mu-k*sigma`` pessimistic objective) leaks into either arm.

Declared simplifications (honest, so the comparator is neither strawman nor
secretly stronger than RIG):

- **TuRBO-1, not TuRBO-m.** We run a SINGLE trust region (m=1). The multi-region
  TuRBO-m allocates several regions with an implicit bandit; at the M2 budget
  (~40 evals, d=2 recipe space) a single region is the standard and strongest
  choice, and TuRBO-m's bandit would only dilute a tiny budget across regions. We
  say so rather than silently run m=1 and call it "TuRBO".
- **Restart keeps history.** On region collapse we reset the TR geometry/counters
  and inject a fresh Sobol exploration batch (counted against budget), but RETAIN
  the observation history for the GP rather than cold-restarting on only new
  points. This can only *help* the comparator (more data), consistent with the
  house rule "no weakening the comparators"; at the M2 budget/dimension a collapse
  is rare in any case.
- **SCBO objective is the shared box-distance scalarization.** The spec *is* the
  box, so the objective ``f=-g`` is redundant with the constraints by
  construction; SCBO here is therefore a constrained-*feasibility* searcher. We
  keep the box-distance objective (rather than a constant) so the objective GP is
  non-degenerate and, among posterior-feasible candidates, Thompson sampling still
  has a gradient toward the box centre.

torch/gpytorch/botorch are the ``[torch]`` optional extra, imported lazily by
``rig.baselines.__init__`` so ``import rig`` stays torch-free. Continuous recipe
spaces only (the M2/MBE recipe vars are all continuous); a compositional variable
raises, matching ``BoTorchBO``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc
from torch.quasirandom import SobolEngine

from rig.active.loop import Trajectory
from rig.baselines.warm_bo import _distance_to_box
from rig.interfaces import CompositionalVariable, ContinuousVariable
from rig.inverse.pessimistic import parse_targets
from rig.transforms import RecipeTransform

_DTYPE = torch.double


# --------------------------------------------------------------------------- #
# Trust-region state machine (canonical BoTorch TuRBO / SCBO tutorial verbatim) #
# --------------------------------------------------------------------------- #


@dataclass
class _TurboState:
    """The TuRBO-1 / SCBO trust-region state (Eriksson 2019 §3; the BoTorch
    ``TurboState`` tutorial dataclass). ``length`` is the edge of the (unit-cube-
    normalized) hyperrectangle; success/failure counters expand it toward
    ``length_max`` or shrink it, and ``restart_triggered`` fires when it collapses
    below ``length_min``. For SCBO ``best_value`` is the best *feasible* objective
    and ``best_constraint_values`` the incumbent's constraint vector (inf until a
    point is seen), so improvement is judged feasibility-first."""

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(default=0)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    best_constraint_values: torch.Tensor | None = None
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        # canonical: ceil(max(4/q, dim/q)) — the number of consecutive no-improve
        # batches tolerated before the region shrinks (Eriksson 2019 / tutorial).
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def _update_tr_length(state: _TurboState) -> None:
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    if state.length < state.length_min:
        state.restart_triggered = True


def _update_state_unconstrained(state: _TurboState, y_next: np.ndarray) -> None:
    """TuRBO update: a batch is a success iff it improves the incumbent by a
    relative 1e-3 margin (BoTorch ``update_state`` verbatim)."""
    best_batch = float(np.max(y_next))
    if best_batch > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    state.best_value = max(state.best_value, best_batch)
    _update_tr_length(state)


def _feasible_mask(C: np.ndarray) -> np.ndarray:
    """(n, n_con) constraint values -> (n,) bool: feasible iff all c <= 0."""
    return np.all(C <= 0.0, axis=-1)


def _best_index_for_batch(y: np.ndarray, C: np.ndarray) -> int:
    """SCBO ``get_best_index_for_batch``: the best FEASIBLE point (max objective),
    else the minimum-total-violation point."""
    feas = _feasible_mask(C)
    if feas.any():
        score = y.copy()
        score[~feas] = -np.inf
        return int(np.argmax(score))
    return int(np.argmin(np.clip(C, 0.0, None).sum(axis=-1)))


def _update_state_constrained(state: _TurboState, y_next: np.ndarray, C_next: np.ndarray) -> None:
    """SCBO update (BoTorch constrained ``update_state`` verbatim): a batch is a
    success iff it yields a new best feasible objective, or (while nothing is yet
    feasible) reduces the incumbent's total constraint violation."""
    n_con = C_next.shape[-1]
    if state.best_constraint_values is None:
        state.best_constraint_values = torch.full((n_con,), float("inf"), dtype=_DTYPE)
    idx = _best_index_for_batch(y_next, C_next)
    y_best = float(y_next[idx])
    c_best = torch.as_tensor(C_next[idx], dtype=_DTYPE)
    bcv = state.best_constraint_values

    if bool((c_best <= 0).all()):  # the batch's chosen point is feasible
        improves = y_best > state.best_value + 1e-3 * math.fabs(state.best_value)
        incumbent_infeasible = bool((bcv > 0).any())
        if improves or incumbent_infeasible:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_best
            state.best_constraint_values = c_best
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:  # nothing feasible in the batch -> judge on total violation
        viol_next = float(c_best.clamp(min=0).sum())
        viol_incumbent = float(bcv.clamp(min=0).sum())
        if viol_next < viol_incumbent:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_best
            state.best_constraint_values = c_best
        else:
            state.success_counter = 0
            state.failure_counter += 1
    _update_tr_length(state)


# --------------------------------------------------------------------------- #
# Shared arm: warm start / objective / query / GP / trust-region candidates    #
# --------------------------------------------------------------------------- #


class _TrustRegionArm:
    """Shared machinery for the two trust-region BoTorch arms — identical to the
    ``BoTorchBO`` fairness contract (warm start, objective, hit rule, budget, GP
    tier, interior search domain). Subclasses implement ``_propose_batch``."""

    def __init__(
        self,
        *,
        machine: Callable[[Mapping[str, float]], np.ndarray],
        in_spec: Callable[[np.ndarray], bool],
        variables: Sequence[ContinuousVariable | CompositionalVariable],
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        spec: Mapping[str, Any],
        cost_recipe: Callable[[Mapping[str, float]], float] = lambda r: 1.0,
        c_batch: float = 0.0,
        budget: int = 40,
        q: int = 4,
        n_seed: int | None = None,
        n_candidates: int | None = None,
        u_bound: float = 5.0,
        success_tolerance: int = 10,
        seed: int = 0,
    ) -> None:
        self.variables = list(variables)
        for v in self.variables:
            if not isinstance(v, ContinuousVariable):
                raise NotImplementedError(
                    f"{type(self).__name__} supports continuous recipe variables only; a "
                    f"compositional variable ({v.name!r}) needs the trust region built in a "
                    "simplex reparameterization (follow-on). Use WarmStartedBO for "
                    "compositional spaces."
                )
        self.machine = machine
        self.in_spec = in_spec
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        box = parse_targets(spec["targets"], self.output_keys)
        self._out_idx = np.array([self.output_keys.index(n) for n in box.output_names], dtype=int)
        self._lower, self._upper = box.lower, box.upper
        self._scale = np.maximum((self._upper - self._lower) / 2.0, 1e-12)  # tol-fair
        self.cost_recipe = cost_recipe
        self.c_batch = float(c_batch)
        self.budget = int(budget)
        self.q = int(q)
        self.u_bound = float(u_bound)
        self.success_tolerance = int(success_tolerance)
        self.seed = int(seed)
        self._rt = RecipeTransform(self.variables)
        self._flat_keys = [v.name for v in self.variables]
        d = self._rt.dim
        self.n_seed = int(n_seed) if n_seed is not None else max(2 * d + 2, 8)
        # Sobol candidate-set size for Thompson sampling (canonical uses
        # min(5000, max(2000, 200*d)); the M2 recipe space is d~2, so a smaller set
        # fully covers the trust region and keeps the arm CPU-friendly).
        self.n_candidates = (
            int(n_candidates) if n_candidates is not None else min(2000, max(400, 200 * d))
        )
        # Interior search box = the box-sigmoid image of u in [-u_bound, u_bound],
        # IDENTICAL to BoTorchBO/WarmStartedBO/the RIG loop (never the exact edge).
        s_lo = 1.0 / (1.0 + np.exp(self.u_bound))
        s_hi = 1.0 / (1.0 + np.exp(-self.u_bound))
        lows = np.array([v.lower for v in self.variables], dtype=float)
        highs = np.array([v.upper for v in self.variables], dtype=float)
        self._bounds = torch.tensor(
            np.stack([lows + (highs - lows) * s_lo, lows + (highs - lows) * s_hi]),
            dtype=_DTYPE,
        )

    # -- warm start / objective / query (identical to BoTorchBO) ------------- #

    def _recipe_to_x(self, recipe: Mapping[str, float]) -> np.ndarray:
        return np.array([recipe[k] for k in self._flat_keys], dtype=float)

    def _sobol_u(self, n: int, seed: int) -> np.ndarray:
        sampler = qmc.Sobol(d=self._rt.dim, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u = sampler.random(max(n, 1))
        return (2.0 * u - 1.0) * self.u_bound

    def _g(self, outcome: np.ndarray) -> float:
        y = np.asarray(outcome, dtype=float)[self._out_idx]
        return _distance_to_box(y, self._lower, self._upper, scale=self._scale)

    def _constraints(self, outcome: np.ndarray) -> np.ndarray:
        """Box as outcome constraints c(x) <= 0: for each finite-bounded spec
        output, ``L - y`` (lower) and ``y - U`` (upper). Infinite sides are dropped
        (trivially satisfied). Returns (n_con,)."""
        y = np.asarray(outcome, dtype=float)[self._out_idx]
        cons: list[float] = []
        for j in range(len(y)):
            if np.isfinite(self._lower[j]):
                cons.append(float(self._lower[j] - y[j]))
            if np.isfinite(self._upper[j]):
                cons.append(float(y[j] - self._upper[j]))
        return np.array(cons, dtype=float)

    def _query(self, recipes: list[Mapping[str, float]]):
        X = np.array([self._recipe_to_x(r) for r in recipes], dtype=float)
        outcomes = [np.asarray(self.machine(r), dtype=float) for r in recipes]
        g = np.array([self._g(o) for o in outcomes], dtype=float)
        C = np.array([self._constraints(o) for o in outcomes], dtype=float)
        cost = self.c_batch + float(sum(self.cost_recipe(r) for r in recipes))
        hits = [bool(self.in_spec(o)) for o in outcomes]
        return X, g, C, cost, hits

    def _x_to_recipes(self, X: np.ndarray) -> list[dict[str, float]]:
        return [{k: float(v) for k, v in zip(self._flat_keys, row, strict=True)} for row in X]

    # -- GP tier (identical to BoTorchBO) ------------------------------------ #

    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> SingleTaskGP:
        """SingleTaskGP on ``y`` (already sign-oriented for maximization) with input
        Normalize over the interior box, outcome Standardize, §20.5 Hvarfner
        sqrt(D) dim-scaled Matern-5/2 prior — the same tier as the RIG GP + the
        other BO arms."""
        torch.manual_seed(self.seed)
        train_x = torch.as_tensor(X, dtype=_DTYPE)
        train_y = torch.as_tensor(y, dtype=_DTYPE).unsqueeze(-1)
        d = train_x.shape[1]
        gp = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d, bounds=self._bounds),
            outcome_transform=Standardize(1),
            covar_module=get_covar_module_with_dim_scaled_prior(
                ard_num_dims=d, use_rbf_kernel=False
            ),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(mll)
        return gp

    # -- trust-region candidate set (canonical TuRBO ``create_candidates``) --- #

    def _to_unit(self, X: np.ndarray) -> torch.Tensor:
        lo, hi = self._bounds[0], self._bounds[1]
        return (torch.as_tensor(X, dtype=_DTYPE) - lo) / (hi - lo)

    def _from_unit(self, U: torch.Tensor) -> np.ndarray:
        lo, hi = self._bounds[0], self._bounds[1]
        return (lo + U * (hi - lo)).detach().cpu().numpy().astype(float)

    def _tr_candidates(
        self, gp: SingleTaskGP, x_center_unit: torch.Tensor, length: float, batch_seed: int
    ) -> torch.Tensor:
        """Lengthscale-shaped Sobol candidate set inside the trust region (unit
        cube), with the canonical per-coordinate perturbation mask so most
        coordinates keep the incumbent's value (Eriksson 2019 §3 / BoTorch
        tutorial). Returns (n_candidates, d) in the UNIT cube."""
        d = x_center_unit.numel()
        ls = gp.covar_module.lengthscale.squeeze().detach()
        ls = ls.reshape(-1)
        if ls.numel() != d:  # scalar lengthscale -> isotropic
            ls = ls.expand(d)
        weights = ls / ls.mean()
        weights = weights / torch.prod(weights.pow(1.0 / d))  # geometric mean 1
        tr_lb = torch.clamp(x_center_unit - weights * length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center_unit + weights * length / 2.0, 0.0, 1.0)

        n = self.n_candidates
        gen = torch.Generator().manual_seed(int(batch_seed))
        sobol = SobolEngine(dimension=d, scramble=True, seed=int(batch_seed))
        pert = sobol.draw(n).to(dtype=_DTYPE)
        pert = tr_lb + (tr_ub - tr_lb) * pert
        prob_perturb = min(20.0 / d, 1.0)
        mask = torch.rand(n, d, generator=gen, dtype=_DTYPE) <= prob_perturb
        # ensure at least one coordinate is perturbed per candidate row
        empty = torch.where(mask.sum(dim=1) == 0)[0]
        if empty.numel() > 0:
            rand_dim = torch.randint(0, d, (empty.numel(),), generator=gen)
            mask[empty, rand_dim] = True
        X_cand = x_center_unit.expand(n, d).clone()
        X_cand[mask] = pert[mask]
        return X_cand

    # subclasses implement the per-batch proposal
    def _propose_batch(self, *args, **kwargs) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# TuRBO-1                                                                       #
# --------------------------------------------------------------------------- #


class TuRBOBaseline(_TrustRegionArm):
    """TuRBO-1 trust-region BO to reach a spec (Eriksson 2019; §9.8/§12.3).

    Drop-in :class:`~rig.eval.m2_sweep` ``MethodFactory``: construct with
    ``(*, machine, in_spec, spec, seed, ...)`` and call ``run() -> Trajectory``.
    Optimizes ``f=-g`` (the shared tolerance-normalized box scalarization) by
    Thompson sampling inside a single trust region whose size adapts via the
    canonical success/failure machine, restarting on collapse."""

    def _propose_batch(
        self,
        gp: SingleTaskGP,
        X_unit_all: torch.Tensor,
        f: np.ndarray,
        length: float,
        k: int,
        batch_seed: int,
    ) -> np.ndarray:
        x_center_unit = X_unit_all[int(np.argmax(f))]
        X_cand = self._tr_candidates(gp, x_center_unit, length, batch_seed)
        torch.manual_seed(int(batch_seed))
        thompson = MaxPosteriorSampling(model=gp, replacement=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_next_unit = thompson(X_cand, num_samples=k)
        return self._from_unit(X_next_unit)  # (k, d) real-space

    def run(self) -> Trajectory:
        # §9.2 warm start: the IDENTICAL Sobol DoE the RIG loop + other BO arms use.
        seed_recipes = [self._rt.forward(u) for u in self._sobol_u(self.n_seed, self.seed)]
        X, g, _C, cost, hits = self._query(seed_recipes)
        cumulative = cost
        traj = Trajectory(hit=False, cost_to_target=float("inf"), n_queries=len(seed_recipes))
        traj.cumulative_cost.append(cumulative)
        seed_hit = any(hits)
        traj.per_batch_hit.append(seed_hit)
        if seed_hit:
            traj.hit = True
            traj.cost_to_target = cumulative
            traj.stop_reason = "target met in seed DoE"
            return traj

        d = self._rt.dim
        state = _TurboState(dim=d, batch_size=self.q, success_tolerance=self.success_tolerance)
        state.best_value = float(-np.min(g))  # incumbent of f=-g from the seed DoE
        gp = self._fit_gp(X, -g)

        while traj.n_queries < self.budget:
            k = min(self.q, self.budget - traj.n_queries)
            batch_seed = self.seed + 100 + traj.n_queries
            if state.restart_triggered:
                # region collapsed: reset geometry/counters and reseed with a fresh
                # Sobol exploration batch (declared simplification: history retained).
                state = _TurboState(
                    dim=d, batch_size=self.q, success_tolerance=self.success_tolerance
                )
                state.best_value = float(-np.min(g))
                fresh = [self._rt.forward(u) for u in self._sobol_u(k, batch_seed)]
                cand_X = np.array([self._recipe_to_x(r) for r in fresh], dtype=float)
            else:
                cand_X = self._propose_batch(gp, self._to_unit(X), -g, state.length, k, batch_seed)
            batch = self._x_to_recipes(cand_X)

            Xb, gb, _Cb, cost, hits = self._query(batch)
            cumulative += cost
            traj.n_queries += len(batch)
            batch_hit = any(hits)
            traj.cumulative_cost.append(cumulative)
            traj.per_batch_hit.append(batch_hit)
            if batch_hit:
                traj.hit = True
                traj.cost_to_target = cumulative
                traj.stop_reason = "target met (TuRBO Thompson proposal in-spec on machine)"
                return traj

            X = np.vstack([X, Xb])
            g = np.concatenate([g, gb])
            _update_state_unconstrained(state, -gb)
            gp = self._fit_gp(X, -g)

        traj.stop_reason = "budget exhausted"
        return traj


# --------------------------------------------------------------------------- #
# SCBO                                                                          #
# --------------------------------------------------------------------------- #


class SCBOBaseline(_TrustRegionArm):
    """SCBO constrained trust-region BO to reach a spec (Eriksson & Poloczek 2021;
    §9.8/§12.3).

    Drop-in :class:`~rig.eval.m2_sweep` ``MethodFactory``. Models the objective
    ``f=-g`` and each of the ``2m`` box constraints ``c(x) <= 0`` with a SEPARATE
    GP (SCBO's defining feature), and selects each batch with
    ``ConstrainedMaxPosteriorSampling`` — constrained Thompson sampling that
    prefers posterior-feasible candidates and falls back to minimum predicted
    violation — inside the TuRBO trust region driven by *feasible* improvement."""

    def _fit_constraint_models(self, X: np.ndarray, C: np.ndarray) -> ModelListGP:
        """One SingleTaskGP per constraint column (SCBO models each c_i separately)."""
        models = [self._fit_gp(X, C[:, j]) for j in range(C.shape[1])]
        return ModelListGP(*models)

    def _propose_batch(
        self,
        obj_gp: SingleTaskGP,
        con_model: ModelListGP,
        X_unit_all: torch.Tensor,
        y: np.ndarray,
        C: np.ndarray,
        length: float,
        k: int,
        batch_seed: int,
    ) -> np.ndarray:
        # trust-region centre = the SCBO best point (feasible-first, else min-violation)
        x_center_unit = X_unit_all[_best_index_for_batch(y, C)]
        X_cand = self._tr_candidates(obj_gp, x_center_unit, length, batch_seed)
        torch.manual_seed(int(batch_seed))
        cmps = ConstrainedMaxPosteriorSampling(
            model=obj_gp, constraint_model=con_model, replacement=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_next_unit = cmps(X_cand, num_samples=k)
        return self._from_unit(X_next_unit)

    def run(self) -> Trajectory:
        # §9.2 warm start: IDENTICAL Sobol DoE.
        seed_recipes = [self._rt.forward(u) for u in self._sobol_u(self.n_seed, self.seed)]
        X, g, C, cost, hits = self._query(seed_recipes)
        cumulative = cost
        traj = Trajectory(hit=False, cost_to_target=float("inf"), n_queries=len(seed_recipes))
        traj.cumulative_cost.append(cumulative)
        seed_hit = any(hits)
        traj.per_batch_hit.append(seed_hit)
        if seed_hit:
            traj.hit = True
            traj.cost_to_target = cumulative
            traj.stop_reason = "target met in seed DoE"
            return traj

        d = self._rt.dim
        state = _TurboState(dim=d, batch_size=self.q, success_tolerance=self.success_tolerance)
        # seed the incumbent from the best seed-DoE point (feasible-first)
        idx0 = _best_index_for_batch(-g, C)
        state.best_value = float(-g[idx0])
        state.best_constraint_values = torch.as_tensor(C[idx0], dtype=_DTYPE)
        obj_gp = self._fit_gp(X, -g)
        con_model = self._fit_constraint_models(X, C)

        while traj.n_queries < self.budget:
            k = min(self.q, self.budget - traj.n_queries)
            batch_seed = self.seed + 100 + traj.n_queries
            if state.restart_triggered:
                state = _TurboState(
                    dim=d, batch_size=self.q, success_tolerance=self.success_tolerance
                )
                idxr = _best_index_for_batch(-g, C)
                state.best_value = float(-g[idxr])
                state.best_constraint_values = torch.as_tensor(C[idxr], dtype=_DTYPE)
                fresh = [self._rt.forward(u) for u in self._sobol_u(k, batch_seed)]
                cand_X = np.array([self._recipe_to_x(r) for r in fresh], dtype=float)
            else:
                cand_X = self._propose_batch(
                    obj_gp, con_model, self._to_unit(X), -g, C, state.length, k, batch_seed
                )
            batch = self._x_to_recipes(cand_X)

            Xb, gb, Cb, cost, hits = self._query(batch)
            cumulative += cost
            traj.n_queries += len(batch)
            batch_hit = any(hits)
            traj.cumulative_cost.append(cumulative)
            traj.per_batch_hit.append(batch_hit)
            if batch_hit:
                traj.hit = True
                traj.cost_to_target = cumulative
                traj.stop_reason = "target met (SCBO constrained-Thompson proposal in-spec)"
                return traj

            X = np.vstack([X, Xb])
            g = np.concatenate([g, gb])
            C = np.vstack([C, Cb])
            _update_state_constrained(state, -gb, Cb)
            obj_gp = self._fit_gp(X, -g)
            con_model = self._fit_constraint_models(X, C)

        traj.stop_reason = "budget exhausted"
        return traj
