"""Warm-started GP-EI Bayesian optimization baseline (implementation-plan §9.8 / §12.3).

The *fair* BO baseline MFL omitted: GP + Expected Improvement, warm-started from
the expert-constrained ranges (Kanarik's cost-halving lever), matched-budget
against the RIG active-learning loop (WP-F) so M2's "beats warm-started BO
in-silico" claim is head-to-head. It targets the SAME spec and returns the SAME
:class:`~rig.active.loop.Trajectory`, so WP-G's difference-in-RMST consumes both.

Formulation: the scalar objective is the distance from the outcome to the spec
box, ``g(x) = ‖relu(L − y, y − U) / w‖₂`` where ``w`` is the per-output box
half-width (0 iff in-spec). A GP is fit to observed ``g`` and EI minimizes it
toward 0. This is the standard "BO to hit a target" practitioner setup — a
genuinely strong, tuned comparator, not a strawman.

The per-output **tolerance normalization** (``/w``) is essential and NOT
optional: a raw ``‖relu(L−y, y−U)‖₂`` is dominated by whichever output has the
largest engineering scale, so on a multi-KPI spec whose outputs span orders of
magnitude (e.g. ``T_center`` ~1e3 K vs ``bow_cooldown_um`` ~1e-4 m in SI) the
scalarization would be numerically blind to the small-scale output and BO could
never satisfy its box — a scalarization artifact, not a method weakness. Measuring
each residual in units of its own tolerance makes the objective scale-invariant
and puts every KPI's box on equal footing (the standard multi-spec scalarization).

numpy/scipy only. The BoTorch qLogEI/qLogNEHVI production baselines are WP-E;
this is the honest GP-tier stand-in with the identical warm-start + budget.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, qmc

from rig.active.loop import Trajectory
from rig.forward.gp import GPForwardModel
from rig.interfaces import CompositionalVariable, ContinuousVariable
from rig.inverse.pessimistic import parse_targets
from rig.transforms import RecipeTransform


def _distance_to_box(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, scale: np.ndarray | None = None
) -> float:
    """L2 distance from outcome ``y`` to the spec box (0 iff inside).

    ``scale`` (per-output, > 0) divides each residual before the norm, so the
    objective is measured in units of each output's own tolerance — scale-fair
    across outputs of disparate engineering magnitude. ``None`` = raw distance."""
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    resid = below + above  # exactly one side is nonzero per output
    if scale is not None:
        resid = resid / scale
    return float(np.sqrt(np.sum(resid**2)))


def expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float) -> np.ndarray:
    """EI for MINIMIZING an objective with incumbent ``best`` (target 0).

    ``EI(x) = (best − μ)·Φ(z) + σ·φ(z)``, ``z = (best − μ)/σ``. σ→0 gives
    ``max(best − μ, 0)`` (the noise-free limit)."""
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    out = np.maximum(best - mu, 0.0)
    pos = sigma > 1e-12
    z = (best - mu[pos]) / sigma[pos]
    out[pos] = (best - mu[pos]) * norm.cdf(z) + sigma[pos] * norm.pdf(z)
    return np.maximum(out, 0.0)


class WarmStartedBO:
    """GP-EI BO to reach a spec, warm-started + matched-budget with WP-F (§12.3).

    Mirrors :class:`~rig.active.loop.ActiveLearningLoop` so trajectories compare
    directly. Drives ``machine(recipe) -> outcome``; ``in_spec`` is the
    ground-truth hit test (τ floored at Gage-R&R)."""

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
        n_pool: int = 256,
        u_bound: float = 5.0,
        acq_optimize: bool = False,
        acq_restarts: int = 8,
        seed: int = 0,
    ) -> None:
        self.machine = machine
        self.in_spec = in_spec
        self.variables = list(variables)
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        box = parse_targets(spec["targets"], self.output_keys)
        self._out_idx = np.array([self.output_keys.index(n) for n in box.output_names], dtype=int)
        self._lower, self._upper = box.lower, box.upper
        # per-output tolerance (box half-width) => scale-fair scalarization.
        self._scale = np.maximum((self._upper - self._lower) / 2.0, 1e-12)
        self.cost_recipe = cost_recipe
        self.c_batch = float(c_batch)
        self.budget = int(budget)
        self.q = int(q)
        self.n_pool = int(n_pool)
        self.u_bound = float(u_bound)
        # acq_optimize: maximize EI CONTINUOUSLY (L-BFGS multistart on the
        # acquisition surface, the optimize_acqf pattern) instead of only ranking
        # a discrete Sobol pool — so BO's search resolution is not capped below the
        # spec-box size (M2 re-validation finding BF-1a). Default off preserves the
        # fixed-pool baseline + existing tests.
        self.acq_optimize = bool(acq_optimize)
        self.acq_restarts = int(acq_restarts)
        self.seed = int(seed)
        self._rt = RecipeTransform(self.variables)
        self._flat_keys = self._build_flat_keys()
        d = self._rt.dim
        self.n_seed = int(n_seed) if n_seed is not None else max(2 * d + 2, 8)

    def _build_flat_keys(self) -> list[str]:
        keys: list[str] = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                keys.append(v.name)
            else:
                keys.extend(f"{v.name}.{c}" for c in v.components)
        return keys

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

    def _query(self, recipes: list[Mapping[str, float]]):
        X = np.array([self._recipe_to_x(r) for r in recipes], dtype=float)
        outcomes = [np.asarray(self.machine(r), dtype=float) for r in recipes]
        g = np.array([self._g(o) for o in outcomes], dtype=float)
        cost = self.c_batch + float(sum(self.cost_recipe(r) for r in recipes))
        hits = [bool(self.in_spec(o)) for o in outcomes]
        return X, g, cost, hits

    def _ei_at(self, X: np.ndarray, gp: Any, best: float) -> np.ndarray:
        """EI at design rows ``X`` (n,d) under the current surrogate ``gp``."""
        dist = gp.predict(np.atleast_2d(X))
        mu = np.ravel(np.asarray(dist.mean))
        sig = np.ravel(
            np.sqrt(np.asarray(dist.aleatoric_sigma) ** 2 + np.asarray(dist.epistemic_sigma) ** 2)
        )
        return expected_improvement(mu, sig, best=best)

    def _neg_ei_u(self, u_flat: np.ndarray, gp: Any, best: float) -> float:
        """-EI at a single reparameterized point ``u`` (for L-BFGS)."""
        recipe = self._rt.forward(np.asarray(u_flat, dtype=float))
        X = self._recipe_to_x(recipe).reshape(1, -1)
        return -float(self._ei_at(X, gp, best)[0])

    def _refine_u(self, u0: np.ndarray, gp: Any, best: float) -> np.ndarray:
        """Continuously maximize EI (L-BFGS-B on -EI) from start ``u0``, box-bounded
        in reparameterized space (the ``optimize_acqf`` pattern)."""
        bounds = [(-self.u_bound, self.u_bound)] * self._rt.dim
        try:
            res = minimize(
                self._neg_ei_u,
                np.asarray(u0, dtype=float),
                args=(gp, best),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 100},
            )
            u = np.asarray(res.x, dtype=float)
            if np.all(np.isfinite(u)):
                return u
        except Exception:  # noqa: BLE001 — refinement must never sink a campaign
            pass
        return np.asarray(u0, dtype=float)

    def _pick_batch(
        self, X: np.ndarray, recipes: list[Mapping[str, float]], ei: np.ndarray, k: int
    ) -> list[Mapping[str, float]]:
        """Top-``k`` by EI with a simple dedup in recipe-vector space."""
        picks: list[int] = []
        for idx in np.argsort(-ei):
            if len(picks) >= k:
                break
            if all(np.linalg.norm(X[idx] - X[p]) > 1e-6 for p in picks):
                picks.append(int(idx))
        return [recipes[i] for i in picks]

    def run(self) -> Trajectory:
        # §9.2 warm start: Sobol DoE over the expert-constrained ranges.
        seed_recipes = [self._rt.forward(u) for u in self._sobol_u(self.n_seed, self.seed)]
        X, g, cost, hits = self._query(seed_recipes)
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

        gp = GPForwardModel(n_restarts=2, seed=self.seed).fit(X, g[:, None])

        while traj.n_queries < self.budget:
            pool_u = self._sobol_u(self.n_pool, self.seed + 100 + traj.n_queries)
            pool_recipes = [self._rt.forward(u) for u in pool_u]
            pool_X = np.array([self._recipe_to_x(r) for r in pool_recipes], dtype=float)
            dist = gp.predict(pool_X)
            mu = np.ravel(np.asarray(dist.mean))
            sig = np.ravel(
                np.sqrt(
                    np.asarray(dist.aleatoric_sigma) ** 2 + np.asarray(dist.epistemic_sigma) ** 2
                )
            )
            best = float(np.min(g))
            ei = expected_improvement(mu, sig, best=best)
            k = min(self.q, self.budget - traj.n_queries)

            if self.acq_optimize:
                # continuous acquisition (BF-1a fix): L-BFGS-refine the top-EI pool
                # starts into the box, then pick among the continuous optima, so
                # BO's resolution is not capped by the discrete pool spacing.
                starts = np.argsort(-ei)[: max(self.acq_restarts, k)]
                ref_u = [self._refine_u(pool_u[s], gp, best) for s in starts]
                ref_recipes = [self._rt.forward(u) for u in ref_u]
                ref_X = np.array([self._recipe_to_x(r) for r in ref_recipes], dtype=float)
                batch = self._pick_batch(ref_X, ref_recipes, self._ei_at(ref_X, gp, best), k)
            else:
                batch = self._pick_batch(pool_X, pool_recipes, ei, k)

            Xb, gb, cost, hits = self._query(batch)
            cumulative += cost
            traj.n_queries += len(batch)
            batch_hit = any(hits)
            traj.cumulative_cost.append(cumulative)
            traj.per_batch_hit.append(batch_hit)
            if batch_hit:
                traj.hit = True
                traj.cost_to_target = cumulative
                traj.stop_reason = "target met (EI proposal in-spec on machine)"
                return traj

            X = np.vstack([X, Xb])
            g = np.concatenate([g, gb])
            gp = GPForwardModel(n_restarts=2, seed=self.seed).fit(X, g[:, None])

        traj.stop_reason = "budget exhausted"
        return traj
