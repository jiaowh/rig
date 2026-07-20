"""The closed active-learning loop (implementation-plan §9.2/§9.4/§9.7) — numpy tier.

Turns the per-query pessimistic inverse (§8, WP-D) + the cost-cooled acquisition
(§9.4) into MFL's missing "given enough (x,z) pairs" answer: an actual
experiment selector. Each batch blends the two coupled §9.1 objectives —
**exploit** (query the inverse's current best recipe on the real machine) and
**explore** (spend the rest of the lot where the surrogate is least trustworthy
IN the spec's pre-image, via EPIG). It warm-starts from a Sobol DoE (§9.2, never
from scratch), refits + re-solves every batch (§9.7 cadence, D6), and stops on
target-met / budget / acquisition-stall (§9.7).

Machine-agnostic: driven by a ``machine(recipe) -> outcome`` callable and a
ground-truth ``in_spec(outcome) -> bool`` (τ floored at Gage-R&R), so it runs on
WP-B's in-silico pathology machine or any process. The amortized-posterior
re-distillation (offline, D6) and the Phase-II qLogNEHVI hand-off are WP-E.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import qmc

from rig.active.acquisition import anneal, cost_cooled_acquisition
from rig.active.batch import select_batch
from rig.forward.gp import GPForwardModel
from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
    Infeasible,
)
from rig.inverse import PessimisticInverseSolver
from rig.transforms import RecipeTransform


@dataclass
class Trajectory:
    """One campaign's cost-to-target trajectory (feeds the WP-G survival stats)."""

    hit: bool
    cost_to_target: float  # inf if the spec was not hit within budget
    n_queries: int
    cumulative_cost: list[float] = field(default_factory=list)  # after each batch
    per_batch_hit: list[bool] = field(default_factory=list)
    stop_reason: str = ""


class ActiveLearningLoop:
    """Closed-loop cost-to-target campaign for ONE spec (§9). Drives ``machine``
    (the real tool ``M``), never differentiates it: gradients flow only through
    the refit surrogate (the standard offline-MBO posture, §3.2)."""

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
        surrogate_factory: Callable[[], Any] | None = None,
        revalidation_model: Any | None = None,
        kappa: float = 1.0,
        z_epi: float = 1.0,
        delta_frac: float = 0.01,
        stall_eps: float = 1e-4,
        u_bound: float = 5.0,
        seed: int = 0,
    ) -> None:
        self.machine = machine
        self.in_spec = in_spec
        self.variables = list(variables)
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.spec = dict(spec)
        self.cost_recipe = cost_recipe
        self.c_batch = float(c_batch)
        self.budget = int(budget)
        self.q = int(q)
        self.n_pool = int(n_pool)
        self.kappa = kappa
        self.z_epi = z_epi
        self.delta_frac = delta_frac
        self.stall_eps = stall_eps
        self.u_bound = u_bound
        self.seed = int(seed)
        self.revalidation_model = revalidation_model
        self._rt = RecipeTransform(self.variables)
        self._flat_keys = self._build_flat_keys()
        d = self._rt.dim
        self.n_seed = int(n_seed) if n_seed is not None else max(2 * d + 2, 8)
        # §9.2: the seed DoE is the FIRST budget draw, not a freebie — n_seed real
        # machine runs fire before any optimization batch. A budget below n_seed
        # would silently overspend (budget=4, n_seed=8 ⇒ 8 runs), so fail loud at
        # construction rather than fire runs the caller never authorized.
        if self.budget < self.n_seed:
            raise ValueError(
                f"budget ({self.budget}) < n_seed ({self.n_seed}): the §9.2 seed DoE "
                f"alone would fire {self.n_seed} real machine runs, overspending the "
                f"declared budget of {self.budget}. Raise budget to at least n_seed, or "
                "lower n_seed."
            )
        if surrogate_factory is None:

            def surrogate_factory() -> GPForwardModel:
                return GPForwardModel(n_restarts=2, seed=self.seed)

        self.surrogate_factory = surrogate_factory

    # -- recipe/vector plumbing (shares the WP-D flat-key convention) ------------

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
        dim = self._rt.dim
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            u = sampler.random(max(n, 1))
        return (2.0 * u - 1.0) * self.u_bound

    def _pool(self, seed: int) -> tuple[np.ndarray, list[dict[str, float]]]:
        recipes = [self._rt.forward(u) for u in self._sobol_u(self.n_pool, seed)]
        X = np.array([self._recipe_to_x(r) for r in recipes], dtype=float)
        return X, recipes

    def _query(self, recipes: list[Mapping[str, float]]) -> tuple[np.ndarray, np.ndarray, float]:
        """Run the machine on a batch; return (X, Y, batch_cost)."""
        X = np.array([self._recipe_to_x(r) for r in recipes], dtype=float)
        Y = np.array([np.asarray(self.machine(r), dtype=float) for r in recipes], dtype=float)
        cost = self.c_batch + float(sum(self.cost_recipe(r) for r in recipes))
        return X, Y, cost

    def _any_in_spec(self, Y: np.ndarray) -> bool:
        """True iff any queried outcome row is in-spec on the real machine."""
        return any(bool(self.in_spec(Y[i])) for i in range(Y.shape[0]))

    # -- the loop ---------------------------------------------------------------

    def run(self) -> Trajectory:
        # §9.2 warm start: Sobol DoE over the search space, never from scratch.
        seed_recipes = [self._rt.forward(u) for u in self._sobol_u(self.n_seed, self.seed)]
        X, Y, cost = self._query(seed_recipes)
        cumulative = cost
        traj = Trajectory(hit=False, cost_to_target=float("inf"), n_queries=len(seed_recipes))
        traj.cumulative_cost.append(cumulative)
        # cost-to-target (§9.1) is the cost when the campaign first produces ANY
        # in-spec recipe on the real machine — the seed DoE and the (q-1) explore
        # runs count, not only the exploit pick (a space-filling hit IS a hit;
        # this also matches the BO baseline so the M2 comparison is fair).
        seed_hit = self._any_in_spec(Y)
        traj.per_batch_hit.append(seed_hit)
        if seed_hit:
            traj.hit = True
            traj.cost_to_target = cumulative
            traj.stop_reason = "target met in seed DoE"
            return traj

        surrogate = self.surrogate_factory().fit(X, Y)
        low_acq_streak = 0

        while traj.n_queries < self.budget:
            progress = traj.n_queries / max(self.budget, 1)
            lam = anneal(progress, 0.2, 0.9)
            beta = anneal(progress, 1.0, 0.0)

            # §5.7/§8 inner-loop cost: run the multi-start inverse + the EPIG/BALD
            # acquisition on the surrogate's FAST view (e.g. the deep-ensemble's
            # SNGP single member) when it offers one — the K-member mixture is not
            # run in the thousands-of-passes inner loop. The GP tier has no fast
            # view (hasattr False) so its behavior is byte-for-byte unchanged. The
            # exploit pick is fired on the REAL machine below, the ultimate
            # re-validation of what the fast surrogate proposed.
            inner = (
                surrogate.inner_loop_surrogate()
                if hasattr(surrogate, "inner_loop_surrogate")
                else surrogate
            )

            # §5.7/§13.2 re-validation: the fast view SCREENS, the FULL model is the
            # arbiter — solve() re-scores the selected candidates on it and drops any
            # the fast member certifies but the full ensemble (+ conformal C(x')⊆Z*)
            # rejects. Without this the gate is inert: the loop shipped whatever the
            # single SNGP member liked. When `inner is not surrogate` a fast/full
            # split genuinely exists, so this is UNCONDITIONAL (`surrogate` is the
            # freshly-refit full model every batch). The GP tier has no fast view
            # (`inner is surrogate`), so re-validation on the same model would be a
            # no-op; there it stays opt-in via the `revalidation_model` ctor arg
            # (default None ⇒ the historical path, byte-for-byte).
            revalidation_model = surrogate if inner is not surrogate else self.revalidation_model

            # NB `u_bound` is forwarded (audit 2026-07-17): it used to be declared
            # here, used for this loop's own Sobol pool, and then silently dropped
            # on the floor while every sibling knob was passed through — so the
            # solver fell back to its own default 8.0. Two consequences: a user
            # narrowing the search with `u_bound=3.0` got a solver still ranging
            # over u∈[−8,8]; and the M2 comparison was reach-asymmetric, since
            # `WarmStartedBO` uses u_bound=5.0 (the RIG exploit could reach
            # T_heater∈[1150.117, 1499.883] against the BO arm's [1152.342,
            # 1497.658]). Matched reach is a fairness requirement here.
            solver = PessimisticInverseSolver(
                inner,
                self.variables,
                self.output_keys,
                X_train=X,
                kappa=self.kappa,
                z_epi=self.z_epi,
                delta_frac=self.delta_frac,
                u_bound=self.u_bound,
                revalidation_model=revalidation_model,
                seed=self.seed,
            )
            res = solver.solve(self.spec)
            if isinstance(res, Infeasible):
                exploit_recipe = dict(res.nearest_achievable)
                star_recipes = [exploit_recipe]
            else:
                exploit_recipe = dict(res[0].recipe)
                star_recipes = [dict(c.recipe) for c in res]
            X_star = np.array([self._recipe_to_x(r) for r in star_recipes], dtype=float)

            # §9.4 cost-cooled explore acquisition over a fresh Sobol fill.
            pool_X, pool_recipes = self._pool(self.seed + 100 + traj.n_queries)
            acq = cost_cooled_acquisition(
                inner,
                pool_X,
                X_star,
                cost_fn=self.cost_recipe,
                recipes=pool_recipes,
                lam=lam,
                beta=beta,
            )
            max_acq = float(np.max(acq)) if acq.size else 0.0

            # batch = 1 exploit (the inverse's best) + (q-1) diverse explore picks.
            n_explore = max(self.q - 1, 0)
            explore_idx = select_batch(acq, pool_X, n_explore, model=inner)
            batch = [exploit_recipe] + [pool_recipes[i] for i in explore_idx]
            batch = batch[: max(self.budget - traj.n_queries, 1)]

            Xb, Yb, cost = self._query(batch)
            cumulative += cost
            traj.n_queries += len(batch)

            # ground-truth: did ANY recipe in this lot hit on the real machine?
            # (exploit or explore — the whole lot is fired and measured together).
            batch_hit = self._any_in_spec(Yb)
            traj.cumulative_cost.append(cumulative)
            traj.per_batch_hit.append(batch_hit)
            if batch_hit:
                traj.hit = True
                traj.cost_to_target = cumulative
                traj.stop_reason = "target met (proposal in-spec on machine)"
                return traj

            # refit on all data (§9.7 warm-start refit cadence).
            X = np.vstack([X, Xb])
            Y = np.vstack([Y, Yb])
            surrogate = self.surrogate_factory().fit(X, Y)

            # §9.7 acquisition-stall stop: max acq below eps for 2 batches.
            low_acq_streak = low_acq_streak + 1 if max_acq < self.stall_eps else 0
            if low_acq_streak >= 2:
                traj.stop_reason = "acquisition stall (max α < ε for 2 batches)"
                return traj

        traj.stop_reason = "budget exhausted"
        return traj
