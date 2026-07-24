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

Opt-in qualification hook (audit F2 remainder, 2026-07-22): construct with
``qualification=<a rig.active.campaign.ConfirmationCampaign>`` to require
independent confirmation-batch certification before a ground-truth in-spec
observation is allowed to stand as a "target met" stop -- see
:class:`ActiveLearningLoop`'s own docstring for the full hook semantics.
Default ``qualification=None`` is byte-identical to every prior release of
this module (test_loop_qualification_none_is_byte_identical_to_no_param).
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
from rig.active.campaign import CampaignResult, ConfirmationCampaign
from rig.forward.gp import GPForwardModel
from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
    Infeasible,
    RecipeCandidate,
)
from rig.inverse import PessimisticInverseSolver
from rig.transforms import RecipeTransform


@dataclass
class Trajectory:
    """One campaign's cost-to-target trajectory (feeds the WP-G survival stats).

    ``qualification_outcome`` / ``qualification_rejections`` are ADDITIVE
    fields (default ``None`` / ``[]``): populated only when the owning
    ``ActiveLearningLoop`` was constructed with ``qualification=<a
    ConfirmationCampaign>`` (audit F2 remainder, 2026-07-22 -- see that
    constructor argument's own docstring for the full hook semantics). Every
    existing consumer that does not know about these two fields keeps
    working unmodified, including the byte-identical ``qualification=None``
    default path, which never touches either one.
    """

    hit: bool
    cost_to_target: float  # inf if the spec was not hit within budget
    n_queries: int
    cumulative_cost: list[float] = field(default_factory=list)  # after each batch
    per_batch_hit: list[bool] = field(default_factory=list)
    stop_reason: str = ""
    # -- F2 remainder: opt-in qualification hook, both additive/optional --
    qualification_outcome: CampaignResult | None = None  # CampaignResult that certified `hit`
    qualification_rejections: list[CampaignResult] = field(
        default_factory=list
    )  # rejected attempts


class ActiveLearningLoop:
    """Closed-loop cost-to-target campaign for ONE spec (§9). Drives ``machine``
    (the real tool ``M``), never differentiates it: gradients flow only through
    the refit surrogate (the standard offline-MBO posture, §3.2).

    Opt-in qualification hook (audit F2 remainder, 2026-07-22)
    ------------------------------------------------------------
    ``rig.active.campaign.ConfirmationCampaign`` wraps
    ``rig.qualification.ConfirmationBatchGate`` into a provenance-logged
    confirmation-batch orchestrator, but nothing called it automatically:
    this loop declared a hit the instant the real machine's outcome landed
    in-spec, with NO independent confirmation -- "a successful solve is a
    recommendation, not a validated recipe" (root ``audit.md``, F2). The
    ``qualification`` constructor argument closes that gap, OPT-IN:

    ``qualification=None`` (the default)
        Byte-identical to every prior release of this class: a ground-truth
        in-spec observation (seed DoE or in-loop batch) is declared a hit
        immediately, exactly as before -- proven by
        ``test_loop_qualification_none_is_byte_identical_to_no_param``.
    ``qualification=<a ConfirmationCampaign>``
        At EVERY point this loop would otherwise declare ``hit=True`` and
        return with a "target met" ``stop_reason`` -- the seed-DoE early
        return AND the in-loop per-batch hit, identically -- every recipe
        that measured in-spec IN THAT LOT is first wrapped as a
        ``RecipeCandidate`` (see ``_hitting_candidates``; only ``.recipe``
        is ever read downstream) and run through
        ``qualification.run(candidates)`` before the hit is allowed to
        stand:

        * **certified** (>=1 candidate passes): the hit stands exactly as
          in the ``None`` path (``hit=True``, ``cost_to_target`` = this
          lot's cumulative SEARCH cost -- unaffected by qualification cost,
          see below), plus the ``CampaignResult`` is attached to the new
          ``Trajectory.qualification_outcome`` field and ``stop_reason``
          gets a "(qualified)" suffix.
        * **rejected** (every hitting candidate fails confirmation): the
          hit is NOT declared -- ``hit`` stays False and the loop does NOT
          stop. The ``CampaignResult`` is appended to
          ``Trajectory.qualification_rejections`` (there can be more than
          one over a campaign's lifetime). The in-spec observation still
          counts as DATA: it was already queried and already sits in
          ``X``/``Y`` before qualification ever runs, the surrogate refits
          on it, and the search continues exactly as if ``_any_in_spec``
          had returned False for this lot -- only the STOP decision is
          gated, never the data.
        * **budget-exhausted**: see "Budget honesty" below -- a distinct
          terminal ``stop_reason``, no gate fired.

    Budget honesty
        A confirmation run is a REAL machine query exactly like a seed-DoE
        or exploit/explore query, so it is charged against the SAME
        ``budget``/``n_queries`` accounting, never a separate pool. Before
        firing, the EXACT cost is computed as ``n_hitting_candidates *
        n_runs`` (every ``ConfirmationBatchGate.certify`` call fires
        precisely ``n_runs`` verifier calls with no early exit, so this is
        exact, not an estimate -- ``_expected_qualification_calls``). If
        ``traj.n_queries + that_cost`` would exceed ``budget``, NOTHING is
        fired (a confirmation run cannot be un-fired) and the loop stops
        immediately with the distinct ``stop_reason`` "unqualified hit,
        budget exhausted" (``hit`` stays False). Otherwise the campaign
        runs and ``traj.n_queries`` is charged for exactly
        ``CampaignResult.n_machine_calls`` afterward -- in BOTH the
        certified and the rejected case (a rejected confirmation batch
        still spent real machine time). ``cost_to_target`` (the $ metric,
        via ``cost_recipe``/``c_batch``) is deliberately NOT inflated by
        qualification cost: it keeps measuring the SEARCH's cost to find
        the candidate, a distinct question from the cost of independently
        validating it -- the validation cost is visible instead via
        ``n_queries`` and ``qualification_outcome.n_machine_calls``.

    Multi-fire safety: ``ConfirmationCampaign`` salts every ``run_id`` and
    timestamp with a per-instance invocation counter (fixed 2026-07-22), so
    firing the SAME ``qualification`` instance more than once over one
    loop's lifetime (e.g. a rejected seed-DoE hit followed by a later
    in-loop hit) can never produce colliding ``RunRecord`` ids, while
    replaying the whole loop still reproduces the same id sequence
    call-for-call.
    """

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
        # Opt-in F2 remainder (audit 2026-07-22): when set, every target-met
        # stop (seed DoE AND in-loop) first requires independent confirmation
        # via this campaign before the hit is allowed to stand. Default None
        # is byte-identical to every prior release -- see the class docstring.
        qualification: ConfirmationCampaign | None = None,
        # §8 binding feasibility policy (F3, audit 2026-07-21): kappa=z_epi=2.0,
        # delta_frac=0.02 — IDENTICAL to PessimisticInverseSolver's defaults, so a loop
        # built with defaults searches under the same conservatism as a direct solve.
        # These previously defaulted to the more permissive 1.0/1.0/0.01 ablation, which
        # silently flipped FEASIBLE/INFEASIBLE relative to the binding policy. The two
        # default sets are pinned equal by test_active_loop's
        # test_loop_feasibility_defaults_match_solver_binding_policy so they cannot drift.
        kappa: float = 2.0,
        z_epi: float = 2.0,
        delta_frac: float = 0.02,
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
        self.qualification = qualification
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

    # -- opt-in qualification hook (audit F2 remainder, 2026-07-22) -------------
    # Only ever called when `self.qualification is not None` AND a lot's
    # ground-truth `_any_in_spec` already came back True -- see the class
    # docstring for the full hook semantics (budget arithmetic, stop reasons,
    # what stays byte-identical).

    def _hitting_candidates(
        self, recipes: Sequence[Mapping[str, float]], Y: np.ndarray
    ) -> list[RecipeCandidate]:
        """Wrap every in-spec recipe from one lot as a RecipeCandidate.

        Only ``.recipe`` is ever read downstream by ConfirmationCampaign /
        ConfirmationBatchGate.certify -- certification never consults
        confidence/support_score/feasibility_flag (D7 non-circularity,
        rig.qualification's own module docstring) -- so the other
        RecipeCandidate fields below are uninformative placeholders, not a
        model opinion: this loop has no surrogate to report one from at the
        seed-DoE stage, and even in-loop it was the machine's MEASUREMENT,
        not a prediction, that triggered this call.
        """
        return [
            RecipeCandidate(
                recipe=dict(recipes[i]),
                confidence=1.0,
                predicted_outcome_interval=None,
                feasibility_flag=True,
                support_score=0.0,
            )
            for i in range(Y.shape[0])
            if bool(self.in_spec(Y[i]))
        ]

    def _expected_qualification_calls(self, n_candidates: int) -> int | None:
        """Predict exactly how many real machine calls firing
        ``self.qualification`` on ``n_candidates`` recipes will cost --
        checked BEFORE firing, because a confirmation run cannot be un-fired.

        ``ConfirmationCampaign`` exposes no public "cost of running this"
        accessor, so this reads its gate configuration directly: exactly one
        of a pre-built ``gate`` or ``gate_params`` is always set
        (``ConfirmationCampaign.__init__`` enforces it), and
        ``ConfirmationBatchGate.certify`` fires EXACTLY ``n_runs`` verifier
        calls per candidate with no early exit -- so
        ``n_candidates * n_runs`` is exact, never an estimate, whenever
        ``n_runs`` is resolvable this way. Returns None on the well-formed-
        but-unresolvable edge case where ``gate_params`` omits ``n_runs``
        (the caller treats None as fail-closed, i.e. as if it WOULD
        overspend -- this loop has no business guessing a confirmation-batch
        size).
        """
        if n_candidates == 0:
            return 0
        campaign = self.qualification
        assert campaign is not None
        gate = getattr(campaign, "_static_gate", None)
        if gate is not None:
            n_runs = getattr(gate, "_n_runs", None)
        else:
            params = getattr(campaign, "_gate_params", None) or {}
            n_runs = params.get("n_runs")
        if n_runs is None:
            return None
        return int(n_candidates) * int(n_runs)

    def _qualify_hit(
        self,
        traj: Trajectory,
        cumulative: float,
        recipes: Sequence[Mapping[str, float]],
        Y: np.ndarray,
        hit_stop_reason: str,
    ) -> bool:
        """Attempt confirmation of one lot's in-spec recipe(s) (F2 remainder).

        Mutates ``traj`` and returns True iff the loop should STOP now:

        * budget cannot afford the confirmation batch -> fires NOTHING,
          ``stop_reason`` = "unqualified hit, budget exhausted", ``hit``
          stays False. Returns True (STOP).
        * >=1 hitting recipe is certified -> ``hit=True``,
          ``cost_to_target=cumulative``, ``stop_reason=hit_stop_reason``,
          ``qualification_outcome`` carries the CampaignResult. Returns
          True (STOP).
        * every hitting recipe is rejected -> the CampaignResult is appended
          to ``qualification_rejections``, ``hit`` stays False. Returns
          False (CONTINUE): the caller must fall through to the ordinary
          non-hit path exactly as if ``_any_in_spec`` had returned False for
          this lot -- the in-spec observation still counts as DATA (X/Y
          already include it), only the STOP decision is gated.

        In every branch ``traj.n_queries`` is charged for exactly the
        confirmation runs actually fired (zero in the budget-exhausted
        branch, since none are fired there) -- budget honesty (the loop's
        ``budget`` is a count of real machine queries, and a confirmation
        run is one).
        """
        assert self.qualification is not None
        candidates = self._hitting_candidates(recipes, Y)
        expected = self._expected_qualification_calls(len(candidates))
        if expected is None or traj.n_queries + expected > self.budget:
            traj.stop_reason = "unqualified hit, budget exhausted"
            return True
        outcome = self.qualification.run(candidates)
        # `candidates` is a non-empty list[RecipeCandidate], never Infeasible,
        # so ConfirmationCampaign.run always takes the CampaignResult branch
        # here (NothingToQualify is only ever returned for an Infeasible
        # input -- see that class's own run() docstring).
        assert isinstance(outcome, CampaignResult)
        traj.n_queries += outcome.n_machine_calls
        if outcome.n_certified > 0:
            traj.hit = True
            traj.cost_to_target = cumulative
            traj.qualification_outcome = outcome
            traj.stop_reason = hit_stop_reason
            return True
        traj.qualification_rejections.append(outcome)
        return False

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
            if self.qualification is None:
                traj.hit = True
                traj.cost_to_target = cumulative
                traj.stop_reason = "target met in seed DoE"
                return traj
            if self._qualify_hit(
                traj, cumulative, seed_recipes, Y, "target met in seed DoE (qualified)"
            ):
                return traj
            # rejected: fall through, continuing exactly as if seed_hit had
            # been False -- the seed observations still count as data below.

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
                if self.qualification is None:
                    traj.hit = True
                    traj.cost_to_target = cumulative
                    traj.stop_reason = "target met (proposal in-spec on machine)"
                    return traj
                if self._qualify_hit(
                    traj,
                    cumulative,
                    batch,
                    Yb,
                    "target met (proposal in-spec on machine, qualified)",
                ):
                    return traj
                # rejected: fall through to the refit + stall-check below,
                # continuing exactly as if batch_hit had been False.

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
