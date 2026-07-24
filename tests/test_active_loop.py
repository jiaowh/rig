"""WP-F: the closed active-learning loop (implementation-plan §9.2/§9.4/§9.7)."""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest

from rig.active import ActiveLearningLoop
from rig.active.campaign import ConfirmationCampaign
from rig.interfaces import ContinuousVariable, Infeasible, PredictiveDistribution
from rig.inverse import PessimisticInverseSolver


def _linear_machine(slope=3.0, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)

    def machine(recipe):
        return np.array([slope * recipe["x"] + noise * rng.standard_normal()])

    return machine


def _loop(spec, budget=40, slope=3.0, **kw):
    lo, hi = spec["targets"]["y"]

    def in_spec(y):
        return lo <= float(y[0]) <= hi

    return ActiveLearningLoop(
        machine=_linear_machine(slope=slope),
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec=spec,
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=budget,
        q=4,
        n_seed=8,
        n_pool=96,
        # kept on the permissive 1.0/1.0/0.01 ablation ON PURPOSE (F3, audit
        # 2026-07-21): these tests assert reachability/cost behaviour tuned for it, so
        # they pin it explicitly rather than ride the new binding-§8 2.0/2.0/0.02
        # default. test_loop_feasibility_defaults_match_solver_binding_policy guards
        # the DEFAULTS separately.
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
        **kw,
    )


def test_loop_feasibility_defaults_match_solver_binding_policy():
    # F3 (audit 2026-07-21): the loop's feasibility knobs must DEFAULT to the binding
    # §8 policy (kappa=z_epi=2.0, delta_frac=0.02) — IDENTICAL to
    # PessimisticInverseSolver's defaults — so a default-constructed loop searches under
    # the same conservatism as a direct solve. They previously drifted (loop 1.0/1.0/0.01
    # vs solver 2.0/2.0/0.02), which silently flips FEASIBLE/INFEASIBLE. Introspect the
    # signature defaults so the guard has no side effects (no machine runs fire).
    loop_defaults = inspect.signature(ActiveLearningLoop.__init__).parameters
    solver_defaults = inspect.signature(PessimisticInverseSolver.__init__).parameters
    for name in ("kappa", "z_epi", "delta_frac"):
        assert loop_defaults[name].default == solver_defaults[name].default, (
            f"{name}: loop default {loop_defaults[name].default} != solver default "
            f"{solver_defaults[name].default} — the two must not drift apart (F3)"
        )
    # pin the binding §8 values explicitly, so a matched-but-wrong drift is caught too.
    assert loop_defaults["kappa"].default == 2.0
    assert loop_defaults["z_epi"].default == 2.0
    assert loop_defaults["delta_frac"].default == 0.02


def test_loop_reaches_reachable_spec():
    # y = 3x on [0,1]; target [1.4,1.6] reachable at x~0.5.
    traj = _loop({"targets": {"y": (1.4, 1.6)}}).run()
    assert traj.hit is True
    assert math.isfinite(traj.cost_to_target)
    assert traj.cost_to_target <= 40
    assert "target met" in traj.stop_reason
    assert traj.cumulative_cost == sorted(traj.cumulative_cost)  # monotone


def test_loop_unreachable_spec_exhausts_budget_without_false_hit():
    # y = 3x maxes at 3 on [0,1]; target [10,11] is unreachable — the loop must
    # NOT report a spurious hit and must stop on budget/stall.
    traj = _loop({"targets": {"y": (10.0, 11.0)}}, budget=24).run()
    assert traj.hit is False
    assert math.isinf(traj.cost_to_target)
    assert traj.stop_reason in {
        "budget exhausted",
        "acquisition stall (max α < ε for 2 batches)",
    }
    assert not any(traj.per_batch_hit)


def test_loop_is_deterministic():
    a = _loop({"targets": {"y": (1.4, 1.6)}}).run()
    b = _loop({"targets": {"y": (1.4, 1.6)}}).run()
    assert a.hit == b.hit
    assert a.cost_to_target == b.cost_to_target
    assert a.n_queries == b.n_queries


def test_loop_records_trajectory_structure():
    traj = _loop({"targets": {"y": (1.4, 1.6)}}).run()
    assert len(traj.cumulative_cost) == len(traj.per_batch_hit)
    assert traj.n_queries >= 8  # at least the seed DoE
    assert traj.cumulative_cost[0] == 8.0  # 8 seed runs at $1 each


def test_loop_cost_to_target_pins_first_in_spec_batch():
    # instrument the machine to log every queried outcome, then assert
    # cost_to_target equals the cumulative cost of the LOT containing the first
    # in-spec run — seed or explore, not just the exploit pick (the review's HIGH
    # bug reported a later cost by only checking Yb[0]).
    log: list[float] = []

    def machine(recipe):
        y = 3.0 * recipe["x"]  # noise-free ⇒ deterministic ordering
        log.append(y)
        return np.array([y])

    def in_spec(y):
        return 1.4 <= float(y[0]) <= 1.6

    loop = ActiveLearningLoop(
        machine=machine,
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (1.4, 1.6)}},
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=40,
        q=4,
        n_seed=8,
        n_pool=96,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    traj = loop.run()
    assert traj.hit
    first = next(i for i, y in enumerate(log) if 1.4 <= y <= 1.6)
    # lot boundaries: [0,8) seed lot, then [8,12), [12,16), … lots of q=4.
    expected = 8.0 if first < 8 else 8.0 + 4.0 * ((first - 8) // 4 + 1)
    assert traj.cost_to_target == expected


def test_loop_credits_seed_doe_hit():
    # a broad spec the space-filling seed necessarily satisfies ⇒ the hit and its
    # cost are credited to the seed lot, not deferred to an optimization batch.
    traj = _loop({"targets": {"y": (0.0, 3.0)}}).run()
    assert traj.hit is True
    assert traj.per_batch_hit[0] is True
    assert traj.cost_to_target == 8.0  # the seed lot cost
    assert "seed DoE" in traj.stop_reason


def test_loop_stall_stop_when_acquisition_collapses(monkeypatch):
    # audit C8: the §9.7 acquisition-stall stop was never exercised (OOD BALD
    # keeps max α above stall_eps), and the unreachable test accepts either stop
    # reason. Force the acquisition to 0 so the stall branch actually fires, and
    # pin stop_reason == 'acquisition stall' with n_queries < budget.
    import rig.active.loop as loopmod

    monkeypatch.setattr(
        loopmod, "cost_cooled_acquisition", lambda model, X, X_star, **kw: np.zeros(len(X))
    )
    traj = _loop({"targets": {"y": (10.0, 11.0)}}, budget=40).run()  # unreachable
    assert traj.hit is False
    assert traj.stop_reason == "acquisition stall (max α < ε for 2 batches)"
    assert traj.n_queries < 40


def test_loop_c_batch_added_to_each_lot_cost():
    # audit D13: the fixed per-batch cost c_batch is added once per lot (incl. the
    # seed DoE). Every non-gated test uses c_batch=0 so this was untested off-sim,
    # yet both shipping adapters feed c_batch=1000 — a regression would corrupt
    # the WP-G/M2 cost-to-target metric.
    lo, hi = 1.4, 1.6
    loop = ActiveLearningLoop(
        machine=_linear_machine(slope=3.0),
        in_spec=lambda y: lo <= float(y[0]) <= hi,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (lo, hi)}},
        cost_recipe=lambda r: 1.0,
        c_batch=5.0,
        budget=40,
        q=4,
        n_seed=8,
        n_pool=96,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    traj = loop.run()
    assert traj.cumulative_cost[0] == 8 * 1.0 + 5.0  # seed lot: n_seed*cost + c_batch
    for prev, cur in zip(traj.cumulative_cost, traj.cumulative_cost[1:], strict=False):
        assert cur - prev == 5.0 + 4 * 1.0  # each later lot: c_batch + q*cost


# --- §5.7/§13.2 re-validation wiring + §9.2 seed-vs-budget guards -------------


class _ToyModel:
    """A constant-mean ForwardModel stub (§3.2 surface) for the wiring guards.

    ``mean`` is out-of/in-spec by construction; support is a flat 0.0 so the §8.2
    floor never rejects and only the spec margin decides feasibility. delta_frac=0
    on the loop keeps ``jacobian`` off the hot path, so it is not needed here."""

    def __init__(self, mean: float) -> None:
        self._mean = float(mean)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> _ToyModel:
        return self

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mu = np.full((Xq.shape[0], 1), self._mean)
        ale = np.full_like(mu, 0.01)
        epi = np.zeros_like(mu)
        if single:
            mu, ale, epi = mu[0], ale[0], epi[0]
        return PredictiveDistribution(
            mean=mu, aleatoric_sigma=ale, epistemic_sigma=epi, conformal_set=None
        )

    def support_score(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        return 0.0 if x.ndim == 1 else np.zeros(x.shape[0])


class _ToyEnsemble(_ToyModel):
    """Full model whose mean is OUT of spec, exposing a distinct fast inner-loop
    view whose mean is IN spec — the fast/full divergence §13.2 re-validation must
    catch. Mirrors ``DeepEnsembleForwardModel.inner_loop_surrogate``'s contract
    (a distinct object identity for the fast view)."""

    def __init__(self) -> None:
        super().__init__(mean=5.0)  # 5.0 is outside [1.4, 1.6]
        self.fast_view = _ToyModel(mean=1.5)  # 1.5 is inside [1.4, 1.6]

    def inner_loop_surrogate(self) -> _ToyModel:
        return self.fast_view


def _wiring_loop(machine, *, budget, n_seed, surrogate_factory):
    return ActiveLearningLoop(
        machine=machine,
        in_spec=lambda y: 1.4 <= float(y[0]) <= 1.6,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (1.4, 1.6)}},
        budget=budget,
        q=4,
        n_seed=n_seed,
        n_pool=16,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.0,
        surrogate_factory=surrogate_factory,
        seed=0,
    )


def test_ensemble_fast_view_wires_full_model_as_revalidation(monkeypatch):
    # Finding A: on the ensemble fast-view path the solver was handed the FAST
    # SNGP member view but never a revalidation_model, so the §13.2 C(x')⊆Z*
    # re-validation gate was inert — a candidate the fast member certifies was
    # returned even when the full ensemble rejects it. Assert (1) the solver
    # receives the FULL model as revalidation_model (identity), and (2) the
    # explicit fast-certifies / full-rejects divergence is caught (Infeasible).
    import rig.active.loop as loopmod

    captured: dict = {}
    real_solver = loopmod.PessimisticInverseSolver

    def spy(model, variables, output_keys, **kw):
        captured["model"] = model
        captured["reval"] = kw.get("revalidation_model")
        solver = real_solver(model, variables, output_keys, **kw)
        orig_solve = solver.solve

        def solve(spec):
            res = orig_solve(spec)
            captured["result"] = res
            return res

        solver.solve = solve
        return solver

    monkeypatch.setattr(loopmod, "PessimisticInverseSolver", spy)
    # isolate the wiring from the acquisition machinery (the toy has no
    # posterior_cov); mirrors test_loop_stall_stop_when_acquisition_collapses.
    monkeypatch.setattr(loopmod, "cost_cooled_acquisition", lambda *a, **k: np.zeros(len(a[1])))
    monkeypatch.setattr(
        loopmod, "select_batch", lambda acq, X, n, **kw: list(range(min(n, len(X))))
    )

    full = _ToyEnsemble()
    _wiring_loop(
        lambda r: np.array([5.0]),  # never in spec ⇒ the loop reaches the solver
        budget=6,
        n_seed=4,
        surrogate_factory=lambda: full,
    ).run()

    assert captured["model"] is full.fast_view  # the inner loop ran on the fast view
    assert captured["reval"] is full  # ...and the FULL model re-validates (identity)
    assert isinstance(captured["result"], Infeasible)  # divergence caught, not certified


def test_seed_doe_exceeding_budget_raises_before_any_machine_call():
    # Finding B: n_seed > budget silently fired n_seed real machine runs (the seed
    # DoE ignored the declared budget). Fail loud instead, and fire NOTHING.
    calls = {"n": 0}

    def counting_machine(r):
        calls["n"] += 1
        return np.array([5.0])

    with pytest.raises(ValueError, match=r"(?s)budget.*4.*n_seed.*8|n_seed.*8.*budget.*4"):
        _wiring_loop(
            counting_machine, budget=4, n_seed=8, surrogate_factory=lambda: _ToyModel(5.0)
        ).run()
    assert calls["n"] == 0  # not one machine run fired


# ---------------------------------------------------------------------------
# Opt-in qualification hook (audit F2 remainder, 2026-07-22): a `qualification=
# ConfirmationCampaign(...)` requires independent confirmation before a
# ground-truth in-spec observation is allowed to stand as a "target met" stop.
# ---------------------------------------------------------------------------


def test_loop_qualification_defaults_to_none():
    # The hook must default to None so every caller unaware of it gets
    # byte-identical behavior. Introspect the signature (no side effects, no
    # machine runs fire) -- mirrors
    # test_loop_feasibility_defaults_match_solver_binding_policy's style.
    default = inspect.signature(ActiveLearningLoop.__init__).parameters["qualification"].default
    assert default is None


def test_loop_qualification_none_is_byte_identical_to_no_param():
    # Reuses test_loop_reaches_reachable_spec's exact setup: a seeded run with
    # `qualification` omitted vs explicitly passed as `qualification=None` must
    # produce an IDENTICAL Trajectory (dataclass `==`, every field) -- proof
    # the hook is a true no-op until configured.
    spec = {"targets": {"y": (1.4, 1.6)}}
    a = _loop(spec).run()
    b = _loop(spec, qualification=None).run()
    assert a == b
    assert a.qualification_outcome is None
    assert a.qualification_rejections == []


def _pass_qualification_loop(*, budget=48, n_runs=5, gate_seed=0):
    """A loop whose whole seed lot hits by construction (constant machine at
    1.5, target [1.4, 1.6]), wired to a ConfirmationCampaign whose verifier
    ALWAYS certifies (same constant 1.5). Returns (loop, verifier_call_counter).
    """
    calls = {"n": 0}

    def verifier(recipe):
        calls["n"] += 1
        return {"y": 1.5}

    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=gate_seed,
        machine=verifier,
        gate_params=dict(
            targets={"y": (1.4, 1.6)},
            n_runs=n_runs,
            min_in_spec_rate=0.5,
            confidence=0.8,
            provenance_source="physics_sim",
        ),
    )
    loop = ActiveLearningLoop(
        machine=lambda recipe: np.array([1.5]),  # constant: every seed recipe is in spec
        in_spec=lambda y: 1.4 <= float(y[0]) <= 1.6,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (1.4, 1.6)}},
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=budget,
        q=4,
        n_seed=8,
        n_pool=16,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
        qualification=campaign,
    )
    return loop, calls


def test_loop_qualification_pass_declares_hit_and_charges_budget_for_confirmation_runs():
    loop, calls = _pass_qualification_loop()
    traj = loop.run()

    assert traj.hit is True
    assert traj.stop_reason == "target met in seed DoE (qualified)"
    assert traj.qualification_outcome is not None
    # constant machine ⇒ all 8 seed recipes are in-spec ⇒ all 8 are submitted
    # for confirmation, and the always-certifying verifier passes every one.
    assert traj.qualification_outcome.n_candidates == 8
    assert traj.qualification_outcome.n_certified == 8
    assert traj.qualification_rejections == []
    # budget honesty: 8 seed queries + (8 hitting candidates * 5 confirmation
    # runs each) = 48 -- exact, not an estimate.
    assert calls["n"] == 8 * 5
    assert traj.n_queries == 8 + 8 * 5 == 48
    assert traj.cost_to_target == 8.0  # unaffected by qualification's own cost


def test_loop_qualification_pass_path_is_deterministic():
    loop_a, _ = _pass_qualification_loop()
    loop_b, _ = _pass_qualification_loop()
    a = loop_a.run()
    b = loop_b.run()

    assert a.hit is True
    assert a.hit == b.hit
    assert a.n_queries == b.n_queries
    assert a.cost_to_target == b.cost_to_target
    assert a.stop_reason == b.stop_reason
    assert a.qualification_outcome == b.qualification_outcome
    a_json = [r.model_dump_json() for r in a.qualification_outcome.all_run_records]
    b_json = [r.model_dump_json() for r in b.qualification_outcome.all_run_records]
    assert a_json == b_json


def test_loop_qualification_rejection_does_not_stop_the_loop():
    # The search machine gets "lucky" ONCE -- the whole seed lot lands in-spec
    # by construction -- but every later query is deliberately far out of
    # spec. Before this hook, the loop would have declared a hit on the raw
    # ground-truth observation; the qualification hook must refuse to declare
    # it (the gate always rejects) and keep the loop running.
    call_count = {"n": 0}

    def machine(recipe):
        call_count["n"] += 1
        if call_count["n"] <= 8:  # the whole seed lot: constant, in spec
            return np.array([1.5])
        return np.array([999.0])  # every later query: far out of spec

    def in_spec(y):
        return 1.4 <= float(y[0]) <= 1.6

    reject_calls = {"n": 0}

    def always_reject_verifier(recipe):
        reject_calls["n"] += 1
        return {"y": 999.0}  # never in [1.4, 1.6] -> the gate always rejects

    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=always_reject_verifier,
        gate_params=dict(
            targets={"y": (1.4, 1.6)},
            n_runs=3,
            min_in_spec_rate=0.5,
            confidence=0.8,
            provenance_source="physics_sim",
        ),
    )
    loop = ActiveLearningLoop(
        machine=machine,
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (1.4, 1.6)}},
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=40,
        q=4,
        n_seed=8,
        n_pool=16,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
        qualification=campaign,
    )
    traj = loop.run()

    assert traj.hit is False  # the lucky seed-lot hit was never confirmed
    assert traj.cost_to_target == float("inf")
    assert traj.per_batch_hit[0] is True  # ground truth: the seed lot DID hit
    assert len(traj.qualification_rejections) == 1
    assert traj.qualification_rejections[0].n_certified == 0
    assert traj.qualification_rejections[0].n_candidates == 8
    assert traj.qualification_outcome is None
    assert traj.stop_reason in {
        "budget exhausted",
        "unqualified hit, budget exhausted",
        "acquisition stall (max α < ε for 2 batches)",
    }
    assert traj.n_queries <= 40
    # exactly one confirmation attempt fired: 8 candidates * 3 runs each.
    assert reject_calls["n"] == 8 * 3


def test_loop_qualification_budget_exhausted_does_not_fire_gate():
    # F2 remainder budget honesty: firing the confirmation batch would spend
    # more real machine calls than remain, so the loop must refuse to fire it
    # at all (zero additional calls) and stop with a DISTINCT stop_reason.
    calls = {"n": 0}

    def verifier(recipe):
        calls["n"] += 1
        return {"y": 1.5}  # would trivially certify if it ever ran

    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=verifier,
        gate_params=dict(
            targets={"y": (0.0, 3.0)},
            n_runs=5,
            min_in_spec_rate=0.5,
            confidence=0.8,
            provenance_source="physics_sim",
        ),
    )
    # budget == n_seed (8) exactly: zero machine-run budget remains after the
    # seed lot, so ANY confirmation run at all would overspend.
    traj = _loop({"targets": {"y": (0.0, 3.0)}}, budget=8, qualification=campaign).run()

    assert traj.hit is False
    assert traj.stop_reason == "unqualified hit, budget exhausted"
    assert calls["n"] == 0  # the gate never fired
    assert traj.qualification_outcome is None
    assert traj.qualification_rejections == []
    assert traj.n_queries == 8  # unchanged by the refused qualification attempt
