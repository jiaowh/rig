"""WP-F: the closed active-learning loop (implementation-plan §9.2/§9.4/§9.7)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rig.active import ActiveLearningLoop
from rig.interfaces import ContinuousVariable, Infeasible, PredictiveDistribution


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
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
        **kw,
    )


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

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "_ToyModel":
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
