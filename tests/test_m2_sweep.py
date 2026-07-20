"""M2 sweep harness (implementation-plan §12.3): the powered RIG-vs-BO cost-to-target
evaluator. CI-safe synthetic machine; asserts the HARNESS is valid (well-formed
panel, correct direction on a deterministic stub) — NOT a flaky "RIG always wins"
(that is the empirical result, produced by examples/run_m2_sweep.py)."""

from __future__ import annotations

import math

import numpy as np

from rig.active import ActiveLearningLoop
from rig.baselines import WarmStartedBO
from rig.eval.m2_sweep import Target, run_m2_sweep
from rig.interfaces import ContinuousVariable

# -- deterministic direction check on stub methods (no GP flakiness) ---------------


class _Stub:
    def __init__(self, time: float, hit: bool) -> None:
        self.hit = hit
        self.cost_to_target = time if hit else float("inf")
        self.cumulative_cost = [time]
        self.n_queries = 8

    def run(self) -> _Stub:
        return self


def test_sweep_math_recovers_direction_and_is_self_consistent():
    # 'fast' always reaches at cost 8, 'slow' at cost 20 => fast is cheaper, so
    # ΔRMST = RMST_fast - RMST_slow must be negative, win-rate 1.0. All-events at
    # a single time t makes RMST == t (area under the KM step).
    def fast(**_kw):
        return _Stub(8.0, True)

    def slow(**_kw):
        return _Stub(20.0, True)

    targets = [
        Target("t1", {"targets": {"y": (0.0, 1.0)}}, lambda y: True),
        Target("t2", {"targets": {"y": (0.0, 1.0)}}, lambda y: True),
    ]
    rep = run_m2_sweep(
        make_machine=lambda seed: lambda recipe: np.array([0.0]),
        methods={"fast": fast, "slow": slow},
        targets=targets,
        seeds=range(6),
        horizon=24.0,
        n_bootstrap=500,
        bootstrap_seed=0,
    )
    assert rep.reference == "fast" and rep.other == "slow"
    assert rep.pooled_rmst["fast"] == 8.0 and rep.pooled_rmst["slow"] == 20.0
    assert rep.pooled_delta_rmst == -12.0  # 8 - 20
    assert rep.pooled_win_rate == 1.0 and rep.pooled_tie_rate == 0.0
    assert rep.prob_reference_better == 1.0
    lo, hi = rep.bootstrap_ci95
    assert lo <= -12.0 <= hi and hi < 0.0  # CI entirely below 0 (fast better)
    assert "cheaper" in rep.verdict
    assert len(rep.campaigns) == 2 * 2 * 6  # methods * targets * seeds


# -- small real RIG-vs-BO smoke: the panel is well-formed --------------------------


def _machine_factory(seed: int):
    rng = np.random.default_rng(seed)

    def machine(recipe):
        x = recipe["x"]
        return np.array([2.0 + 1.5 * np.sin(5.0 * x) + 0.02 * rng.standard_normal()])

    return machine


def _rig(*, machine, in_spec, spec, seed):
    return ActiveLearningLoop(
        machine=machine,
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec=spec,
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=20,
        q=4,
        n_seed=8,
        n_pool=48,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=seed,
    )


def _bo(*, machine, in_spec, spec, seed):
    return WarmStartedBO(
        machine=machine,
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec=spec,
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=20,
        q=4,
        n_seed=8,
        n_pool=48,
        seed=seed,
    )


def _target(tid, lo, hi):
    return Target(
        tid, {"targets": {"y": (lo, hi)}}, lambda y, lo=lo, hi=hi: lo <= float(y[0]) <= hi
    )


def test_m2_sweep_produces_well_formed_panel():
    rep = run_m2_sweep(
        make_machine=_machine_factory,
        methods={"rig": _rig, "bo": _bo},
        targets=[_target("mid", 2.6, 3.0), _target("upper", 3.0, 3.4)],
        seeds=range(5),
        horizon=20.0,
        n_bootstrap=400,
        bootstrap_seed=1,
    )
    assert rep.methods == ["rig", "bo"]
    assert len(rep.per_target) == 2
    for tv in rep.per_target:
        for m in ("rig", "bo"):
            assert math.isfinite(tv.rmst[m]) and tv.rmst[m] > 0
            assert 0.0 <= tv.hit_rate[m] <= 1.0
        # self-consistency: delta is RMST[ref] - RMST[other]
        assert tv.delta_rmst == tv.rmst["rig"] - tv.rmst["bo"]
    # pooled panel is finite / valid (no performance claim — harness validity only)
    assert 0.0 <= rep.pooled_p_value <= 1.0
    assert math.isfinite(rep.pooled_delta_rmst) and math.isfinite(rep.pooled_delta_se)
    lo, hi = rep.bootstrap_ci95
    assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi
    assert 0.0 <= rep.prob_reference_better <= 1.0
    assert len(rep.campaigns) == 2 * 2 * 5
    # determinism: identical config reproduces the pooled statistic exactly.
    rep2 = run_m2_sweep(
        make_machine=_machine_factory,
        methods={"rig": _rig, "bo": _bo},
        targets=[_target("mid", 2.6, 3.0), _target("upper", 3.0, 3.4)],
        seeds=range(5),
        horizon=20.0,
        n_bootstrap=400,
        bootstrap_seed=1,
    )
    assert rep2.pooled_delta_rmst == rep.pooled_delta_rmst
