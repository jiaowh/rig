"""WP-G: inverse-recipe metrics + diversity (implementation-plan §12.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rig.constraints import BoxConstraint, ConstraintSet
from rig.eval.diversity import mean_pairwise_l2, mode_count, vendi_score
from rig.eval.inverse_metrics import (
    TargetOutcome,
    constraint_satisfaction_rate,
    false_abstention_rate,
    false_success_rate,
    feasibility_flag_accuracy,
    robust_hit_rate,
    success_rate_at_budget,
    target_hit_rate,
)


def _o(tid, feasible, declared_infeasible, hit, cost):
    return TargetOutcome(tid, feasible, declared_infeasible, hit, cost)


def _sample_outcomes():
    return [
        _o("f_hit", True, False, True, 3.0),  # feasible, hit cheaply
        _o("f_hit2", True, False, True, 8.0),  # feasible, hit later
        _o("f_miss", True, False, False, 10.0),  # feasible, unhit (censored)
        _o("f_refused", True, True, False, 10.0),  # feasible but WRONGLY refused
        _o("inf_ok", False, True, False, 10.0),  # infeasible, correctly refused
        _o("inf_bad", False, False, True, 5.0),  # infeasible but "hit" = a bug
    ]


def test_target_hit_rate_feasible_only():
    outs = _sample_outcomes()
    # feasible targets: f_hit, f_hit2, f_miss, f_refused -> 2/4 hit
    assert target_hit_rate(outs) == pytest.approx(0.5)
    # including infeasible: hits = f_hit, f_hit2, inf_bad = 3/6
    assert target_hit_rate(outs, feasible_only=False) == pytest.approx(0.5)


def test_success_rate_at_budget():
    outs = _sample_outcomes()
    # feasible hits within budget 5: only f_hit (cost 3) -> 1/4
    assert success_rate_at_budget(outs, 5.0) == pytest.approx(0.25)
    # budget 8: f_hit + f_hit2 -> 2/4
    assert success_rate_at_budget(outs, 8.0) == pytest.approx(0.5)


def test_false_success_rate_flags_infeasible_hits():
    outs = _sample_outcomes()
    # infeasible targets: inf_ok, inf_bad; false success = inf_bad -> 1/2
    assert false_success_rate(outs) == pytest.approx(0.5)


def test_false_abstention_rate():
    outs = _sample_outcomes()
    # feasible targets: 4; wrongly declared infeasible = f_refused -> 1/4
    assert false_abstention_rate(outs) == pytest.approx(0.25)


def test_feasibility_flag_accuracy():
    outs = _sample_outcomes()
    # correct flags: f_hit(no), f_hit2(no), f_miss(no), f_refused(WRONG),
    # inf_ok(yes-correct), inf_bad(no-WRONG) -> correct = 4/6
    assert feasibility_flag_accuracy(outs) == pytest.approx(4.0 / 6.0)


def test_contradictory_outcome_rejected():
    with pytest.raises(ValueError):
        _o("bad", True, True, True, 1.0)  # declared infeasible AND hit


def test_metrics_return_nan_on_empty_pool():
    only_feasible = [_o("f", True, False, True, 1.0)]
    assert math.isnan(false_success_rate(only_feasible))  # no infeasible targets
    only_infeasible = [_o("i", False, True, False, 1.0)]
    assert math.isnan(target_hit_rate(only_infeasible))  # no feasible targets


def test_constraint_satisfaction_rate():
    cs = ConstraintSet(box=(BoxConstraint("x", 0.0, 1.0),))
    recipes = [{"x": 0.5}, {"x": 0.9}, {"x": 1.5}]  # last violates
    assert constraint_satisfaction_rate(recipes, cs) == pytest.approx(2.0 / 3.0)


def test_robust_hit_rate_rewards_flat_basins():
    # machine y = x; tolerance |y-5|<=0.5. A recipe at x=5 with small actuation
    # noise stays in tol most of the time; large noise falls out more often.
    def machine(r):
        return np.array([r["x"]])

    def in_tol(y):
        return abs(float(y[0]) - 5.0) <= 0.5

    tight = robust_hit_rate(
        {"x": 5.0}, machine, in_tol, actuation_noise={"x": 0.1}, n_samples=200, seed=0
    )
    loose = robust_hit_rate(
        {"x": 5.0}, machine, in_tol, actuation_noise={"x": 1.0}, n_samples=200, seed=0
    )
    assert tight > loose
    assert tight > 0.95  # 0.1 noise rarely exceeds the 0.5 tolerance


# ---------------------------------------------------------------------------
# diversity
# ---------------------------------------------------------------------------


def test_vendi_identical_is_one():
    assert vendi_score(np.ones((8, 3))) == pytest.approx(1.0, abs=1e-6)


def test_vendi_scale_invariant():
    # standardized + median-bandwidth kernel ⇒ Vendi is invariant to global
    # rescaling (a deliberate feature for heterogeneous-unit recipes).
    rng = np.random.default_rng(0)
    base = rng.standard_normal((12, 2))
    assert vendi_score(base * 5.0) == pytest.approx(vendi_score(base * 0.5), rel=1e-6)


def test_vendi_higher_for_spread_than_clustered():
    # structural monotonicity: 3 tight clusters (effective ~3) vs 12 spread
    # points (effective higher). This is what Vendi is meant to distinguish.
    rng = np.random.default_rng(0)
    centres = rng.standard_normal((3, 2)) * 10.0
    clustered = np.repeat(centres, 4, axis=0) + 0.01 * rng.standard_normal((12, 2))
    spread = rng.standard_normal((12, 2))
    assert vendi_score(spread) > vendi_score(clustered)
    assert vendi_score(clustered) < 5.0  # ~3 effective clusters


def test_vendi_bounds():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 3))
    v = vendi_score(X)
    assert 1.0 <= v <= 10.0 + 1e-9


def test_mode_count_and_pairwise():
    # two tight clusters far apart -> ~2 modes
    a = np.zeros((5, 2))
    b = np.full((5, 2), 20.0)
    X = np.vstack([a, b])
    assert mode_count(X, radius=0.5) == 2
    assert mean_pairwise_l2(X) > 0.0
    assert math.isnan(mean_pairwise_l2(np.zeros((1, 2))))
