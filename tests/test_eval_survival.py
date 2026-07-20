"""WP-G: cost-to-target survival analysis (implementation-plan §12.2). Hand-verified values."""

from __future__ import annotations

import numpy as np
import pytest

from rig.eval.survival import (
    kaplan_meier,
    rmst,
    rmst_difference_test,
    split_feasible,
)


def test_km_all_events_matches_hand_computation():
    km = kaplan_meier([1, 2, 3, 4, 5], [True] * 5)
    np.testing.assert_allclose(km.surv, [0.8, 0.6, 0.4, 0.2, 0.0])
    np.testing.assert_array_equal(km.t, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(km.n_at_risk, [5, 4, 3, 2, 1])
    # audit D11: pin the Greenwood variance (previously unasserted + unconsumed,
    # so a Greenwood typo shipped green). At t=1: S=0.8, sum d/(n(n-d))=1/(5·4),
    # var = 0.8^2 · 0.05 = 0.032.
    np.testing.assert_allclose(km.var[0], 0.032)


def test_km_with_censoring_leaves_risk_set():
    # times [1,2,3], events [hit, censored, hit]: the censored subject at t=2
    # leaves the risk set without an event, so at t=3 only 1 is at risk.
    km = kaplan_meier([1, 2, 3], [True, False, True])
    np.testing.assert_allclose(km.surv, [2.0 / 3.0, 0.0])
    np.testing.assert_array_equal(km.t, [1, 3])
    np.testing.assert_array_equal(km.n_at_risk, [3, 1])


def test_km_tie_censor_and_event_same_time_stays_in_risk_set():
    # audit (guard, this file): pins the documented tie convention (module
    # docstring + kaplan_meier docstring) — a censoring TIED with an event at
    # the same time t stays IN the risk set for that event (it has not yet
    # left when the event is recorded), so n_at_risk at t counts it. This is
    # the case a wrong-convention mutant (censor leaves the risk set BEFORE
    # the tied event is tallied) silently breaks while every other test in
    # this file + test_m2_sweep.py stays green — proven by sandbox mutation,
    # see FINDINGS/BUILD_LOG. Hand computation: times=[2,2,3],
    # events=[hit, censored, hit]. n=3.
    #   t=2: n_at_risk = |{times>=2}| = 3 (censored subject still at risk),
    #        d = |{time==2 & event}| = 1  -> S = 1*(1-1/3) = 2/3
    #   t=3: n_at_risk = |{times>=3}| = 1, d = 1 -> S = 2/3*(1-1/1) = 0
    # The wrong convention instead excludes the tied censor from n_at_risk at
    # t=2 (n_at_risk=2), giving S=1-1/2=0.5 at t=2 -- a different number.
    km = kaplan_meier([2, 2, 3], [True, False, True])
    np.testing.assert_allclose(km.surv, [2.0 / 3.0, 0.0])
    np.testing.assert_array_equal(km.t, [2, 3])
    np.testing.assert_array_equal(km.n_at_risk, [3, 1])


def test_km_budget_tie_censor_and_hit_shaped_like_m2_campaign():
    # audit (guard, this file): the M2-shaped case named in the finding --
    # cost-to-target campaigns commonly have several subjects land EXACTLY at
    # the sweep budget, some hitting the spec right at budget (event=True)
    # and some exhausting the budget unhit (censored=False) at that same
    # value. times=[5,10,10,10], events=[hit@5, hit@10, hit@10, censored@10].
    #   t=5:  n_at_risk=4, d=1 -> S = 1*(1-1/4) = 0.75
    #   t=10: n_at_risk = |{times>=10}| = 3 (the budget-exhausted subject is
    #         still at risk at the tied event time), d = |{time==10 & event}|
    #         = 2 -> S = 0.75*(1-2/3) = 0.25
    # A convention that drops the tied censor from the risk set first gives
    # n_at_risk=2 at t=10 (both remaining are hits), d=2 -> S=0.0: a visibly
    # wrong "everyone eventually hits" curve collapse vs. the correct 0.25.
    km = kaplan_meier([5, 10, 10, 10], [True, True, True, False])
    np.testing.assert_allclose(km.surv, [0.75, 0.25])
    np.testing.assert_array_equal(km.t, [5, 10])
    np.testing.assert_array_equal(km.n_at_risk, [4, 3])
    np.testing.assert_array_equal(km.n_events, [1, 2])


def test_km_step_evaluate():
    km = kaplan_meier([1, 2, 3, 4, 5], [True] * 5)
    # S=1 before first event, right-continuous steps, 0 past the last.
    np.testing.assert_allclose(
        km.evaluate(np.array([0.0, 0.5, 2.5, 4.9, 10.0])), [1.0, 1.0, 0.6, 0.2, 0.0]
    )


def test_rmst_matches_hand_computation():
    r = rmst([1, 2, 3, 4, 5], [True] * 5, 5.0)
    assert r.rmst == pytest.approx(3.0)  # 1 + .8 + .6 + .4 + .2
    assert r.se > 0.0
    r2 = rmst([1, 2, 3], [True, False, True], 3.0)
    assert r2.rmst == pytest.approx(7.0 / 3.0)  # 1*1 + (2/3)*2


def test_rmst_horizon_shorter_than_events():
    # horizon before some events: only the covered area counts.
    r = rmst([1, 2, 3, 4, 5], [True] * 5, 2.5)
    # S=1 on [0,1), 0.8 on [1,2), 0.6 on [2,2.5): 1 + 0.8 + 0.6*0.5 = 2.1
    assert r.rmst == pytest.approx(2.1)


def test_rmst_se_shrinks_with_n():
    rng = np.random.default_rng(0)
    small = rmst(rng.exponential(2, 20), [True] * 20, 6.0)
    large = rmst(rng.exponential(2, 400), [True] * 400, 6.0)
    assert large.se < small.se


def test_rmst_variance_pinned_by_hand():
    # times [1,2,3] all events; KM S=[2/3, 1/3, 0], RMST(3)=1+2/3+1/3=2.0.
    # Var = Σ_{t_i≤3} [∫_{t_i}^3 S]² d_i/(n_i(n_i−d_i)):
    #   t=1: n=3,d=1, ∫_1^3 S = 2/3+1/3 = 1.0 → 1² · 1/(3·2) = 1/6
    #   t=2: n=2,d=1, ∫_2^3 S = 1/3       → (1/3)² · 1/(2·1) = 1/18
    #   t=3: n=1,d=1 → n==d, skipped
    # Var = 1/6 + 1/18 = 2/9 ⇒ se = sqrt(2/9). This pins the variance path that
    # drives the §12.2 PRIMARY difference-in-RMST p-value.
    r = rmst([1, 2, 3], [True, True, True], 3.0)
    assert r.rmst == pytest.approx(2.0)
    assert r.se == pytest.approx(np.sqrt(2.0 / 9.0))


def test_rmst_difference_pinned_two_sample():
    # a fixed tiny case pins delta, se, and the p-value together.
    a_r = rmst([1, 2, 3], [True, True, True], 3.0)  # rmst 2.0, var 2/9
    b_r = rmst([2, 4, 6], [True, True, True], 3.0)  # see below
    d = rmst_difference_test([1, 2, 3], [True] * 3, [2, 4, 6], [True] * 3, 3.0)
    assert d.delta == pytest.approx(a_r.rmst - b_r.rmst)
    assert d.se == pytest.approx(np.sqrt(a_r.se**2 + b_r.se**2))
    assert 0.0 <= d.p_value <= 1.0


def test_rmst_difference_identical_is_null():
    d = rmst_difference_test([1, 2, 3, 4, 5], [True] * 5, [1, 2, 3, 4, 5], [True] * 5, 5.0)
    assert d.delta == pytest.approx(0.0)
    assert d.p_value == pytest.approx(1.0)


def test_rmst_difference_detects_cheaper_method():
    rng = np.random.default_rng(1)
    a = rng.exponential(2.0, 100)  # cheaper (smaller cost-to-target)
    b = rng.exponential(6.0, 100)
    d = rmst_difference_test(a, [True] * 100, b, [True] * 100, 15.0)
    assert d.delta < 0.0  # a's restricted-mean cost is lower
    assert d.p_value < 0.01
    assert d.rmst_a < d.rmst_b


def test_split_feasible_excludes_infeasible_targets():
    # the §12.2 (i) correction: infeasible targets are NOT censored into the KM.
    times = [1.0, 2.0, 3.0, 99.0]
    events = [True, True, False, False]
    feasible = [True, True, True, False]
    t, e = split_feasible(times, events, feasible)
    np.testing.assert_array_equal(t, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(e, [True, True, False])


def test_km_rejects_bad_input():
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [True])  # length mismatch
    with pytest.raises(ValueError):
        kaplan_meier([-1.0], [True])  # negative time
    with pytest.raises(ValueError):
        rmst([1, 2], [True, True], 0.0)  # non-positive horizon
