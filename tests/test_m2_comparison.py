"""M2 in-silico comparison harness: RIG active-learning loop vs warm-started BO,
scored by difference-in-RMST (implementation-plan §12.2). CI-safe synthetic machine.

This ties the stack together — WP-D inverse + WP-F loop + §12.3 BO baseline +
WP-G survival — into the M2 measurement: cost-to-target trajectories over seeds
fed to the difference-in-RMST test. It asserts the HARNESS produces a valid
head-to-head (finite RMSTs, a valid p-value) and that both methods actually
reach feasible targets — NOT a flaky "RIG always wins" (that is an empirical
result for the real campaign, not a unit test).
"""

from __future__ import annotations

import math

import numpy as np

from rig.active import ActiveLearningLoop
from rig.baselines import WarmStartedBO
from rig.eval.survival import rmst, rmst_difference_test
from rig.interfaces import ContinuousVariable

BUDGET = 48


def _machine_factory(seed):
    rng = np.random.default_rng(seed)

    # a non-monotone response so hitting the spec needs a decent surrogate, not
    # just luck: y = 2 + 1.5*sin(5x) over x in [0,1].
    def machine(recipe):
        x = recipe["x"]
        return np.array([2.0 + 1.5 * np.sin(5.0 * x) + 0.02 * rng.standard_normal()])

    return machine


def _run(method_cls, spec, seed, **extra):
    def in_spec(y):
        lo, hi = spec["targets"]["y"]
        return lo <= float(y[0]) <= hi

    m = method_cls(
        machine=_machine_factory(seed),
        in_spec=in_spec,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec=spec,
        cost_recipe=lambda r: 1.0,
        c_batch=0.0,
        budget=BUDGET,
        q=4,
        n_seed=8,
        n_pool=96,
        seed=seed,
        **extra,
    )
    traj = m.run()
    # cost-to-target as survival data: event = hit, else right-censored at budget.
    time = traj.cost_to_target if traj.hit else float(traj.cumulative_cost[-1])
    return time, traj.hit


def test_m2_rig_vs_bo_difference_in_rmst_harness():
    # a reachable mid-range target of y = 2 + 1.5 sin(5x): y in [2.6, 3.0].
    spec = {"targets": {"y": (2.6, 3.0)}}
    seeds = range(10)
    rig = [_run(ActiveLearningLoop, spec, s, kappa=1.0, z_epi=1.0, delta_frac=0.01) for s in seeds]
    bo = [_run(WarmStartedBO, spec, s) for s in seeds]

    rig_t = [t for t, _ in rig]
    rig_e = [h for _, h in rig]
    bo_t = [t for t, _ in bo]
    bo_e = [h for _, h in bo]

    # both methods reach the reachable target on most seeds.
    assert sum(rig_e) >= 7, rig_e
    assert sum(bo_e) >= 7, bo_e

    # the difference-in-RMST harness produces a valid head-to-head.
    horizon = float(BUDGET)
    r_rig = rmst(rig_t, rig_e, horizon)
    r_bo = rmst(bo_t, bo_e, horizon)
    assert math.isfinite(r_rig.rmst) and r_rig.rmst > 0
    assert math.isfinite(r_bo.rmst) and r_bo.rmst > 0

    d = rmst_difference_test(rig_t, rig_e, bo_t, bo_e, horizon)
    assert 0.0 <= d.p_value <= 1.0
    assert math.isfinite(d.delta) and math.isfinite(d.se)
    # delta = RMST_rig - RMST_bo; negative would mean RIG is cheaper. We only
    # assert the comparison is well-formed (a signed, finite effect).
    assert d.delta == r_rig.rmst - r_bo.rmst


def test_m2_adaptive_phase_contributes_and_direction_is_surfaceable():
    # audit C9/D14: the harness reach bars (sum(e) >= 7) are met partly by the
    # shared 8-point seed DoE, so the ADAPTIVE loop was untested and a RIG
    # regression could flip the result silently. (a) require at least one hit for
    # EACH method to arrive strictly AFTER the seed lot (cost > n_seed*cost_recipe)
    # — proving the adaptive phase, not just the seed, did work; (b) confirm the
    # difference-in-RMST direction is meaningful, so a regression that makes RIG
    # costlier than BO WOULD surface as a positive delta.
    spec = {"targets": {"y": (2.6, 3.0)}}
    seeds = range(10)
    rig = [_run(ActiveLearningLoop, spec, s, kappa=1.0, z_epi=1.0, delta_frac=0.01) for s in seeds]
    bo = [_run(WarmStartedBO, spec, s) for s in seeds]

    # (a) adaptive-phase hits exist for both methods (cost strictly above the
    # 8-run seed lot). A broken inverse/EI that only ever hit via the seed would
    # fail this.
    assert any(t > 8.0 for t, h in rig if h), rig
    assert any(t > 8.0 for t, h in bo if h), bo

    # (b) the comparator can SURFACE a direction: if RIG were uniformly costlier
    # than BO, difference-in-RMST (delta = RMST_rig - RMST_bo) must be positive.
    horizon = float(BUDGET)
    worse = rmst_difference_test([40] * 10, [True] * 10, [10] * 10, [True] * 10, horizon)
    assert worse.delta > 0.0  # RIG(t=40) - BO(t=10) > 0 => a RIG regression is visible


def test_m2_infeasible_target_excluded_from_survival():
    # an unreachable target (y maxes ~3.5): both methods censor at budget; the
    # feasibility layer (not survival) owns it — here we just confirm neither
    # method reports a false hit.
    spec = {"targets": {"y": (10.0, 11.0)}}
    for cls in (ActiveLearningLoop, WarmStartedBO):
        extra = (
            {"kappa": 1.0, "z_epi": 1.0, "delta_frac": 0.01} if cls is ActiveLearningLoop else {}
        )
        _, hit = _run(cls, spec, 0, **extra)
        assert hit is False
