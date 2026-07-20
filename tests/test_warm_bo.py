"""M2 baseline: warm-started GP-EI BO (implementation-plan §9.8/§12.3).

audit C10: the WarmStartedBO acquisition core (expected_improvement) had NO test
— it was exercised only indirectly by test_m2_comparison, whose reach assertions
are satisfied by the shared seed DoE, so a sign-flipped or mis-scaled EI passed
the whole suite. That leaves the M2 "RIG beats warm-started BO" comparison
resting on an unverified comparator. These pin the EI formula directly and smoke
the loop end-to-end.
"""

from __future__ import annotations

import math

import numpy as np

from rig.baselines import WarmStartedBO
from rig.baselines.warm_bo import expected_improvement
from rig.interfaces import ContinuousVariable


def test_expected_improvement_known_answers():
    # EI for MINIMIZATION toward incumbent `best`.
    # sigma -> 0: EI -> max(best - mu, 0) (noise-free improvement).
    ei0 = expected_improvement(np.array([-1.0, 0.5, 2.0]), np.zeros(3), best=1.0)
    np.testing.assert_allclose(ei0, [2.0, 0.5, 0.0])
    # a sign-flipped/maximizing EI could not reproduce this exact vector.

    # strictly increasing in sigma at fixed mu < best.
    ei_lo = expected_improvement(np.array([0.0]), np.array([0.1]), best=1.0)
    ei_hi = expected_improvement(np.array([0.0]), np.array([1.0]), best=1.0)
    assert ei_hi[0] > ei_lo[0] > 0.0

    # mu far ABOVE best with tiny sigma: essentially no expected improvement.
    assert expected_improvement(np.array([10.0]), np.array([1e-6]), best=0.0)[0] < 1e-6

    # EI is always non-negative.
    assert np.all(expected_improvement(np.array([5.0, -5.0]), np.full(2, 0.3), best=0.0) >= 0.0)


def test_expected_improvement_prefers_the_lower_predicted_mean():
    # at equal sigma, a candidate with a smaller predicted objective (closer to
    # the target box) must have the higher EI — the ranking the BO loop relies on.
    ei = expected_improvement(np.array([0.2, 0.8]), np.array([0.3, 0.3]), best=1.0)
    assert ei[0] > ei[1]


def test_warm_bo_reaches_reachable_target():
    def machine(recipe):
        return np.array([3.0 * recipe["x"]])  # y = 3x on [0,1]; [1.4,1.6] at x~0.5

    bo = WarmStartedBO(
        machine=machine,
        in_spec=lambda y: 1.4 <= float(y[0]) <= 1.6,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (1.4, 1.6)}},
        cost_recipe=lambda r: 1.0,
        budget=40,
        q=4,
        n_seed=8,
        n_pool=96,
        seed=0,
    )
    traj = bo.run()
    assert traj.hit is True
    assert math.isfinite(traj.cost_to_target) and traj.cost_to_target <= 40


def test_warm_bo_unreachable_target_no_false_hit():
    def machine(recipe):
        return np.array([3.0 * recipe["x"]])  # maxes at 3 on [0,1]

    bo = WarmStartedBO(
        machine=machine,
        in_spec=lambda y: 10.0 <= float(y[0]) <= 11.0,
        variables=[ContinuousVariable("x", 0.0, 1.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (10.0, 11.0)}},
        budget=24,
        q=4,
        n_seed=8,
        n_pool=96,
        seed=0,
    )
    traj = bo.run()
    assert traj.hit is False
    assert math.isinf(traj.cost_to_target)
    assert traj.stop_reason == "budget exhausted"
