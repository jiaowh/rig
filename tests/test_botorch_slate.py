"""WP-E slate: SCBO + TuRBO trust-region BoTorch comparator tests (§9.8 / §12.3).

The two BoTorch families a reviewer of the M2 "RIG reaches spec ~2x cheaper than
BO" claim would demand, added FAITHFULLY. These tests pin (a) the FAIRNESS
contract shared with ``WarmStartedBO``/``BoTorchBO`` — bit-identical warm start,
exact budget accounting (no hidden machine evaluations — the fairness-leak bug
class), determinism, same Trajectory / drop-in-to-m2_sweep — and (b) a
discriminating KNOWN-ANSWER steelman: on a tight-box bowl the seed DoE misses,
both arms converge via their optimization loop where pure random search cannot, so
a silently-broken comparator (which would fake a RIG win) fails the test.

torch is the optional ``[torch]`` extra, so the module skips when it is absent.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip if the [torch] extra is absent

from rig.active.loop import Trajectory  # noqa: E402
from rig.baselines import BoTorchBO, SCBOBaseline, TuRBOBaseline, WarmStartedBO  # noqa: E402
from rig.eval.m2_sweep import Target, run_m2_sweep  # noqa: E402
from rig.interfaces import CompositionalVariable, ContinuousVariable  # noqa: E402
from rig.transforms import RecipeTransform  # noqa: E402

ARMS = [TuRBOBaseline, SCBOBaseline]

# --- a 1-D reach problem (shared with the BoTorchBO tests) ------------------

VARS_1D = [ContinuousVariable(name="x", lower=0.0, upper=4.0)]


def _machine_1d(recipe):
    """y = 0.5x (noise-free): the box (0.9,1.1) is hit at x~2."""
    return np.array([0.5 * recipe["x"]])


def _in_spec_1d(y):
    return bool(0.9 <= y[0] <= 1.1)


def _arm_1d(cls, **kw):
    base = dict(
        machine=_machine_1d,
        in_spec=_in_spec_1d,
        variables=VARS_1D,
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (0.9, 1.1)}},
        budget=20,
        q=4,
        n_seed=8,
        seed=0,
    )
    base.update(kw)
    return cls(**base)


# --- a 2-D bowl KNOWN-ANSWER problem the seed DoE misses ---------------------

VARS_2D = [ContinuousVariable("x1", 0.0, 1.0), ContinuousVariable("x2", 0.0, 1.0)]
_BOWL_CENTER = (0.7, 0.3)
_BOWL_TOL = 6e-4  # feasible disk radius ~0.024; area fraction ~1.8e-3


def _machine_bowl(recipe):
    return np.array([(recipe["x1"] - _BOWL_CENTER[0]) ** 2 + (recipe["x2"] - _BOWL_CENTER[1]) ** 2])


def _in_spec_bowl(y):
    return bool(y[0] <= _BOWL_TOL)


def _arm_bowl(cls, seed, budget=60):
    return cls(
        machine=_machine_bowl,
        in_spec=_in_spec_bowl,
        variables=VARS_2D,
        input_keys=["x1", "x2"],
        output_keys=["y"],
        spec={"targets": {"y": (0.0, _BOWL_TOL)}},
        budget=budget,
        q=4,
        n_seed=8,
        seed=seed,
    )


def _random_sobol_hits_bowl(seed, budget=60):
    """Pure random (scrambled-Sobol) search at the SAME budget/domain — the
    'no optimization' null. A silently-broken TR-BO would match this."""
    from scipy.stats import qmc

    rt = RecipeTransform(VARS_2D)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = (2.0 * qmc.Sobol(d=2, scramble=True, seed=seed).random(budget) - 1.0) * 5.0
    return any(_in_spec_bowl(_machine_bowl(rt.forward(ui))) for ui in u)


# --------------------------------------------------------------------------- #
# fairness contract                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ARMS)
def test_warm_start_bit_identical_across_arms(cls):
    """§9.2/§12.3: the seed DoE MUST be bit-identical to WarmStartedBO/BoTorchBO —
    same RecipeTransform + campaign seed — else the warm start is not a fair
    comparison."""
    common = dict(
        machine=_machine_1d,
        in_spec=_in_spec_1d,
        variables=VARS_1D,
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (0.9, 1.1)}},
        budget=20,
        q=4,
        seed=7,
    )
    arm = cls(**common)
    wbo = WarmStartedBO(**common)
    bto = BoTorchBO(**common)
    assert arm.n_seed == wbo.n_seed == bto.n_seed
    u_arm = arm._sobol_u(arm.n_seed, arm.seed)
    assert np.array_equal(u_arm, wbo._sobol_u(wbo.n_seed, wbo.seed))
    assert np.array_equal(u_arm, bto._sobol_u(bto.n_seed, bto.seed))


@pytest.mark.parametrize("cls", ARMS)
def test_returns_trajectory_and_finds_reachable_target(cls):
    traj = _arm_1d(cls).run()
    assert isinstance(traj, Trajectory)
    assert traj.hit is True
    assert traj.cost_to_target < float("inf")
    assert traj.cumulative_cost == sorted(traj.cumulative_cost)  # monotone cost


@pytest.mark.parametrize("cls", ARMS)
def test_deterministic_same_seed(cls):
    """§13.4: torch Sobol/MC sampling seeded off the campaign seed -> two runs of
    the same arm at the same seed are byte-identical."""
    a = _arm_bowl(cls, seed=3).run()
    b = _arm_bowl(cls, seed=3).run()
    assert (a.hit, a.cost_to_target, a.n_queries) == (b.hit, b.cost_to_target, b.n_queries)
    assert a.cumulative_cost == b.cumulative_cost
    assert a.per_batch_hit == b.per_batch_hit


@pytest.mark.parametrize("cls", ARMS)
def test_budget_accounting_exact_no_hidden_evaluations(cls):
    """Fairness-leak guard: the number of MACHINE calls must EXACTLY equal
    ``traj.n_queries`` (warm start counted, no hidden evaluations), and on an
    unreachable target n_queries lands exactly on ``budget``. Red-proof: inserting
    any extra internal machine(r) call makes ``calls > n_queries`` and this fails."""
    calls = {"n": 0}

    def counting_machine(recipe):
        calls["n"] += 1
        return _machine_bowl(recipe)

    # unreachable box (bowl minimum is 0; require y in [5,6]) => always exhausts.
    arm = cls(
        machine=counting_machine,
        in_spec=lambda y: bool(5.0 <= y[0] <= 6.0),
        variables=VARS_2D,
        input_keys=["x1", "x2"],
        output_keys=["y"],
        spec={"targets": {"y": (5.0, 6.0)}},
        budget=20,
        q=4,
        n_seed=8,
        seed=0,
    )
    traj = arm.run()
    assert traj.hit is False
    assert traj.stop_reason == "budget exhausted"
    assert traj.n_queries == 20  # (budget - n_seed) divisible by q -> exact
    assert calls["n"] == traj.n_queries  # NO hidden machine evaluations


@pytest.mark.parametrize("cls", ARMS)
def test_budget_accounting_counts_warm_start_on_early_hit(cls):
    """On a hit, machine calls still equal n_queries exactly (the seed DoE is
    counted, not free)."""
    calls = {"n": 0}

    def counting_machine(recipe):
        calls["n"] += 1
        return _machine_1d(recipe)

    arm = _arm_1d(cls, machine=counting_machine)
    traj = arm.run()
    assert calls["n"] == traj.n_queries
    assert traj.n_queries <= arm.budget


# --------------------------------------------------------------------------- #
# steelman: discriminating known-answer sanity                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ARMS)
def test_known_answer_optimizes_where_random_cannot(cls):
    """STEELMAN: on a tight-box bowl the 8-point seed DoE misses, the arm must
    converge via its optimization loop (Thompson / constrained-Thompson inside the
    trust region), reaching the known optimum region where pure random search at
    the same budget cannot. A silently-broken comparator would look like random and
    fake a RIG win — this is the check that would catch it."""
    seeds = range(3)
    # null: pure random search essentially never hits this tight box at this budget.
    rand_hits = sum(_random_sobol_hits_bowl(s) for s in seeds)
    assert rand_hits == 0, f"sanity box not tight enough; random hit {rand_hits}"

    trajs = [_arm_bowl(cls, s).run() for s in seeds]
    assert all(t.hit for t in trajs), f"{cls.__name__} failed to optimize the bowl"
    # the hit must come from the LOOP, not the seed DoE (else the box wasn't tight).
    assert all("seed DoE" not in t.stop_reason for t in trajs)


@pytest.mark.parametrize("cls", ARMS)
def test_unreachable_target_no_false_hit(cls):
    traj = _arm_1d(
        cls,
        spec={"targets": {"y": (5.0, 6.0)}},  # y=0.5x maxes at 2.0 on x in [0,4]
        in_spec=lambda y: bool(5.0 <= y[0] <= 6.0),
        budget=16,
    ).run()
    assert traj.hit is False
    assert math.isinf(traj.cost_to_target)
    assert traj.stop_reason == "budget exhausted"


@pytest.mark.parametrize("cls", ARMS)
def test_compositional_variable_rejected(cls):
    with pytest.raises(NotImplementedError, match="compositional"):
        cls(
            machine=_machine_1d,
            in_spec=_in_spec_1d,
            variables=[CompositionalVariable(name="alloy", components=("ga", "in"))],
            input_keys=["alloy.ga", "alloy.in"],
            output_keys=["y"],
            spec={"targets": {"y": (0.9, 1.1)}},
            seed=0,
        )


# --------------------------------------------------------------------------- #
# drop-in to the M2 harness                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ARMS)
def test_plugs_into_m2_sweep(cls):
    """Both arms are drop-in m2_sweep MethodFactories: a tiny RIG-harness-style
    pairwise sweep (arm vs numpy BO) produces a well-formed M2Report — proving the
    §12.3 slate can now include SCBO/TuRBO."""

    def make_machine(_seed):
        return _machine_1d

    def arm_factory(*, machine, in_spec, spec, seed):
        return cls(
            machine=machine,
            in_spec=in_spec,
            variables=VARS_1D,
            input_keys=["x"],
            output_keys=["y"],
            spec=spec,
            budget=12,
            q=4,
            n_seed=8,
            seed=seed,
        )

    def wbo_factory(*, machine, in_spec, spec, seed):
        return WarmStartedBO(
            machine=machine,
            in_spec=in_spec,
            variables=VARS_1D,
            input_keys=["x"],
            output_keys=["y"],
            spec=spec,
            budget=12,
            q=4,
            n_seed=8,
            seed=seed,
        )

    target = Target(
        id="reach",
        spec={"targets": {"y": (0.9, 1.1)}},
        in_spec=lambda y: bool(0.9 <= y[0] <= 1.1),
    )
    report = run_m2_sweep(
        make_machine=make_machine,
        methods={cls.__name__: arm_factory, "warm": wbo_factory},
        targets=[target],
        seeds=[0, 1],
        horizon=13.0,
        n_bootstrap=200,
    )
    d = report.to_dict()
    assert len(report.campaigns) == 2 * 1 * 2
    assert 0.0 <= d["pooled_hit_rate"][cls.__name__] <= 1.0
    assert np.isfinite(d["pooled_delta_rmst"])
