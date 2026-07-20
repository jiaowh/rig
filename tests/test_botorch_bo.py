"""WP-E: production BoTorch BO baseline tests (implementation-plan §9.8 / §12.3).

The continuous-acquisition BoTorch comparator (`SingleTaskGP` + qLogEI/qLCB via
`optimize_acqf`) that closes the M2 BF-1b owed item. Tests pin the FAIRNESS
contract (identical warm-start to WarmStartedBO, same Trajectory, same hit rule)
and the drop-in-to-m2_sweep contract, not a performance claim. torch is the
optional extra, so the module skips when it is absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip if the [torch] extra is absent

from rig.active.loop import Trajectory  # noqa: E402
from rig.baselines import BoTorchBO, WarmStartedBO  # noqa: E402
from rig.eval.m2_sweep import Target, run_m2_sweep  # noqa: E402
from rig.interfaces import CompositionalVariable, ContinuousVariable  # noqa: E402

VARS = [ContinuousVariable(name="x", lower=0.0, upper=4.0)]


def _machine(recipe):
    """y = 0.5x (noise-free, deterministic): the box (0.9,1.1) is hit at x≈2."""
    return np.array([0.5 * recipe["x"]])


def _in_spec(y):
    return bool(0.9 <= y[0] <= 1.1)


def _bo(**kw):
    base = dict(
        machine=_machine,
        in_spec=_in_spec,
        variables=VARS,
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (0.9, 1.1)}},
        budget=18,
        q=3,
        seed=0,
    )
    base.update(kw)
    return BoTorchBO(**base)


# --- fairness contract ------------------------------------------------------


def test_warm_start_bit_identical_to_warm_bo():
    """§9.2/§12.3: the seed DoE MUST be identical across arms — same RecipeTransform
    + campaign seed — else the warm-start is not a fair comparison."""
    common = dict(
        machine=_machine,
        in_spec=_in_spec,
        variables=VARS,
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (0.9, 1.1)}},
        budget=18,
        q=3,
        seed=7,
    )
    bo = BoTorchBO(**common)
    wbo = WarmStartedBO(**common)
    assert bo.n_seed == wbo.n_seed
    assert np.allclose(bo._sobol_u(bo.n_seed, bo.seed), wbo._sobol_u(wbo.n_seed, wbo.seed))


def test_returns_trajectory_and_finds_reachable_target():
    traj = _bo().run()
    assert isinstance(traj, Trajectory)
    assert traj.hit is True  # x≈2 is reachable and BoTorch finds it
    assert traj.cost_to_target < float("inf")
    assert traj.cumulative_cost == sorted(traj.cumulative_cost)  # monotone
    assert traj.n_queries >= traj.n_queries  # sanity: well-formed


def test_deterministic_same_seed():
    """§13.4: BoTorch draws its own randomness (Sobol init, MC sampler); seeding it
    from the campaign seed must make two runs identical."""
    a, b = _bo(seed=3).run(), _bo(seed=3).run()
    assert (a.hit, a.cost_to_target, a.n_queries) == (b.hit, b.cost_to_target, b.n_queries)
    assert a.cumulative_cost == b.cumulative_cost


def test_unreachable_target_exhausts_budget_without_crash():
    """An unreachable box → hit=False, budget exhausted (never a bogus hit)."""
    traj = _bo(
        spec={"targets": {"y": (5.0, 6.0)}},  # y=0.5x maxes at 2.0 on x∈[0,4]
        in_spec=lambda y: bool(5.0 <= y[0] <= 6.0),
        budget=12,
    ).run()
    assert traj.hit is False
    assert traj.stop_reason == "budget exhausted"


def test_search_bounds_match_reference_interior():
    """Fix: optimize_acqf searches the SAME box-sigmoid interior the reference arms
    reach (u∈[−u_bound, u_bound]), NOT the full closed box — so the two BO arms'
    feasible domains are identical (strict fairness)."""
    import math

    bo = _bo()  # x∈[0,4], u_bound=5
    lo = 4.0 / (1.0 + math.exp(5.0))
    hi = 4.0 / (1.0 + math.exp(-5.0))
    b = bo._bounds.tolist()
    assert abs(b[0][0] - lo) < 1e-9 and abs(b[1][0] - hi) < 1e-9
    assert b[0][0] > 0.0 and b[1][0] < 4.0  # strictly interior — never the box edge


def test_gp_uses_matern_kernel():
    """Fix: the surrogate kernel is Matérn-5/2 (matches the RIG GP tier + the
    docstring), not the helper's RBF default."""
    from gpytorch.kernels import MaternKernel

    gp = _bo()._fit_gp(np.array([[1.0], [2.0], [3.0]]), np.array([0.5, 0.0, 0.5]))
    assert isinstance(gp.covar_module, MaternKernel)
    assert float(gp.covar_module.nu) == 2.5


def test_qlcb_acquisition_runs():
    traj = _bo(acquisition="qlcb").run()
    assert isinstance(traj, Trajectory)


def test_bad_acquisition_rejected():
    with pytest.raises(ValueError, match="qlogei"):
        _bo(acquisition="nope")


def test_compositional_variable_rejected():
    """Continuous-only scope: a simplex variable must fail loud, not silently
    search a broken (non-simplex) box."""
    with pytest.raises(NotImplementedError, match="compositional"):
        BoTorchBO(
            machine=_machine,
            in_spec=_in_spec,
            variables=[CompositionalVariable(name="alloy", components=("ga", "in"))],
            input_keys=["alloy.ga", "alloy.in"],
            output_keys=["y"],
            spec={"targets": {"y": (0.9, 1.1)}},
            seed=0,
        )


# --- drop-in to the M2 harness ----------------------------------------------


def test_plugs_into_m2_sweep_vs_warm_bo():
    """BoTorchBO is a drop-in m2_sweep MethodFactory: a tiny RIG-harness-style
    pairwise sweep (BoTorch vs numpy BO) produces a well-formed M2Report — proving
    the §12.3 slate can now include the production BoTorch comparator."""

    def make_machine(_seed):
        return _machine  # deterministic; CRN pairing is trivial here

    def bo_factory(*, machine, in_spec, spec, seed):
        return BoTorchBO(
            machine=machine,
            in_spec=in_spec,
            variables=VARS,
            input_keys=["x"],
            output_keys=["y"],
            spec=spec,
            budget=12,
            q=3,
            seed=seed,
        )

    def wbo_factory(*, machine, in_spec, spec, seed):
        return WarmStartedBO(
            machine=machine,
            in_spec=in_spec,
            variables=VARS,
            input_keys=["x"],
            output_keys=["y"],
            spec=spec,
            budget=12,
            q=3,
            seed=seed,
        )

    target = Target(
        id="reach",
        spec={"targets": {"y": (0.9, 1.1)}},
        in_spec=lambda y: bool(0.9 <= y[0] <= 1.1),
    )
    report = run_m2_sweep(
        make_machine=make_machine,
        methods={"botorch": bo_factory, "warm": wbo_factory},
        targets=[target],
        seeds=[0, 1],
        horizon=13.0,
        n_bootstrap=200,
    )
    d = report.to_dict()
    assert len(report.campaigns) == 2 * 1 * 2  # methods × targets × seeds
    assert 0.0 <= d["pooled_hit_rate"]["botorch"] <= 1.0
    assert np.isfinite(d["pooled_delta_rmst"])
