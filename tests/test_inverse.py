"""WP-D: per-query pessimistic inverse solver tests (implementation-plan §8). Synthetic +
in-silico GP; no adapter import except the sim-gated MBE integration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rig.constraints import (
    BoxConstraint,
    ConstraintSet,
    LinearConstraint,
    MonotoneConstraint,
    SimplexConstraint,
)
from rig.forward import GPForwardModel, MultiToolGPForwardModel
from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
    Infeasible,
    PredictiveDistribution,
    RecipeCandidate,
)
from rig.inverse import PessimisticInverseSolver, SpecBox, parse_targets
from rig.transforms import SimplexTransform

# ---------------------------------------------------------------------------
# controllable analytic ForwardModel — lets us assert the pessimism math
# ---------------------------------------------------------------------------


class _AnalyticModel:
    """Controllable ForwardModel. Mean is A@x+b (linear) OR a callable
    ``mean_fn``/``jac_fn`` pair (nonlinear, for genuine multi-pre-image cases);
    constant aleatoric; callable epistemic and support. Fully implements the
    ForwardModel protocol so we can drive each §8 pessimism channel
    independently."""

    def __init__(
        self,
        A=None,
        b=None,
        ale=None,
        epi_fn=None,
        support_fn=None,
        mean_fn=None,
        jac_fn=None,
        m=None,
    ):
        self.A = None if A is None else np.asarray(A, float)  # (m, d)
        self.b = None if b is None else np.asarray(b, float)  # (m,)
        self.mean_fn = mean_fn
        self.jac_fn = jac_fn
        self._m = m if m is not None else (self.b.shape[0] if self.b is not None else 1)
        self.ale = np.asarray(ale, float)  # (m,)
        self.epi_fn = epi_fn or (lambda x: np.zeros(self._m))
        self.support_fn = support_fn or (lambda x: 0.0)

    def _mean_row(self, xi):
        if self.mean_fn is not None:
            return np.asarray(self.mean_fn(xi), float)
        return self.A @ xi + self.b

    def predict(self, x) -> PredictiveDistribution:
        x = np.asarray(x, float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mean = np.stack([self._mean_row(xi) for xi in Xq])  # (n, m)
        ale = np.broadcast_to(self.ale, mean.shape).copy()
        epi = np.stack([np.asarray(self.epi_fn(xi), float) for xi in Xq])  # (n, m)
        if single:
            mean, ale, epi = mean[0], ale[0], epi[0]
        return PredictiveDistribution(mean, ale, epi, None)

    def jacobian(self, x) -> np.ndarray:
        xi = np.asarray(x, float)
        if self.jac_fn is not None:
            return np.atleast_2d(np.asarray(self.jac_fn(xi), float))
        return self.A  # constant for a linear mean

    def support_score(self, x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            return float(self.support_fn(x))
        return np.array([float(self.support_fn(xi)) for xi in x])

    def update(self, records) -> None:  # pragma: no cover - not exercised
        pass


def _var(name, lo, hi):
    return ContinuousVariable(name, lo, hi)


# ---------------------------------------------------------------------------
# parse_targets
# ---------------------------------------------------------------------------


def test_parse_targets_all_forms():
    box = parse_targets(
        {
            "a": (1.0, 2.0),
            "b": {"lower": 0.0},
            "c": {"upper": 5.0},
            "d": {"target": 3.0, "tol": 0.5},
        },
        ["a", "b", "c", "d", "unused"],
    )
    assert isinstance(box, SpecBox)
    assert box.output_names == ("a", "b", "c", "d")
    np.testing.assert_allclose(box.lower, [1.0, 0.0, -np.inf, 2.5])
    np.testing.assert_allclose(box.upper, [2.0, np.inf, 5.0, 3.5])


def test_parse_targets_rejects_unknown_output():
    with pytest.raises(KeyError):
        parse_targets({"zzz": (0, 1)}, ["a", "b"])


def test_parse_targets_rejects_inverted_and_unbounded():
    with pytest.raises(ValueError):
        parse_targets({"a": (2.0, 1.0)}, ["a"])
    with pytest.raises(ValueError):
        parse_targets({"a": {}}, ["a"])  # constrains neither bound


def test_parse_targets_rejects_zero_width_point_target():
    # a bare {'target': t} (no tol) is a zero-width box → fail loud, not a
    # silent always-INFEASIBLE (review finding).
    with pytest.raises(ValueError, match="zero-width"):
        parse_targets({"a": {"target": 7.0}}, ["a"])
    with pytest.raises(ValueError, match="zero-width"):
        parse_targets({"a": (5.0, 5.0)}, ["a"])


# ---------------------------------------------------------------------------
# constructor fail-closed (§8.2 anti-reward-hacking)
# ---------------------------------------------------------------------------


def test_requires_support_floor_or_xtrain():
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    with pytest.raises(ValueError, match="support_floor or X_train"):
        PessimisticInverseSolver(model, [_var("x", 0, 10)], ["y"])


def test_solve_requires_targets():
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    solver = PessimisticInverseSolver(model, [_var("x", 0, 10)], ["y"], support_floor=-10.0)
    with pytest.raises(KeyError, match="targets"):
        solver.solve({})


# ---------------------------------------------------------------------------
# analytic recovery + interface contract
# ---------------------------------------------------------------------------


def test_recovers_feasible_recipe():
    # y = 2*x + 1 over x in [0, 10]; target y in [10, 12] => x in [4.5, 5.5].
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.0,
        seed=0,
    )
    res = solver.solve({"targets": {"y": (10.0, 12.0)}})
    assert isinstance(res, list) and res, "expected feasible candidates"
    for c in res:
        assert isinstance(c, RecipeCandidate)
        assert c.feasibility_flag is True
        assert 0.0 <= c.confidence <= 1.0
        assert 4.0 <= c.recipe["x"] <= 6.0  # inside the pre-image (± band)
        lo, hi = c.predicted_outcome_interval["y"]
        assert lo <= hi
    # best candidate should sit near the box centre (max robust margin).
    best = max(res, key=lambda c: c.confidence)
    assert abs(2.0 * best.recipe["x"] + 1.0 - 11.0) < 1.0


def test_box_constraint_by_construction_never_violated():
    model = _AnalyticModel(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0], ale=[0.1, 0.1])
    solver = PessimisticInverseSolver(
        model,
        [_var("x1", -1.0, 1.0), _var("x2", 5.0, 7.0)],
        ["y1", "y2"],
        support_floor=-10.0,
        delta_frac=0.0,
        seed=1,
    )
    res = solver.solve({"targets": {"y1": (-0.5, 0.5), "y2": (5.5, 6.5)}})
    assert isinstance(res, list) and res
    for c in res:
        assert -1.0 <= c.recipe["x1"] <= 1.0
        assert 5.0 <= c.recipe["x2"] <= 7.0


def test_simplex_constraint_by_construction():
    # 2 outputs equal to the two mole fractions; target both near 0.5.
    A = [[1.0, 0.0], [0.0, 1.0]]
    model = _AnalyticModel(A=A, b=[0.0, 0.0], ale=[0.05, 0.05])
    comp = CompositionalVariable("alloy", ("ga", "in"))
    solver = PessimisticInverseSolver(
        model,
        [comp],
        ["f_ga", "f_in"],
        support_floor=-10.0,
        delta_frac=0.0,
        seed=2,
    )
    res = solver.solve({"targets": {"f_ga": (0.3, 0.7), "f_in": (0.3, 0.7)}})
    assert isinstance(res, list) and res
    for c in res:
        s = c.recipe["alloy.ga"] + c.recipe["alloy.in"]
        assert abs(s - 1.0) < 1e-6  # sum-to-1 exact
        assert c.recipe["alloy.ga"] >= 0 and c.recipe["alloy.in"] >= 0


# ---------------------------------------------------------------------------
# pessimism actually penalizes epistemic uncertainty (the headline invariant)
# ---------------------------------------------------------------------------


def test_epistemic_region_is_penalized():
    # y = x over [0, 10]; target y in [4.5, 5.5]. Epistemic is HUGE near x=5
    # (right where the mean hits the target). The mean CAN reach the spec (at
    # x=5) but the surrogate is far too uncertain there, so the robust solver
    # returns INFEASIBLE — and the §8.8 diagnostic probe must correctly label it
    # EPISTEMIC-limited ("collect more runs"), reporting the mean-feasible point
    # x≈5, NOT "genuinely unreachable".
    def epi(x):
        return np.array([5.0 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)])

    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1], epi_fn=epi)
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        z_epi=2.0,
        delta_frac=0.0,
        seed=3,
    )
    res = solver.solve({"targets": {"y": (4.5, 5.5)}})
    assert isinstance(res, Infeasible)
    assert "epistemic" in res.reason.lower()
    assert "unreachable" not in res.reason.lower()
    assert 4.0 <= res.nearest_achievable["x"] <= 6.0  # the mean-feasible point
    assert res.distance_to_feasible > 0.0


def test_low_epistemic_preimage_chosen_over_high_epistemic_twin():
    # y = -(x-2)(x-8): a FOLDED response with two GENUINE pre-images of y≈0
    # (x≈2 and x≈8), BOTH mean-feasible for target [-0.5, 0.5]. Epistemic is
    # high only near x=8. Pessimism must DEMOTE the x≈8 twin and return x≈2 —
    # and with z_epi=0 the x≈8 twin is accepted, so epistemic is the load-bearing
    # discriminator (the earlier y=x version had no real twin — review finding).
    def mean_fn(xi):
        return np.array([-(xi[0] - 2.0) * (xi[0] - 8.0)])

    def jac_fn(xi):
        return np.array([[-2.0 * xi[0] + 10.0]])

    def epi(xi):
        return np.array([3.0 * np.exp(-((xi[0] - 8.0) ** 2) / 0.5)])

    model = _AnalyticModel(ale=[0.1], mean_fn=mean_fn, jac_fn=jac_fn, epi_fn=epi, m=1)
    common = dict(support_floor=-10.0, delta_frac=0.0, kappa=1.0, seed=14)
    spec = {"targets": {"y": (-0.5, 0.5)}, "max_candidates": 4}

    # pessimistic (z_epi=2): only the low-epistemic x≈2 pre-image survives.
    res = PessimisticInverseSolver(model, [_var("x", 0.0, 10.0)], ["y"], z_epi=2.0, **common).solve(
        spec
    )
    assert isinstance(res, list) and res
    assert all(c.recipe["x"] < 5.0 for c in res)  # x≈2 branch only

    # non-pessimistic (z_epi=0): the high-epistemic x≈8 twin is now acceptable,
    # so at least one returned recipe sits on it — proving epistemic pessimism
    # was the load-bearing factor above.
    res0 = PessimisticInverseSolver(
        model, [_var("x", 0.0, 10.0)], ["y"], z_epi=0.0, **common
    ).solve(spec)
    assert isinstance(res0, list)
    assert any(c.recipe["x"] > 5.0 for c in res0)


# ---------------------------------------------------------------------------
# input-tolerance (δ) robustness — §8.5
# ---------------------------------------------------------------------------


def test_delta_shrinks_feasibility_on_steep_response():
    # Steep response y = 20*x over [0,1]; target [9,11] => x in [0.45,0.55].
    # With delta_frac=0 it is feasible; with a large delta the ‖J‖·Δ term eats
    # the whole margin and it becomes infeasible.
    model = _AnalyticModel(A=[[20.0]], b=[0.0], ale=[0.2])
    common = dict(support_floor=-10.0, z_epi=0.0, seed=5)
    r0 = PessimisticInverseSolver(
        model, [_var("x", 0.0, 1.0)], ["y"], delta_frac=0.0, **common
    ).solve({"targets": {"y": (9.0, 11.0)}})
    r1 = PessimisticInverseSolver(
        model, [_var("x", 0.0, 1.0)], ["y"], delta_frac=0.10, **common
    ).solve({"targets": {"y": (9.0, 11.0)}})
    assert isinstance(r0, list) and r0
    assert isinstance(r1, Infeasible)


def test_delta_range_scaling_uses_variable_range():
    # audit D5: Δ_i = delta_frac * (upper - lower). Every existing delta test
    # uses a range of exactly 1.0, so the range factor was untested — dropping
    # it would 10x-under-count the ‖J‖·Δ penalty here. Pin both Δ and s.
    model = _AnalyticModel(A=[[20.0]], b=[0.0], ale=[1.0])
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.1,
        z_epi=0.0,
        seed=0,
    )
    np.testing.assert_allclose(solver._delta_raw, [1.0])  # 0.1 * (10 - 0), NOT 0.1
    box = parse_targets({"y": (5.0, 15.0)}, ["y"])
    out_idx = np.array([0])
    *_rest, s, _support, _sc = solver._margins(np.array([0.5]), box, out_idx)
    # s = z_epi·σ_epi + |J|·Δ = 0 + 20 * 1.0 = 20 (would be 2.0 without range scaling)
    np.testing.assert_allclose(s[out_idx], [20.0])


def test_predicted_interval_equals_mu_plus_minus_s_plus_kappa_sigma():
    # audit D6: the operator-facing worst-case credited interval must equal
    # [μ − s − κσ_ale, μ + s + κσ_ale] (only lo<=hi was previously tested). With
    # delta_frac=0 and z_epi=0, s=0, so the interval is exactly μ ± κ·σ_ale.
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    kappa = 2.0
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.0,
        z_epi=0.0,
        kappa=kappa,
        seed=0,
    )
    res = solver.solve({"targets": {"y": (10.0, 12.0)}})
    assert isinstance(res, list) and res
    c = res[0]
    mu = 2.0 * c.recipe["x"] + 1.0
    lo, hi = c.predicted_outcome_interval["y"]
    np.testing.assert_allclose([lo, hi], [mu - kappa * 0.05, mu + kappa * 0.05], atol=1e-6)
    assert 10.0 <= lo and hi <= 12.0  # honest pessimism: interval ⊆ spec box


# ---------------------------------------------------------------------------
# support floor hard-reject — §8.2
# ---------------------------------------------------------------------------


def test_offsupport_optimum_is_rejected():
    # y = x, target [4.5,5.5] reachable at x=5, but support collapses there
    # (a far-OOD hole the mean happens to fit). Everything is off-support =>
    # the §8.2 hard reject fires and we get the "outside support" verdict.
    def support(x):
        return -abs(x[0] - 9.0)  # only x≈9 is on-support; target needs x≈5

    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1], support_fn=support)
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-1.0,
        z_epi=0.0,
        delta_frac=0.0,
        seed=6,
    )
    res = solver.solve({"targets": {"y": (4.5, 5.5)}})
    assert isinstance(res, Infeasible)
    assert "support" in res.reason.lower()


# ---------------------------------------------------------------------------
# genuine infeasibility — unreachable target box
# ---------------------------------------------------------------------------


def test_unreachable_target_is_infeasible_with_nearest_point():
    # y = x over [0,10] can never reach 100. Nearest achievable is x≈10. A hard
    # spec conflict must NOT be mislabeled epistemic even with nonzero epistemic
    # uncertainty (the epi_dominated regression the review caught) — more data
    # cannot fix a target the mean misses by ~90 units.
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1], epi_fn=lambda x: np.array([0.5]))
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-100.0,
        z_epi=2.0,
        delta_frac=0.0,
        seed=7,
    )
    res = solver.solve({"targets": {"y": (99.0, 101.0)}})
    assert isinstance(res, Infeasible)
    assert res.nearest_achievable["x"] > 9.0  # pinned to the top of the box
    assert res.distance_to_feasible > 0.0
    assert "unreachable" in res.reason.lower()
    assert "epistemic" not in res.reason.lower()  # not mislabeled "collect runs"
    assert "relaxation" in res.reason.lower()


def test_multi_output_only_one_violated():
    # y1 = x1 (easily met), y2 = x2 (unreachable). Exercises the per-output
    # relaxation dict (only y2), the confidence product driven to 0 by one
    # output, and worst-output selection (review coverage-gap finding).
    model = _AnalyticModel(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0], ale=[0.1, 0.1])
    solver = PessimisticInverseSolver(
        model,
        [_var("x1", 0.0, 5.0), _var("x2", 0.0, 5.0)],
        ["y1", "y2"],
        support_floor=-100.0,
        z_epi=0.0,
        delta_frac=0.0,
        seed=8,
    )
    res = solver.solve({"targets": {"y1": (2.0, 3.0), "y2": (99.0, 101.0)}})
    assert isinstance(res, Infeasible)
    # relaxation must mention ONLY the violated output y2, never the satisfied y1.
    assert "y2" in res.reason
    assert "y1 by" not in res.reason
    assert res.distance_to_feasible > 0.0


def test_partial_epistemic_narrow_box_not_mislabeled_unreachable():
    # mean=x hits the box centre (x=5 in [4.9,5.1]) but the box is narrower than
    # the ±κσ band AND epistemic is nonzero: infeasible, yet the mean IS in the
    # box and collecting runs reduces the needed relaxation. Must NOT be called
    # "genuinely unreachable / more data won't help" (re-review Finding 1).
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1], epi_fn=lambda x: np.array([0.3]))
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-100.0,
        kappa=2.0,
        z_epi=2.0,
        delta_frac=0.0,
        seed=9,
    )
    res = solver.solve({"targets": {"y": (4.9, 5.1)}})
    assert isinstance(res, Infeasible)
    assert 4.8 <= res.nearest_achievable["x"] <= 5.2  # mean-in-box point
    assert "epistemic" in res.reason.lower()  # data helps
    assert "unreachable" not in res.reason.lower()  # mean is NOT out of box


def test_aleatoric_narrow_box_not_labeled_delta_or_epistemic():
    # same narrow box, but NO epistemic and NO δ: the only cause is the box being
    # tighter than the ±κσ aleatoric band. Message must name aleatoric, not
    # claim process variation δ (delta_frac=0) or epistemic (re-review Finding 2).
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-100.0,
        kappa=2.0,
        z_epi=0.0,
        delta_frac=0.0,
        seed=10,
    )
    res = solver.solve({"targets": {"y": (4.9, 5.1)}})
    assert isinstance(res, Infeasible)
    assert "aleatoric" in res.reason.lower()
    assert "epistemic" not in res.reason.lower()
    assert "unreachable" not in res.reason.lower()


# ---------------------------------------------------------------------------
# determinism + diversity
# ---------------------------------------------------------------------------


def test_determinism_same_seed():
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    kw = dict(support_floor=-10.0, delta_frac=0.0, seed=11)
    spec = {"targets": {"y": (10.0, 12.0)}}
    a = PessimisticInverseSolver(model, [_var("x", 0, 10)], ["y"], **kw).solve(spec)
    b = PessimisticInverseSolver(model, [_var("x", 0, 10)], ["y"], **kw).solve(spec)
    assert isinstance(a, list) and isinstance(b, list)
    assert [c.recipe["x"] for c in a] == [c.recipe["x"] for c in b]


def test_returns_distinct_candidates():
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    solver = PessimisticInverseSolver(
        model, [_var("x", 0, 10)], ["y"], support_floor=-10.0, delta_frac=0.0, seed=12
    )
    res = solver.solve({"targets": {"y": (5.0, 17.0)}, "max_candidates": 3})
    assert isinstance(res, list)
    xs = [c.recipe["x"] for c in res]
    # audit D7: the wide [5,17] pre-image over x∈[0,10] must yield >= 2 distinct
    # candidates — without this bound the distinctness check passes vacuously if
    # diversity regressed to a single anchor.
    assert len(xs) >= 2
    assert len(xs) == len({round(x, 3) for x in xs})  # all distinct


def test_max_candidates_nonpositive_raises():
    # audit D3: q < 1 must fail loud, not return [] a caller misreads as INFEASIBLE.
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    solver = PessimisticInverseSolver(
        model, [_var("x", 0.0, 10.0)], ["y"], support_floor=-10.0, seed=0
    )
    with pytest.raises(ValueError, match="max_candidates must be >= 1"):
        solver.solve({"targets": {"y": (10.0, 12.0)}, "max_candidates": 0})


def test_diverse_preimage_spreads_the_manifold():
    # y = 2a − b over [0,5]²; target y∈[3.5,4.5] has a 1-D line pre-image. The
    # farthest-point selection must return recipes that SPREAD the line, not q
    # copies of one optimum (§8.7 non-injectivity / MFL weakness #2).
    model = _AnalyticModel(A=[[2.0, -1.0]], b=[0.0], ale=[0.1])
    solver = PessimisticInverseSolver(
        model,
        [_var("a", 0.0, 5.0), _var("b", 0.0, 5.0)],
        ["y"],
        support_floor=-10.0,
        z_epi=0.0,
        delta_frac=0.0,
        kappa=1.0,
        seed=13,
    )
    res = solver.solve({"targets": {"y": (3.5, 4.5)}, "max_candidates": 4})
    assert isinstance(res, list) and len(res) >= 3
    a_vals = np.array([c.recipe["a"] for c in res])
    # all lie on the pre-image (2a−b≈4) yet span a wide range of `a`.
    for c in res:
        assert abs(2.0 * c.recipe["a"] - c.recipe["b"] - 4.0) <= 1.0
    assert a_vals.max() - a_vals.min() > 1.0  # genuinely spread, not clustered


# ---------------------------------------------------------------------------
# GP integration (CI-safe synthetic) — the real backbone end to end
# ---------------------------------------------------------------------------


def _fit_gp_1d(seed=0, n=40):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, n)[:, None]
    y = (3.0 * X[:, 0] + 0.2 * np.sin(6.0 * X[:, 0]))[:, None]
    y = y + 0.02 * rng.standard_normal((n, 1))
    gp = GPForwardModel(n_restarts=2, seed=seed, max_iter=80).fit(X, y)
    return gp, X


def test_gp_inverse_reachable_then_verify():
    gp, X = _fit_gp_1d()
    solver = PessimisticInverseSolver(
        gp,
        [_var("x", 0.0, 1.0)],
        ["y"],
        X_train=X,
        z_epi=1.0,
        delta_frac=0.01,
        kappa=1.0,
        seed=0,
    )
    # y ~ 3x on [0,1]; target ~1.5 => x ~ 0.5, well inside the trained cloud.
    res = solver.solve({"targets": {"y": {"target": 1.5, "tol": 0.3}}})
    assert isinstance(res, list) and res
    for c in res:
        mu = float(np.atleast_1d(gp.predict(np.array([c.recipe["x"]])).mean)[0])
        assert 1.2 <= mu <= 1.8  # the surrogate agrees the recipe hits spec


def test_gp_inverse_out_of_range_is_infeasible():
    gp, X = _fit_gp_1d()
    solver = PessimisticInverseSolver(
        gp,
        [_var("x", 0.0, 1.0)],
        ["y"],
        X_train=X,
        z_epi=1.0,
        delta_frac=0.01,
        kappa=1.0,
        seed=0,
    )
    # y maxes near 3 on the trained range; ask for 5 => unreachable.
    res = solver.solve({"targets": {"y": {"target": 5.0, "tol": 0.2}}})
    assert isinstance(res, Infeasible)
    assert res.distance_to_feasible > 0.0


# ---------------------------------------------------------------------------
# tool-conditioned inversion (§8.3 split-plot / WP-I handoff)
# ---------------------------------------------------------------------------


def test_tool_bound_inversion():
    rng = np.random.default_rng(0)
    # tool A: y = 3x ; tool B: y = 3x + 0.5. Build records for both, fit the
    # multitask GP, then invert GIVEN tool B via for_tool.
    Xg = np.linspace(0.0, 1.0, 24)[:, None]
    XA, yA = Xg, (3.0 * Xg[:, 0])[:, None] + 0.02 * rng.standard_normal((24, 1))
    XB, yB = Xg, (3.0 * Xg[:, 0] + 0.5)[:, None] + 0.02 * rng.standard_normal((24, 1))
    X = np.vstack([XA, XB])
    Y = np.vstack([yA, yB])
    tools = ["A"] * 24 + ["B"] * 24
    mt = MultiToolGPForwardModel(n_restarts=2, seed=0, max_iter=80)
    mt.fit(X, Y, tools)

    boundB = mt.for_tool("B")
    solver = PessimisticInverseSolver(
        boundB,
        [_var("x", 0.0, 1.0)],
        ["y"],
        X_train=X,
        z_epi=1.0,
        delta_frac=0.01,
        kappa=1.0,
        seed=0,
    )
    # tool B has y = 3x + 0.5; target 2.0 => x ~ 0.5 on B (on A it'd be 0.667).
    res = solver.solve({"targets": {"y": {"target": 2.0, "tol": 0.3}}, "tool_id": "B"})
    assert isinstance(res, list) and res
    for c in res:
        mu = float(np.atleast_1d(boundB.predict(np.array([c.recipe["x"]])).mean)[0])
        assert 1.7 <= mu <= 2.3


# ---------------------------------------------------------------------------
# §5.7 / §13.2 full-ensemble re-validation (WP-E) — regression guards for the
# two adversarial-review findings (both LOW, both fixed).
# ---------------------------------------------------------------------------


class _ConformalStub:
    """ForwardModel whose predict carries a FIXED conformal_set (for the §13.2
    gate). Mean/support are constant and reach the box; only the conformal band
    can spill it."""

    def __init__(self, mean, ale, cs):
        self._mean = np.asarray(mean, float)
        self._ale = np.asarray(ale, float)
        self._cs = np.asarray(cs, float)  # (m, 2)

    def predict(self, x) -> PredictiveDistribution:
        x = np.asarray(x, float)
        single = x.ndim == 1
        n = 1 if single else x.shape[0]
        mean = np.broadcast_to(self._mean, (n, self._mean.shape[0])).copy()
        ale = np.broadcast_to(self._ale, mean.shape).copy()
        epi = np.zeros_like(mean)
        cs = np.broadcast_to(self._cs, (n, *self._cs.shape)).copy()
        if single:
            return PredictiveDistribution(mean[0], ale[0], epi[0], cs[0])
        return PredictiveDistribution(mean, ale, epi, cs)

    def support_score(self, x):
        x = np.asarray(x, float)
        return 1.0 if x.ndim == 1 else np.ones(x.shape[0])

    def jacobian(self, x) -> np.ndarray:
        return np.zeros((self._mean.shape[0], np.asarray(x, float).shape[0]))

    def update(self, records) -> None:  # pragma: no cover
        pass


def test_revalidation_floor_is_per_model():
    """The re-validation support floor is derived from the RE-VALIDATION model
    (support_score is a per-model quantity), so a full model on a DIFFERENT
    support scale than the fast surrogate does not spuriously reject a genuinely
    feasible recipe. (Pre-fix, the fast model's floor gated the full model's
    scores → false INFEASIBLE.)"""
    variables = [_var("x0", 0.0, 10.0), _var("x1", 0.0, 10.0)]
    X_train = np.array([[5, 5], [4, 6], [6, 4], [5.5, 4.5]], float)
    A, b, ale = [[1.0, 1.0]], [0.0], [0.5]
    fast = _AnalyticModel(A=A, b=b, ale=ale, support_fn=lambda x: 0.0)  # floor 0.0
    full = _AnalyticModel(A=A, b=b, ale=ale, support_fn=lambda x: -10.0)  # offset scale
    solver = PessimisticInverseSolver(
        fast,
        variables,
        ["y"],
        kappa=1.0,
        delta_frac=0.0,
        X_train=X_train,
        revalidation_model=full,
        seed=0,
    )
    res = solver.solve({"targets": {"y": (8.0, 12.0)}})
    assert isinstance(res, list) and len(res) >= 1  # NOT spuriously INFEASIBLE


def test_revalidation_conformal_rejection_diagnosed():
    """A rejection caused ONLY by the §13.2 conformal gate (the candidate is
    margin-feasible + on-support on the full model) yields a conformal-cause
    INFEASIBLE with a NONZERO distance_to_feasible — not the contradictory
    distance 0.0 + an epistemic 'collect runs' message."""
    variables = [_var("x0", 0.0, 10.0), _var("x1", 0.0, 10.0)]
    fast = _AnalyticModel(A=[[1.0, 1.0]], b=[0.0], ale=[0.5], support_fn=lambda x: 1.0)
    # full model: mean 10 (margin-feasible in [8,12]) but conformal band [5,15] spills
    full = _ConformalStub(mean=[10.0], ale=[0.5], cs=[[5.0, 15.0]])
    solver = PessimisticInverseSolver(
        fast,
        variables,
        ["y"],
        kappa=1.0,
        delta_frac=0.0,
        support_floor=0.0,
        revalidation_model=full,
        seed=0,
    )
    res = solver.solve({"targets": {"y": (8.0, 12.0)}})
    assert isinstance(res, Infeasible)
    assert res.distance_to_feasible > 0.0  # honest spill, not the contradictory 0.0
    assert "conformal" in res.reason
    assert "collect runs" not in res.reason  # not the epistemic mis-diagnosis


# ---------------------------------------------------------------------------
# F9 (audit 2026-07-17): the inverse above 2 input dimensions.
#
# Nothing in this repo had EVER run the inverse above d=2 — every result (MBE
# recipe=2 vars, sputtering=power/pressure, the M3 toy) sat at 2-D, so
# "dimension-agnostic" was an untested claim about the part of the problem that
# is hardest exactly where dimensionality bites. These tests pin it down, and
# they score against GROUND TRUTH (evaluate the TRUE function at the returned
# recipe) rather than against the model's own opinion, which is the only way to
# tell a real inverse from a surrogate agreeing with itself.
# ---------------------------------------------------------------------------


def _nd_truth(d: int, seed: int = 0):
    """Smooth d-dim, 2-output process with EVERY input dim active."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=d) / np.sqrt(d)
    V = rng.normal(size=d) / np.sqrt(d)

    def f(X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        y0 = 5.0 + 3.0 * np.tanh(X @ W) + 0.6 * np.sin(X @ V)
        y1 = 8.0 + 2.0 * (X @ V) + 0.4 * np.cos(X @ W)
        return np.stack([y0, y1], axis=-1)

    return f


def _nd_setup(d: int, n_train: int, seed: int = 0):
    from scipy.stats import qmc

    lo, hi = -2.0, 2.0
    truth = _nd_truth(d, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        X = qmc.scale(qmc.Sobol(d=d, scramble=True, seed=seed).random(n_train), [lo] * d, [hi] * d)
    Y = truth(X) + np.random.default_rng(seed + 1).normal(0, 0.05, size=(n_train, 2))
    model = GPForwardModel(n_restarts=2, seed=seed).fit(X, Y)
    variables = [ContinuousVariable(f"x{i}", lo, hi) for i in range(d)]
    return truth, X, model, variables


@pytest.mark.parametrize("d", [4, 8])
def test_inverse_returns_ground_truth_hits_above_two_dimensions(d):
    """The F9 gap, closed for real: solve in d dims, then evaluate the TRUE function
    at the returned recipe. Measured across d=2,4,6,8,10,15 during the audit — all
    FEASIBLE with 3/3 genuine ground-truth hits. Two dims are pinned here to keep the
    suite quick."""
    from scipy.stats import qmc

    truth, X, model, variables = _nd_setup(d, 12 * d)
    x_ref = qmc.scale(qmc.Sobol(d=d, scramble=True, seed=99).random(1), [-2.0] * d, [2.0] * d)[0]
    y_ref = truth(x_ref)[0]
    tol = 0.8
    spec = {
        "targets": {
            "y0": (y_ref[0] - tol, y_ref[0] + tol),
            "y1": (y_ref[1] - tol, y_ref[1] + tol),
        },
        "max_candidates": 3,
    }
    solver = PessimisticInverseSolver(
        model, variables=variables, output_keys=["y0", "y1"], X_train=X, seed=0
    )
    res = solver.solve(spec)
    assert not isinstance(res, Infeasible), f"d={d}: unexpectedly INFEASIBLE"
    for cand in res:
        xv = np.array([cand.recipe[f"x{i}"] for i in range(d)])
        y_true = truth(xv)[0]  # GROUND TRUTH, not model.predict
        assert y_ref[0] - tol <= y_true[0] <= y_ref[0] + tol, f"d={d}: y0 missed truth"
        assert y_ref[1] - tol <= y_true[1] <= y_ref[1] + tol, f"d={d}: y1 missed truth"


def test_restart_budget_scales_with_search_dimension():
    """Regression (audit 2026-07-17, F9): `n_restarts` was a FIXED 48 for every
    dimension — dense in 2-D, vanishing in 20-D. A starved multi-start degrades into a
    FALSE INFEASIBLE: we fail to FIND a recipe and report that none EXISTS, the exact
    confusion §8.8 exists to prevent. dim=2 MUST still be exactly 48 or every existing
    2-D result (M2, the AL loop) silently moves."""
    X = np.random.default_rng(0).uniform(0.0, 1.0, size=(30, 2))
    model = GPForwardModel(n_restarts=1, seed=0).fit(X, X.sum(axis=1, keepdims=True))

    def budget(d):
        vs = [ContinuousVariable(f"x{i}", 0.0, 1.0) for i in range(d)]
        Xd = np.random.default_rng(0).uniform(0.0, 1.0, size=(30, d))
        m = GPForwardModel(n_restarts=1, seed=0).fit(Xd, Xd.sum(axis=1, keepdims=True))
        return PessimisticInverseSolver(
            m, variables=vs, output_keys=["y"], X_train=Xd, seed=0
        ).n_restarts

    assert budget(2) == 48, "dim=2 must reproduce the historical default exactly"
    assert budget(1) == 48, "floored"
    assert budget(5) == 120
    assert budget(20) == 480
    # an explicit budget still wins
    vs = [ContinuousVariable(f"x{i}", 0.0, 1.0) for i in range(2)]
    s = PessimisticInverseSolver(
        model, variables=vs, output_keys=["y"], X_train=X, n_restarts=7, seed=0
    )
    assert s.n_restarts == 7


def test_simplex_restart_budget_uses_u_space_dimension():
    """The budget must scale with the FREE-coordinate count the optimizer actually
    searches (K-1 per simplex), not the recipe key count."""
    d_comp = 4  # a 4-component simplex contributes 3 free u-coords
    X = np.random.default_rng(0).uniform(0.1, 0.4, size=(30, d_comp))
    X = X / X.sum(axis=1, keepdims=True)
    model = GPForwardModel(n_restarts=1, seed=0).fit(X, X[:, :1])
    vs = [CompositionalVariable("alloy", tuple("abcd"))]
    s = PessimisticInverseSolver(
        model, variables=vs, output_keys=["y"], X_train=X, support_floor=-10.0, seed=0
    )
    assert s._rt.dim == d_comp - 1  # 4 components -> 3 free u-coords, not 4
    assert s.n_restarts == 72  # max(_MIN_RESTARTS, 24 * 3) -> 72, keyed on u-dim


# ---------------------------------------------------------------------------
# §8.3 declared couplings: soft barrier + HARD reject.
#
# Until 2026-07-17 the module docstring claimed linear couplings were "enforced by
# a soft penalty + reject" while the solver took no ConstraintSet at all — a process
# declaring a LinearConstraint got VIOLATING recipes back with feasibility_flag=True,
# a silent wrong answer. These tests pin the claim to the code. The load-bearing
# invariant is the REJECT, not the barrier: the barrier is an optimization aid and is
# allowed to be tuned badly or switched off, and a violating recipe must STILL never
# be certified.
# ---------------------------------------------------------------------------


def _coupled_setup():
    """y = a + b over [0,5]²; target y∈[4.5,5.5] has the line a+b≈5 as its pre-image.
    The coupling a−b ≥ 2 cuts that line to a≈[3.5,5] — so an unconstrained solve
    provably lands on BOTH sides of it."""
    model = _AnalyticModel(A=[[1.0, 1.0]], b=[0.0], ale=[0.1])
    variables = [_var("a", 0.0, 5.0), _var("b", 0.0, 5.0)]
    cs = ConstraintSet(linear=(LinearConstraint({"a": 1.0, "b": -1.0}, lower=2.0),))
    spec = {"targets": {"y": (4.5, 5.5)}, "max_candidates": 4}
    common = dict(support_floor=-10.0, z_epi=0.0, delta_frac=0.0, kappa=1.0, seed=21)
    return model, variables, cs, spec, common


def test_constrained_solve_returns_only_satisfying_recipes():
    """Every returned candidate satisfies ConstraintSet.is_satisfied — and the
    unconstrained solve of the SAME problem returns violating ones, so the wiring is
    what does it (without that second half the assertion passes vacuously).

    SCOPE, measured not assumed: this guards the WIRING, not the reject. Verified by
    mutation — deleting the ConstraintSet from the ctor (the audit's actual defect)
    turns it red, but deleting the hard reject alone leaves it GREEN, because the
    barrier by itself already steers this problem clear of the coupling. Do not read a
    pass here as evidence the reject works;
    `test_hard_reject_holds_with_the_barrier_switched_off` is the one that isolates it."""
    model, variables, cs, spec, common = _coupled_setup()

    free = PessimisticInverseSolver(model, variables, ["y"], **common).solve(spec)
    assert isinstance(free, list) and free
    assert any(not cs.is_satisfied(c.recipe) for c in free), (
        "the unconstrained solve must violate the coupling, else this test is vacuous"
    )

    res = PessimisticInverseSolver(model, variables, ["y"], constraints=cs, **common).solve(spec)
    assert isinstance(res, list) and res
    for c in res:
        assert c.feasibility_flag is True
        assert cs.is_satisfied(c.recipe), f"certified a violating recipe: {c.recipe}"
        assert c.recipe["a"] - c.recipe["b"] >= 2.0 - 1e-9


def test_hard_reject_holds_with_the_barrier_switched_off():
    """The reject is the safety property; the barrier is only an optimization aid. At
    constraint_penalty=0 the objective is IDENTICAL to the unconstrained one — the
    solver optimizes straight into the violating region — and not one violating recipe
    may still come back certified. A guard that only ever ran at the default weight
    could not tell 'rejected' apart from 'the penalty happened to steer clear'.

    This is THE reject guard, and it is the only one: verified by mutation that deleting
    the hard reject turns this test red while every other coupling test stays green."""
    model, variables, cs, spec, common = _coupled_setup()
    res = PessimisticInverseSolver(
        model, variables, ["y"], constraints=cs, constraint_penalty=0.0, **common
    ).solve(spec)
    if isinstance(res, Infeasible):
        return  # abstaining is honest; certifying a violator is not
    for c in res:
        # Independent re-derivation of the coupling a − b ≥ 2, NOT via
        # cs.is_satisfied — that is the same checker the hard reject calls, so if
        # ConstraintSet's linear branch ever regressed to always-true, an
        # is_satisfied-only assertion would pass vacuously and only the negative
        # attribution tests would catch it (audit finding, 2026-07-17). Computing
        # A@x − b here makes THIS safety test redden on a broken checker too.
        excursion = 2.0 - (c.recipe["a"] - c.recipe["b"])  # > 0 ⇒ violates a−b ≥ 2
        assert excursion <= 1e-9, (
            f"penalty=0 leaked a violating recipe past the hard reject: {c.recipe} "
            f"(a−b = {c.recipe['a'] - c.recipe['b']:.6f}, needs ≥ 2)"
        )
        assert cs.is_satisfied(c.recipe)  # and the checker agrees


def test_barrier_recovers_a_recipe_the_reject_alone_would_abstain_on():
    """The barrier earns its keep: pit it against a soft support reward that pulls
    straight OUT of the admissible half-plane. With the barrier off every restart
    slides out and the reject leaves nothing → a FALSE INFEASIBLE. With it on, the
    admissible pre-image is found. This is why the penalty exists at all — and why the
    reject is what makes it safe for the penalty to be merely a heuristic."""
    _model, variables, cs, spec, common = _coupled_setup()
    # support falls off as `a` grows, so λ_m·support pushes toward small a — exactly
    # the direction the coupling forbids.
    pushed = _AnalyticModel(A=[[1.0, 1.0]], b=[0.0], ale=[0.1], support_fn=lambda x: -x[0])
    off = PessimisticInverseSolver(
        pushed, variables, ["y"], constraints=cs, constraint_penalty=0.0, **common
    ).solve(spec)
    assert isinstance(off, Infeasible)  # reject alone: nothing admissible survives

    on = PessimisticInverseSolver(pushed, variables, ["y"], constraints=cs, **common).solve(spec)
    assert isinstance(on, list) and on
    assert all(cs.is_satisfied(c.recipe) for c in on)


def test_constraint_barrier_does_not_collapse_preimage_diversity():
    """`_CONSTRAINT_TAU` is a §8.7 knob, not merely a numerical one. A log-sigmoid is a
    smooth hinge, so its pull decays inward but never to zero — and the spec-feasible
    plateau is FLAT, so that residual is the only gradient along it and drags the whole
    pre-image into the constraint's interior. At τ_c=20 the returned set COLLAPSES to a
    single corner point; the shipped 60 keeps it spread. Pinned because a future
    "simplify the constants" would quietly cost the diverse pre-image §8.7 exists for."""
    model, variables, cs, spec, common = _coupled_setup()
    res = PessimisticInverseSolver(model, variables, ["y"], constraints=cs, **common).solve(spec)
    assert isinstance(res, list)
    a_vals = np.array([c.recipe["a"] for c in res])
    assert len(res) >= 3, f"barrier collapsed the pre-image to {len(res)} candidate(s)"
    assert a_vals.max() - a_vals.min() > 0.1, "candidates are one point in disguise"


def test_constraint_makes_target_unreachable_is_infeasible_naming_it():
    """A coupling that makes the target unreachable is a first-class INFEASIBLE whose
    reason NAMES the violated constraint — never a clipped point, and never the §8.8
    "genuinely unreachable ... relax the target" mis-diagnosis, which would send the
    operator off to change a process whose real problem is the declared coupling.

    y1 = a+b, y2 = b over [0,5]². The spec forces b≥4 ⇒ a≤1.5 ⇒ a−b ≤ −2.5, so the
    coupling a−b ≥ 2 excludes every solution — while the admissible half-plane is LARGE
    and is genuinely explored. Without the coupling the same spec is feasible."""
    variables = [_var("a", 0.0, 5.0), _var("b", 0.0, 5.0)]
    model = _AnalyticModel(A=[[1.0, 1.0], [0.0, 1.0]], b=[0.0, 0.0], ale=[0.1, 0.1])
    cs = ConstraintSet(linear=(LinearConstraint({"a": 1.0, "b": -1.0}, lower=2.0),))
    spec = {"targets": {"y1": (4.5, 5.5), "y2": (4.0, 5.0)}}
    common = dict(support_floor=-10.0, z_epi=0.0, delta_frac=0.0, kappa=1.0, seed=21)

    free = PessimisticInverseSolver(model, variables, ["y1", "y2"], **common).solve(spec)
    assert isinstance(free, list) and free, "without the coupling the spec is reachable"

    res = PessimisticInverseSolver(model, variables, ["y1", "y2"], constraints=cs, **common).solve(
        spec
    )
    assert isinstance(res, Infeasible)
    assert "constraint-blocked" in res.reason
    assert "linear" in res.reason and "'a'" in res.reason  # names the violated coupling
    assert "genuinely unreachable" not in res.reason  # not the §8.8 mis-diagnosis
    # the reported point is labelled as violating, not offered as usable.
    assert not cs.is_satisfied(res.nearest_achievable)
    assert "NOT a usable recipe" in res.reason
    assert res.distance_to_feasible > 0.0


def test_constraint_infeasible_never_certifies_the_nearest_point():
    """The OTHER constraint verdict branch: nothing admissible survives at all (here the
    barrier and the output term balance just outside a+b=6), so there is no
    spec-reaching violator to point at. §8.7: the abstention must still come back as an
    Infeasible whose reason NAMES the coupling — never a RecipeCandidate with the
    coupling quietly clipped.

    SCOPE: like the test above this guards the wiring and the verdict's attribution (it
    reddens when the ConstraintSet is ignored, or when `_constraint_blocked` stops
    attributing), NOT the reject in isolation — removing the reject alone leaves it
    green, because these restarts miss the spec regardless."""
    variables = [_var("a", 0.0, 5.0), _var("b", 0.0, 5.0)]
    model = _AnalyticModel(A=[[1.0, 1.0]], b=[0.0], ale=[0.1])
    cs = ConstraintSet(linear=(LinearConstraint({"a": 1.0, "b": 1.0}, upper=6.0),))
    res = PessimisticInverseSolver(
        model,
        variables,
        ["y"],
        constraints=cs,
        support_floor=-10.0,
        z_epi=0.0,
        delta_frac=0.0,
        kappa=1.0,
        seed=21,
    ).solve({"targets": {"y": (9.0, 11.0)}})
    assert isinstance(res, Infeasible)
    assert "constraint" in res.reason
    assert "linear" in res.reason


# -- constructor: what the transform already guarantees is VERIFIED, not re-enforced --


def test_box_constraint_narrower_than_its_free_variable_raises():
    """Two sources of truth for one interval. The §8.3 transform makes the VARIABLE's
    range exact, so a narrower declared box means the solver searches a space the
    process forbids and the reject silently bins the excess. Fail loud at construction
    and name the fix (narrow the variable) rather than quietly degrade."""
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    cs = ConstraintSet(box=(BoxConstraint("x", 2.0, 8.0),))
    with pytest.raises(ValueError, match="Two sources of truth for one interval"):
        PessimisticInverseSolver(
            model, [_var("x", 0.0, 10.0)], ["y"], support_floor=-10.0, constraints=cs
        )


def test_agreeing_box_and_simplex_declarations_add_no_penalty_row():
    """Box + simplex the transform already realizes are exact by construction: verified,
    then dropped. They must NOT be re-enforced — a barrier on them would cost search
    quality to guard a constraint that cannot be violated in the first place."""
    model = _AnalyticModel(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0], ale=[0.05, 0.05])
    variables = [_var("x", 0.0, 10.0), CompositionalVariable("alloy", ("ga", "in"))]
    cs = ConstraintSet(
        box=(BoxConstraint("x", 0.0, 10.0),),
        simplex=(SimplexConstraint(("alloy.ga", "alloy.in")),),
    )
    solver = PessimisticInverseSolver(
        model, variables, ["y1", "y2"], support_floor=-10.0, constraints=cs
    )
    assert solver._penalty_rows == ()


def test_simplex_the_transform_cannot_realize_raises_not_implemented():
    """A sum-to-total EQUALITY cannot ride on the barrier+reject built for
    inequalities: every restart would miss it and the solver would abstain
    unconditionally. Say NOT IMPLEMENTED loudly rather than ship a constraint path that
    always answers INFEASIBLE."""
    model = _AnalyticModel(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0], ale=[0.05, 0.05])
    variables = [CompositionalVariable("alloy", ("ga", "in"))]
    cs = ConstraintSet(simplex=(SimplexConstraint(("alloy.ga", "alloy.in"), total=2.0),))
    with pytest.raises(NotImplementedError, match="NOT exact by construction"):
        PessimisticInverseSolver(
            model, variables, ["f_ga", "f_in"], support_floor=-10.0, constraints=cs
        )


def test_constraint_on_a_non_free_variable_raises():
    """A coupling touching a hard-to-change / conditioning factor (§8.3) is not
    checkable on a candidate recipe, which carries only the free variables. Silently
    skipping it is exactly the silent wrong answer this wiring exists to kill."""
    model = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    cs = ConstraintSet(linear=(LinearConstraint({"x": 1.0, "gap": 2.0}, upper=5.0),))
    with pytest.raises(ValueError, match="not one of this solver's free variables"):
        PessimisticInverseSolver(
            model, [_var("x", 0.0, 10.0)], ["y"], support_floor=-10.0, constraints=cs
        )


def test_monotone_declarations_are_accepted_and_explicitly_not_enforced():
    """NOT IMPLEMENTED, pinned so nobody reads the constraint wiring as covering it.
    A MonotoneConstraint relates an OUTPUT to an INPUT, so it is not a property of a
    recipe at all — ConstraintSet.validate skips it and so does this solver. Here the
    model is strictly INCREASING in x while the declaration says decreasing: the solve
    proceeds and is_satisfied is True regardless, i.e. passing is_satisfied says
    NOTHING about monotonicity."""
    model = _AnalyticModel(A=[[2.0]], b=[1.0], ale=[0.05])
    cs = ConstraintSet(monotone=(MonotoneConstraint("y", "x", "decreasing"),))
    solver = PessimisticInverseSolver(
        model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.0,
        constraints=cs,
        seed=0,
    )
    assert solver._penalty_rows == ()  # nothing pointwise-checkable to enforce
    res = solver.solve({"targets": {"y": (10.0, 12.0)}})
    assert isinstance(res, list) and res
    for c in res:
        assert cs.is_satisfied(c.recipe)  # True despite the contradicted declaration


# -- constraints=None must not move a single existing number --------------------


def test_constraints_none_is_byte_identical_to_the_unwired_solver():
    """VALUE PIN. `constraints=None` is the default, and every existing result (M2, the
    AL loop, the M3 acceptance run, the dimensionality study) was produced through it,
    so the coupling wiring must not perturb it by a single ULP. These numbers were taken
    from the PRE-wiring solver and verified bit-identical afterwards across the feasible
    / diverse / unreachable / epistemic-limited / off-support / δ / multi-output /
    simplex / warm-start branches. A leak of the barrier or the reject into the default
    path moves them."""
    model = _AnalyticModel(A=[[2.0, -1.0]], b=[0.0], ale=[0.1])
    variables = [_var("a", 0.0, 5.0), _var("b", 0.0, 5.0)]
    solver = PessimisticInverseSolver(
        model,
        variables,
        ["y"],
        support_floor=-10.0,
        z_epi=0.0,
        delta_frac=0.0,
        kappa=1.0,
        seed=13,
    )
    assert solver.constraints is None
    assert solver._penalty_rows == ()  # no barrier term is added to the objective
    res = solver.solve({"targets": {"y": (3.5, 4.5)}, "max_candidates": 4})
    assert isinstance(res, list)
    assert [c.recipe["a"] for c in res] == [
        2.000133347735353,
        4.511713300703435,
        3.2704990939131373,
        3.8582482858393536,
    ]
    assert [c.recipe["b"] for c in res] == [
        0.0016767506523323907,
        4.998040991994523,
        2.537660432580658,
        3.705064242714137,
    ]

    # the INFEASIBLE path too — it routes through the nominal probe and the §8.8
    # taxonomy, both of which the wiring reaches into.
    epi_model = _AnalyticModel(
        A=[[1.0]],
        b=[0.0],
        ale=[0.1],
        epi_fn=lambda x: np.array([5.0 * np.exp(-((x[0] - 5.0) ** 2) / 0.5)]),
    )
    inf = PessimisticInverseSolver(
        epi_model,
        [_var("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        z_epi=2.0,
        delta_frac=0.0,
        seed=3,
    ).solve({"targets": {"y": (4.5, 5.5)}})
    assert isinstance(inf, Infeasible)
    assert inf.nearest_achievable["x"] == 5.0
    assert inf.distance_to_feasible == 97.0


# ---------------------------------------------------------------------------
# analytic objective gradient (opt-in, 2026-07-17) — §8.6 solver cost
#
# The headline risk here is NOT slowness, it is a WRONG gradient: a subtly bad
# descent direction degrades the multi-start silently, and in this system a
# degraded search surfaces as a FALSE INFEASIBLE (we failed to FIND a recipe and
# reported that none EXISTS) — the exact confusion §8.8 exists to prevent. So the
# gradient is verified against finite differences term by term AND end to end,
# before any speed claim is entertained.
#
# On the choice of fixture — this cost a real investigation, so it is written down.
# The gradient checks below do NOT use `_nd_setup`, and that is deliberate. Its
# `y1 = 8 + 2·(X@V) + 0.4·cos(X@W)` is very nearly LINEAR, so the GP fits ARD
# lengthscales of ~30–90 in standardized units; K is then near-singular, and
# `predict` carries ~1e-11 of its own floating-point noise. Central differences
# divide that noise by h, so on that fixture the FD "reference" is accurate to only
# ~1e-6 at h=1e-6 and gets WORSE as h shrinks (measured: 9.7e-8 at h=1e-3 → 1.9e-4
# at h=1e-7, the textbook error ∝ 1/h roundoff signature). Verified by the same
# ladder that `GPForwardModel.jacobian` — long-shipped and independently correct —
# "disagrees" with FD of its own `predict` by 2.7e-5 there. Checking a 1e-6-accurate
# analytic gradient against a 1e-6-accurate reference measures the reference. So
# `_grad_fixture` below fits a genuinely nonlinear process at a moderate output
# scale, where the FD reference is good to ~1e-8 and a 1e-6 assertion has teeth.
# ---------------------------------------------------------------------------

# The FD path's answer to `test_analytic_grad_is_off_by_default...`, recorded from
# the default (finite-difference) solver. This is NOT independent evidence that the
# feature left the FD path alone — the pre-existing tests above, several of which pin
# exact FD-path values, are that. What this pins is the DEFAULT: flip `analytic_grad`
# to True by default, or build the provider unconditionally, and these numbers move.
_FD_PIN = {"x0": 0.18990728432361959, "x1": 0.07718734317709641, "confidence": 0.9994639092674249}


def _grad_fixture(d: int, seed: int = 0):
    """A well-conditioned GP: genuinely nonlinear in every input, outputs O(1), so
    the fitted lengthscales stay moderate and `predict` is accurate to ~1e-15. See
    the section comment for why this matters."""
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, 2)) / np.sqrt(d)
    V = rng.normal(size=(d, 2)) / np.sqrt(d)

    def truth(X):
        X = np.atleast_2d(X)
        return np.tanh(X @ W) + 0.5 * np.sin(X @ V)

    X = rng.uniform(-1.0, 1.0, size=(12 * d, d))
    Y = truth(X) + 0.05 * rng.normal(size=(12 * d, 2))
    model = GPForwardModel(n_restarts=2, seed=seed).fit(X, Y)
    variables = [ContinuousVariable(f"x{i}", -1.0, 1.0) for i in range(d)]
    return truth, X, model, variables


def _grad_setup(d: int, seed: int = 0, delta: float = 0.02, constraints=None):
    _, X, model, variables = _grad_fixture(d, seed=seed)
    solver = PessimisticInverseSolver(
        model,
        variables=variables,
        output_keys=["y0", "y1"],
        X_train=X,
        delta_frac=delta,
        analytic_grad=True,
        constraints=constraints,
        seed=seed,
    )
    box = parse_targets(
        {"y0": {"target": 0.0, "tol": 0.8}, "y1": {"target": 0.0, "tol": 0.8}}, ["y0", "y1"]
    )
    return solver, box, np.array([0, 1])


def _central_fd(f, u, i, h):
    up, dn = np.array(u, float), np.array(u, float)
    up[i] += h
    dn[i] -= h
    return (f(up) - f(dn)) / (2.0 * h)


def _fd_grad(f, u, h=1e-4):
    """Richardson-extrapolated central differences: O(h⁴) truncation instead of
    O(h²), so the reference lands ~1e-8 and the analytic gradient is the more
    accurate of the two rather than the thing being measured."""
    g = np.empty(u.size)
    for i in range(u.size):
        d1 = _central_fd(f, u, i, h)
        d2 = _central_fd(f, u, i, h / 2.0)
        g[i] = (4.0 * d2 - d1) / 3.0
    return g


def _rel_grad_err(solver, box, out_idx, u, ignore_epi=False):
    """Normwise ‖g − g_fd‖_∞ / ‖g_fd‖_∞.

    Normwise rather than per-component on purpose: a per-component ratio divides the
    reference's noise floor by whatever that component happens to be, so a gradient
    entry near zero manufactures an arbitrarily large "relative error" that reports
    the reference's precision rather than the correctness of the maths.
    """
    _, g = solver._neg_objective_grad(u, box, out_idx, ignore_epi, False)
    g_fd = _fd_grad(lambda z: solver._neg_objective(z, box, out_idx, ignore_epi, False), u)
    return float(np.max(np.abs(g - g_fd))) / max(float(np.max(np.abs(g_fd))), 1.0)


@pytest.mark.parametrize("d", [1, 2, 5, 10])
@pytest.mark.parametrize("delta", [0.0, 0.02])
@pytest.mark.parametrize("ignore_epi", [False, True])
def test_analytic_gradient_matches_finite_differences(d, delta, ignore_epi):
    """THE headline guard: the closed-form ∂objective/∂u equals its finite-difference
    reference at randomly drawn u — for the δ-free case (no Hessian needed) AND the δ
    case (which needs ∂J/∂x, the Matérn-5/2 posterior-mean Hessian), on both the
    pessimistic objective and the §8.8 epistemic-free probe objective.

    Measured max normwise relative error over d ∈ {1,2,3,4,5,6,8,10,15,20} × δ ∈
    {0, 0.02} × ignore_epi × with/without couplings, 4 random u per cell: **5.41e-07**.
    """
    solver, box, out_idx = _grad_setup(d, seed=d, delta=delta)
    rng = np.random.default_rng(1000 + d)
    for _ in range(4):
        u = rng.uniform(-4.0, 4.0, size=d)
        rel = _rel_grad_err(solver, box, out_idx, u, ignore_epi)
        assert rel < 1e-6, f"d={d} delta={delta} ignore_epi={ignore_epi}: rel={rel:.3e}"


def test_analytic_gradient_error_tracks_the_reference_not_the_maths():
    """The load-bearing check behind every tolerance in this section, and the one that
    cannot be gamed by picking a friendly h.

    If the analytic gradient were WRONG, the discrepancy would plateau at the size of
    the bug however good the reference got. If it is RIGHT, the discrepancy is the
    REFERENCE's error and shrinks as the reference is refined. So: sweep h and assert
    the FD estimate converges TOWARD the analytic gradient. This is what licenses the
    claim that the analytic gradient is more accurate than the FD path it replaces.
    """
    solver, box, out_idx = _grad_setup(5, seed=5)
    u = np.random.default_rng(1005).uniform(-4.0, 4.0, size=5)
    _, g = solver._neg_objective_grad(u, box, out_idx, False, False)

    def f(z):
        return solver._neg_objective(z, box, out_idx, False, False)

    # plain central differences, deliberately un-extrapolated: error should fall ~h²
    errs = [
        float(np.max(np.abs(g - np.array([_central_fd(f, u, i, h) for i in range(5)]))))
        for h in (1e-2, 1e-3, 1e-4)
    ]
    assert errs[0] > errs[1] > errs[2], f"FD does not converge to the analytic g: {errs}"
    assert errs[2] < 1e-5, errs
    # ~h² truncation: a 10× smaller step should cut the error by ~100×. Loosely
    # bounded (>20×) because roundoff is already creeping in at h=1e-4.
    assert errs[0] / errs[1] > 20.0, errs


def test_analytic_gradient_matches_finite_differences_with_couplings():
    """The §8.3 barrier is part of the objective, so it is part of the gradient. Its
    log-sigmoid rows are the one term whose derivative is not routed through the model,
    so they get their own check."""
    cs = ConstraintSet(
        linear=[LinearConstraint(coefficients={"x0": 1.0, "x1": -0.5}, lower=-0.4, upper=0.4)]
    )
    solver, box, out_idx = _grad_setup(4, seed=4, constraints=cs)
    assert solver._penalty_rows, "vacuous unless a barrier row actually exists"
    rng = np.random.default_rng(77)
    for _ in range(4):
        u = rng.uniform(-3.0, 3.0, size=4)
        assert _rel_grad_err(solver, box, out_idx, u) < 1e-6


def _grad_fixture_simplex(seed: int = 11):
    """Well-conditioned 3-composition simplex GP: interior-weighted samples
    (Dirichlet conc=2, not the vertex-jamming conc=1), genuinely nonlinear O(1)
    output, 60 points. Keeps fitted lengthscales moderate so the FD reference is
    good to ~1e-8 and a 1e-6 assertion has teeth — the simplex analogue of
    `_grad_fixture`. The earlier inline fixture here (conc=1, sin/x² outputs, 40
    points) drove objective gradients to ~1e4 at near-vertex u, where central FD
    cannot verify to 1e-6 regardless of whether the maths is right — the simplex
    twin of the near-linear trap the `_grad_fixture` comment documents. Verified
    against a 60-digit mpmath reference: analytic gradient is correct to rel 5e-8
    even at the vertex-adjacent point the old test tripped on; worst normwise rel
    err on this fixture 2.4e-7 over 8 interior draws."""
    rng = np.random.default_rng(seed)
    X = rng.dirichlet(np.full(3, 2.0), size=60)
    W = rng.normal(size=(3, 1)) / np.sqrt(3)
    Y = np.tanh(X @ W) + 0.5 * np.sin(3.0 * X @ W) + 0.02 * rng.normal(size=(60, 1))
    model = GPForwardModel(n_restarts=2, seed=0).fit(X, Y)
    return model, X


def test_softmax_jacobian_formula_is_the_derivative_of_the_simplex_transform():
    """The one simplex-specific term, in TOTAL ISOLATION of the GP: the fixed-gauge
    softmax Jacobian x_a(δ_ab − x_b) over the K−1 free coords IS the derivative of
    `SimplexTransform.forward`. Fixture-independent, so it pins the formula even at
    the vertex-adjacent u=[0.95,0.005] that the full-objective FD reference cannot
    resolve (matches to ~1e-10 there). This is the guard that survives even if the
    e2e fixture below ever drifts — the softmax branch of `_dx_du` is otherwise
    exercised by nothing green (every other analytic test uses box variables only)."""
    st = SimplexTransform(3)
    assert st.dim == 2, "K−1 free coords"
    h = 1e-6
    for u in (np.array([0.95, 0.005]), np.array([0.3, -0.4]), np.array([-1.5, 2.0])):
        x = st.forward(u)  # length-K composition
        j_formula = np.array(
            [[x[a] * ((1.0 if a == b else 0.0) - x[b]) for b in range(2)] for a in range(3)]
        )
        j_fd = np.column_stack(
            [
                (st.forward(u + e) - st.forward(u - e)) / (2.0 * h)
                for e in (np.array([h, 0.0]), np.array([0.0, h]))
            ]
        )
        assert np.max(np.abs(j_formula - j_fd)) < 1e-8, u


def test_analytic_gradient_covers_the_simplex_block():
    """∂objective/∂u through the softmax parametrization end to end, on a
    WELL-CONDITIONED simplex fixture where the FD reference is trustworthy to ~1e-8.
    The softmax Jacobian x_a(δ_ab − x_b) — not the box sigmoid — is the one term a
    mixed-block recipe would otherwise get a silently wrong descent direction on, and
    it is guarded ONLY here (plus the isolated formula test above). Corrupting the
    source `_dx_du` softmax block — dropping the (−x_b) coupling — makes this fail at
    rel err 0.42, verified. The old inline fixture made the assertion unverifiable
    (~1e4 gradients); this one keeps a real 1e-6 assertion on that branch."""
    model, X = _grad_fixture_simplex()
    solver = PessimisticInverseSolver(
        model,
        variables=[CompositionalVariable("alloy", ["ga", "in", "al"])],
        output_keys=["y"],
        X_train=X,
        analytic_grad=True,
        seed=0,
    )
    box = parse_targets({"y": {"target": 0.0, "tol": 0.5}}, ["y"])
    assert solver._rt.dim == 2, "K−1 free coords"
    rng = np.random.default_rng(99)
    for _ in range(8):
        u = rng.uniform(-2.0, 2.0, size=2)
        assert _rel_grad_err(solver, box, np.array([0]), u) < 1e-6


def test_analytic_objective_value_matches_the_finite_difference_objective():
    """The analytic path must optimize the SAME objective, not merely a similar one.
    `_neg_objective_grad` re-derives the value alongside the gradient, so a drift in
    either would move the answer while every gradient check still passed. Measured max
    relative gap: 3.9e-15."""
    for d in (2, 6):
        solver, box, out_idx = _grad_setup(d, seed=d)
        rng = np.random.default_rng(21 + d)
        for _ in range(6):
            u = rng.uniform(-4.0, 4.0, size=d)
            for ie in (False, True):
                v_an, _ = solver._neg_objective_grad(u, box, out_idx, ie, False)
                v_fd = solver._neg_objective(u, box, out_idx, ie, False)
                assert abs(v_an - v_fd) <= 1e-11 * max(1.0, abs(v_fd))


def test_analytic_grad_terms_agree_with_the_models_public_methods():
    """The provider reads the GP's fitted state directly, so it could drift from what
    `predict`/`jacobian`/`support_score` actually return — and then the solver would be
    optimizing a model that does not ship. Values are held to 1e-12 relative (not
    bitwise: same maths, differently associated), and every derivative is checked
    against FD of the model's OWN public method, which localizes a break to a single
    formula instead of averaging it into one objective number."""
    from rig.inverse.pessimistic import _GPTermProvider

    for d in (2, 5):
        _, X, model, _ = _grad_fixture(d, seed=d)
        provider = _GPTermProvider(model, need_hessian=True)
        rng = np.random.default_rng(300 + d)
        for _ in range(3):
            x = rng.uniform(-0.9, 0.9, size=d)
            t = provider.terms(x)
            dist = model.predict(x)
            assert np.allclose(t.mu, dist.mean, rtol=1e-12, atol=1e-14)
            assert np.allclose(t.sig_epi, dist.epistemic_sigma, rtol=1e-12, atol=1e-14)
            assert np.allclose(t.sig_ale, dist.aleatoric_sigma, rtol=1e-12, atol=1e-14)
            assert np.allclose(t.jac, model.jacobian(x), rtol=1e-12, atol=1e-14)
            assert abs(t.support - model.support_score(x)) < 1e-14
            for i in range(d):
                # ∂σ_ale/∂x ≡ 0 is an ASSUMPTION the gradient leans on (§10.3 floor v0
                # fits ONE noise scalar per output). Assert it rather than trust it: an
                # input-dependent-noise backend must ADD a term here, not silently
                # inherit a zero.
                assert (
                    abs(
                        _central_fd(
                            lambda z, m=model: float(m.predict(z).aleatoric_sigma[0]), x, i, 1e-5
                        )
                    )
                    < 1e-12
                )
                fd_mu = _fd_grad_at(lambda z, m=model: m.predict(z).mean, x, i)
                assert np.max(np.abs(t.jac[:, i] - fd_mu)) < 1e-6
                fd_epi = _fd_grad_at(lambda z, m=model: m.predict(z).epistemic_sigma, x, i)
                assert np.max(np.abs(t.d_sig_epi[:, i] - fd_epi)) < 1e-6
                fd_sup = _fd_grad_at(lambda z, m=model: m.support_score(z), x, i)
                assert abs(t.d_support[i] - fd_sup) < 1e-6
                # the Hessian column, vs FD of the model's own analytic Jacobian
                fd_J = _fd_grad_at(lambda z, m=model: m.jacobian(z), x, i)
                assert np.max(np.abs(t.hess[:, :, i] - fd_J)) < 1e-6


def _fd_grad_at(f, x, i, h=1e-4):
    """Richardson central difference of a vector-valued f along axis i."""
    d1 = _central_fd(f, x, i, h)
    d2 = _central_fd(f, x, i, h / 2.0)
    return (4.0 * np.asarray(d2) - np.asarray(d1)) / 3.0


def test_analytic_grad_and_fd_paths_are_both_ground_truth_valid():
    """The two paths need NOT return the same NUMBER of recipes. An exact gradient
    drives L-BFGS-B to different — and generally MORE — distinct optima than a
    finite-difference estimate of it, and §8.7 diversity selection returns all the
    distinct ones it finds. Measured on this fixture: the FD path finds 1 recipe, the
    analytic path finds 3, and ALL are in-box on the true function; i.e. the FD path
    UNDER-EXPLORES the pre-image (documented, BUILD_LOG 2026-07-17). An earlier
    version asserted `len(fd) == len(an)`, which encoded the false premise that the
    two gradients converge to the same-sized optimum set.

    The real correctness property — and the one that actually matters for a
    pessimistic solver — is that NEITHER path ever presents a recipe that misses on
    GROUND TRUTH, and that the top-ranked recipe is valid on both. Scored against
    `truth()`, never the GP that proposed the recipe (audit F2)."""
    truth, X, model, variables = _grad_fixture(6, seed=6)
    y_ref = truth(np.full(6, 0.3))[0]
    tol = 0.4
    spec = {
        "targets": {"y0": (y_ref[0] - tol, y_ref[0] + tol), "y1": (y_ref[1] - tol, y_ref[1] + tol)},
        "max_candidates": 3,
    }
    kw = dict(variables=variables, output_keys=["y0", "y1"], X_train=X, seed=0)
    fd = PessimisticInverseSolver(model, **kw).solve(spec)
    an = PessimisticInverseSolver(model, analytic_grad=True, **kw).solve(spec)
    assert not isinstance(fd, Infeasible), "fixture must be feasible or this is vacuous"
    assert not isinstance(an, Infeasible), "analytic path abstained where FD did not"

    def in_box(cand):
        y = truth(np.array([cand.recipe[f"x{i}"] for i in range(6)]))[0]
        return abs(y[0] - y_ref[0]) <= tol and abs(y[1] - y_ref[1]) <= tol

    # THE false-success guard, applied to BOTH paths: no presented recipe misses on
    # ground truth. (The old test only checked the analytic path's recipes.)
    for cand in list(fd) + list(an):
        assert in_box(cand), f"presented a recipe that MISSES ground truth: {cand.recipe}"
    # Headline stability: the top-ranked recipe of each path is valid and confident.
    assert in_box(fd[0]) and in_box(an[0])
    assert fd[0].confidence > 0.9 and an[0].confidence > 0.9
    # Pin the under-exploration rather than hide it: the exact-gradient path finds at
    # least as many distinct optima as the FD path. If this ever inverts, the
    # under-exploration story has changed and it should be re-investigated.
    assert len(an) >= len(fd)


def test_analytic_grad_is_off_by_default_and_leaves_two_dim_results_untouched():
    """M2 and the AL loop are 2-D and were produced on the FD path, so the default must
    stay FD. Flipping `analytic_grad` to True by default — or building the provider
    unconditionally — moves these numbers, and this is where that gets caught rather
    than in a published result."""
    _, X, model, variables = _nd_setup(2, 24, seed=2)
    solver = PessimisticInverseSolver(
        model, variables=variables, output_keys=["y0", "y1"], X_train=X, seed=0
    )
    assert solver.analytic_grad is False
    assert solver._terms is None, "the default path must not build a gradient provider"
    res = solver.solve({"targets": {"y0": (4.5, 5.5), "y1": (7.5, 8.5)}, "max_candidates": 2})
    assert not isinstance(res, Infeasible)
    assert res[0].recipe["x0"] == pytest.approx(_FD_PIN["x0"], abs=1e-9)
    assert res[0].recipe["x1"] == pytest.approx(_FD_PIN["x1"], abs=1e-9)
    assert res[0].confidence == pytest.approx(_FD_PIN["confidence"], abs=1e-9)


def test_analytic_grad_refuses_a_model_it_cannot_differentiate():
    """Fail loud, never fall back. A silent revert to finite differences would leave the
    solve CORRECT but the speedup absent — an unfalsifiable claim, and the same shape as
    the 'command returned success unconditionally' defect this repo already ate once."""
    m = _AnalyticModel(A=[[1.0]], b=[0.0], ale=[0.1])
    with pytest.raises(ValueError, match="analytic_grad=True"):
        PessimisticInverseSolver(
            m,
            variables=[ContinuousVariable("x", 0.0, 1.0)],
            output_keys=["y"],
            support_floor=-10.0,
            analytic_grad=True,
        )


def test_analytic_grad_refuses_an_unfitted_gp():
    """An unfitted GP has no alpha/L to differentiate. Refuse at construction rather
    than raise something opaque from inside the hot loop."""
    with pytest.raises(ValueError, match="analytic_grad=True"):
        PessimisticInverseSolver(
            GPForwardModel(),
            variables=[ContinuousVariable("x", 0.0, 1.0)],
            output_keys=["y"],
            support_floor=-10.0,
            analytic_grad=True,
        )


def test_analytic_grad_sees_through_the_conformal_wrapper():
    """The §5.6 conformal wrapper only ADDS `conformal_set`; μ/σ_ale/σ_epi — the only
    fields the §8 margins read — pass through untouched. So differentiating the base GP
    is differentiating exactly what the objective consumes."""
    _, X, model, variables = _grad_fixture(3, seed=3)

    class _Wrap:
        def __init__(self, base):
            self.base = base

        def predict(self, x):
            return self.base.predict(x)

        def jacobian(self, x):
            return self.base.jacobian(x)

        def support_score(self, x):
            return self.base.support_score(x)

        def update(self, records):  # pragma: no cover - not exercised
            pass

    solver = PessimisticInverseSolver(
        _Wrap(model),
        variables=variables,
        output_keys=["y0", "y1"],
        X_train=X,
        analytic_grad=True,
        seed=0,
    )
    box = parse_targets(
        {"y0": {"target": 0.0, "tol": 0.8}, "y1": {"target": 0.0, "tol": 0.8}}, ["y0", "y1"]
    )
    assert _rel_grad_err(solver, box, np.array([0, 1]), np.array([0.4, -1.1, 0.7])) < 1e-6


def test_analytic_grad_cuts_model_evaluations_per_gradient():
    """The WHY, counted deterministically rather than timed: SciPy's finite-difference
    gradient costs `dim+1` objective evaluations (each a `predict` AND a `jacobian`),
    the analytic one costs 1. Wall-clock speedup is reported in BUILD_LOG — a timing
    assertion on a shared box is a flake generator, but the call count is exact and is
    the mechanism the speedup comes from."""
    d = 10
    _, X, model, variables = _grad_fixture(d, seed=d)
    counts = {"n": 0}
    real_predict = model.predict

    def counting_predict(x):
        counts["n"] += 1
        return real_predict(x)

    kw = dict(
        variables=variables, output_keys=["y0", "y1"], X_train=X, seed=0, n_restarts=2, max_iter=10
    )
    spec = {"targets": {"y0": {"target": 0.0, "tol": 0.8}, "y1": {"target": 0.0, "tol": 0.8}}}
    fd_solver = PessimisticInverseSolver(model, **kw)
    an_solver = PessimisticInverseSolver(model, analytic_grad=True, **kw)
    # patch AFTER construction: the §8.2 support floor is derived in __init__ and
    # would otherwise be counted, unequally, against the two paths.
    model.predict = counting_predict
    try:
        counts["n"] = 0
        fd_solver.solve(spec)
        fd_calls = counts["n"]
        counts["n"] = 0
        an_solver.solve(spec)
        an_calls = counts["n"]
    finally:
        model.predict = real_predict
    # the analytic path calls `predict` only from `_evaluate` (once per restart); the
    # hot loop is served by the provider, so the ratio is large and stable.
    assert an_calls * 5 < fd_calls, f"fd={fd_calls} analytic={an_calls}"
