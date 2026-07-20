"""Property tests for constraint-by-construction transforms (implementation-plan §8.3, §13.3)."""

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from rig.interfaces import CompositionalVariable, ContinuousVariable
from rig.transforms import BoxTransform, RecipeTransform, SimplexTransform

finite_u = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


# ---------------------------------------------------------------------------
# BoxTransform: feasible for ALL u
# ---------------------------------------------------------------------------


# too_slow suppressed: it fires on Hypothesis INPUT-GENERATION timing, which is
# an artifact of CPU load when the (heavy) WP-E torch tests run in the same
# session — not a property failure. The invariant under test is unaffected.
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(u=arrays(np.float64, 3, elements=finite_u))
def test_box_output_always_in_bounds(u):
    t = BoxTransform(lower=[0.0, -5.0, 100.0], upper=[1.0, 5.0, 900.0])
    x = t.forward(u)
    assert np.all(x >= t.lower)
    assert np.all(x <= t.upper)


@given(u=arrays(np.float64, 2, elements=st.floats(-20, 20)))
def test_box_roundtrip_idempotent(u):
    """u -> x -> u' -> x'' must reproduce x (transform-level idempotence)."""
    t = BoxTransform(lower=[0.0, 200.0], upper=[10.0, 1200.0])
    x = t.forward(u)
    x2 = t.forward(t.inverse(x))
    np.testing.assert_allclose(x2, x, rtol=1e-9, atol=1e-9)


def test_box_extreme_u_saturates_not_nan():
    t = BoxTransform(lower=[0.0], upper=[1.0])
    for u in (np.array([1e300]), np.array([-1e300])):
        x = t.forward(u)
        assert np.all(np.isfinite(x)) and 0.0 <= x[0] <= 1.0


def test_box_inverse_rejects_out_of_bounds():
    # audit D1: an out-of-bounds recipe value must raise, not be silently
    # clamped to a fabricated finite u.
    t = BoxTransform(lower=[0.0, 200.0], upper=[10.0, 1200.0])
    with pytest.raises(ValueError, match="outside"):
        t.inverse(np.array([12.0, 500.0]))  # 12 > upper 10
    with pytest.raises(ValueError, match="outside"):
        t.inverse(np.array([5.0, 100.0]))  # 100 < lower 200


def test_box_inverse_accepts_exact_bounds():
    # the saturated image of forward() (exact lo/hi) round-trips without raising.
    t = BoxTransform(lower=[0.0], upper=[1.0])
    assert np.isfinite(t.inverse(np.array([1.0])))  # p == 1 exactly, allowed
    assert np.isfinite(t.inverse(np.array([0.0])))  # p == 0 exactly, allowed


def test_recipe_transform_inverse_rejects_out_of_bounds():
    # audit D1: RecipeTransform.inverse inherits the guard for continuous vars.
    rt = RecipeTransform([ContinuousVariable("temp", 200.0, 1200.0)])
    with pytest.raises(ValueError, match="outside"):
        rt.inverse({"temp": 1500.0})


def test_box_inverse_rejects_non_finite():
    # finding A: NaN compares False against both `p < -tol` and `p > 1+tol`, so
    # the D1 fail-loud guard was silently bypassed and a NaN u fell through.
    # +/-inf already raised (inf comparisons are well-defined) but are included
    # here as a fixed regression fence alongside NaN.
    t = BoxTransform(lower=[0.0], upper=[1.0])
    for bad in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="outside|non-finite"):
            t.inverse(np.array([bad]))


# ---------------------------------------------------------------------------
# SimplexTransform: non-negative, sum-to-1, exact by construction
# ---------------------------------------------------------------------------


@given(u=arrays(np.float64, 3, elements=finite_u))
def test_simplex_output_on_simplex(u):
    t = SimplexTransform(n_components=4)
    x = t.forward(u)
    assert np.all(x >= 0.0)
    assert abs(x.sum() - 1.0) < 1e-9


@given(u=arrays(np.float64, 2, elements=st.floats(-15, 15)))
def test_simplex_roundtrip_idempotent(u):
    t = SimplexTransform(n_components=3)
    x = t.forward(u)
    x2 = t.forward(t.inverse(x))
    np.testing.assert_allclose(x2, x, rtol=1e-9, atol=1e-9)


def test_simplex_boundary_inverse_finite():
    t = SimplexTransform(n_components=3)
    u = t.inverse(np.array([1.0, 0.0, 0.0]))  # boundary point
    assert np.all(np.isfinite(u))
    x = t.forward(u)
    assert abs(x.sum() - 1.0) < 1e-9


def test_simplex_inverse_rejects_out_of_simplex():
    # finding B: a genuinely out-of-simplex input (negative share, or sum far
    # from 1) must raise like BoxTransform.inverse does, not be silently
    # clamped/renormalized into a fabricated finite u.
    t = SimplexTransform(n_components=3)
    with pytest.raises(ValueError, match="simplex"):
        t.inverse(np.array([-0.1, 0.5, 0.6]))  # negative component
    with pytest.raises(ValueError, match="simplex"):
        t.inverse(np.array([1.0, 0.5, 0.5]))  # sum == 2, far from 1


def test_simplex_inverse_tolerates_float_drift():
    # float-drift-sized deviations (same magnitude as BoxTransform's own edge
    # tolerance, §8.3) must still round-trip -- this is the exact size of noise
    # RecipeTransform.forward()'s own softmax normalization and downstream
    # torch round-trips (amortized.py, pessimistic.py warm starts) produce.
    t = SimplexTransform(n_components=3)
    x = np.array([0.5, 0.3, 0.2]) + 1e-12
    u = t.inverse(x)
    assert np.all(np.isfinite(u))
    x2 = t.forward(u)
    np.testing.assert_allclose(x2, x / x.sum(), rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# RecipeTransform: composed u-vector <-> typed recipe dict
# ---------------------------------------------------------------------------

VARS = [
    ContinuousVariable(name="temperature", lower=200.0, upper=1200.0, unit="K"),
    CompositionalVariable(name="alloy", components=("ga", "al", "in")),
    ContinuousVariable(name="pressure", lower=1e-8, upper=1e-4, unit="Pa"),
]


@settings(max_examples=200)
@given(u=arrays(np.float64, 4, elements=st.floats(-15, 15)))
def test_recipe_transform_valid_and_roundtrip(u):
    t = RecipeTransform(VARS)
    assert t.dim == 4  # 1 + (3-1) + 1
    r = t.forward(u)
    assert 200.0 <= r["temperature"] <= 1200.0
    assert 1e-8 <= r["pressure"] <= 1e-4
    comps = [r["alloy.ga"], r["alloy.al"], r["alloy.in"]]
    assert all(c >= 0 for c in comps)
    assert abs(sum(comps) - 1.0) < 1e-9
    r2 = t.forward(t.inverse(r))
    for k in r:
        np.testing.assert_allclose(r2[k], r[k], rtol=1e-9, atol=1e-9)


def test_recipe_transform_rejects_categorical():
    import pytest

    from rig.interfaces import CategoricalVariable

    with pytest.raises(TypeError):
        RecipeTransform([CategoricalVariable(name="chamber", levels=("A", "B"))])
