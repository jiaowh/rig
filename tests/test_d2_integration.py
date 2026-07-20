"""D2 inverse-engine integration (implementation-plan §2.2 D2, §14.3): the amortized
proposal seeds the ONE per-query pessimistic §8 refinement. These tests are torch-free
— they drive the solver's ``warm_start_recipes`` path and the :class:`AmortizedRefiner`
composition with a controllable analytic forward model and a fake proposer, so they run
without the ``[torch]`` extra. The real-generator end-to-end M3 acceptance (the gate
'amortized proposal matches per-query quality after refinement') lives in
``test_m3_acceptance.py``."""

from __future__ import annotations

import numpy as np

from rig.interfaces import (
    ContinuousVariable,
    Infeasible,
    InverseSolver,
    PredictiveDistribution,
    RecipeCandidate,
)
from rig.inverse import AmortizedRefiner, PessimisticInverseSolver


class _PlateauModel:
    """1-D forward with a FLAT mean away from x≈8 (``y = 5·(1+tanh(4·(x−8)))``), so the
    gradient at the box centre x=5 is ~1e-9 → L-BFGS-B from the centre cannot climb to
    the pre-image. Only a start already near x≈9 reaches ``y≈10``. Lets us prove a warm
    start is genuinely consumed + refined (cold-from-centre stays INFEASIBLE)."""

    def __init__(self, ale=0.1):
        self.ale = float(ale)

    def _y(self, xi):
        return 5.0 * (1.0 + np.tanh(4.0 * (xi[0] - 8.0)))

    def predict(self, x) -> PredictiveDistribution:
        x = np.asarray(x, float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mean = np.array([[self._y(xi)] for xi in Xq])
        ale = np.full_like(mean, self.ale)
        epi = np.zeros_like(mean)
        if single:
            mean, ale, epi = mean[0], ale[0], epi[0]
        return PredictiveDistribution(mean, ale, epi, None)

    def jacobian(self, x) -> np.ndarray:
        xi = np.asarray(x, float)
        sech2 = 1.0 - np.tanh(4.0 * (xi[0] - 8.0)) ** 2
        return np.array([[5.0 * 4.0 * sech2]])

    def support_score(self, x):
        x = np.asarray(x, float)
        return 1.0 if x.ndim == 1 else np.ones(x.shape[0])

    def update(self, records) -> None:  # pragma: no cover
        pass


class _FakeProposer:
    """Stands in for the §14.3 amortized generator: returns fixed recipe dicts,
    ignoring the spec — so the test controls exactly what seeds the refinement."""

    def __init__(self, recipes):
        self._recipes = recipes
        self.calls: list[int] = []

    def sample(self, spec, n):
        self.calls.append(n)
        return [dict(r) for r in self._recipes[:n]]


def _solver(*, n_restarts, max_iter=200, seed=0):
    return PessimisticInverseSolver(
        _PlateauModel(),
        [ContinuousVariable("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.0,
        n_restarts=n_restarts,
        max_iter=max_iter,
        seed=seed,
    )


_TARGET = {"targets": {"y": (9.0, 11.0)}}  # needs x well above 8 (mean → 10)


# --- solver warm-start mechanics --------------------------------------------


def test_cold_single_start_from_centre_is_infeasible():
    """Baseline: with only the box-centre start (x=5, on the flat plateau) the cold
    solver cannot reach the pre-image → INFEASIBLE. This is the gap a warm start fills."""
    res = _solver(n_restarts=1).solve(_TARGET)
    assert isinstance(res, Infeasible)


def test_warm_start_recipe_is_refined_into_a_candidate():
    """A warm-start recipe near the pre-image (x≈9) turns the SAME cold-infeasible
    query FEASIBLE — proof the seed is consumed and refined by the §8 objective."""
    res = _solver(n_restarts=1).solve({**_TARGET, "warm_start_recipes": [{"x": 9.0}]})
    assert isinstance(res, list) and res
    assert all(isinstance(c, RecipeCandidate) for c in res)
    assert all(8.0 <= c.recipe["x"] <= 10.0 for c in res)  # stayed in the feasible basin


def test_warm_start_absent_is_byte_identical_to_cold():
    """No key, empty list, and None all take the cold Sobol path unchanged."""
    solver = _solver(n_restarts=8)
    base = solver.solve(_TARGET)
    for warm in ([], None):
        other = solver.solve({**_TARGET, "warm_start_recipes": warm})
        assert type(other) is type(base)
        if isinstance(base, list):
            assert [c.recipe["x"] for c in other] == [c.recipe["x"] for c in base]


def test_warm_start_never_worsens_the_result():
    """Warm seeds are a SUPERSET of the cold starts, so the best pessimistic confidence
    can only rise or tie — never fall (the D2 'matches or beats' guarantee, math-level)."""
    solver = _solver(n_restarts=8)
    cold = solver.solve(_TARGET)
    warm = solver.solve({**_TARGET, "warm_start_recipes": [{"x": 9.0}, {"x": 8.7}]})
    assert isinstance(warm, list) and warm  # warm is feasible
    if isinstance(cold, list):
        assert max(c.confidence for c in warm) >= max(c.confidence for c in cold) - 1e-9


# --- AmortizedRefiner (D2 engine) -------------------------------------------


def test_refiner_conforms_to_inverse_solver_protocol():
    refiner = AmortizedRefiner(_FakeProposer([{"x": 9.0}]), _solver(n_restarts=1))
    assert isinstance(refiner, InverseSolver)  # structural: has solve(spec)


def test_refiner_routes_proposals_as_warm_starts():
    """The engine turns the cold-infeasible query FEASIBLE by seeding the refiner with
    the generator's proposals — and draws exactly n_proposals of them."""
    prop = _FakeProposer([{"x": 9.0}, {"x": 8.5}, {"x": 9.3}])
    refiner = AmortizedRefiner(prop, _solver(n_restarts=1), n_proposals=3)
    res = refiner.solve(_TARGET)
    assert isinstance(res, list) and res
    assert prop.calls == [3]  # drew n_proposals proposals, once


def test_refiner_propose_passthrough():
    prop = _FakeProposer([{"x": 9.0}, {"x": 8.5}])
    refiner = AmortizedRefiner(prop, _solver(n_restarts=1), n_proposals=2)
    assert refiner.propose(_TARGET) == [{"x": 9.0}, {"x": 8.5}]


def test_refiner_rejects_bad_n_proposals():
    import pytest

    with pytest.raises(ValueError, match="n_proposals"):
        AmortizedRefiner(_FakeProposer([]), _solver(n_restarts=1), n_proposals=0)


def test_rescue_is_attributable_to_proposal_quality():
    """The D2 rescue must come from proposal QUALITY, not merely from 'an extra start'.
    On the cold-infeasible query: a GOOD proposer (x≈9, in the pre-image) rescues it;
    a GARBAGE proposer (x≈1, on the trapped flat plateau) does NOT — so a passing D2
    result is genuinely attributable to what the generator proposes, rebutting the
    'garbage proposals would pass too' objection."""
    good = AmortizedRefiner(_FakeProposer([{"x": 9.0}]), _solver(n_restarts=1), n_proposals=1)
    garbage = AmortizedRefiner(_FakeProposer([{"x": 1.0}]), _solver(n_restarts=1), n_proposals=1)
    assert isinstance(good.solve(_TARGET), list)  # good proposal → feasible
    assert isinstance(garbage.solve(_TARGET), Infeasible)  # garbage proposal → still infeasible
