"""M3 acceptance gate (implementation-plan §15.4, line: *'amortized proposal matches
per-query quality after refinement'*). End-to-end D2: a real §14.3 zuko flow generator
proposes recipes, the ONE per-query §8 pessimistic solver refines them. We assert the D2
engine reaches the same per-query QUALITY as the canonical cold multi-start solver — and
does so on a LIGHT refinement budget (1 cold start) where the cold solver alone fails,
which is the whole point of amortization. torch/zuko are the optional extra → skips when
absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from rig.interfaces import (  # noqa: E402
    ContinuousVariable,
    Infeasible,
    PredictiveDistribution,
)
from rig.inverse import (  # noqa: E402
    AmortizedInverseGenerator,
    AmortizedRefiner,
    PessimisticInverseSolver,
)


def _plateau(x):
    """Flat mean away from x≈8: y = 5·(1+tanh(4·(x−8))). Cold L-BFGS-B from the box
    centre (x=5) is trapped on the plateau; only a start near the x≈9 pre-image reaches
    y≈10. This makes the amortized warm-start's contribution measurable."""
    return 5.0 * (1.0 + np.tanh(4.0 * (np.asarray(x, float) - 8.0)))


class _PlateauForward:
    """The solver's analytic forward surrogate for the plateau process (1 output)."""

    def __init__(self, ale=0.1):
        self.ale = float(ale)

    def predict(self, x) -> PredictiveDistribution:
        x = np.asarray(x, float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mean = _plateau(Xq[:, 0])[:, None]
        ale = np.full_like(mean, self.ale)
        epi = np.zeros_like(mean)
        if single:
            return PredictiveDistribution(mean[0], ale[0], epi[0], None)
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


@pytest.fixture(scope="module")
def trained_generator():
    """A real zuko flow generator trained on the plateau process (x → y)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 10.0, size=(1600, 1))
    Y = _plateau(X[:, 0])[:, None] + 0.1 * rng.standard_normal((1600, 1))
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 10.0)],
        ["y"],
        n_members=2,
        transforms=3,
        hidden=(64, 64),
        max_epochs=150,
        region_hw=(0.25, 2.0),
        seed=0,
    )
    return gen.fit(X, Y)


def _solver(*, n_restarts, seed=0):
    return PessimisticInverseSolver(
        _PlateauForward(),
        [ContinuousVariable("x", 0.0, 10.0)],
        ["y"],
        support_floor=-10.0,
        delta_frac=0.0,
        n_restarts=n_restarts,
        seed=seed,
    )


_SPEC = {"targets": {"y": (9.0, 11.0)}}  # pre-image x ≳ 8.3


def test_amortized_proposals_land_in_the_feasible_basin(trained_generator):
    """Sanity: the flow's proposals for a high-y box concentrate at high x (the
    pre-image), i.e. it learned the inverse — so they are useful refinement seeds."""
    S = trained_generator.sample_array(_SPEC, 64)[:, 0]
    assert float(np.median(S)) > 8.0  # median proposal is in the x≳8 pre-image


def test_d2_matches_or_beats_cold_reference(trained_generator):
    """D2 (proposals + the SAME cold starts) is a SUPERSET of the cold multi-start, so
    it matches or beats the canonical solver's quality on every query."""
    solver = _solver(n_restarts=8)
    cold = solver.solve(_SPEC)
    d2 = AmortizedRefiner(trained_generator, solver, n_proposals=8).solve(_SPEC)
    assert isinstance(d2, list) and d2
    if isinstance(cold, list):
        assert max(c.confidence for c in d2) >= max(c.confidence for c in cold) - 1e-9


def test_d2_light_budget_matches_heavy_cold(trained_generator):
    """THE M3 GATE: amortized proposal matches per-query quality after refinement.
    D2 with a LIGHT budget (1 cold start + amortized proposals) reaches the same
    feasibility + quality as the HEAVY cold reference (24 Sobol starts), while the
    light cold solver alone is INFEASIBLE — amortization pays for the refinement budget."""
    cold_light = _solver(n_restarts=1).solve(_SPEC)
    cold_heavy = _solver(n_restarts=24).solve(_SPEC)
    d2_light = AmortizedRefiner(trained_generator, _solver(n_restarts=1), n_proposals=8).solve(
        _SPEC
    )

    assert isinstance(cold_light, Infeasible)  # 1 cold start (box centre) is trapped
    assert isinstance(cold_heavy, list) and cold_heavy  # 24 starts find the pre-image
    assert isinstance(d2_light, list) and d2_light  # D2 with 1 start + proposals succeeds
    # quality parity: D2-light's best pessimistic confidence matches heavy cold's
    assert max(c.confidence for c in d2_light) >= max(c.confidence for c in cold_heavy) - 0.02
    # and every refined recipe is a genuine feasible recipe in the pre-image basin
    assert all(c.feasibility_flag and c.recipe["x"] >= 8.0 for c in d2_light)
