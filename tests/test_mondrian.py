"""Mondrian / group-conditional conformal (implementation-plan §5.6, §20). Synthetic only.

Covers, per the build spec:
  - the exact per-group ``ceil((1-alpha)(n_g+1))`` quantile rule, INCLUDING the
    honest per-group ``+inf`` small-group branch and the ``min_group_n`` pooled
    fallback trigger;
  - interface parity: the §8 solver's default §13.2 gate consumes
    :class:`MondrianConformalForwardModel` UNMODIFIED (a wrapped solve returns
    conformal-checked candidates);
  - marginal coverage preserved on an exchangeable synthetic;
  - the SELECTED-POINT mechanism twin: a solver-selected point in a region the
    POOLED quantile under-covers is rejected by the Mondrian gate but admitted by
    the pooled gate (the d=8-style false-success miss);
  - determinism;
  - group_fn-uses-predicted-only (no y leakage).

Two red-proofs are inlined as commented mutations with the value they produce
(the per-group quantile rule and the selected-point mechanism), so a reviewer can
re-break them and watch the assertion go red.
"""

from __future__ import annotations

import numpy as np
import pytest

from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.calibration.mondrian import (
    MondrianConformalCalibrator,
    MondrianConformalForwardModel,
    finite_quantile_floor,
    predicted_magnitude_group_fn,
)
from rig.interfaces import (
    ContinuousVariable,
    Infeasible,
    PredictiveDistribution,
    RecipeCandidate,
)
from rig.inverse import PessimisticInverseSolver

ALPHA = 0.1


# ---------------------------------------------------------------------------
# controllable stub models
# ---------------------------------------------------------------------------


class _MagnitudeModel:
    """Single-output model whose predicted mean == x (identity), constant sigma.

    Lets a test place a calibration point in a chosen magnitude group simply by
    choosing x, and control the residual it contributes (via the y passed to fit).
    """

    def __init__(self, ale: float = 1.0, epi: float = 0.0) -> None:
        self.ale = float(ale)
        self.epi = float(epi)

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mu = Xq[:, 0:1]  # mean = first coordinate
        ale = np.full_like(mu, self.ale)
        epi = np.full_like(mu, self.epi)
        if single:
            return PredictiveDistribution(mu[0], ale[0], epi[0], None)
        return PredictiveDistribution(mu, ale, epi, None)

    def support_score(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        return 0.0 if x.ndim == 1 else np.zeros(x.shape[0])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[1.0]])

    def update(self, records) -> None:  # pragma: no cover
        pass


def _tertile_edges_from(values: np.ndarray) -> np.ndarray:
    """(1, 2) interior tertile edges (1/3, 2/3 quantiles) for one output."""
    e = np.quantile(np.asarray(values, dtype=float), [1.0 / 3.0, 2.0 / 3.0])
    return e[None, :]


# ---------------------------------------------------------------------------
# per-group quantile rule: pooled fallback + honest +inf branch
# ---------------------------------------------------------------------------


def test_finite_quantile_floor_matches_conformal_rule():
    # ceil((1-alpha)(n+1)) <= n first holds at n=9 for alpha=0.1, n=4 for 0.2.
    assert finite_quantile_floor(0.1) == 9
    assert finite_quantile_floor(0.2) == 4
    for alpha in (0.05, 0.1, 0.2, 0.33):
        n = finite_quantile_floor(alpha)
        assert np.ceil((1.0 - alpha) * (n + 1)) <= n
        assert np.ceil((1.0 - alpha) * n) > (n - 1)  # n-1 would overflow


def test_per_group_quantile_exact_rule_and_pooled_fallback():
    # Two groups by predicted magnitude: "low" (x<0) rich, "high" (x>=0) SPARSE.
    # min_group_n=10 forces the sparse high group to borrow the pooled quantile,
    # while the rich low group keeps its own exact ceil-rule quantile.
    model = _MagnitudeModel(ale=1.0)
    edges = np.array(
        [[0.0, 0.0]]
    )  # digitize -> {<0:low(0), ==0:mid(1) unused, >0:high} ; use 2 edges
    # Build a 3-label partition but only populate low/high; mid stays empty.
    group_fn = predicted_magnitude_group_fn(edges, labels=("low", "mid", "high"))

    rng = np.random.default_rng(0)
    x_low = np.linspace(-5.0, -0.1, 30)[:, None]
    x_high = np.linspace(0.1, 5.0, 12)[:, None]
    X = np.vstack([x_low, x_high])
    # residual = y - mu = y - x ; set residuals with KNOWN spreads per group.
    resid_low = rng.normal(0.0, 1.0, size=(30, 1))
    resid_high = rng.normal(0.0, 5.0, size=(12, 1))  # high group much noisier
    Y = X + np.vstack([resid_low, resid_high])

    cal = MondrianConformalCalibrator(group_fn, alpha=ALPHA, min_group_n=10)
    cal.fit(model, X, Y)

    scores = np.abs(Y - X)[:, 0]  # sigma=1 -> standardized score == |resid|
    low_scores = np.sort(scores[:30])
    all_scores = np.sort(scores)  # pooled

    table, pooled = cal._kappa_table(None)
    d = table[0]

    # low group (n_g=30 >= min_group_n=10): its OWN exact quantile, k=ceil(.9*31)=28.
    k_low = int(np.ceil((1.0 - ALPHA) * (30 + 1)))
    assert k_low == 28
    assert d["low"] == pytest.approx(low_scores[k_low - 1])
    # RED-PROOF: k_low-1 -> k_low (off-by-one) makes this fail (picks the 29th score).

    # high group (n_g=12 >= min_group_n=10) keeps its OWN quantile, k=ceil(.9*13)=12.
    high_scores = np.sort(scores[30:])
    k_high = int(np.ceil((1.0 - ALPHA) * (12 + 1)))
    assert k_high == 12
    assert d["high"] == pytest.approx(high_scores[k_high - 1])

    # pooled quantile over all 42 points, k=ceil(.9*43)=39 (the fallback band).
    k_pool = int(np.ceil((1.0 - ALPHA) * (42 + 1)))
    assert k_pool == 39
    assert pooled[0] == pytest.approx(all_scores[k_pool - 1])


def test_pooled_fallback_triggers_below_min_group_n():
    # A high group with n_g=6 and min_group_n=10 -> pooled fallback (finite),
    # NOT a per-group +inf (which 6 points at alpha=0.1 would give: ceil(.9*7)=7>6).
    model = _MagnitudeModel(ale=1.0)
    edges = np.array([[0.0, 0.0]])
    group_fn = predicted_magnitude_group_fn(edges, labels=("low", "mid", "high"))
    rng = np.random.default_rng(1)
    x_low = np.linspace(-5.0, -0.1, 30)[:, None]
    x_high = np.linspace(0.1, 5.0, 6)[:, None]  # only 6 -> underpowered
    X = np.vstack([x_low, x_high])
    Y = X + rng.normal(0.0, 1.0, size=X.shape)
    cal = MondrianConformalCalibrator(group_fn, alpha=ALPHA, min_group_n=10)
    cal.fit(model, X, Y)
    table, pooled = cal._kappa_table(None)
    assert np.isfinite(pooled[0])
    assert table[0]["high"] == pooled[0]  # borrowed pooled, finite
    assert np.isfinite(table[0]["high"])


def test_honest_per_group_inf_branch_when_min_group_n_disabled():
    # Set min_group_n=1 (disable the pooled fallback) and give the high group only
    # 6 points: ceil(.9*7)=7 > 6 -> the honest per-group +inf, not a borrowed band.
    model = _MagnitudeModel(ale=1.0)
    edges = np.array([[0.0, 0.0]])
    group_fn = predicted_magnitude_group_fn(edges, labels=("low", "mid", "high"))
    rng = np.random.default_rng(2)
    x_low = np.linspace(-5.0, -0.1, 30)[:, None]
    x_high = np.linspace(0.1, 5.0, 6)[:, None]
    X = np.vstack([x_low, x_high])
    Y = X + rng.normal(0.0, 1.0, size=X.shape)
    cal = MondrianConformalCalibrator(group_fn, alpha=ALPHA, min_group_n=1)
    cal.fit(model, X, Y)
    table, _ = cal._kappa_table(None)
    assert np.isinf(table[0]["high"])  # honest small-group +inf
    assert np.isfinite(table[0]["low"])
    # and the band at a high-group point is unbounded (the §13.2 gate would abstain)
    band = cal.interval(np.array([2.0]))  # (1, 2)
    assert np.isneginf(band[0, 0]) and np.isposinf(band[0, 1])


# ---------------------------------------------------------------------------
# grouping uses the PREDICTED mean only (no y leakage)
# ---------------------------------------------------------------------------


def test_group_fn_uses_predicted_mean_not_observed_y():
    # Construct a case where grouping by observed y would DIFFER from grouping by
    # predicted mean, and pin that the Mondrian path follows the PREDICTED mean.
    # model mean == x. Two calibration clusters:
    #   cluster A: x=-3 (predicted low), but y=+10 (observed high) — big residual.
    #   cluster B: x=+3 (predicted high), y=+3 (observed high) — small residual.
    # If grouping used observed y, A and B would share the "high" group and its
    # quantile would be dominated by A's huge residual. Grouping by PREDICTED mean
    # puts A in "low", so the "high" group's quantile reflects only B's small
    # residuals — that is what we assert.
    model = _MagnitudeModel(ale=1.0)
    edges = np.array([[0.0, 0.0]])
    group_fn = predicted_magnitude_group_fn(edges, labels=("low", "mid", "high"))

    xA = np.full((15, 1), -3.0)
    yA = xA + 10.0  # residual +10, predicted-low
    xB = np.full((15, 1), 3.0)
    rng = np.random.default_rng(3)
    yB = xB + rng.normal(0.0, 0.5, size=(15, 1))  # small residual, predicted-high
    X = np.vstack([xA, xB])
    Y = np.vstack([yA, yB])

    cal = MondrianConformalCalibrator(group_fn, alpha=ALPHA, min_group_n=5)
    cal.fit(model, X, Y)
    # groups stored on PREDICTED mean: A -> low, B -> high (never mixed by y).
    assert set(cal.groups_[:15, 0]) == {"low"}
    assert set(cal.groups_[15:, 0]) == {"high"}
    table, _ = cal._kappa_table(None)
    # high-group kappa reflects B's small residuals (< 2), not A's +10.
    assert table[0]["high"] < 2.0
    # low-group kappa carries A's huge residual (~10).
    assert table[0]["low"] > 5.0
    # RED-PROOF: grouping by observed y (both clusters -> "high") would push the
    # high-group kappa above 5 and this < 2.0 assertion would fail.


# ---------------------------------------------------------------------------
# marginal coverage preserved on an exchangeable synthetic
# ---------------------------------------------------------------------------


def _cheap_gp(seed: int):
    from rig.forward import GPForwardModel

    return GPForwardModel(n_restarts=1, seed=seed, max_iter=60)


def test_marginal_coverage_preserved_on_exchangeable_data():
    # On exchangeable data Mondrian must NOT break marginal coverage: pooled PICP
    # stays near nominal (same band the split calibrator would give, refined by
    # group). Grouping by PREDICTED-mean tertile; edges frozen from the cal slice.
    covered = total = 0
    for trial in range(40):
        rng = np.random.default_rng(200 + trial)
        Xtr = rng.uniform(0.0, 3.0, size=(40, 1))
        ytr = np.sin(2.0 * Xtr[:, 0]) + 0.15 * rng.standard_normal(40)
        model = _cheap_gp(trial).fit(Xtr, ytr[:, None])

        Xc = rng.uniform(0.0, 3.0, size=(60, 1))
        yc = (np.sin(2.0 * Xc[:, 0]) + 0.15 * rng.standard_normal(60))[:, None]
        mu_c = np.asarray(model.predict(Xc).mean).reshape(-1)
        edges = _tertile_edges_from(mu_c)
        gf = predicted_magnitude_group_fn(edges)
        cal = MondrianConformalCalibrator(gf, alpha=ALPHA)
        cal.fit(model, Xc, yc)

        Xte = rng.uniform(0.0, 3.0, size=(30, 1))
        yte = (np.sin(2.0 * Xte[:, 0]) + 0.15 * rng.standard_normal(30))[:, None]
        itv = cal.interval(Xte)  # (30, 1, 2)
        hit = (yte >= itv[..., 0]) & (yte <= itv[..., 1])
        covered += int(hit.sum())
        total += hit.size
    coverage = covered / total
    assert 0.85 <= coverage <= 0.97, coverage


def test_wrapper_matches_split_conformal_shape_contract():
    rng = np.random.default_rng(7)
    Xtr = rng.uniform(0.0, 3.0, size=(30, 1))
    ytr = (np.sin(2.0 * Xtr[:, 0]) + 0.15 * rng.standard_normal(30))[:, None]
    base = _cheap_gp(0).fit(Xtr, ytr)
    Xc = rng.uniform(0.0, 3.0, size=(30, 1))
    yc = (np.sin(2.0 * Xc[:, 0]) + 0.15 * rng.standard_normal(30))[:, None]
    edges = _tertile_edges_from(np.asarray(base.predict(Xc).mean).reshape(-1))
    cal = MondrianConformalCalibrator(predicted_magnitude_group_fn(edges), alpha=ALPHA)
    cal.fit(base, Xc, yc)
    wrapped = MondrianConformalForwardModel(base, cal)

    x = np.array([1.5])
    dist = wrapped.predict(x)
    assert isinstance(dist, PredictiveDistribution)
    np.testing.assert_array_equal(dist.mean, base.predict(x).mean)
    assert dist.conformal_set.shape == (1, 2)  # (m, 2), same as ConformalForwardModel
    lo, hi = dist.conformal_set[0]
    assert lo < dist.mean[0] < hi
    assert wrapped.support_score(x) == base.support_score(x)
    np.testing.assert_array_equal(wrapped.jacobian(x), base.jacobian(x))
    batch = wrapped.predict(Xc[:4])
    assert batch.conformal_set.shape == (4, 1, 2)  # (n, m, 2)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def test_determinism_double_fit_identical_bands():
    rng = np.random.default_rng(11)
    Xtr = rng.uniform(0.0, 3.0, size=(30, 1))
    ytr = (np.sin(2.0 * Xtr[:, 0]) + 0.15 * rng.standard_normal(30))[:, None]
    Xc = rng.uniform(0.0, 3.0, size=(60, 1))
    yc = (np.sin(2.0 * Xc[:, 0]) + 0.15 * rng.standard_normal(60))[:, None]
    Xte = rng.uniform(0.0, 3.0, size=(25, 1))

    def band(seed_model: int) -> np.ndarray:
        base = _cheap_gp(seed_model).fit(Xtr, ytr)
        edges = _tertile_edges_from(np.asarray(base.predict(Xc).mean).reshape(-1))
        cal = MondrianConformalCalibrator(predicted_magnitude_group_fn(edges), alpha=ALPHA)
        cal.fit(base, Xc, yc)
        return cal.interval(Xte)

    np.testing.assert_array_equal(band(0), band(0))


# ---------------------------------------------------------------------------
# interface parity: the §8 default §13.2 gate consumes the wrapper UNMODIFIED
# ---------------------------------------------------------------------------


def _solver(model, **kw) -> PessimisticInverseSolver:
    return PessimisticInverseSolver(
        model,
        [ContinuousVariable("x", 0.0, 5.0)],
        ["y"],
        X_train=np.linspace(0.0, 5.0, 12)[:, None],
        kappa=2.0,
        z_epi=2.0,
        delta_frac=0.0,
        seed=0,
        **kw,
    )


class _Overconfident:
    """mean = slope*x, deliberately tiny raw aleatoric sigma (overconfident)."""

    def __init__(self, slope: float, ale: float) -> None:
        self.slope = float(slope)
        self.ale = float(ale)

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mu = (self.slope * Xq[:, 0])[:, None]
        ale = np.full_like(mu, self.ale)
        epi = np.zeros_like(mu)
        if single:
            return PredictiveDistribution(mu[0], ale[0], epi[0], None)
        return PredictiveDistribution(mu, ale, epi, None)

    def support_score(self, x: np.ndarray):
        x = np.asarray(x, dtype=float)
        return 0.0 if x.ndim == 1 else np.zeros(x.shape[0])

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return np.array([[self.slope]])

    def update(self, records) -> None:  # pragma: no cover
        pass


def test_solver_default_gate_consumes_mondrian_wrapper_unmodified():
    # A band-fitting box: the wrapped Mondrian solve returns candidates the §13.2
    # gate certified — labelled "conformal-checked" — with NO solver edits.
    base = _Overconfident(slope=2.0, ale=0.02)
    rng = np.random.default_rng(0)
    Xc = np.linspace(0.2, 4.8, 60)[:, None]
    Yc = 2.0 * Xc + rng.normal(0.0, 0.3, size=Xc.shape)
    edges = _tertile_edges_from(np.asarray(base.predict(Xc).mean).reshape(-1))
    cal = MondrianConformalCalibrator(predicted_magnitude_group_fn(edges), alpha=ALPHA)
    cal.fit(base, Xc, Yc)
    conf = MondrianConformalForwardModel(base, cal)

    x_star = np.array([2.5])
    mu_star = float(base.predict(x_star).mean[0])
    cs = conf.predict(x_star).conformal_set  # (1, 2)
    band_half = float((cs[0, 1] - cs[0, 0]) / 2.0)
    box_half = 2.5 * band_half  # comfortably wider than the calibrated band
    spec = {"targets": {"y": (mu_star - box_half, mu_star + box_half)}, "max_candidates": 3}

    res = _solver(conf).solve(spec)
    assert isinstance(res, list) and res, "a band-fitting box must stay feasible"
    for c in res:
        assert isinstance(c, RecipeCandidate)
        assert c.feasibility_flag is True
        assert c.calibration_status == "conformal-checked"  # the default gate ran on the wrapper


# ---------------------------------------------------------------------------
# the SELECTED-POINT mechanism twin (the d=8 false-success miss)
# ---------------------------------------------------------------------------


def test_mondrian_rejects_selected_point_the_pooled_gate_admits():
    """Mechanism twin for THIS feature. The solver hands the conformal gate a
    SELECTED point; here the selected region (high predicted magnitude) is one the
    POOLED marginal quantile UNDER-covers (its residuals are much wider there) but
    the pooled band, dominated by the dense low-magnitude region, is narrow enough
    to fit a tight spec box — a marginal miss that slips through. The Mondrian gate,
    using the HIGH group's own (wide) quantile, rejects the same box.

    Model mean == x, so the solver drives x up toward the box; the box sits in the
    high-magnitude region. Pooled quantile ~ dominated by the 40 low/mid points
    (tight residuals); high group's 20 points have 6x the spread.
    """
    model = _MagnitudeModel(ale=1.0)
    rng = np.random.default_rng(4)
    # calibration: dense tight low/mid region, sparse WIDE high region.
    x_lowmid = np.linspace(0.0, 6.0, 40)[:, None]
    x_high = np.linspace(6.2, 10.0, 20)[:, None]
    X = np.vstack([x_lowmid, x_high])
    resid = np.vstack(
        [
            rng.normal(0.0, 0.15, size=(40, 1)),  # tight where data is dense
            rng.normal(0.0, 2.5, size=(20, 1)),  # WIDE in the high tail
        ]
    )
    Y = X + resid
    # tertile edges from the PREDICTED means (== x) of the calibration slice.
    edges = _tertile_edges_from(X[:, 0])
    gf = predicted_magnitude_group_fn(edges)

    # spec box in the HIGH region, sized to fit the POOLED band but NOT the high band.
    x_star = np.array([9.0])
    mu_star = 9.0
    split = SplitConformalCalibrator(alpha=ALPHA)
    split.fit(model, X, Y)
    pooled_half = float(split.kappa()[0])  # sigma=1 -> half-width == kappa

    mond = MondrianConformalCalibrator(gf, alpha=ALPHA)
    mond.fit(model, X, Y)
    mond_band = mond.interval(x_star)  # (1, 2)
    high_half = float(mond_band[0, 1] - mond_band[0, 0]) / 2.0

    # precondition: the high group's band is materially wider than the pooled band.
    assert high_half > pooled_half, (high_half, pooled_half)
    box_half = 0.5 * (pooled_half + high_half)  # fits pooled, spills the high band
    # and the box must be raw-feasible (>= kappa*sigma_ale) so the gate is the
    # discriminator, not the raw margin — true here since box_half ~ 3-4 >> kappa=2.
    assert box_half >= 2.0, box_half
    lo, hi = mu_star - box_half, mu_star + box_half
    spec = {"targets": {"y": (lo, hi)}, "max_candidates": 3}

    # solver over x in [0, 10] so it can actually reach the high region (mean == x).
    def hi_solver(m):
        return PessimisticInverseSolver(
            m,
            [ContinuousVariable("x", 0.0, 10.0)],
            ["y"],
            X_train=np.linspace(0.0, 10.0, 20)[:, None],
            kappa=2.0,
            z_epi=2.0,
            delta_frac=0.0,
            seed=0,
        )

    # POOLED (split-conformal) gate: admits the selected point (marginal miss slips).
    pooled_wrap = ConformalForwardModel(model, split)
    pooled_res = hi_solver(pooled_wrap).solve(spec)
    assert isinstance(pooled_res, list) and pooled_res, "pooled gate should admit the box"
    assert all(c.calibration_status == "conformal-checked" for c in pooled_res)

    # MONDRIAN gate: the high group's own quantile rejects the same box.
    mond_wrap = MondrianConformalForwardModel(model, mond)
    mond_res = hi_solver(mond_wrap).solve(spec)
    assert isinstance(mond_res, Infeasible), "Mondrian gate must reject the selected-point miss"
    assert "conformal" in mond_res.reason
    assert mond_res.distance_to_feasible > 0.0
    # RED-PROOF: make MondrianConformalCalibrator fall back to the pooled quantile
    # internally (e.g. min_group_n=10_000 so every group borrows pooled) and this
    # Infeasible assertion flips to a list — the pooled gate admits it. Restored by
    # the default min_group_n (per-group quantiles).
