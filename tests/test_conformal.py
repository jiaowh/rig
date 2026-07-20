"""WP-C: conformal calibration tests (implementation-plan §5.6, D4). Synthetic only."""

from __future__ import annotations

import numpy as np
import pytest

from rig.calibration import (
    ACIController,
    ConformalForwardModel,
    JackknifePlusCalibrator,
    SplitConformalCalibrator,
)
from rig.calibration.conformal import conformal_quantile
from rig.forward import GPForwardModel
from rig.interfaces import PredictiveDistribution

ALPHA = 0.1
N_TRIALS = 200


def _f(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * x)


def _make_data(rng, n: int, noise: float = 0.15):
    X = rng.uniform(0.0, 3.0, size=(n, 1))
    y = _f(X[:, 0]) + noise * rng.standard_normal(n)
    return X, y[:, None]


def _cheap_gp(seed: int) -> GPForwardModel:
    return GPForwardModel(n_restarts=1, seed=seed, max_iter=60)


class _ConstantModel:
    """Fixed-prediction stub: goes stale by construction (drift testing)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu, self.sigma = mu, sigma

    def fit(self, X: np.ndarray, Y: np.ndarray) -> _ConstantModel:
        return self

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        shape = (1,) if x.ndim == 1 else (x.shape[0], 1)
        return PredictiveDistribution(
            mean=np.full(shape, self.mu),
            aleatoric_sigma=np.full(shape, self.sigma),
            epistemic_sigma=np.zeros(shape),
            conformal_set=None,
        )


# ---------------------------------------------------------------------------
# coverage over many seeded trials
# ---------------------------------------------------------------------------


def test_split_conformal_coverage_over_trials():
    covered = total = 0
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(1000 + trial)
        X_tr, Y_tr = _make_data(rng, 30)
        X_cal, Y_cal = _make_data(rng, 30)
        X_te, Y_te = _make_data(rng, 25)
        model = _cheap_gp(trial).fit(X_tr, Y_tr)
        cal = SplitConformalCalibrator(alpha=ALPHA)
        cal.fit(model, X_cal, Y_cal)
        itv = cal.interval(X_te)  # (25, 1, 2)
        hit = (Y_te >= itv[..., 0]) & (Y_te <= itv[..., 1])
        covered += int(hit.sum())
        total += hit.size
    coverage = covered / total
    assert 0.85 <= coverage <= 0.97, coverage


def test_jackknife_plus_cv_coverage_over_trials():
    covered = total = 0
    for trial in range(N_TRIALS):
        rng = np.random.default_rng(5000 + trial)
        X, Y = _make_data(rng, 50)  # n > 40 -> CV+ path (the D4 switch point)
        X_te, Y_te = _make_data(rng, 10)
        jk = JackknifePlusCalibrator(alpha=ALPHA, k_folds=5, seed=trial)
        jk.fit(lambda t=trial: _cheap_gp(t), X, Y)
        assert jk.mode_ == "cv+"
        itv = jk.interval(X_te)  # (10, 1, 2)
        hit = (Y_te >= itv[..., 0]) & (Y_te <= itv[..., 1])
        covered += int(hit.sum())
        total += hit.size
    coverage = covered / total
    assert 0.85 <= coverage <= 0.97, coverage


def test_jackknife_plus_loo_path_tiny_n():
    rng = np.random.default_rng(42)
    X, Y = _make_data(rng, 20)  # n <= 40 -> LOO
    X_te, Y_te = _make_data(rng, 20)
    jk = JackknifePlusCalibrator(alpha=ALPHA)
    jk.fit(lambda: _cheap_gp(3), X, Y)
    assert jk.mode_ == "loo"
    assert len(jk.models_) == 20
    itv = jk.interval(X_te)
    assert itv.shape == (20, 1, 2)
    assert np.all(itv[..., 0] < itv[..., 1])
    hit = (Y_te >= itv[..., 0]) & (Y_te <= itv[..., 1])
    assert hit.mean() >= 0.6  # single-trial sanity, not a coverage claim
    single = jk.interval(X_te[0])
    assert single.shape == (1, 2)


def test_conformal_quantile_small_n_returns_inf():
    scores = np.abs(np.random.default_rng(0).standard_normal((5, 1)))
    # ceil(0.9 * 6) = 6 > 5: no finite 90% claim from 5 scores (D4 honesty)
    assert np.isinf(conformal_quantile(scores, 0.1)[0])
    assert np.isfinite(conformal_quantile(scores, 0.5)[0])


class _ConstantModel2:
    """Two-output fixed-prediction stub with DISTINCT per-output means/scales."""

    def __init__(self, mus=(0.0, 10.0), sigmas=(1.0, 3.0)) -> None:
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> _ConstantModel2:
        return self

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        n = 1 if x.ndim == 1 else x.shape[0]
        mean = np.broadcast_to(self.mus, (n, 2)).copy()
        ale = np.broadcast_to(self.sigmas, (n, 2)).copy()
        if x.ndim == 1:
            mean, ale = mean[0], ale[0]
        return PredictiveDistribution(
            mean=mean, aleatoric_sigma=ale, epistemic_sigma=np.zeros_like(mean), conformal_set=None
        )


def test_jackknife_plus_tiny_n_returns_inf():
    # audit C3: the JackknifePlusCalibrator interval must return [-inf, +inf] at
    # tiny n where no finite (1-alpha) band is honest (D4). The only existing
    # jackknife tests land in the finite regime (n=20/50). An off-by-one in the
    # k_lo<1 / k_hi>n guard would ship a bogus finite band, undetected.
    for n in (3, 5, 8):
        X = np.linspace(0.0, 1.0, n)[:, None]
        Y = np.linspace(-1.0, 1.0, n)[:, None]
        jk = JackknifePlusCalibrator(alpha=0.1)
        jk.fit(lambda: _ConstantModel(mu=0.0, sigma=1.0), X, Y)
        itv = jk.interval(np.array([0.5]))  # (m, 2)
        assert np.isneginf(itv[0, 0]) and np.isposinf(itv[0, 1]), (n, itv)


def test_jackknife_plus_multi_output_broadcasts_per_output():
    # audit C3: every conformal test is single-output; verify per-output
    # broadcasting — a 2-output model with distinct residual scales must yield
    # per-output bands (output 1's band wider than output 0's) and cover both.
    rng = np.random.default_rng(5)
    n = 40
    X = rng.uniform(0.0, 1.0, (n, 1))
    Y = np.stack([rng.normal(0.0, 1.0, n), rng.normal(10.0, 3.0, n)], axis=-1)
    jk = JackknifePlusCalibrator(alpha=0.1)
    jk.fit(_ConstantModel2, X, Y)
    Xte = rng.uniform(0.0, 1.0, (30, 1))
    Yte = np.stack([rng.normal(0.0, 1.0, 30), rng.normal(10.0, 3.0, 30)], axis=-1)
    itv = jk.interval(Xte)
    assert itv.shape == (30, 2, 2)
    assert np.all(itv[..., 0] < itv[..., 1])
    for j in range(2):
        cov = float(np.mean((Yte[:, j] >= itv[:, j, 0]) & (Yte[:, j] <= itv[:, j, 1])))
        assert 0.75 <= cov <= 1.0, (j, cov)
    w0 = float(np.mean(itv[:, 0, 1] - itv[:, 0, 0]))
    w1 = float(np.mean(itv[:, 1, 1] - itv[:, 1, 0]))
    assert w1 > w0  # output 1 (sigma 3) must have wider bands than output 0 (sigma 1)


# ---------------------------------------------------------------------------
# ACI under drift (§5.6: rolling coverage is the drift detector)
# ---------------------------------------------------------------------------


def test_aci_recovers_coverage_under_drift_while_static_degrades():
    rng = np.random.default_rng(99)
    model = _ConstantModel(mu=0.0, sigma=1.0)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    X_cal = np.zeros((100, 1))
    Y_cal = rng.normal(0.0, 1.0, size=(100, 1))
    cal.fit(model, X_cal, Y_cal)
    kappa_static = float(cal.kappa()[0])  # frozen pre-drift band

    aci = ACIController(cal, alpha_target=ALPHA, gamma=0.05, window=100)
    n_pre, n_post = 200, 500
    y_pre = rng.normal(0.0, 1.0, size=n_pre)
    y_post = rng.normal(4.0, 1.0, size=n_post)  # the base model is now stale
    x = np.zeros(1)
    for y in y_pre:
        aci.observe(x, np.array([y]))
    for y in y_post:
        aci.observe(x, np.array([y]))

    # ACI adapts: rolling coverage back within +/-0.05 of nominal 0.9
    rolling = float(np.asarray(aci.rolling_coverage).reshape(-1)[0])
    assert abs(rolling - (1.0 - ALPHA)) <= 0.05, rolling
    # the static band never adapts: its post-shift coverage collapses
    static_cov = float(np.mean(np.abs(y_post) <= kappa_static))
    assert static_cov < 0.7, static_cov
    # alpha_t respected its clip range throughout (checked at the end state)
    a = np.asarray(aci.alpha_t).reshape(-1)
    assert np.all(a >= 0.001) and np.all(a <= 0.5)


def test_aci_alpha_update_rule_exact():
    """One step of the Gibbs & Candès update, verified numerically."""
    model = _ConstantModel(mu=0.0, sigma=1.0)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    rng = np.random.default_rng(1)
    cal.fit(model, np.zeros((60, 1)), rng.normal(0, 1, (60, 1)))
    aci = ACIController(cal, alpha_target=0.1, gamma=0.05, update_scores=False)
    err = aci.observe(np.zeros(1), np.array([100.0]))  # certain miss
    assert err[0] == 1.0
    np.testing.assert_allclose(aci.alpha_t, 0.1 + 0.05 * (0.1 - 1.0))
    err = aci.observe(np.zeros(1), np.array([0.0]))  # certain hit
    assert err[0] == 0.0
    np.testing.assert_allclose(aci.alpha_t, 0.055 + 0.05 * 0.1)


def test_aci_err_scored_against_preupdate_interval():
    # audit D12: err_t must be scored against the PRE-update band. Choose y
    # BETWEEN the pre-update and (narrower) post-update band edges: a correct
    # impl reports err=0 (pre-update covers it) and bumps alpha; an
    # update-then-score bug would report err=1 against the narrowed band.
    rng = np.random.default_rng(7)
    model = _ConstantModel(mu=0.0, sigma=1.0)
    cal = SplitConformalCalibrator(alpha=0.2)
    cal.fit(model, np.zeros((200, 1)), rng.normal(0.0, 1.0, (200, 1)))
    k_pre = float(cal.kappa(0.2)[0])
    k_post = float(cal.kappa(0.3)[0])  # higher alpha -> narrower band
    assert k_post < k_pre, (k_pre, k_post)
    y = np.array([0.5 * (k_pre + k_post)])  # inside pre band, outside post band
    aci = ACIController(cal, alpha_target=0.2, gamma=0.5, update_scores=False)
    # alpha_t starts at alpha_target=0.2; a hit bumps it to 0.2+0.5*(0.2-0)=0.3.
    err = aci.observe(np.zeros(1), y)
    assert err[0] == 0.0  # PRE-update band [-k_pre, k_pre] covers y
    np.testing.assert_allclose(np.asarray(aci.alpha_t).reshape(-1)[0], 0.3)


# ---------------------------------------------------------------------------
# the ForwardModel wrapper
# ---------------------------------------------------------------------------


def test_conformal_forward_model_fills_conformal_set():
    rng = np.random.default_rng(7)
    X_tr, Y_tr = _make_data(rng, 30)
    X_cal, Y_cal = _make_data(rng, 30)
    base = _cheap_gp(0).fit(X_tr, Y_tr)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(base, X_cal, Y_cal)
    wrapped = ConformalForwardModel(base, cal)

    x = np.array([1.5])
    dist = wrapped.predict(x)
    assert isinstance(dist, PredictiveDistribution)
    np.testing.assert_array_equal(dist.mean, base.predict(x).mean)
    assert dist.conformal_set is not None
    assert dist.conformal_set.shape == (1, 2)
    lo, hi = dist.conformal_set[0]
    assert lo < dist.mean[0] < hi
    # delegation
    assert wrapped.support_score(x) == base.support_score(x)
    np.testing.assert_array_equal(wrapped.jacobian(x), base.jacobian(x))
    # batch shape
    batch = wrapped.predict(X_cal[:4])
    assert batch.conformal_set.shape == (4, 1, 2)


def test_conformal_forward_model_with_aci_controller():
    rng = np.random.default_rng(8)
    X_tr, Y_tr = _make_data(rng, 30)
    X_cal, Y_cal = _make_data(rng, 30)
    base = _cheap_gp(0).fit(X_tr, Y_tr)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(base, X_cal, Y_cal)
    aci = ACIController(cal, alpha_target=ALPHA)
    wrapped = ConformalForwardModel(base, cal, controller=aci)
    x = np.array([1.0])
    dist = wrapped.predict(x)
    assert dist.conformal_set.shape == (1, 2)
    err = wrapped.observe(x, np.array([_f(1.0)]))
    assert err.shape == (1,)
    assert aci.t == 1


def test_aci_observe_nan_scores_as_miss_fail_closed():
    # A non-finite observation (dead/dropped sensor reading) must NOT be
    # scored as "covered" — that raises alpha_t (narrows bands: the unsafe
    # direction) and hides the dead stream behind rolling_coverage == 1.0.
    # Fail-closed: NaN counts as a MISS (widens bands, tanks rolling
    # coverage to 0.0) and is never appended to the calibration score buffer.
    rng = np.random.default_rng(0)
    model = _ConstantModel(mu=0.0, sigma=1.0)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, np.zeros((60, 1)), rng.normal(0.0, 1.0, size=(60, 1)))
    n_scores_before = cal.scores_.shape[0]

    aci = ACIController(cal, alpha_target=ALPHA, gamma=0.05, window=50)
    x = np.zeros(1)
    for _ in range(20):
        err = aci.observe(x, np.array([np.nan]))
        assert err[0] == 1.0  # NaN must score as a miss, not a hit

    # widening direction: alpha_t must move BELOW alpha_target, never above
    a = float(np.asarray(aci.alpha_t).reshape(-1)[0])
    assert a < ALPHA, a

    rolling = float(np.asarray(aci.rolling_coverage).reshape(-1)[0])
    assert rolling == 0.0, rolling

    # NaN scores must not pollute the calibration buffer
    assert cal.scores_.shape[0] == n_scores_before


def test_jackknife_needs_minimum_points():
    jk = JackknifePlusCalibrator()
    with pytest.raises(ValueError, match="at least 3"):
        jk.fit(lambda: _cheap_gp(0), np.zeros((2, 1)), np.zeros((2, 1)))
