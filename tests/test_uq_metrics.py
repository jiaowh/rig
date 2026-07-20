"""WP-C: §5.8 UQ metric tests — closed forms vs numerics, known Gaussians."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr, ndtri

from rig.metrics import (
    crps_gaussian,
    interval_score,
    mae,
    mpiw,
    picp,
    pit_values,
    quantile_calibration_error,
    rmse,
    uq_report,
)

RNG_SEED = 20260715


# ---------------------------------------------------------------------------
# CRPS: closed form (Gneiting et al. 2005) vs numerical integration
# ---------------------------------------------------------------------------


def _crps_numeric(mu: float, sigma: float, y: float) -> float:
    def integrand(t: float) -> float:
        return (ndtr((t - mu) / sigma) - float(t >= y)) ** 2

    lo, hi = min(mu - 12 * sigma, y - 12 * sigma), max(mu + 12 * sigma, y + 12 * sigma)
    val, _ = quad(integrand, lo, hi, points=[y, mu], limit=200)
    return val


def test_crps_matches_numerical_integration():
    triples = [(0.0, 1.0, 0.3), (1.0, 2.0, -0.5), (-2.0, 0.5, -2.0), (3.0, 0.2, 5.0)]
    for mu, sigma, y in triples:
        closed = float(crps_gaussian(np.array(mu), np.array(sigma), np.array(y)))
        numeric = _crps_numeric(mu, sigma, y)
        np.testing.assert_allclose(closed, numeric, rtol=1e-3)


def test_crps_vectorized_shape_and_positivity():
    rng = np.random.default_rng(RNG_SEED)
    mu = rng.standard_normal((50, 3))
    sigma = np.abs(rng.standard_normal((50, 3))) + 0.1
    y = rng.standard_normal((50, 3))
    c = crps_gaussian(mu, sigma, y)
    assert c.shape == (50, 3)
    assert np.all(c > 0.0)
    # CRPS is minimized (over y) at y = mu
    at_mean = crps_gaussian(np.array(0.0), np.array(1.0), np.array(0.0))
    off_mean = crps_gaussian(np.array(0.0), np.array(1.0), np.array(2.0))
    assert at_mean < off_mean


# ---------------------------------------------------------------------------
# interval metrics on known Gaussians
# ---------------------------------------------------------------------------


def test_picp_mpiw_on_known_gaussian():
    rng = np.random.default_rng(RNG_SEED)
    n = 20000
    y = rng.normal(0.0, 1.0, size=n)
    z90 = ndtri(0.95)  # central 90% band
    lo, hi = np.full(n, -z90), np.full(n, z90)
    assert abs(float(picp(lo, hi, y)[0]) - 0.90) < 0.01
    np.testing.assert_allclose(float(mpiw(lo, hi)[0]), 2.0 * z90)


def test_interval_score_hand_cases():
    # inside: just the width
    assert float(interval_score(np.array(0.0), np.array(2.0), np.array(1.0), 0.1)) == 2.0
    # below lower by 0.5: width + (2/alpha)*0.5 = 2 + 10 = 12
    np.testing.assert_allclose(
        float(interval_score(np.array(0.0), np.array(2.0), np.array(-0.5), 0.1)), 12.0
    )
    # above upper by 1: width + 20*1
    np.testing.assert_allclose(
        float(interval_score(np.array(0.0), np.array(2.0), np.array(3.0), 0.1)), 22.0
    )


# ---------------------------------------------------------------------------
# PIT / quantile calibration
# ---------------------------------------------------------------------------


def test_pit_uniform_when_calibrated_and_qce_detects_miscalibration():
    rng = np.random.default_rng(RNG_SEED)
    n = 5000
    mu = rng.standard_normal(n)
    sigma = np.abs(rng.standard_normal(n)) + 0.5
    y = mu + sigma * rng.standard_normal(n)  # perfectly calibrated

    pit = pit_values(mu, sigma, y)
    assert pit.shape == (n,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))
    qce_good = float(quantile_calibration_error(mu, sigma, y)[0])
    assert qce_good < 0.02, qce_good

    # overconfident model (sigma halved) must score much worse
    qce_bad = float(quantile_calibration_error(mu, 0.5 * sigma, y)[0])
    assert qce_bad > 0.1, qce_bad
    assert qce_bad > 5.0 * qce_good


# ---------------------------------------------------------------------------
# point metrics + report bundle
# ---------------------------------------------------------------------------


def test_rmse_mae_per_output():
    mu = np.array([[0.0, 1.0], [0.0, 1.0]])
    y = np.array([[1.0, 1.0], [-1.0, 3.0]])
    np.testing.assert_allclose(rmse(mu, y), [1.0, np.sqrt(2.0)])
    np.testing.assert_allclose(mae(mu, y), [1.0, 1.0])
    # 1-D input -> single-output (1,) aggregate
    assert rmse(np.zeros(4), np.ones(4)).shape == (1,)


def test_uq_report_bundle_on_calibrated_gaussian():
    rng = np.random.default_rng(RNG_SEED)
    n, m = 4000, 2
    mu = rng.standard_normal((n, m))
    sigma = np.full((n, m), 0.7)
    y = mu + sigma * rng.standard_normal((n, m))
    report = uq_report(mu, sigma, y)
    assert set(report) == {"rmse", "mae", "crps", "quantile_calibration_error", "levels"}
    assert report["rmse"].shape == (m,)
    for level in (0.5, 0.8, 0.9, 0.95):
        cov = report["levels"][level]["picp"]
        assert cov.shape == (m,)
        assert np.all(np.abs(cov - level) < 0.03), (level, cov)
        assert np.all(report["levels"][level]["mpiw"] > 0.0)
        assert np.all(report["levels"][level]["interval_score"] > 0.0)
