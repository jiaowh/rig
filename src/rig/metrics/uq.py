"""UQ metrics — report all, per split, per output dimension (implementation-plan §5.8).

Conventions (uniform across this module):

- Predictions-first argument order: ``(mu, sigma, y)`` / ``(lower, upper, y)``.
- Inputs are ``(n,)`` or ``(n, m)`` arrays; 1-D inputs are treated as a
  single output column, and every aggregate comes back as an ``(m,)`` array
  (so ``[0]`` indexes the single-output case). Per-sample quantities
  (``crps_gaussian``, ``interval_score``, ``pit_values``) keep the input
  shape.
- No plotting anywhere: ``pit_values`` returns the raw PIT array; histogram
  rendering is the caller's problem (§5.8 asks for the values, not a plot).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import ndtr, ndtri  # standard normal CDF / quantile, vectorized

_INV_SQRT_PI = 1.0 / np.sqrt(np.pi)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _norm_pdf(z: np.ndarray) -> np.ndarray:
    return _INV_SQRT_2PI * np.exp(-0.5 * z * z)


def _col(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


# ---------------------------------------------------------------------------
# point accuracy
# ---------------------------------------------------------------------------


def rmse(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Root-mean-square error per output, shape (m,)."""
    return np.sqrt(np.mean((_col(mu) - _col(y)) ** 2, axis=0))


def mae(mu: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mean absolute error per output, shape (m,)."""
    return np.mean(np.abs(_col(mu) - _col(y)), axis=0)


# ---------------------------------------------------------------------------
# probabilistic accuracy
# ---------------------------------------------------------------------------


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form CRPS of a Gaussian predictive (Gneiting et al. 2005).

    CRPS(N(mu, sigma^2), y) = sigma * [z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi)],
    z = (y - mu)/sigma. Per-sample; keeps the input shape.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    y = np.asarray(y, dtype=float)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * ndtr(z) - 1.0) + 2.0 * _norm_pdf(z) - _INV_SQRT_PI)


def pit_values(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Probability integral transform Phi((y - mu)/sigma), input shape kept.

    Uniform(0, 1) iff the Gaussian predictive is perfectly calibrated.
    """
    return ndtr(
        (np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / np.asarray(sigma, dtype=float)
    )


def quantile_calibration_error(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    levels: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Regression quantile-calibration error (Kuleshov et al. 2018 style).

    Mean over a quantile grid q of |empirical_freq(PIT <= q) - q|, per
    output; 0 = perfectly calibrated. Default grid: 0.05..0.95 step 0.05.
    """
    if levels is None:
        levels = np.arange(0.05, 0.951, 0.05)
    q = np.asarray(levels, dtype=float)
    pit = _col(pit_values(mu, sigma, y))  # (n, m)
    emp = np.mean(pit[:, :, None] <= q[None, None, :], axis=0)  # (m, len(q))
    return np.mean(np.abs(emp - q[None, :]), axis=1)


# ---------------------------------------------------------------------------
# interval metrics
# ---------------------------------------------------------------------------


def interval_score(lower: np.ndarray, upper: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Winkler interval score at miscoverage alpha (proper; lower = better).

    IS = (u - l) + (2/alpha)(l - y) 1{y < l} + (2/alpha)(y - u) 1{y > u}.
    Per-sample; keeps the input shape.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    return (
        (upper - lower)
        + (2.0 / alpha) * (lower - y) * (y < lower)
        + (2.0 / alpha) * (y - upper) * (y > upper)
    )


def picp(lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Prediction-interval coverage probability per output, shape (m,)."""
    lower, upper, y = _col(lower), _col(upper), _col(y)
    return np.mean((y >= lower) & (y <= upper), axis=0)


def mpiw(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Mean prediction-interval width per output, shape (m,)."""
    return np.mean(_col(upper) - _col(lower), axis=0)


def uq_report(
    mu: np.ndarray,
    sigma: np.ndarray,
    y: np.ndarray,
    levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
) -> dict[str, np.ndarray | dict[float, dict[str, np.ndarray]]]:
    """The §5.8 bundle for Gaussian predictives, per output.

    Central intervals at each coverage level use the Gaussian quantile
    mu ± z_{(1+level)/2} * sigma. Returns rmse/mae/mean CRPS/QCE plus, per
    level, PICP / MPIW / mean interval score. PIT values are separate
    (:func:`pit_values`) — they are per-sample, not aggregates.
    """
    mu2, sig2, y2 = _col(mu), _col(sigma), _col(y)
    out: dict[str, np.ndarray | dict[float, dict[str, np.ndarray]]] = {
        "rmse": rmse(mu2, y2),
        "mae": mae(mu2, y2),
        "crps": np.mean(crps_gaussian(mu2, sig2, y2), axis=0),
        "quantile_calibration_error": quantile_calibration_error(mu2, sig2, y2),
    }
    per_level: dict[float, dict[str, np.ndarray]] = {}
    for level in levels:
        z = ndtri(0.5 * (1.0 + level))
        lo, hi = mu2 - z * sig2, mu2 + z * sig2
        per_level[level] = {
            "picp": picp(lo, hi, y2),
            "mpiw": mpiw(lo, hi),
            "interval_score": np.mean(interval_score(lo, hi, y2, 1.0 - level), axis=0),
        }
    out["levels"] = per_level
    return out
