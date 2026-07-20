"""Shared exact-GP machinery for the numpy forward surrogates (PRIVATE).

Extracted from ``rig.forward.gp`` (WP-C) so the WP-I multi-task model reuses
the same kernel / Cholesky / multi-start / standardization pieces without
copy-paste. Nothing in this module is public API and GPForwardModel's public
behavior is unchanged by the extraction (its tests pass untouched).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import cholesky
from scipy.optimize import minimize

SQRT5 = np.sqrt(5.0)

# log-parameter bounds shared by the single- and multi-task GPs:
# [log lengthscales..., log signal_var, log noise_var]
LOG_ELL_BOUNDS = (np.log(1e-3), np.log(1e4))
LOG_SF2_BOUNDS = (np.log(1e-6), np.log(1e4))
LOG_SN2_BOUNDS = (np.log(1e-8), np.log(1e2))


def cholesky_with_jitter(K: np.ndarray) -> tuple[np.ndarray, float]:
    """Lower Cholesky of K with escalating diagonal jitter. The guarded loop
    tries 0, then 1e-10 .. 1e-4 of the mean diagonal; if all fail, a final
    (un-guarded) attempt uses 1e-3 of the mean diagonal and its LinAlgError, if
    any, propagates. Returns (L, jitter_used)."""
    scale = float(np.mean(np.diag(K)))
    jitter = 0.0
    for _ in range(8):
        try:
            L = cholesky(K + jitter * np.eye(K.shape[0]), lower=True)
            return L, jitter
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10.0, 1e-10 * scale)
    return cholesky(K + jitter * np.eye(K.shape[0]), lower=True), jitter


def matern52(
    X1: np.ndarray, X2: np.ndarray, ell: np.ndarray, sf2: float
) -> tuple[np.ndarray, np.ndarray]:
    """Matérn-5/2 ARD kernel. Returns (K, r) where r is the scaled distance."""
    diff = X1[:, None, :] - X2[None, :, :]
    r2 = np.sum((diff / ell) ** 2, axis=-1)
    r = np.sqrt(np.maximum(r2, 0.0))
    K = sf2 * (1.0 + SQRT5 * r + (5.0 / 3.0) * r2) * np.exp(-SQRT5 * r)
    return K, r


def matern52_grad_x(xs: np.ndarray, X: np.ndarray, ell: np.ndarray, sf2: float) -> np.ndarray:
    """Rows ``dk(xs, X_i)/dx`` of the Matérn-5/2 ARD kernel at a single point.

    dk/dx_j = -(5/3) sf2 (1 + sqrt5 r) exp(-sqrt5 r) (x_j - x_ij)/ell_j^2
    (no 1/r singularity — the closed form is smooth at r=0). Shape (n, d).
    """
    diff = xs[None, :] - X  # (n, d)
    r = np.sqrt(np.maximum(np.sum((diff / ell) ** 2, axis=1), 0.0))  # (n,)
    coeff = -(5.0 / 3.0) * sf2 * (1.0 + SQRT5 * r) * np.exp(-SQRT5 * r)  # (n,)
    return coeff[:, None] * diff / (ell**2)[None, :]  # (n, d)


def multistart_minimize(
    fun_and_grad: Callable[[np.ndarray], tuple[float, np.ndarray]],
    starts: Sequence[np.ndarray],
    bounds: Sequence[tuple[float, float]],
    max_iter: int,
) -> np.ndarray:
    """L-BFGS-B multi-start over analytic (value, gradient) objectives.

    Returns the best theta found (the first start if every run fails to
    improve on +inf — matching WP-C's original loop exactly).
    """
    best_val, best_theta = np.inf, np.asarray(starts[0])
    for theta0 in starts:
        res = minimize(
            fun_and_grad,
            theta0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        if res.fun < best_val:
            best_val, best_theta = float(res.fun), res.x
    return best_theta


def standardize_stats(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise (mean, scale) with zero-variance columns pinned to scale 1
    — TRAIN statistics only (§5.3 leakage guard is the caller's job)."""
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    return mean, np.where(std > 0.0, std, 1.0)


def regularized_cov_inv(Xs: np.ndarray) -> np.ndarray:
    """Inverse of the regularized covariance of standardized inputs — the
    §8.2 cheap Mahalanobis support-score fallback."""
    d = Xs.shape[1]
    cov = np.cov(Xs, rowvar=False) if Xs.shape[0] > 1 else np.eye(d)
    cov = np.atleast_2d(cov)
    cov += 1e-6 * max(float(np.trace(cov)) / d, 1.0) * np.eye(d)
    return np.linalg.inv(cov)
