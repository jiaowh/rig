"""Exact GP forward surrogate — the D3 little-data backbone (implementation-plan §5.2).

One independent exact GP per output dimension, Matérn-5/2 kernel with ARD
lengthscales, signal variance, and a Gaussian noise variance that is the
constant-per-output aleatoric floor v0 (the §10.3 identifiability-honest
choice at small n: run-to-run noise and metrology noise are not separable
without replicates, so v0 fits one total-noise scalar per output).

Hyperparameters by exact marginal-likelihood maximization (analytic
gradients, L-BFGS-B on log-params, seeded multi-start). numpy/scipy only —
exact GP at n <= ~300 is numerically trivial and is the plan's own primary
in this regime (D3); the torch stack is WP-E.

Standardization uses TRAIN statistics only (§5.3 leakage guard) and is
purely internal: ``predict`` speaks raw (SI-magnitude) units.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from rig.forward._gp_common import (
    LOG_ELL_BOUNDS as _LOG_ELL_BOUNDS,
)
from rig.forward._gp_common import (
    LOG_SF2_BOUNDS as _LOG_SF2_BOUNDS,
)
from rig.forward._gp_common import (
    LOG_SN2_BOUNDS as _LOG_SN2_BOUNDS,
)
from rig.forward._gp_common import (
    SQRT5 as _SQRT5,
)
from rig.forward._gp_common import (
    cholesky_with_jitter as _cholesky_with_jitter,
)
from rig.forward._gp_common import (
    matern52 as _matern52,
)
from rig.forward._gp_common import (
    matern52_grad_x as _matern52_grad_x,
)
from rig.forward._gp_common import (
    multistart_minimize as _multistart_minimize,
)
from rig.forward._gp_common import (
    regularized_cov_inv as _regularized_cov_inv,
)
from rig.forward._gp_common import (
    standardize_stats as _standardize_stats,
)
from rig.forward.data import records_to_arrays
from rig.interfaces import PredictiveDistribution


@dataclass
class _GPHyper:
    ell: np.ndarray  # (d,) ARD lengthscales, standardized-input units
    sf2: float  # signal variance, standardized-output units
    sn2: float  # noise variance (aleatoric floor v0), standardized units


class _SingleOutputGP:
    """Exact GP for one standardized output. Internal to GPForwardModel."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X  # (n, d) standardized
        self.y = y  # (n,) standardized
        self.hyper: _GPHyper | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    # -- marginal likelihood ------------------------------------------------

    def _nlml_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        n, d = self.X.shape
        ell = np.exp(theta[:d])
        sf2 = float(np.exp(theta[d]))
        sn2 = float(np.exp(theta[d + 1]))
        try:
            Kf, r = _matern52(self.X, self.X, ell, sf2)
            K = Kf + sn2 * np.eye(n)
            L, _ = _cholesky_with_jitter(K)
        except np.linalg.LinAlgError:
            return 1e25, np.zeros_like(theta)
        alpha = cho_solve((L, True), self.y)
        nlml = (
            0.5 * float(self.y @ alpha)
            + float(np.sum(np.log(np.diag(L))))
            + 0.5 * n * np.log(2.0 * np.pi)
        )
        # dNLML/dtheta_j = 0.5 tr(A dK/dtheta_j), A = K^-1 - alpha alpha^T
        Kinv = cho_solve((L, True), np.eye(n))
        A = Kinv - np.outer(alpha, alpha)
        grad = np.empty_like(theta)
        # d K / d log ell_k = (5/3) sf2 (1 + sqrt5 r) exp(-sqrt5 r) * D_k / ell_k^2
        base = (5.0 / 3.0) * sf2 * (1.0 + _SQRT5 * r) * np.exp(-_SQRT5 * r)
        for k in range(d):
            Dk = (self.X[:, None, k] - self.X[None, :, k]) ** 2
            grad[k] = 0.5 * float(np.sum(A * (base * Dk / ell[k] ** 2)))
        grad[d] = 0.5 * float(np.sum(A * Kf))  # d K / d log sf2 = Kf
        grad[d + 1] = 0.5 * sn2 * float(np.trace(A))  # d K / d log sn2 = sn2 I
        return nlml, grad

    def fit(self, n_restarts: int, seed: int, max_iter: int) -> None:
        n, d = self.X.shape
        rng = np.random.default_rng(seed)
        bounds = [_LOG_ELL_BOUNDS] * d + [_LOG_SF2_BOUNDS, _LOG_SN2_BOUNDS]
        # deterministic first start: unit lengthscales / unit signal on
        # standardized data, modest noise
        starts = [np.concatenate([np.zeros(d), [np.log(1.0)], [np.log(0.05)]])]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(
                np.concatenate(
                    [
                        rng.uniform(np.log(0.1), np.log(10.0), size=d),
                        [rng.uniform(np.log(0.1), np.log(10.0))],
                        [rng.uniform(np.log(1e-4), np.log(1.0))],
                    ]
                )
            )
        best_theta = _multistart_minimize(self._nlml_and_grad, starts, bounds, max_iter)
        ell = np.exp(best_theta[:d])
        sf2 = float(np.exp(best_theta[d]))
        sn2 = float(np.exp(best_theta[d + 1]))
        self.hyper = _GPHyper(ell=ell, sf2=sf2, sn2=sn2)
        Kf, _ = _matern52(self.X, self.X, ell, sf2)
        self._L, _ = _cholesky_with_jitter(Kf + sn2 * np.eye(n))
        self._alpha = cho_solve((self._L, True), self.y)

    # -- posterior ------------------------------------------------------------

    def predict(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and FUNCTION-SPACE variance (epistemic) at Xs (m, d)."""
        assert self.hyper is not None and self._L is not None and self._alpha is not None
        h = self.hyper
        Ks, _ = _matern52(Xs, self.X, h.ell, h.sf2)  # (m, n)
        mu = Ks @ self._alpha
        V = solve_triangular(self._L, Ks.T, lower=True)  # (n, m)
        var = np.maximum(h.sf2 - np.sum(V * V, axis=0), 0.0)
        return mu, var

    def mean_grad(self, xs: np.ndarray) -> np.ndarray:
        """Analytic gradient of the posterior mean at a single point xs (d,).

        d mu/d x = sum_i alpha_i dk(x, x_i)/dx with, for Matérn-5/2,
        dk/dx_j = -(5/3) sf2 (1 + sqrt5 r) exp(-sqrt5 r) (x_j - x_ij)/ell_j^2
        (no 1/r singularity — the closed form is smooth at r=0).
        """
        assert self.hyper is not None and self._alpha is not None
        h = self.hyper
        dk = _matern52_grad_x(xs, self.X, h.ell, h.sf2)  # (n, d)
        return self._alpha @ dk  # (d,)

    def cov(self, Xs1: np.ndarray, Xs2: np.ndarray) -> np.ndarray:
        """Posterior covariance of the LATENT function between two standardized
        input sets: Cov(f(Xs1), f(Xs2)) = K12 − K1ᵀ K⁻¹ K2, shape (n1, n2).

        This is the joint (epistemic) covariance the acquisition layer needs for
        EPIG (§9.4) — how much observing f at one recipe informs f at another.
        """
        assert self.hyper is not None and self._L is not None
        h = self.hyper
        K12, _ = _matern52(Xs1, Xs2, h.ell, h.sf2)  # (n1, n2)
        K1, _ = _matern52(Xs1, self.X, h.ell, h.sf2)  # (n1, n)
        K2, _ = _matern52(Xs2, self.X, h.ell, h.sf2)  # (n2, n)
        V1 = solve_triangular(self._L, K1.T, lower=True)  # (n, n1)
        V2 = solve_triangular(self._L, K2.T, lower=True)  # (n, n2)
        return K12 - V1.T @ V2  # (n1, n2)


class GPForwardModel:
    """Exact-GP ForwardModel (implementation-plan §3.2 protocol; §5.2 little-data D3 primary).

    - ``predict(x) -> PredictiveDistribution(mean, aleatoric_sigma,
      epistemic_sigma, conformal_set)`` with ``conformal_set=None`` when
      unwrapped (the §5.6 calibration wrapper fills it).
      ``x`` may be a single point ``(d,)`` -> fields shaped ``(m,)``, or a
      batch ``(n, d)`` -> fields shaped ``(n, m)``.
    - ``epistemic_sigma`` = GP posterior std of the latent function;
      ``aleatoric_sigma`` = fitted constant noise std per output (floor v0).
    - ``support_score(x)`` = negative Mahalanobis distance to the training
      set in standardized input space (regularized covariance) — the §8.2
      cheap fallback. Higher = more in-distribution.
    - ``jacobian(x)`` = analytic posterior-mean Jacobian, shape ``(m, d)``,
      raw units.
    - ``update(records)`` refits on old + new data (requires ``input_keys``
      and ``output_keys`` at construction to map RunRecords).
    """

    def __init__(
        self,
        input_keys: Sequence[str] | None = None,
        output_keys: Sequence[str] | None = None,
        n_restarts: int = 5,
        seed: int = 0,
        max_iter: int = 100,
    ) -> None:
        self.input_keys = list(input_keys) if input_keys is not None else None
        self.output_keys = list(output_keys) if output_keys is not None else None
        self.n_restarts = n_restarts
        self.seed = seed
        self.max_iter = max_iter
        self._gps: list[_SingleOutputGP] = []
        self._X_raw: np.ndarray | None = None
        self._Y_raw: np.ndarray | None = None

    # -- fitting ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return bool(self._gps)

    @property
    def n_train_(self) -> int:
        return 0 if self._X_raw is None else int(self._X_raw.shape[0])

    def fit(self, X: np.ndarray, Y: np.ndarray) -> GPForwardModel:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d); got shape {X.shape}")
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
        self._X_raw, self._Y_raw = X, Y

        # standardize with TRAIN statistics only (§5.3 leakage guard)
        self._x_mean, self._x_scale = _standardize_stats(X)
        self._y_mean, self._y_scale = _standardize_stats(Y)
        Xs = (X - self._x_mean) / self._x_scale
        Ys = (Y - self._y_mean) / self._y_scale

        # regularized covariance of standardized inputs for support_score (§8.2)
        self._support_cov_inv = _regularized_cov_inv(Xs)

        self._gps = []
        for j in range(Ys.shape[1]):
            gp = _SingleOutputGP(Xs, Ys[:, j])
            gp.fit(self.n_restarts, self.seed + 1000 * j, self.max_iter)
            self._gps.append(gp)
        return self

    def update(self, records: Iterable[Any]) -> None:
        """Ingest RunRecords and refit on old + new data (invariant 2d)."""
        if self.input_keys is None or self.output_keys is None:
            raise ValueError(
                "update(records) needs input_keys/output_keys at construction "
                "to map RunRecords to arrays; use fit(X, Y) for raw matrices"
            )
        X_new, Y_new = records_to_arrays(records, self.input_keys, self.output_keys)
        if self._X_raw is not None:
            X_new = np.vstack([self._X_raw, X_new])
            Y_new = np.vstack([self._Y_raw, Y_new])
        self.fit(X_new, Y_new)

    # -- fitted-parameter views (raw units) -------------------------------------

    @property
    def noise_std_(self) -> np.ndarray:
        """Fitted aleatoric noise std per output, raw output units."""
        self._require_fitted()
        return np.array(
            [np.sqrt(gp.hyper.sn2) * s for gp, s in zip(self._gps, self._y_scale, strict=True)]
        )

    @property
    def lengthscales_(self) -> np.ndarray:
        """ARD lengthscales per output, raw input units, shape (m, d)."""
        self._require_fitted()
        return np.stack([gp.hyper.ell * self._x_scale for gp in self._gps])

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("GPForwardModel is not fitted; call fit(X, Y) first")

    # -- ForwardModel protocol ---------------------------------------------------

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        Xs = (Xq - self._x_mean) / self._x_scale
        means, epi = [], []
        for gp, ym, ys in zip(self._gps, self._y_mean, self._y_scale, strict=True):
            mu, var = gp.predict(Xs)
            means.append(ym + ys * mu)
            epi.append(ys * np.sqrt(var))
        mean = np.stack(means, axis=-1)  # (n, m)
        epistemic = np.stack(epi, axis=-1)  # (n, m)
        aleatoric = np.broadcast_to(self.noise_std_, mean.shape).copy()
        if single:
            mean, aleatoric, epistemic = mean[0], aleatoric[0], epistemic[0]
        return PredictiveDistribution(
            mean=mean,
            aleatoric_sigma=aleatoric,
            epistemic_sigma=epistemic,
            conformal_set=None,  # filled by the §5.6 calibration wrapper
        )

    def support_score(self, x: np.ndarray) -> float | np.ndarray:
        """Negative Mahalanobis distance to the training set, standardized
        input space (§8.2 cheap fallback). Float for a single (d,) point;
        (n,) array for a batch."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale  # train mean is 0 here
        d2 = np.einsum("ij,jk,ik->i", Xs, self._support_cov_inv, Xs)
        score = -np.sqrt(np.maximum(d2, 0.0))
        return float(score[0]) if single else score

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Analytic d(mean)/dx at a single point x (d,), raw units, shape (m, d)."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("jacobian(x) takes a single point of shape (d,)")
        xs = (x - self._x_mean) / self._x_scale
        rows = []
        for gp, ys in zip(self._gps, self._y_scale, strict=True):
            # chain rule through both standardizations:
            # y = y_mean + y_scale * mu(x_std), x_std = (x - x_mean)/x_scale
            rows.append(ys * gp.mean_grad(xs) / self._x_scale)
        return np.stack(rows)

    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Per-output posterior LATENT covariance Cov(f(X1), f(X2)) in raw output
        units, shape ``(m, n1, n2)`` (the acquisition layer's EPIG needs this,
        §9.4). ``X1``/``X2`` are ``(n, d)`` raw recipe vectors."""
        self._require_fitted()
        Xs1 = (np.atleast_2d(np.asarray(X1, dtype=float)) - self._x_mean) / self._x_scale
        Xs2 = (np.atleast_2d(np.asarray(X2, dtype=float)) - self._x_mean) / self._x_scale
        return np.stack(
            [ys * ys * gp.cov(Xs1, Xs2) for gp, ys in zip(self._gps, self._y_scale, strict=True)]
        )
