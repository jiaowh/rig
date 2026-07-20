"""Conformal calibration layer — implements D4 (implementation-plan §5.6).

Three pieces, per-output throughout:

- :class:`JackknifePlusCalibrator` — jackknife+ (Barber et al. 2021) for
  tiny n (LOO at n <= 40), switching to K-fold CV+ above. Distribution-free
  ~(1 - 2*alpha) guarantee; the honest small-n path.
- :class:`SplitConformalCalibrator` — split conformal on the standardized
  residual score |y - mu| / sigma_total (a CQR-lite: variance-scaled, hence
  input-adaptive width). Needs a held-out calibration block, so it is the
  abundant-data / Phase-0 path (D4).
- :class:`ACIController` — online Adaptive Conformal Inference (Gibbs &
  Candès 2021) keyed on run index: alpha_{t+1} = alpha_t +
  gamma * (alpha_target - err_t). The PRIMARY real-data coverage path under
  drift; its rolling-coverage statistic is the §5.6 concrete drift detector.

:class:`ConformalForwardModel` wraps any ForwardModel and fills the
canonical ``conformal_set`` field. Band semantics: for the split calibrator
the band is ``mean ± kappa * sigma_total(x)`` with kappa the conformal
quantile of standardized residuals (the plan's ``q_hat(x) = kappa * sigma``,
§2.3); jackknife+ bands come from LOO/fold-model order statistics.
Default coverage target alpha = 0.1.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np

from rig.interfaces import ForwardModel, PredictiveDistribution

DEFAULT_ALPHA = 0.1


def _as_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def _mean_2d(mean: np.ndarray, nq: int) -> np.ndarray:
    """Normalize a model's predicted mean to shape (nq, m).

    Disambiguates 1-D returns: (m,) for a single query vs (nq,) for a batch
    from a single-output model.
    """
    mean = np.asarray(mean, dtype=float)
    if mean.ndim == 1:
        return mean[None, :] if nq == 1 else mean[:, None]
    return mean


def _sigma_total(dist: PredictiveDistribution) -> np.ndarray:
    return np.sqrt(np.asarray(dist.aleatoric_sigma) ** 2 + np.asarray(dist.epistemic_sigma) ** 2)


def conformal_quantile(scores: np.ndarray, alpha: float | np.ndarray) -> np.ndarray:
    """Finite-sample conformal quantile per output.

    ``scores``: (n, m) nonconformity scores. Returns the
    ceil((1-alpha)(n+1))-th smallest score per output, or +inf when that
    order statistic exceeds n (small-n honesty: no coverage claim possible).
    ``alpha`` may be a scalar or per-output (m,) array (ACI feeds the latter).
    """
    scores = _as_2d(scores)
    n, m = scores.shape
    alpha_arr = np.broadcast_to(np.asarray(alpha, dtype=float), (m,))
    k = np.ceil((1.0 - alpha_arr) * (n + 1)).astype(int)  # 1-based order stat
    sorted_scores = np.sort(scores, axis=0)
    out = np.empty(m)
    for j in range(m):
        out[j] = np.inf if k[j] > n else sorted_scores[k[j] - 1, j]
    return out


class SplitConformalCalibrator:
    """Split conformal with standardized-residual score s = |y - mu|/sigma_total.

    Variance-scaled (CQR-lite): band width adapts to the model's own
    input-dependent sigma_total, so intervals widen where the surrogate is
    uncertain. ``fit`` consumes a HELD-OUT calibration split — never the
    training data (§5.3/§5.6 leakage guard is the caller's responsibility).
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        self.model: ForwardModel | None = None
        self.scores_: np.ndarray | None = None  # (n_cal, m)

    def fit(self, model: ForwardModel, X_cal: np.ndarray, Y_cal: np.ndarray) -> None:
        self.model = model
        self.scores_ = self.score(np.asarray(X_cal, dtype=float), _as_2d(Y_cal))

    def score(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Nonconformity score(s) |y - mu| / sigma_total at x."""
        assert self.model is not None, "fit() first"
        dist = self.model.predict(x)
        y = np.asarray(y, dtype=float)
        mu, sig = np.asarray(dist.mean, dtype=float), _sigma_total(dist)
        if y.ndim == 2:  # batch: normalize model output to (n, m)
            mu, sig = _mean_2d(mu, y.shape[0]), _mean_2d(sig, y.shape[0])
        return np.abs(y - mu) / sig

    def append(self, x: np.ndarray, y: np.ndarray) -> None:
        """Online score update (consumed by ACIController.observe)."""
        assert self.scores_ is not None, "fit() first"
        s = np.atleast_2d(self.score(x, y))
        self.scores_ = np.vstack([self.scores_, s])

    def kappa(self, alpha: float | np.ndarray | None = None) -> np.ndarray:
        """Per-output band multiplier: q_hat(x) = kappa * sigma_total(x)."""
        assert self.scores_ is not None, "fit() first"
        return conformal_quantile(self.scores_, self.alpha if alpha is None else alpha)

    def interval(self, x: np.ndarray, alpha: float | np.ndarray | None = None) -> np.ndarray:
        """Calibrated interval per output: (m, 2) for a (d,) point, (n, m, 2)
        for a batch."""
        assert self.model is not None, "fit() first"
        dist = self.model.predict(x)
        half = self.kappa(alpha) * _sigma_total(dist)
        return np.stack([dist.mean - half, dist.mean + half], axis=-1)


class JackknifePlusCalibrator:
    """Jackknife+ / CV+ intervals (Barber et al. 2021) around any ForwardModel.

    LOO mode for n <= ``loo_max_n`` (default 40, per the plan's tiny-n
    regime); K-fold CV+ above (default K=10). ``fit`` takes a zero-arg
    ``model_factory`` returning a FRESH unfitted model, because jackknife+
    needs leave-out refits — the wrapped production model itself is not
    touched. Guarantee: >= 1 - 2*alpha coverage distribution-free; empirically
    ~ 1 - alpha.
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        loo_max_n: int = 40,
        k_folds: int = 10,
        seed: int = 0,
    ) -> None:
        self.alpha = alpha
        self.loo_max_n = loo_max_n
        self.k_folds = k_folds
        self.seed = seed
        self.models_: list[Any] = []
        self.fold_of_: np.ndarray | None = None  # (n,) index into models_
        self.residuals_: np.ndarray | None = None  # (n, m) LOO/OOF |residual|
        self.mode_: str | None = None

    def fit(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        Y: np.ndarray,
    ) -> None:
        X = np.asarray(X, dtype=float)
        Y = _as_2d(Y)
        n = X.shape[0]
        if n < 3:
            raise ValueError("jackknife+ needs at least 3 points")
        if n <= self.loo_max_n:
            self.mode_ = "loo"
            folds = [np.array([i]) for i in range(n)]
        else:
            self.mode_ = "cv+"
            k = min(self.k_folds, n)
            perm = np.random.default_rng(self.seed).permutation(n)
            folds = [perm[f::k] for f in range(k)]
        self.models_ = []
        self.fold_of_ = np.empty(n, dtype=int)
        self.residuals_ = np.empty_like(Y)
        for f, heldout in enumerate(folds):
            train = np.setdiff1d(np.arange(n), heldout)
            model = model_factory()
            model.fit(X[train], Y[train])
            self.models_.append(model)
            self.fold_of_[heldout] = f
            mu = _mean_2d(model.predict(X[heldout]).mean, len(heldout))
            self.residuals_[heldout] = np.abs(Y[heldout] - mu)

    def interval(self, x: np.ndarray, alpha: float | np.ndarray | None = None) -> np.ndarray:
        """Jackknife+/CV+ interval: order statistics of mu_{-i}(x) ± R_i.

        (m, 2) for a single (d,) point; (nq, m, 2) for a batch.
        """
        assert self.residuals_ is not None and self.fold_of_ is not None, "fit() first"
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        n, m = self.residuals_.shape
        alpha_arr = np.broadcast_to(
            np.asarray(self.alpha if alpha is None else alpha, dtype=float), (m,)
        )
        # fold-model predictions at the query points: (n_folds, nq, m)
        mu_folds = np.stack([_mean_2d(mdl.predict(Xq).mean, Xq.shape[0]) for mdl in self.models_])
        mu_i = mu_folds[self.fold_of_]  # (n, nq, m): mu_{-i}(x)
        lo_vals = np.sort(mu_i - self.residuals_[:, None, :], axis=0)
        hi_vals = np.sort(mu_i + self.residuals_[:, None, :], axis=0)
        k_lo = np.floor(alpha_arr * (n + 1)).astype(int)  # 1-based
        k_hi = np.ceil((1.0 - alpha_arr) * (n + 1)).astype(int)
        nq = Xq.shape[0]
        out = np.empty((nq, m, 2))
        for j in range(m):
            out[:, j, 0] = -np.inf if k_lo[j] < 1 else lo_vals[k_lo[j] - 1, :, j]
            out[:, j, 1] = np.inf if k_hi[j] > n else hi_vals[k_hi[j] - 1, :, j]
        return out[0] if single else out


class ACIController:
    """Online Adaptive Conformal Inference (Gibbs & Candès 2021), run-indexed.

    Wraps either calibrator's quantile with a per-output adaptive miscoverage
    level: ``alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)``,
    clipped to ``alpha_clip``. ``observe(x, y)`` scores the CURRENT interval
    first (so err_t uses the pre-update alpha_t), then updates alpha and —
    when the calibrator supports online scores (split conformal) — appends
    the new residual score, which is what lets bands outgrow a stale
    calibration set under drift.

    ``rolling_coverage`` over the last ``window`` observations is the §5.6
    concrete drift detector: sustained coverage below nominal = drift.
    """

    def __init__(
        self,
        calibrator: SplitConformalCalibrator | JackknifePlusCalibrator,
        alpha_target: float = DEFAULT_ALPHA,
        gamma: float = 0.05,
        alpha_clip: tuple[float, float] = (0.001, 0.5),
        window: int = 50,
        update_scores: bool = True,
    ) -> None:
        self.calibrator = calibrator
        self.alpha_target = alpha_target
        self.gamma = gamma
        self.alpha_clip = alpha_clip
        self.update_scores = update_scores
        self.alpha_t: np.ndarray = np.asarray(float(alpha_target))
        self._errs: deque[np.ndarray] = deque(maxlen=window)
        self.t = 0

    def interval(self, x: np.ndarray) -> np.ndarray:
        """Calibrated interval at the current adaptive alpha_t."""
        return self.calibrator.interval(x, alpha=self.alpha_t)

    def observe(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ingest one run (run-index keyed): score coverage, adapt alpha_t.

        Returns the per-output miscoverage indicator err_t (m,).
        """
        itv = self.interval(np.asarray(x, dtype=float))  # (m, 2) up to squeeze
        y = np.asarray(y, dtype=float).reshape(-1)
        lo, hi = itv[..., 0].reshape(-1), itv[..., 1].reshape(-1)
        finite = np.isfinite(y)
        # Fail-closed: a non-finite observation (dead/dropped sensor) cannot
        # be shown to be covered, so it counts as a MISS. Scoring it as a hit
        # (the strict-inequality comparisons below are False on NaN) would
        # raise alpha_t and narrow bands — the unsafe direction — and mask a
        # dead stream behind rolling_coverage == 1.0. Design decision: for a
        # multi-output y, any non-finite entry withholds the append (a score
        # row is per-observation, not per-output) but each output's err/alpha
        # update is still scored independently.
        err = np.where(finite, ((y < lo) | (y > hi)).astype(float), 1.0)
        self.alpha_t = np.clip(
            self.alpha_t + self.gamma * (self.alpha_target - err), *self.alpha_clip
        )
        self._errs.append(err)
        self.t += 1
        if self.update_scores and hasattr(self.calibrator, "append") and np.all(finite):
            self.calibrator.append(x, y)
        return err

    @property
    def rolling_coverage(self) -> np.ndarray:
        """Empirical coverage per output over the trailing window (drift detector)."""
        if not self._errs:
            return np.asarray(np.nan)
        return 1.0 - np.mean(np.stack(self._errs), axis=0)


class ConformalForwardModel:
    """ForwardModel wrapper that fills the canonical ``conformal_set`` field.

    Delegates mean/sigmas/support_score/jacobian to the base model;
    ``conformal_set`` is the calibrated per-output interval — shape (m, 2)
    for a single point, (n, m, 2) for a batch. When an :class:`ACIController`
    is supplied, its adaptive alpha_t governs the band (the drift-robust
    path); otherwise the calibrator's static alpha does.

    ``update(records)`` refits the BASE model only. Static calibrations go
    stale by construction (their scores came from the old model); re-fit the
    calibrator on fresh held-out data after an update, or run under an
    ACIController whose online score stream tracks the model as it changes.
    """

    def __init__(
        self,
        base: ForwardModel,
        calibrator: SplitConformalCalibrator | JackknifePlusCalibrator,
        controller: ACIController | None = None,
    ) -> None:
        self.base = base
        self.calibrator = calibrator
        self.controller = controller

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        dist = self.base.predict(x)
        source = self.controller if self.controller is not None else self.calibrator
        return PredictiveDistribution(
            mean=dist.mean,
            aleatoric_sigma=dist.aleatoric_sigma,
            epistemic_sigma=dist.epistemic_sigma,
            conformal_set=source.interval(x),
        )

    def support_score(self, x: np.ndarray) -> float:
        return self.base.support_score(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.base.jacobian(x)

    def update(self, records: Iterable[Any]) -> None:
        self.base.update(records)

    def observe(self, x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
        """Feed one realized run to the ACI controller (no-op without one)."""
        if self.controller is None:
            return None
        return self.controller.observe(x, y)
