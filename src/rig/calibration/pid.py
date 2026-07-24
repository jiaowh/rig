"""Conformal-PID online calibration -- the §20.2 drift ENDPOINT (D4).

implementation-plan §20.2 designates *conformal-PID / decaying-step online CP*
as the online/drift endpoint that **supersedes bare ACI (which becomes a
component)**. :class:`ConformalPIDController` implements it per

- Angelopoulos, Candès & Tibshirani, "Conformal PID Control for Time Series
  Prediction" (NeurIPS 2023) -- the P (quantile-tracker) + I (integrator) +
  D (scorecaster) decomposition; and
- Angelopoulos, Barber & Bates, "Online conformal prediction with decaying
  step sizes" (ICML 2024) -- the ``step="decaying"`` schedule.

It mirrors :class:`~rig.calibration.conformal.ACIController` so the M1 runner
can treat the two symmetrically (``interval`` / ``observe`` / ``rolling_coverage``),
but the state it tracks is different -- and that difference is the whole point.

Threshold tracking, not alpha tracking (the anti-infinite-width property)
------------------------------------------------------------------------
ACI adapts a per-output *miscoverage level* ``alpha_t`` and then asks the split
calibrator for the conformal quantile at that level; that quantile is a sample
ORDER STATISTIC, and it is ``+inf`` whenever ``ceil((1-alpha_t)(n+1)) > n`` (the
index runs off the end of the calibration scores). So ACI "covers" for free via
unbounded intervals when ``alpha_t`` drops below ``1/(n_scores+1)`` -- the
documented infinite-width trap (``n_infinite_width`` in the Empa results).

Conformal-PID instead tracks the score THRESHOLD ``q_t`` directly, as a real
number, and builds the band as ``mean(x) ± q_t · sigma_total(x)``. Because
``q_t`` is a finite real and ``sigma_total(x)`` is finite and positive, **every
emitted interval is finite at every step, by construction** -- there is no
order-statistic index to overflow. Under sustained miscoverage the P term grows
``q_t`` linearly (finite at every finite t) and the I term adds a *clamped*
(finite) correction; the interval widens, as it should, but never to ``inf``.
This is asserted in ``observe`` and tested (``tests/test_pid.py`` finiteness /
all-miss cases).

Band semantics reuse the split calibrator (comparability)
---------------------------------------------------------
:class:`~rig.calibration.conformal.SplitConformalCalibrator` scores on the
STANDARDIZED residual ``s = |y - mu| / sigma_total`` and bands as
``mean ± kappa · sigma_total`` with ``kappa`` the conformal quantile of those
scores (the plan's ``q_hat(x) = kappa · sigma(x)``, §2.3). This controller
tracks ``q_t`` on that SAME standardized-residual scale and bands as
``mean ± q_t · sigma_total(x)`` -- so PID, split-conformal, and ACI are directly
comparable (same input-adaptive band shape; only the multiplier differs), and a
single set of library-default constants transfers across every output/campaign
because the score scale is already normalized by construction (no per-dataset
``proportional_lr`` needed, unlike the paper's raw-score experiments).

Update (per output, independent controllers -- exactly like ACI)
----------------------------------------------------------------
With ``err_t = 1`` iff the PRE-update band missed (``s_t > q_t``) and target
miscoverage ``alpha``:

- **P (quantile tracker), the online quantile-regression step.**
  ``qts_{t+1} = qts_t + eta_t · (err_t - alpha)``. This is the paper's
  ``qts[t+1] = qts[t] - lr·grad`` with ``grad = alpha`` on a hit /
  ``-(1-alpha)`` on a miss (algebraically identical). It is also exactly ACI's
  update rule, but on the threshold scale instead of the alpha scale.
- **I (integrator), the paper's log-time tan saturation.**
  ``integrator_{t+1} = KI · tan_clamped( S_t · ln(t+1) / (Csat·(t+1)) )`` with
  ``S_t = Σ_{i≤t}(err_i - alpha)`` the running signed coverage error. This is
  the paper's ``saturation_fn_log`` verbatim EXCEPT that the paper's ``mytan``
  returns ``±inf`` at ``±pi/2``; we clamp the argument strictly inside
  ``(-pi/2, pi/2)`` so the integrator -- and hence the interval -- stays finite
  (see above). The integrator corrects PERSISTENT one-sided miscoverage that the
  proportional term alone corrects only slowly; it is a memoryless function of
  the running error sum, added on top of the tracker (``q = qts + integrator``),
  so setting ``KI=0`` recovers the pure P controller.
- **D (scorecaster), OFF by default.** A forecast ``ŝ_{t+1}`` of the next score,
  added to ``q`` (``q = qts + integrator + ŝ``). It needs a trained forecasting
  model (the paper uses a Theta model); the **P+I core is the robust endpoint**
  and D is an optional accelerator. We expose only the hook signature
  ``scorecaster(t_pred: int, x: np.ndarray) -> np.ndarray`` (per output).

``q_t`` is finally clamped to be non-negative (a threshold on a non-negative
``|residual|`` score; a negative threshold would be a vacuously-empty set) --
this is a floor only, there is no upper clamp, so the response can widen without
bound while remaining finite.

Library defaults (uniform, fixed before any outcome -- NO tuning-to-pass)
------------------------------------------------------------------------
- ``alpha_target = 0.1`` (nominal 90% coverage), matching the D4 default.
- ``eta = 0.1`` -- the P learning rate on the standardized-residual scale. The
  paper's quantile-method ``lr`` grid is ``{1, 0.5, 0.1, 0.05}`` on *raw* score
  scales; 0.1 is the scale-appropriate round value on the O(1) standardized
  scale (each miss widens ``q`` by ``eta(1-alpha)=0.09`` ≈ 6% of a typical
  warm-start ``q0≈1.6``; each hit tightens by ``eta·alpha=0.01``). It is the
  threshold-scale analogue of ACI's gentle ``gamma=0.05``.
- ``KI = 2.0`` -- integrator gain: the max integrator contribution is
  ``KI·tan_clamped(→pi/2) ``; in the responsive regime it reaches ≈2 standardized
  score units within a few dozen sustained one-sided steps, enough to visibly
  accelerate drift correction (proven load-bearing by the mutation test) without
  dominating the tracker.
- ``Csat = 7.0`` -- saturation timescale. The paper uses ``Csat=1`` paired with
  a ``T_burnin≈100`` (integrator disabled for 100 steps) on raw scores; we use
  NO burn-in (the warm-start ``q0`` already supplies the scale) on SHORT
  standardized streams (~100 rows), so we widen ``Csat`` to keep the integrator
  in its near-linear, sane regime over realistic horizons -- it only approaches
  the finite saturation clamp under sustained one-sided miscoverage of order
  ``exp(Csat·(pi/2)/(1-alpha)) ≈ 2e5`` steps, i.e. effectively never, while
  still "biting" (adding ~1 unit) by t≈30-50. This rescaling is a documented,
  uniform default, NOT tuned to any campaign.
- decaying step: ``eta_t = eta · t^{-(1/2 + eps)}`` with ``eps = 0.1`` (exponent
  0.6), satisfying the Robbins-Monro conditions ``Σ eta_t = ∞`` and
  ``Σ eta_t^2 < ∞`` that give convergence to a fixed population quantile when one
  exists (the 2024 paper's improvement over constant-step ACI: near-target
  coverage at every time under stability, not merely on average). Default is
  ``step="fixed"``.

Score-buffer decision (explicit)
--------------------------------
``observe`` does **not** append scores to any calibration buffer (no
``SplitConformalCalibrator.append`` analogue of ACI's ``update_scores``). With
direct threshold tracking a growing score buffer is unnecessary: the controller
adapts by MOVING ``q_t``, not by re-widening a stale calibration set. The
calibration scores are used only (a) to warm-start ``q_0`` and (b) as the fixed
reference ECDF for the ``effective_alpha`` reporting map. Consequence: the online
path is pure threshold dynamics; the fitted calibrator's ``scores_`` are never
mutated (the runner's static/ACI blocks are untouched), and there is no stale
buffer to overflow into an infinite quantile -- which is exactly why the finite
property holds.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np

from rig.calibration.conformal import (
    DEFAULT_ALPHA,
    SplitConformalCalibrator,
    _mean_2d,
    _sigma_total,
)

# Clamp the saturation-tan argument strictly inside (-pi/2, pi/2). The paper's
# mytan returns +/-inf at the boundary; we keep it finite -- the anti-infinite-
# width guarantee. tan(pi/2 - 1e-3) ~ 1e3, so the integrator is bounded but can
# still push hard; the proportional term supplies the (finite) unbounded widening.
_TAN_ARG_LIMIT = np.pi / 2.0 - 1e-3


def _sat_tan(arg: np.ndarray) -> np.ndarray:
    """Finite (clamped) tangent -- the paper's saturation with mytan's inf removed."""
    return np.tan(np.clip(arg, -_TAN_ARG_LIMIT, _TAN_ARG_LIMIT))


class ConformalPIDController:
    """Conformal-PID online calibrator (§20.2 endpoint); see the module docstring.

    Constructed from a *fitted* :class:`SplitConformalCalibrator`; warm-starts the
    per-output threshold ``q_0`` from that calibrator's conformal quantile at
    ``alpha_target`` (with a finite fallback to the max calibration score should
    the quantile be ``+inf`` at tiny n -- so ``q_0`` is finite by construction).

    API mirrors :class:`ACIController`: :meth:`interval` at the current state,
    :meth:`observe` (scores the PRE-update band, then adapts), a
    :attr:`rolling_coverage` window statistic for the §5.6 drift detector, and
    :attr:`q_t` / :meth:`effective_alpha` for the reporting trace.
    """

    def __init__(
        self,
        calibrator: SplitConformalCalibrator,
        alpha_target: float = DEFAULT_ALPHA,
        eta: float = 0.1,
        KI: float = 2.0,
        Csat: float = 7.0,
        window: int = 50,
        step: str = "fixed",
        decay_eps: float = 0.1,
        integrate: bool = True,
        scorecaster: Callable[[int, np.ndarray], np.ndarray] | None = None,
        q_floor: float = 0.0,
    ) -> None:
        if calibrator.scores_ is None or calibrator.model is None:
            raise ValueError("ConformalPIDController needs a fitted SplitConformalCalibrator")
        if step not in ("fixed", "decaying"):
            raise ValueError(f"step must be 'fixed' or 'decaying', got {step!r}")
        self.calibrator = calibrator
        self.alpha_target = float(alpha_target)
        self.eta = float(eta)
        self.KI = float(KI)
        self.Csat = float(Csat)
        self.step = step
        self.decay_eps = float(decay_eps)
        self.integrate = bool(integrate)
        self.scorecaster = scorecaster
        self.q_floor = float(q_floor)

        scores = np.asarray(calibrator.scores_, dtype=float)  # (n_cal, m)
        self._m = scores.shape[1]
        # Warm start q_0 = split-conformal quantile at alpha_target, made finite.
        q0 = np.asarray(calibrator.kappa(self.alpha_target), dtype=float).reshape(-1).copy()
        inf = ~np.isfinite(q0)
        if inf.any():  # tiny-n: the order statistic ran off the end -> use max score
            q0[inf] = scores.max(axis=0)[inf]
        q0 = np.maximum(q0, self.q_floor)
        assert np.all(np.isfinite(q0)), "warm-start q_0 must be finite"

        self.q_t: np.ndarray = q0  # (m,) threshold the NEXT interval() will use
        self.qts: np.ndarray = q0.copy()  # (m,) proportional-tracker state
        self.err_sum: np.ndarray = np.zeros(self._m)  # (m,) running Σ(err_i - alpha)
        self._errs: deque[np.ndarray] = deque(maxlen=window)
        self.window = int(window)
        self.t = 0

    # -- step-size schedule ---------------------------------------------------
    def _eta_at(self, t_pred: int) -> float:
        """P learning rate at prediction index ``t_pred`` (1-based)."""
        if self.step == "decaying":
            return self.eta * float(max(t_pred, 1)) ** (-(0.5 + self.decay_eps))
        return self.eta

    def _integrator(self, err_sum: np.ndarray, t_pred: int) -> np.ndarray:
        """The paper's log-time tan integrator (I term), finite by clamp."""
        if not self.integrate or self.KI == 0.0:
            return np.zeros(self._m)
        arg = err_sum * np.log(t_pred + 1) / (self.Csat * (t_pred + 1))
        return self.KI * _sat_tan(arg)

    # -- interval / observe (mirror ACIController) ----------------------------
    def interval(self, x: np.ndarray) -> np.ndarray:
        """Calibrated band at the CURRENT threshold: ``mean ± q_t · sigma_total``.

        ``(m, 2)`` for a single ``(d,)`` point, ``(n, m, 2)`` for a batch.
        Finite at every step by construction (``q_t`` finite, ``sigma_total`` finite).
        """
        x = np.asarray(x, dtype=float)
        dist = self.calibrator.model.predict(x)
        mu = np.asarray(dist.mean, dtype=float)
        sig = _sigma_total(dist)
        if x.ndim == 1:
            half = self.q_t * sig  # (m,)
            return np.stack([mu - half, mu + half], axis=-1)
        mu = _mean_2d(mu, x.shape[0])
        sig = _mean_2d(sig, x.shape[0])
        half = self.q_t[None, :] * sig  # (n, m)
        return np.stack([mu - half, mu + half], axis=-1)

    def observe(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ingest one run: score the PRE-update band, then update ``q_t`` (P + I + D).

        Returns the per-output miscoverage indicator ``err_t`` (m,), scored
        against the band ``interval()`` would have returned BEFORE this call --
        an observation never influences its own interval. Fail-closed on a
        non-finite ``y`` (dead sensor): it counts as a MISS (widens), never a hit.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)  # (m,)
        score = np.asarray(self.calibrator.score(x, y), dtype=float).reshape(-1)  # |y-mu|/sig
        finite = np.isfinite(y)
        with np.errstate(invalid="ignore"):
            miss = score > self.q_t  # y outside [mu - q_t*sig, mu + q_t*sig]
        err = np.where(finite, miss.astype(float), 1.0)  # (m,), fail-closed

        # P: proportional / quantile-tracker step (== ACI's rule on the q scale)
        eta_t = self._eta_at(self.t + 1)
        qts_next = self.qts + eta_t * (err - self.alpha_target)

        # advance time; I: integrator over the running signed coverage error
        self.t += 1
        self.err_sum = self.err_sum + (err - self.alpha_target)
        integ = self._integrator(self.err_sum, self.t)

        # D: optional scorecaster (off by default)
        sc = np.zeros(self._m)
        if self.scorecaster is not None:
            sc = np.asarray(self.scorecaster(self.t, x), dtype=float).reshape(-1)

        q_next = np.maximum(self.q_floor, qts_next + integ + sc)
        if not np.all(np.isfinite(q_next)):  # must be impossible (anti-infinite-width)
            raise AssertionError("ConformalPIDController produced a non-finite threshold")
        self.qts = qts_next
        self.q_t = q_next
        self._errs.append(err)
        return err

    # -- reporting ------------------------------------------------------------
    @property
    def rolling_coverage(self) -> np.ndarray:
        """Empirical coverage per output over the trailing window (§5.6 drift detector)."""
        if not self._errs:
            return np.asarray(np.nan)
        return 1.0 - np.mean(np.stack(self._errs), axis=0)

    def effective_alpha(self, q: np.ndarray | None = None) -> np.ndarray:
        """Miscoverage the threshold ``q`` implies on the fixed calibration ECDF.

        The PID analogue of ACI's ``alpha_t``: the fraction of calibration scores
        that exceed ``q`` (per output). As ``q_t`` rises the band widens and this
        falls -- comparable, for the trace, to ACI's ``alpha_t`` dropping.
        """
        q_arr = self.q_t if q is None else np.asarray(q, dtype=float).reshape(-1)
        scores = np.asarray(self.calibrator.scores_, dtype=float)  # (n, m)
        return np.mean(scores > q_arr[None, :], axis=0)
