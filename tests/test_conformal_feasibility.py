"""F1 regression (audit 2026-07-21): the §13.2 conformal containment gate must be
part of the DEFAULT feasibility decision whenever the solver's model is conformal-
wrapped — not only when an explicit ``revalidation_model`` is injected.

Mechanism twin, NOT the real run. The recorded d=20 false success
(docs/dimensionality-2026-07-17.md) is a real, deterministic case where the GP's
fitted RAW σ was optimistic, so the pessimistic κ·σ margins certified a recipe the
calibrated conformal band would have rejected — but "``conformal_set`` is not part of
the feasibility decision at all". Reproducing that exact 20-D GP run is too slow for a
unit test, so we build its MECHANISM here with a controllable model whose raw σ is
deliberately optimistic (tiny), wrapped by the REAL §5.6 conformal stack
(:class:`SplitConformalCalibrator` + :class:`ConformalForwardModel`) calibrated on
GENUINE residuals whose spread is far wider than that σ. The calibrated band is then
wider than a spec box the raw margins happily certify — exactly the d=20 geometry.

Assertions:
  (a) the UNWRAPPED model returns FEASIBLE candidates, labelled ``"model-feasible"``
      (raw-σ pessimism only — explicitly NOT a calibrated guarantee);
  (b) the SAME model conformal-WRAPPED, on the DEFAULT path (no revalidation_model),
      REJECTS them — an INFEASIBLE with the conformal (aleatoric/coverage) reason —
      and, when the box is wide enough for the band to fit, returns only candidates
      labelled ``"conformal-checked"``.
"""

from __future__ import annotations

import numpy as np

from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.interfaces import ContinuousVariable, Infeasible, PredictiveDistribution, RecipeCandidate
from rig.inverse import PessimisticInverseSolver

_KAPPA = 2.0
_Z_EPI = 2.0
_ALE = 0.02  # the deliberately OPTIMISTIC raw aleatoric σ (overconfident)
_RESID_SIGMA = 0.3  # the GENUINE residual spread the conformal calibrator sees
_SLOPE = 2.0
_ALPHA = 0.1


class _Overconfident:
    """Linear-mean ForwardModel (``mean = slope·x``) with a deliberately OPTIMISTIC
    (tiny) aleatoric σ and zero epistemic. The raw §8 margins will therefore certify a
    box far narrower than the genuine residual spread the conformal calibrator is fitted
    on — the mechanism twin of the d=20 false success (there the GP's FITTED σ was
    optimistic; here we set it so, deterministically). ``conformal_set`` is None: it is a
    bare model, exactly the historical default-path input."""

    def __init__(self, slope: float, ale: float) -> None:
        self.slope = float(slope)
        self.ale = float(ale)

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mu = (self.slope * Xq[:, 0])[:, None]  # (n, 1)
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

    def update(self, records) -> None:  # pragma: no cover - not exercised
        pass


def _wrapped(base: _Overconfident, X_cal: np.ndarray, Y_cal: np.ndarray) -> ConformalForwardModel:
    """Wrap ``base`` with the REAL §5.6 split-conformal stack, calibrated on the given
    (genuine-residual) held-out block."""
    cal = SplitConformalCalibrator(alpha=_ALPHA)
    cal.fit(base, X_cal, Y_cal)
    return ConformalForwardModel(base, cal)


def _calibration_block(seed: int = 0):
    """Deterministic held-out calibration block whose residuals have the GENUINE spread
    ``_RESID_SIGMA`` — much wider than the model's optimistic raw σ ``_ALE``."""
    rng = np.random.default_rng(seed)
    X_cal = np.linspace(0.2, 4.8, 60)[:, None]
    Y_cal = _SLOPE * X_cal + rng.normal(0.0, _RESID_SIGMA, size=X_cal.shape)
    return X_cal, Y_cal


def _solver(model, **kw) -> PessimisticInverseSolver:
    return PessimisticInverseSolver(
        model,
        [ContinuousVariable("x", 0.0, 5.0)],
        ["y"],
        X_train=np.linspace(0.0, 5.0, 12)[:, None],
        kappa=_KAPPA,
        z_epi=_Z_EPI,
        delta_frac=0.0,  # keep the margin math to μ/σ only (no Jacobian δ term)
        seed=0,
        **kw,
    )


def _band_half_width(conf: ConformalForwardModel, x_star: np.ndarray) -> float:
    cs = np.asarray(conf.predict(x_star).conformal_set)  # (m, 2)
    return float((cs[0, 1] - cs[0, 0]) / 2.0)


def test_default_conformal_gate_rejects_overconfident_feasible():
    # The mechanism twin. Raw σ is tiny (0.02); genuine residuals are ~0.3, so the
    # calibrated band is ~0.5 wide — far wider than a box the raw κ·σ margins certify.
    base = _Overconfident(slope=_SLOPE, ale=_ALE)
    X_cal, Y_cal = _calibration_block()
    conf = _wrapped(base, X_cal, Y_cal)

    x_star = np.array([2.5])
    mu_star = float(base.predict(x_star).mean[0])  # 5.0
    raw_half = _KAPPA * _ALE  # z_epi·σ_epi(=0) + κ·σ_ale = the credited raw half-band
    band_half = _band_half_width(conf, x_star)  # calibrated, ~0.5
    # the mechanism precondition: the raw credited band is far tighter than the
    # calibrated band, so a box between them is raw-feasible yet conformally infeasible.
    assert raw_half < band_half, (raw_half, band_half)
    box_half = 0.5 * (raw_half + band_half)  # sits strictly inside (raw_half, band_half)
    lo, hi = mu_star - box_half, mu_star + box_half
    spec = {"targets": {"y": (lo, hi)}, "max_candidates": 3}

    # (a) UNWRAPPED: the raw-σ pessimism certifies the box -> FEASIBLE, "model-feasible".
    raw_res = _solver(base).solve(spec)
    assert isinstance(raw_res, list) and raw_res, "raw-σ solver should certify the box"
    for c in raw_res:
        assert isinstance(c, RecipeCandidate)
        assert c.feasibility_flag is True
        assert c.calibration_status == "model-feasible"  # NOT a calibrated guarantee

    # (b) WRAPPED, DEFAULT path (no revalidation_model): the calibrated band spills the
    # box for every candidate -> INFEASIBLE with the conformal (aleatoric/coverage)
    # reason, NOT the epistemic 'collect runs' one, and an honest nonzero spill.
    conf_res = _solver(conf).solve(spec)
    assert isinstance(conf_res, Infeasible), "default-path conformal gate must reject"
    assert "conformal" in conf_res.reason
    assert "C(x) ⊄ Z*" in conf_res.reason
    assert conf_res.distance_to_feasible > 0.0  # the real band spill, not a bare 0.0
    assert "collect runs" not in conf_res.reason  # not the epistemic mis-diagnosis


def test_default_conformal_gate_accepts_when_band_fits():
    # Positive path: a box WIDE enough that the calibrated band fits -> the wrapped
    # default solve returns candidates, now labelled "conformal-checked" (the gate ran
    # and passed on self.model), never silently "model-feasible".
    base = _Overconfident(slope=_SLOPE, ale=_ALE)
    X_cal, Y_cal = _calibration_block()
    conf = _wrapped(base, X_cal, Y_cal)

    x_star = np.array([2.5])
    mu_star = float(base.predict(x_star).mean[0])
    band_half = _band_half_width(conf, x_star)
    box_half = 2.5 * band_half  # comfortably wider than the calibrated band
    spec = {"targets": {"y": (mu_star - box_half, mu_star + box_half)}, "max_candidates": 3}

    conf_res = _solver(conf).solve(spec)
    assert isinstance(conf_res, list) and conf_res, "a band-fitting box must stay feasible"
    for c in conf_res:
        assert c.feasibility_flag is True
        assert c.calibration_status == "conformal-checked"
        # the returned candidate's calibrated band really does sit inside the box.
        cs = np.asarray(conf.predict(np.array([c.recipe["x"]])).conformal_set)  # (1, 2)
        assert cs[0, 0] >= mu_star - box_half - 1e-9
        assert cs[0, 1] <= mu_star + box_half + 1e-9
