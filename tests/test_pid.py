"""Conformal-PID controller tests (implementation-plan §20.2 endpoint, D4).

Deterministic, synthetic (no sim), fast. Covers, per the build tasking:
(a) exchangeable long-run coverage ~ 1-alpha; (b) the REPAIR test -- static
split-conformal under-covers post-drift while PID recovers; (c) err scored
against the PRE-update band; (d) finiteness incl. adversarial all-miss (the
anti-infinite-width property); (e) the integrator is load-bearing (P+I recovers
faster than P-only via the KI=0 mutation); (f) decaying step holds coverage with
falling threshold volatility; (g) determinism.
"""

from __future__ import annotations

import numpy as np

from rig.calibration import ConformalPIDController, SplitConformalCalibrator
from rig.interfaces import PredictiveDistribution

ALPHA = 0.1


class _Const:
    """Fixed-prediction stub (mu, sigma): goes stale by construction (drift)."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu, self.sigma = mu, sigma

    def fit(self, X: np.ndarray, Y: np.ndarray) -> _Const:
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


def _fresh_cal(seed: int = 0, n: int = 100, mu: float = 0.0, sigma: float = 1.0):
    """A split calibrator fitted on N(0,1) residuals against a mu=0/sigma=1 model."""
    rng = np.random.default_rng(seed)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(_Const(mu, sigma), np.zeros((n, 1)), rng.normal(mu, sigma, size=(n, 1)))
    return cal


# ---------------------------------------------------------------------------
# (a) exchangeable stream -> long-run coverage ~ 1 - alpha
# ---------------------------------------------------------------------------


def test_exchangeable_long_run_coverage():
    cal = _fresh_cal()
    pid = ConformalPIDController(cal, alpha_target=ALPHA)
    # warm-start is the split-conformal quantile, and is finite
    assert np.isfinite(pid.q_t).all()
    np.testing.assert_allclose(pid.q_t, cal.kappa(ALPHA))

    rng = np.random.default_rng(123)
    x = np.zeros(1)
    errs = []
    for _ in range(3000):
        itv = pid.interval(x)
        assert np.all(np.isfinite(itv))
        y = rng.normal(0.0, 1.0, size=1)
        errs.append(pid.observe(x, y))
    coverage = 1.0 - float(np.mean(errs))
    assert 0.86 <= coverage <= 0.94, coverage


# ---------------------------------------------------------------------------
# (b) THE REPAIR TEST: static under-covers post-drift; PID recovers to ~1-alpha
# ---------------------------------------------------------------------------


def test_repair_static_undercovers_pid_recovers_under_drift():
    cal = _fresh_cal(seed=99)
    kappa_static = float(cal.kappa()[0])  # frozen pre-drift band multiplier
    pid = ConformalPIDController(cal, alpha_target=ALPHA)

    rng = np.random.default_rng(99)
    x = np.zeros(1)
    y_pre = rng.normal(0.0, 1.0, size=100)
    y_post = rng.normal(2.0, 1.0, size=900)  # sustained +2 sigma mean shift
    for y in y_pre:
        pid.observe(x, np.array([y]))
    post_err = []
    for y in y_post:
        itv = pid.interval(x)
        assert np.all(np.isfinite(itv))
        post_err.append(float(pid.observe(x, np.array([y]))[0]))
    post_err = np.asarray(post_err)

    # static split conformal (mu=0, sigma=1 => band |y| <= kappa) collapses post-shift
    static_post_cov = float(np.mean(np.abs(y_post) <= kappa_static))
    assert static_post_cov < 0.60, static_post_cov
    # PID's threshold climbs and steady-state (last 400) coverage recovers to ~0.90
    pid_steady_cov = 1.0 - float(np.mean(post_err[-400:]))
    assert 0.83 <= pid_steady_cov <= 0.96, pid_steady_cov
    assert pid.q_t[0] > kappa_static  # it widened to chase the drift


# ---------------------------------------------------------------------------
# (c) err_t is scored against the PRE-update interval (guard mirroring ACI's)
# ---------------------------------------------------------------------------


def test_err_scored_against_preupdate_interval():
    # Choose y strictly BETWEEN the (narrower) post-update band edge and the
    # pre-update edge. A correct impl scores the PRE-update band -> hit (err=0)
    # and THEN narrows; a score-after-update bug would report a miss (err=1)
    # against the narrowed band. eta large + KI=0 isolates the proportional step.
    cal = _fresh_cal(seed=1)
    pid = ConformalPIDController(cal, alpha_target=ALPHA, eta=0.5, KI=0.0)
    x = np.zeros(1)
    q_pre = float(pid.q_t[0])
    q_post_on_hit = q_pre + 0.5 * (0.0 - ALPHA)  # qts + eta*(err - alpha), err=0
    assert q_post_on_hit < q_pre
    s = 0.5 * (q_pre + q_post_on_hit)  # score in the gap; mu=0,sigma=1 => y=s
    y = np.array([s])

    itv = pid.interval(x)
    lo, hi = float(itv[0, 0]), float(itv[0, 1])
    assert lo < y[0] < hi  # PRE-update band covers y

    err = pid.observe(x, y)
    assert err[0] == 0.0  # scored against the PRE-update band -> hit
    # and it DID update afterwards (narrowed to the predicted post-hit threshold)
    np.testing.assert_allclose(pid.q_t[0], q_post_on_hit, atol=1e-9)


# ---------------------------------------------------------------------------
# (d) FINITENESS by construction -- the anti-infinite-width property
# ---------------------------------------------------------------------------


def test_intervals_finite_including_first_steps_and_all_miss():
    cal = _fresh_cal(seed=7)
    pid = ConformalPIDController(cal, alpha_target=ALPHA)
    x = np.zeros(1)

    # finite before any observation and on the very first steps
    assert np.all(np.isfinite(pid.interval(x)))

    n_infinite_width = 0
    for _t in range(400):
        itv = pid.interval(x)
        lo, hi = float(itv[0, 0]), float(itv[0, 1])
        if not np.isfinite(hi - lo):
            n_infinite_width += 1
        # adversarial: a long all-miss stretch -- ACI would go +inf here
        pid.observe(x, np.array([1.0e7]))
        assert np.all(np.isfinite(pid.q_t))
    assert n_infinite_width == 0  # the contract the runner records
    assert np.isfinite(pid.q_t[0]) and pid.q_t[0] > 0  # widened but finite


def test_infinite_calibration_quantile_warm_starts_finite():
    # Tiny calibration set: ceil((1-alpha)(n+1)) > n so the split quantile is +inf.
    # The controller must still warm-start FINITE (fallback to the max score).
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(_Const(0.0, 1.0), np.zeros((5, 1)), np.array([[-1.0], [0.5], [2.0], [-0.3], [1.2]]))
    assert not np.isfinite(cal.kappa(ALPHA)[0])  # split quantile is +inf at n=5
    pid = ConformalPIDController(cal, alpha_target=ALPHA)
    assert np.isfinite(pid.q_t[0])
    assert np.all(np.isfinite(pid.interval(np.zeros(1))))


# ---------------------------------------------------------------------------
# (e) the integrator (I) term is LOAD-BEARING -- KI=0 mutation proof
# ---------------------------------------------------------------------------


def _transient_misses(KI: float, up: float, n: int, seed: int) -> int:
    cal = _fresh_cal(seed=seed)
    pid = ConformalPIDController(cal, alpha_target=ALPHA, KI=KI)
    rng = np.random.default_rng(1000 + seed)
    x = np.zeros(1)
    misses = 0
    for _ in range(n):
        _ = pid.interval(x)
        y = np.array([rng.normal(up, 1.0)])  # sustained one-sided (upward) shift
        misses += int(pid.observe(x, y)[0])
    return misses


def test_integrator_term_is_load_bearing():
    # On a stream with sustained one-sided miscoverage, the full P+I controller
    # (KI=2, the default) must recover FASTER than the P-only controller (KI=0):
    # fewer misses over the transient. Same stream, only the integrator toggled.
    up, n = 3.0, 120
    gap_total = 0
    for seed in range(6):
        m_p_only = _transient_misses(KI=0.0, up=up, n=n, seed=seed)
        m_pi = _transient_misses(KI=2.0, up=up, n=n, seed=seed)
        assert m_pi <= m_p_only, (seed, m_pi, m_p_only)  # never worse
        gap_total += m_p_only - m_pi
    assert gap_total >= 10, gap_total  # materially fewer misses (observed ~20)


# ---------------------------------------------------------------------------
# (f) decaying step: coverage holds; threshold volatility falls over time
# ---------------------------------------------------------------------------


def _run_coverage_and_qtrace(step: str, seed: int = 3, n: int = 1500):
    cal = _fresh_cal()
    pid = ConformalPIDController(cal, alpha_target=ALPHA, step=step)
    rng = np.random.default_rng(seed)
    x = np.zeros(1)
    errs, qs = [], []
    for _ in range(n):
        errs.append(float(pid.observe(x, rng.normal(0.0, 1.0, size=1))[0]))
        qs.append(float(pid.q_t[0]))
    return 1.0 - np.mean(errs), np.asarray(qs)


def test_decaying_step_holds_coverage_and_reduces_volatility():
    cov_fixed, q_fixed = _run_coverage_and_qtrace("fixed")
    cov_decay, q_decay = _run_coverage_and_qtrace("decaying")

    # coverage holds for both on the exchangeable stream
    assert 0.85 <= cov_fixed <= 0.95, cov_fixed
    assert 0.85 <= cov_decay <= 0.95, cov_decay

    # decaying step: threshold volatility falls over time, and is far below the
    # (roughly constant) fixed-step volatility late in the stream
    early_decay = q_decay[100:300].std()
    late_decay = q_decay[-200:].std()
    late_fixed = q_fixed[-200:].std()
    assert late_decay < early_decay, (late_decay, early_decay)
    assert late_decay < 0.5 * late_fixed, (late_decay, late_fixed)


# ---------------------------------------------------------------------------
# (g) determinism: identical runs -> identical traces
# ---------------------------------------------------------------------------


def _q_and_err_trace(seed: int):
    cal = _fresh_cal()
    pid = ConformalPIDController(cal, alpha_target=ALPHA)
    rng = np.random.default_rng(seed)
    x = np.zeros(1)
    qs, errs = [], []
    for _ in range(500):
        qs.append(pid.q_t.copy())
        errs.append(pid.observe(x, rng.normal(0.3, 1.2, size=1)))
    return np.array(qs), np.array(errs)


def test_determinism_identical_traces():
    q1, e1 = _q_and_err_trace(2024)
    q2, e2 = _q_and_err_trace(2024)
    np.testing.assert_array_equal(q1, q2)
    np.testing.assert_array_equal(e1, e2)


# ---------------------------------------------------------------------------
# extra: per-output independence and the (off-by-default) scorecaster hook
# ---------------------------------------------------------------------------


class _Const2:
    """Two-output stub with distinct per-output means/scales."""

    def __init__(self, mus=(0.0, 10.0), sigmas=(1.0, 3.0)) -> None:
        self.mus = np.asarray(mus, dtype=float)
        self.sigmas = np.asarray(sigmas, dtype=float)

    def fit(self, X, Y):
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        n = 1 if x.ndim == 1 else x.shape[0]
        mean = np.broadcast_to(self.mus, (n, 2)).copy()
        ale = np.broadcast_to(self.sigmas, (n, 2)).copy()
        if x.ndim == 1:
            mean, ale = mean[0], ale[0]
        return PredictiveDistribution(
            mean=mean, aleatoric_sigma=ale, epistemic_sigma=np.zeros_like(mean), conformal_set=None
        )


def test_per_output_independent_controllers():
    rng = np.random.default_rng(5)
    n = 120
    X = np.zeros((n, 1))
    Y = np.stack([rng.normal(0.0, 1.0, n), rng.normal(10.0, 3.0, n)], axis=-1)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(_Const2(), X, Y)
    pid = ConformalPIDController(cal, alpha_target=ALPHA)
    assert pid.q_t.shape == (2,)

    x = np.zeros(1)
    errs = []
    rng2 = np.random.default_rng(6)
    for _ in range(1500):
        itv = pid.interval(x)
        assert itv.shape == (2, 2)
        y = np.array([rng2.normal(0.0, 1.0), rng2.normal(10.0, 3.0)])
        errs.append(pid.observe(x, y))
    cov = 1.0 - np.mean(np.stack(errs), axis=0)
    assert np.all((cov >= 0.85) & (cov <= 0.95)), cov


def test_scorecaster_hook_shifts_threshold():
    # The D hook is OFF by default; when supplied it is added to q (per output).
    cal = _fresh_cal(seed=2)
    base = ConformalPIDController(cal, alpha_target=ALPHA)
    shifted = ConformalPIDController(
        cal, alpha_target=ALPHA, scorecaster=lambda t_pred, x: np.array([0.5])
    )
    x = np.zeros(1)
    y = np.array([0.2])
    base.observe(x, y)
    shifted.observe(x, y)
    np.testing.assert_allclose(shifted.q_t[0] - base.q_t[0], 0.5, atol=1e-9)
