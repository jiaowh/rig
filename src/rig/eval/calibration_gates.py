"""Posterior-recovery gates: SBC + TARP (implementation-plan §12.2 — the second MFL-killer).

MFL never checks that the inverse recovers the *full* posterior over recipes
(multimodality, coverage). §12.2 mandates two distribution-free gates:

- **Simulation-Based Calibration** (Talts et al. 2018): if the posterior is
  calibrated, the rank of each ground-truth θ among its posterior samples is
  Uniform{0..M}. Deviations diagnose miscalibration (∪-shaped ⇒ overconfident,
  ∩-shaped ⇒ underconfident, slope ⇒ bias). Uniformity is judged with a
  **simultaneous ECDF band** (the Säilynoja et al. 2022 idea), computed here by
  simulation (a KS-type sup-deviation calibrated under the discrete-uniform
  null) — exact and assumption-light; the analytic band is an optimization.
- **TARP** (Lemos et al. 2023): tests posterior *coverage* via distance to
  random reference points. Reduces to a uniformity test on the per-trial
  credibility, so it shares the same band machinery.

These are a reusable HARNESS. The amortized posterior sampler they consume is
the torch-era normalizing flow (WP-E); until then they are validated on
synthetic posteriors (a calibrated one passes, an overconfident one fails). The
GP-tier inverse's diverse candidate set (§8.7) is a coarse posterior sample and
can also be fed here. numpy/scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# SBC
# ---------------------------------------------------------------------------


def sbc_ranks(theta_true: np.ndarray, posterior_samples: np.ndarray) -> np.ndarray:
    """SBC rank statistics (Talts et al. 2018).

    ``theta_true``: ``(L, dim)`` ground-truth parameters (one per simulated
    dataset). ``posterior_samples``: ``(L, M, dim)`` posterior draws for each.
    Returns ``(L, dim)`` ranks in ``{0..M}``: ``rank[l,d] = #{m :
    posterior_samples[l,m,d] < theta_true[l,d]}``. Under a calibrated posterior
    each column is Discrete-Uniform{0..M}.
    """
    theta_true = np.asarray(theta_true, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    if theta_true.ndim == 1:
        theta_true = theta_true[:, None]
    if posterior_samples.ndim == 2:
        posterior_samples = posterior_samples[:, :, None]
    L, dim = theta_true.shape
    if posterior_samples.shape[0] != L or posterior_samples.shape[2] != dim:
        raise ValueError(
            f"shape mismatch: theta_true {theta_true.shape}, "
            f"posterior_samples {posterior_samples.shape}"
        )
    return np.sum(posterior_samples < theta_true[:, None, :], axis=1).astype(int)


def _uniformity_sup_deviation(u: np.ndarray) -> float:
    """KS-type sup |ECDF(u) − t| for values ``u`` in [0,1] against Uniform[0,1]."""
    u = np.sort(np.asarray(u, dtype=float))
    n = u.size
    ecdf_hi = np.arange(1, n + 1) / n
    ecdf_lo = np.arange(0, n) / n
    return float(max(np.max(np.abs(ecdf_hi - u)), np.max(np.abs(ecdf_lo - u))))


def _sup_deviation_null(n: int, n_sim: int, seed: int) -> np.ndarray:
    """Null distribution of the sup-deviation statistic for ``n`` Uniform[0,1]
    draws (simulation)."""
    rng = np.random.default_rng(seed)
    stats = np.empty(n_sim)
    for i in range(n_sim):
        stats[i] = _uniformity_sup_deviation(rng.random(n))
    return stats


@dataclass(frozen=True)
class UniformityResult:
    """Simultaneous ECDF-band uniformity test result."""

    passed: bool
    statistic: float  # observed sup-deviation
    critical: float  # simultaneous critical value at ``confidence``
    p_value: float  # simulation p-value
    confidence: float


def uniformity_test(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    n_sim: int = 2000,
    seed: int = 0,
) -> UniformityResult:
    """Simultaneous (KS-type) test that ``values`` ⊂ [0,1] are Uniform[0,1],
    calibrated by simulation — the shared engine for SBC and TARP. ``passed`` iff
    the observed sup-deviation is within the ``confidence`` simultaneous band."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size == 0:
        raise ValueError("need at least one value")
    if np.any((values < -1e-9) | (values > 1 + 1e-9)):
        raise ValueError("values must lie in [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    stat = _uniformity_sup_deviation(values)
    null = _sup_deviation_null(values.size, n_sim, seed)
    critical = float(np.quantile(null, confidence))
    p_value = float(np.mean(null >= stat))
    return UniformityResult(
        passed=bool(stat <= critical),
        statistic=stat,
        critical=critical,
        p_value=p_value,
        confidence=confidence,
    )


def sbc_test(
    ranks: np.ndarray,
    n_posterior_samples: int,
    *,
    confidence: float = 0.95,
    n_sim: int = 2000,
    seed: int = 0,
) -> list[UniformityResult]:
    """Per-dimension SBC uniformity test. ``ranks``: ``(L, dim)`` from
    :func:`sbc_ranks`; ``n_posterior_samples`` = M. Maps ranks to
    ``(r + U(0,1)) / (M + 1)`` (jitter removes the discrete-grid artifact so the
    continuous-uniform band applies) and tests each dimension. Returns one
    :class:`UniformityResult` per dimension."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim == 1:
        ranks = ranks[:, None]
    M = int(n_posterior_samples)
    rng = np.random.default_rng(seed)
    out: list[UniformityResult] = []
    for d in range(ranks.shape[1]):
        # continuity-corrected normalized rank ~ Uniform[0,1] under calibration
        u = (ranks[:, d] + rng.random(ranks.shape[0])) / (M + 1)
        out.append(uniformity_test(u, confidence=confidence, n_sim=n_sim, seed=seed + 1 + d))
    return out


# ---------------------------------------------------------------------------
# TARP
# ---------------------------------------------------------------------------


def tarp_credibilities(
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    *,
    references: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """TARP per-trial credibilities (Lemos et al. 2023).

    For each trial ``l`` and a random reference point ``r_l``, the credibility is
    the posterior mass closer to ``r_l`` than ``theta_true[l]`` is:
    ``mean_m [ ||θ_m − r|| < ||θ_true − r|| ]``. Under a calibrated posterior
    these are Uniform[0,1] (feed to :func:`uniformity_test`). ``references``
    ``(L, dim)`` may be supplied; else they are drawn uniformly over the pooled
    posterior-sample bounding box (seeded)."""
    theta_true = np.asarray(theta_true, dtype=float)
    posterior_samples = np.asarray(posterior_samples, dtype=float)
    if theta_true.ndim == 1:
        theta_true = theta_true[:, None]
    if posterior_samples.ndim == 2:
        posterior_samples = posterior_samples[:, :, None]
    L, _, dim = posterior_samples.shape
    if references is None:
        lo = posterior_samples.reshape(-1, dim).min(axis=0)
        hi = posterior_samples.reshape(-1, dim).max(axis=0)
        rng = np.random.default_rng(seed)
        references = lo + (hi - lo) * rng.random((L, dim))
    references = np.asarray(references, dtype=float)
    if references.shape != (L, dim):
        raise ValueError(f"references must be {(L, dim)}, got {references.shape}")

    d_true = np.linalg.norm(theta_true - references, axis=1)  # (L,)
    d_samp = np.linalg.norm(posterior_samples - references[:, None, :], axis=2)  # (L, M)
    return (d_samp < d_true[:, None]).mean(axis=1)  # (L,)


@dataclass(frozen=True)
class TARPResult:
    """TARP expected-coverage result."""

    passed: bool
    max_calibration_error: float  # sup |ECP(c) − c|
    uniformity: UniformityResult
    credibility_grid: np.ndarray
    expected_coverage: np.ndarray  # ECP(c) = ECDF of credibilities


def tarp_test(
    theta_true: np.ndarray,
    posterior_samples: np.ndarray,
    *,
    references: np.ndarray | None = None,
    confidence: float = 0.95,
    n_grid: int = 21,
    n_sim: int = 2000,
    seed: int = 0,
) -> TARPResult:
    """TARP coverage gate (Lemos et al. 2023). Expected Coverage Probability
    ``ECP(c)`` = ECDF of the per-trial credibilities; a calibrated posterior has
    ``ECP(c) = c`` (the diagonal). ``passed`` iff the credibilities pass the
    simultaneous uniformity band; ``max_calibration_error`` = sup|ECP(c)−c|.

    The credibility of a trial is ``K/M`` with ``K``~Binomial(M, c), so under
    calibration it is Discrete-Uniform on ``{0, 1/M, …, 1}`` — NOT continuous
    Uniform[0,1]. The discrete grid carries a built-in sup-deviation ~1/(M+1)
    from the continuous diagonal, which for small M (the coarse GP-tier §8.7
    candidate-set feed) would make the continuous-null uniformity test reject a
    genuinely-calibrated posterior almost always. We therefore apply the SAME
    continuity-correction jitter ``(K + U(0,1))/(M+1)`` that :func:`sbc_test`
    uses before the band test (Talts et al. 2018). The reported ECP curve stays
    on the raw credibilities (it is a diagnostic, not the pass/fail gate)."""
    cred = tarp_credibilities(theta_true, posterior_samples, references=references, seed=seed)
    m_samples = int(np.asarray(posterior_samples).shape[1])
    k = np.rint(cred * m_samples).astype(int)  # recover the integer count K
    jitter = np.random.default_rng(seed + 104729).random(cred.shape[0])
    u = (k + jitter) / (m_samples + 1)
    uni = uniformity_test(u, confidence=confidence, n_sim=n_sim, seed=seed)
    grid = np.linspace(0.0, 1.0, n_grid)
    ecp = np.array([np.mean(cred <= c) for c in grid])
    max_err = float(np.max(np.abs(ecp - grid)))
    return TARPResult(
        passed=uni.passed,
        max_calibration_error=max_err,
        uniformity=uni,
        credibility_grid=grid,
        expected_coverage=ecp,
    )
