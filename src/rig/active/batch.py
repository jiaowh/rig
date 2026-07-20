"""Batch active learning with diversity (implementation-plan §9.5).

Real runs come in lots of ``q≈4-8``. Naive top-q picks near-duplicates, because
correlated points carry redundant information. §9.5 uses **BatchBALD** (Kirsch
et al. 2019, greedy submodular) + a diversity kernel. At the numpy/GP tier we
realize the submodular greedy directly: pick the top-acquisition candidate, then
repeatedly add the candidate that maximizes ``acq − w_div · (max predictive
correlation to the already-picked set)`` — so a candidate redundant with one
already chosen is penalized (diminishing returns). The predictive correlation
comes from the GP joint posterior (``posterior_cov``), the principled redundancy
signal; input distance is the fallback when no joint model is available.

**Split-plot (§9.5, §11.3):** a batch must share whole-plot (hard-to-change:
tool/chamber) factors. This module selects WITHIN one whole-plot group — the
caller (the loop) passes candidates that already share those factors, or groups
first and batches per group. The k-DPP / BADGE-gradient-embedding refinement and
TuRBO batch Thompson sampling are later work; this is the honest greedy core.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class _JointModel(Protocol):
    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...


def _predictive_correlation(model: _JointModel, X: np.ndarray) -> np.ndarray:
    """(n, n) mean-over-output predictive correlation from the GP joint posterior.
    Redundant (highly-correlated) candidates → near 1; independent → near 0."""
    cov = model.posterior_cov(X, X)  # (m, n, n)
    m, n, _ = cov.shape
    corr = np.zeros((n, n))
    for j in range(m):
        c = cov[j]
        d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
        corr += c / np.outer(d, d)
    corr /= m
    return np.clip(corr, -1.0, 1.0)


def _input_correlation(X: np.ndarray, bandwidth: float | None) -> np.ndarray:
    """RBF similarity on standardized inputs — the no-joint-model fallback."""
    X = np.asarray(X, dtype=float)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    sq = np.sum(Z**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0.0)
    if bandwidth is None:
        iu = np.triu_indices(Z.shape[0], k=1)
        med = np.median(np.sqrt(d2[iu])) if iu[0].size else 1.0
        bandwidth = float(med) if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * bandwidth**2))


def select_batch(
    acq_scores: np.ndarray,
    candidate_X: np.ndarray,
    q: int,
    *,
    model: _JointModel | None = None,
    w_div: float = 0.5,
    bandwidth: float | None = None,
) -> list[int]:
    """Greedy submodular batch selection (§9.5). Returns ``q`` indices into the
    candidate pool: the top-acquisition candidate, then repeatedly the candidate
    maximizing ``acq − w_div · max_{picked} correlation``.

    ``model`` (a GP exposing ``posterior_cov``) supplies the predictive
    correlation; without it, input-space RBF similarity is the fallback. Assumes
    candidates already share whole-plot factors (split-plot; the caller groups).
    """
    acq = np.asarray(acq_scores, dtype=float)
    X = np.atleast_2d(np.asarray(candidate_X, dtype=float))
    n = X.shape[0]
    if acq.shape[0] != n:
        raise ValueError("acq_scores length must match candidate_X rows")
    q = min(int(q), n)
    if q <= 0:
        return []
    sim = (
        np.abs(_predictive_correlation(model, X))
        if model is not None
        else _input_correlation(X, bandwidth)
    )
    picked: list[int] = [int(np.argmax(acq))]
    while len(picked) < q:
        redundancy = sim[:, picked].max(axis=1)  # (n,) max sim to any picked
        adjusted = acq - w_div * redundancy
        adjusted[picked] = -np.inf  # never repick
        nxt = int(np.argmax(adjusted))
        if not np.isfinite(adjusted[nxt]):
            break
        picked.append(nxt)
    return picked
