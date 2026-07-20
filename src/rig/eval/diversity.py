"""Diversity of valid solutions (implementation-plan §12.2 — the Vendi score + cross-checks).

MFL returns one recipe; a core claim here is a DIVERSE set of *valid* recipes
(non-injectivity, §8.7). §12.2 measures it with the **Vendi score** (Friedman &
Dieng 2023) — the effective number of distinct samples — plus a mode-count and
mean pairwise-L2 cross-check. Two subtleties the plan insists on:

- **Condition on in-tolerance.** Diversity must be measured over VALID solutions
  only, else a method games it with diverse garbage. The caller filters the
  recipe rows to the in-tolerance subset BEFORE passing them here.
- **Score what deploys, not just the proposal.** The multi-start-pessimism +
  farthest-point selection that produces the SHIPPED recipes can mode-collapse,
  so run this on the deployed set too, not only an amortized proposal cloud.

numpy only. Recipe vectors are standardized per-column before the kernel so the
score is scale-invariant across heterogeneous variables.
"""

from __future__ import annotations

import numpy as np


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be (n, d)")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)  # constant column ⇒ contributes no distance
    return (X - mu) / sd


def _rbf_kernel(X: np.ndarray, bandwidth: float | None) -> np.ndarray:
    """Unit-diagonal RBF similarity on standardized rows. ``bandwidth`` None ⇒
    the median pairwise distance heuristic (robust default)."""
    Z = _standardize(X)
    # pairwise squared distances
    sq = np.sum(Z**2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    if bandwidth is None:
        iu = np.triu_indices(Z.shape[0], k=1)
        med = np.median(np.sqrt(d2[iu])) if iu[0].size else 1.0
        bandwidth = float(med) if med > 0 else 1.0
    return np.exp(-d2 / (2.0 * bandwidth**2))


def vendi_score(X: np.ndarray, bandwidth: float | None = None) -> float:
    """Vendi score (Friedman & Dieng 2023): the effective number of distinct
    recipes in ``X`` ``(n, d)``.

    = exp(−Σ_i λ_i log λ_i), where λ_i are the eigenvalues of ``K/n`` for the
    unit-diagonal similarity kernel ``K`` (so Σ λ_i = 1). Ranges in ``[1, n]``:
    n identical rows → 1; n mutually dissimilar rows → n. Invariant to sample
    duplication only in the limit; robust median-heuristic bandwidth by default.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0
    K = _rbf_kernel(X, bandwidth)
    w = np.linalg.eigvalsh(K / n)
    w = w[w > 1e-12]  # drop numerical-zero eigenvalues (0 log 0 = 0)
    entropy = -float(np.sum(w * np.log(w)))
    return float(np.exp(entropy))


def mode_count(X: np.ndarray, radius: float = 0.5) -> int:
    """Number of distinct modes: greedy clustering in standardized space where a
    new row starts a mode iff it is farther than ``radius`` (in standardized-L2)
    from every existing mode centre. A coarse, robust cross-check on the Vendi
    score (both should move together)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 0:
        return 0
    Z = _standardize(X)
    centres: list[np.ndarray] = []
    for z in Z:
        if all(np.linalg.norm(z - c) > radius for c in centres):
            centres.append(z)
    return len(centres)


def mean_pairwise_l2(X: np.ndarray, standardize: bool = True) -> float:
    """Mean pairwise L2 distance (standardized by default) — the third §12.2
    diversity cross-check. nan for fewer than 2 rows."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] < 2:
        return float("nan")
    Z = _standardize(X) if standardize else X
    iu = np.triu_indices(Z.shape[0], k=1)
    d = np.linalg.norm(Z[iu[0]] - Z[iu[1]], axis=1)
    return float(np.mean(d))
