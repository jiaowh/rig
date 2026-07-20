"""Multi-tool ICM GP forward surrogate — implementation-plan §10.4 level (a), GP-era primary.

Intrinsic-coregionalization multi-task GP (Bonilla et al. 2008): one ICM GP
per output with kernel

    k((x, s), (x', t)) = k_Matern52_ARD(x, x') * B[s, t],
    B = W W^T + diag(v),   W in R^{T x rank},  v > 0 (log-parameterized),

so B is PSD by construction and carries the per-tool signal variances
(the Matérn factor is unit-variance; sf2 would be redundant with B's scale).
Hyperparameters (log ell, W, log v, log sn2) by exact NLML maximization with
ANALYTIC gradients, using the same L-BFGS-B multi-start machinery as
GPForwardModel (shared via ``rig.forward._gp_common``). The noise variance
sn2 is one constant per output shared across tools — the same §10.3
identifiability-honest aleatoric floor v0 as WP-C.

Partial pooling in one sentence: tools share the input kernel and borrow
strength through B's off-diagonal, so a new tool with a handful of runs
inherits the fleet's response surface and only has to learn its own row of B
— the §10.4 "chamber matching" few-shot path, validated in-silico (WP-B
pathology machine); real-data claims stay gated on M0.

Unknown-tool fallback (NEVER silently treated as a known tool). For a
``tool_id`` with zero training runs (or ``tool_id=None``), per output:

    w_t        ∝ Σ_s max(B[t, s], 0)          (B-weighted; uniform if all 0)
    μ_u(x)     = Σ_t w_t μ_t(x)                (population average mean)
    σ²_epi,u(x) = max_t σ²_epi,t(x)            (at least the worst known tool)
                 + Σ_t w_t (μ_t(x) − μ_u(x))²  (between-tool disagreement)
                 + (1 − ρ̄²) · mean_t B[t, t]   (irreducible new-tool variance)

with ρ̄ = mean pairwise tool correlation from B, clipped to [0, 1] (ρ̄ = 0
when only one tool is fitted — a single tool tells us nothing about
tool-to-tool variation, so the full mean prior variance is added). This is
deliberately conservative: unknown-tool epistemic is strictly above every
known tool's epistemic at the same x, which is what makes the §5.8
leave-one-tool-out check pass by construction rather than by luck.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from rig.forward._gp_common import (
    LOG_ELL_BOUNDS,
    LOG_SF2_BOUNDS,
    LOG_SN2_BOUNDS,
    SQRT5,
    cholesky_with_jitter,
    matern52,
    matern52_grad_x,
    multistart_minimize,
    regularized_cov_inv,
    standardize_stats,
)
from rig.forward.data import records_to_arrays_with_tools
from rig.interfaces import PredictiveDistribution

logger = logging.getLogger(__name__)

# W entries live in standardized-output units (B ~ O(1) after y-standardization).
_W_BOUNDS = (-30.0, 30.0)
_LOG_V_BOUNDS = LOG_SF2_BOUNDS  # v is a per-tool signal-variance component


@dataclass
class _ICMHyper:
    ell: np.ndarray  # (d,) ARD lengthscales, standardized-input units
    W: np.ndarray  # (T, rank) coregionalization factors
    v: np.ndarray  # (T,) per-tool diagonal variance (>0)
    sn2: float  # noise variance (aleatoric floor v0), standardized units

    @property
    def B(self) -> np.ndarray:
        return self.W @ self.W.T + np.diag(self.v)


class _ICMSingleOutputGP:
    """Exact ICM GP for one standardized output. Internal to the multi-tool model.

    theta layout: [log ell (d), W.ravel() (T*rank), log v (T), log sn2].
    """

    def __init__(
        self, X: np.ndarray, y: np.ndarray, tool_idx: np.ndarray, n_tools: int, rank: int
    ) -> None:
        self.X = X  # (n, d) standardized
        self.y = y  # (n,) standardized
        self.ix = np.asarray(tool_idx, dtype=int)  # (n,) tool index per run
        self.T = int(n_tools)
        self.rank = int(rank)
        self.hyper: _ICMHyper | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None
        # tool one-hot (n, T) for the per-tool block sums in the v-gradient
        self._P = np.zeros((self.X.shape[0], self.T))
        self._P[np.arange(self.X.shape[0]), self.ix] = 1.0

    # -- theta packing ---------------------------------------------------------

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        d = self.X.shape[1]
        tr = self.T * self.rank
        ell = np.exp(theta[:d])
        W = theta[d : d + tr].reshape(self.T, self.rank)
        v = np.exp(theta[d + tr : d + tr + self.T])
        sn2 = float(np.exp(theta[-1]))
        return ell, W, v, sn2

    # -- marginal likelihood -----------------------------------------------------

    def _nlml_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        n, d = self.X.shape
        ell, W, v, sn2 = self._unpack(theta)
        B = W @ W.T + np.diag(v)
        Bg = B[self.ix][:, self.ix]  # (n, n) coregionalization gram
        try:
            Kx, r = matern52(self.X, self.X, ell, 1.0)  # unit-variance input kernel
            K = Kx * Bg + sn2 * np.eye(n)
            L, _ = cholesky_with_jitter(K)
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
        # lengthscales: dK/dlog ell_k = (dKx/dlog ell_k) ∘ Bg
        AB = A * Bg
        base = (5.0 / 3.0) * (1.0 + SQRT5 * r) * np.exp(-SQRT5 * r)
        for k in range(d):
            Dk = (self.X[:, None, k] - self.X[None, :, k]) ** 2
            grad[k] = 0.5 * float(np.sum(AB * (base * Dk / ell[k] ** 2)))
        # W: dK_ij/dW_{t,r} = Kx_ij (δ_{s_i,t} W[s_j,r] + δ_{s_j,t} W[s_i,r])
        #    ⇒ grad_W[t,r] = Σ_{i: s_i=t} (M @ W[s])_{i,r},  M = A ∘ Kx (symmetric)
        M = A * Kx
        G = M @ W[self.ix]  # (n, rank)
        grad_W = np.zeros((self.T, self.rank))
        np.add.at(grad_W, self.ix, G)
        tr = self.T * self.rank
        grad[d : d + tr] = grad_W.ravel()
        # v: dK_ij/dlog v_t = Kx_ij δ_{s_i,t} δ_{s_j,t} v_t
        block = np.sum(self._P * (M @ self._P), axis=0)  # (T,) per-tool block sums of M
        grad[d + tr : d + tr + self.T] = 0.5 * v * block
        grad[-1] = 0.5 * sn2 * float(np.trace(A))  # dK/dlog sn2 = sn2 I
        return nlml, grad

    def fit(self, n_restarts: int, seed: int, max_iter: int) -> None:
        n, d = self.X.shape
        rng = np.random.default_rng(seed)
        tr = self.T * self.rank
        bounds = (
            [LOG_ELL_BOUNDS] * d + [_W_BOUNDS] * tr + [_LOG_V_BOUNDS] * self.T + [LOG_SN2_BOUNDS]
        )
        # Deterministic first start: unit lengthscales; W encodes STRONG prior
        # inter-tool correlation (B off-diag ≈ 0.81, diag ≈ 0.91) — the chamber-
        # matching inductive bias: tools are similar until the data disagrees.
        W0 = np.full((self.T, self.rank), 0.9 / np.sqrt(self.rank))
        W0 += 0.05 * np.arange(self.rank)[None, :]  # break column symmetry at rank>1
        starts = [
            np.concatenate([np.zeros(d), W0.ravel(), np.full(self.T, np.log(0.1)), [np.log(0.05)]])
        ]
        for _ in range(max(0, n_restarts - 1)):
            starts.append(
                np.concatenate(
                    [
                        rng.uniform(np.log(0.1), np.log(10.0), size=d),
                        0.7 * rng.standard_normal(tr),
                        rng.uniform(np.log(1e-3), np.log(1.0), size=self.T),
                        [rng.uniform(np.log(1e-4), np.log(1.0))],
                    ]
                )
            )
        best_theta = multistart_minimize(self._nlml_and_grad, starts, bounds, max_iter)
        ell, W, v, sn2 = self._unpack(best_theta)
        self.hyper = _ICMHyper(ell=ell, W=W, v=v, sn2=sn2)
        Kx, _ = matern52(self.X, self.X, ell, 1.0)
        Bg = self.hyper.B[self.ix][:, self.ix]
        self._L, _ = cholesky_with_jitter(Kx * Bg + sn2 * np.eye(n))
        self._alpha = cho_solve((self._L, True), self.y)

    # -- posterior -----------------------------------------------------------------

    def predict_tool(self, Xs: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and latent variance at Xs (m, d) for KNOWN tool index t."""
        assert self.hyper is not None and self._L is not None and self._alpha is not None
        h = self.hyper
        B = h.B
        kx, _ = matern52(Xs, self.X, h.ell, 1.0)  # (m, n)
        Q = kx * B[t, self.ix][None, :]  # cross-covariance rows
        mu = Q @ self._alpha
        V = solve_triangular(self._L, Q.T, lower=True)  # (n, m)
        var = np.maximum(B[t, t] - np.sum(V * V, axis=0), 0.0)  # k_x(x,x) = 1
        return mu, var

    def mean_grad_tool(self, xs: np.ndarray, t: int) -> np.ndarray:
        """Analytic d(mean)/dx at a single standardized point for tool t. (d,)"""
        assert self.hyper is not None and self._alpha is not None
        h = self.hyper
        dk = matern52_grad_x(xs, self.X, h.ell, 1.0)  # (n, d)
        return (self._alpha * h.B[t, self.ix]) @ dk

    def cov_tool(self, Xs1: np.ndarray, Xs2: np.ndarray, t: int) -> np.ndarray:
        """Posterior LATENT covariance Cov(f_t(Xs1), f_t(Xs2)) for KNOWN tool
        index ``t``, standardized-output units, shape (n1, n2). The ICM prior
        cross-cov is ``k_x(x, x')·B[t,t]`` conditioned on the pooled data via the
        shared Cholesky — the multitask analogue of ``_SingleOutputGP.cov`` that
        EPIG/BatchBALD (§9.4) need to onboard a new chamber (§10.4). Its diagonal
        equals ``predict_tool``'s variance, so acquisition stays self-consistent
        with ``predict``."""
        assert self.hyper is not None and self._L is not None
        h = self.hyper
        B = h.B
        k12, _ = matern52(Xs1, Xs2, h.ell, 1.0)  # (n1, n2), unit-variance factor
        K12 = k12 * B[t, t]
        Q1 = matern52(Xs1, self.X, h.ell, 1.0)[0] * B[t, self.ix][None, :]  # (n1, n)
        Q2 = matern52(Xs2, self.X, h.ell, 1.0)[0] * B[t, self.ix][None, :]  # (n2, n)
        V1 = solve_triangular(self._L, Q1.T, lower=True)  # (n, n1)
        V2 = solve_triangular(self._L, Q2.T, lower=True)  # (n, n2)
        return K12 - V1.T @ V2


class MultiToolGPForwardModel:
    """Tool-aware ICM multi-task GP ForwardModel (implementation-plan §10.4 level (a)).

    Same public conventions as :class:`GPForwardModel` plus tool handling:

    - ``fit(X, Y, tools)`` — ``tools`` is a per-row sequence of tool-id
      strings; the model maintains a tool→index map (first-appearance order).
    - ``predict(x, tool_id=...)`` -> canonical ``PredictiveDistribution``
      obeying the WP-C shape contract: ``(d,)`` -> fields ``(m,)``,
      ``(n, d)`` -> ``(n, m)``; ``conformal_set=None`` when unwrapped. A
      KNOWN tool uses its own B row; an UNKNOWN tool (zero runs, or
      ``tool_id=None``) gets the B-weighted population average with inflated
      epistemic — see the module docstring for the exact formula.
    - ``support_score(x, tool_id=...)`` — per-tool negative Mahalanobis when
      the tool has >= d+2 runs, else the global one (same semantics as WP-C:
      higher = more in-distribution, max 0 at the mean).
    - ``update(records)`` — refits on old + new data; tool ids are read from
      ``RunRecord.tool_id`` and unseen ids are registered implicitly.
    - ``adapt_to_tool(tool_id, X_new, Y_new)`` — few-shot onboarding of a
      new tool: fold its runs in and refit (full refit is fine at this n).
    - ``for_tool(tool_id)`` — a ForwardModel-conformant view with the tool
      bound, so tool-blind consumers (e.g. ``ConformalForwardModel``) work
      unchanged: ``ConformalForwardModel(model.for_tool("B"), calibrator)``.
    """

    def __init__(
        self,
        input_keys: Sequence[str] | None = None,
        output_keys: Sequence[str] | None = None,
        rank: int = 1,
        n_restarts: int = 5,
        seed: int = 0,
        max_iter: int = 100,
    ) -> None:
        if rank < 1:
            raise ValueError(f"rank must be >= 1 (1-2 recommended at small T); got {rank}")
        self.input_keys = list(input_keys) if input_keys is not None else None
        self.output_keys = list(output_keys) if output_keys is not None else None
        self.rank = int(rank)
        self.n_restarts = n_restarts
        self.seed = seed
        self.max_iter = max_iter
        self._gps: list[_ICMSingleOutputGP] = []
        self._X_raw: np.ndarray | None = None
        self._Y_raw: np.ndarray | None = None
        self._tools_raw: list[str] = []
        self._tool_index: dict[str, int] = {}  # tools WITH training data (fitted)
        self._declared_tools: set[str] = set()  # add_tool()-registered, may lack data

    # -- tool bookkeeping ---------------------------------------------------------

    @property
    def tools_(self) -> list[str]:
        """Tool ids the fitted model has training data for (index order)."""
        return list(self._tool_index)

    @property
    def tool_counts_(self) -> dict[str, int]:
        """Training-run count per fitted tool."""
        counts: dict[str, int] = dict.fromkeys(self._tool_index, 0)
        for t in self._tools_raw:
            counts[t] = counts.get(t, 0) + 1
        return counts

    def add_tool(self, tool_id: str) -> None:
        """Declare a tool id ahead of data. A declared tool with zero training
        runs still takes the unknown-tool fallback path in ``predict`` — being
        in the map is not the same as being learned (never silently pretend)."""
        self._declared_tools.add(str(tool_id))

    # -- fitting --------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return bool(self._gps)

    @property
    def n_train_(self) -> int:
        return 0 if self._X_raw is None else int(self._X_raw.shape[0])

    def fit(self, X: np.ndarray, Y: np.ndarray, tools: Sequence[str]) -> MultiToolGPForwardModel:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d); got shape {X.shape}")
        if Y.ndim == 1:
            Y = Y[:, None]
        tools = [str(t) for t in tools]
        if not (X.shape[0] == Y.shape[0] == len(tools)):
            raise ValueError(
                f"X ({X.shape[0]} rows), Y ({Y.shape[0]}), and tools ({len(tools)}) disagree"
            )
        self._X_raw, self._Y_raw, self._tools_raw = X, Y, tools

        # tool -> index map, stable first-appearance order across refits
        self._tool_index = {}
        for t in tools:
            if t not in self._tool_index:
                self._tool_index[t] = len(self._tool_index)
        self._declared_tools |= set(self._tool_index)
        ix = np.array([self._tool_index[t] for t in tools], dtype=int)
        n_tools = len(self._tool_index)

        # standardize with TRAIN statistics only, POOLED across tools (§5.3) —
        # per-tool y-standardization would erase exactly the tool offsets B
        # must learn.
        self._x_mean, self._x_scale = standardize_stats(X)
        self._y_mean, self._y_scale = standardize_stats(Y)
        Xs = (X - self._x_mean) / self._x_scale
        Ys = (Y - self._y_mean) / self._y_scale

        # support-score stats: global always; per-tool when the tool has
        # >= d+2 runs (else its covariance is rank-deficient/meaningless)
        d = Xs.shape[1]
        self._support_global_inv = regularized_cov_inv(Xs)
        self._support_tool: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for tool, tix in self._tool_index.items():
            rows = Xs[ix == tix]
            if rows.shape[0] >= d + 2:
                self._support_tool[tool] = (rows.mean(axis=0), regularized_cov_inv(rows))

        self._gps = []
        for j in range(Ys.shape[1]):
            gp = _ICMSingleOutputGP(Xs, Ys[:, j], ix, n_tools, self.rank)
            gp.fit(self.n_restarts, self.seed + 1000 * j, self.max_iter)
            self._gps.append(gp)
        return self

    def update(self, records: Iterable[Any]) -> None:
        """Ingest RunRecords (tools from ``RunRecord.tool_id``) and refit on
        old + new data; unseen tool ids are registered implicitly."""
        if self.input_keys is None or self.output_keys is None:
            raise ValueError(
                "update(records) needs input_keys/output_keys at construction "
                "to map RunRecords to arrays; use fit(X, Y, tools) for raw matrices"
            )
        X_new, Y_new, tools_new = records_to_arrays_with_tools(
            records, self.input_keys, self.output_keys
        )
        for t in tools_new:
            if t not in self._tool_index:
                self.add_tool(t)
        if self._X_raw is not None:
            X_new = np.vstack([self._X_raw, X_new])
            Y_new = np.vstack([self._Y_raw, Y_new])
            tools_new = self._tools_raw + tools_new
        self.fit(X_new, Y_new, tools_new)

    def adapt_to_tool(
        self, tool_id: str, X_new: np.ndarray, Y_new: np.ndarray
    ) -> MultiToolGPForwardModel:
        """Few-shot onboarding of ``tool_id`` from its first handful of runs.

        Full refit including the new tool's runs (exact refit is cheap at this
        n; the method exists for API clarity — it is THE §10.4 new-chamber
        path). Logs how many runs the new tool has after the refit.
        """
        self._require_fitted()
        tool_id = str(tool_id)
        X_new = np.asarray(X_new, dtype=float)
        Y_new = np.asarray(Y_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new[None, :]
        if Y_new.ndim == 1:
            Y_new = Y_new[:, None] if Y_new.shape[0] == X_new.shape[0] else Y_new[None, :]
        X = np.vstack([self._X_raw, X_new])
        Y = np.vstack([self._Y_raw, Y_new])
        tools = self._tools_raw + [tool_id] * X_new.shape[0]
        self.fit(X, Y, tools)
        n_tool = self.tool_counts_.get(tool_id, 0)
        logger.info(
            "adapt_to_tool(%r): refit with %d run(s) for this tool (%d total runs, %d tools)",
            tool_id,
            n_tool,
            self.n_train_,
            len(self._tool_index),
        )
        return self

    # -- fitted-parameter views ---------------------------------------------------

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("MultiToolGPForwardModel is not fitted; call fit(X, Y, tools) first")

    @property
    def noise_std_(self) -> np.ndarray:
        """Fitted aleatoric noise std per output, raw output units."""
        self._require_fitted()
        return np.array(
            [np.sqrt(gp.hyper.sn2) * s for gp, s in zip(self._gps, self._y_scale, strict=True)]
        )

    @property
    def tool_covariance_(self) -> np.ndarray:
        """Learned B per output, shape (m, T, T), standardized-output units."""
        self._require_fitted()
        return np.stack([gp.hyper.B for gp in self._gps])

    @property
    def tool_correlation_(self) -> np.ndarray:
        """Tool-correlation matrices per output, shape (m, T, T) — scale-free
        view of B (the §10.4 'how similar are my chambers' readout)."""
        B = self.tool_covariance_
        s = np.sqrt(np.einsum("mtt->mt", B))
        return B / (s[:, :, None] * s[:, None, :])

    # -- unknown-tool fallback pieces -----------------------------------------------

    def _tool_weights(self, B: np.ndarray) -> np.ndarray:
        """B-weighted population weights: w_t ∝ Σ_s max(B[t,s], 0)."""
        w = np.clip(B, 0.0, None).sum(axis=1)
        total = float(w.sum())
        if total <= 0.0:
            return np.full(B.shape[0], 1.0 / B.shape[0])
        return w / total

    def _mean_pairwise_corr(self, B: np.ndarray) -> float:
        """Mean pairwise tool correlation, clipped to [0, 1]; 0 for T = 1."""
        T = B.shape[0]
        if T < 2:
            return 0.0
        s = np.sqrt(np.diag(B))
        C = B / np.outer(s, s)
        iu = np.triu_indices(T, k=1)
        return float(np.clip(np.mean(C[iu]), 0.0, 1.0))

    # -- ForwardModel protocol ----------------------------------------------------

    def predict(self, x: np.ndarray, tool_id: str | None = None) -> PredictiveDistribution:
        """Canonical PredictiveDistribution at x, conditioned on a tool.

        KNOWN ``tool_id`` (has training runs): that tool's B row. UNKNOWN
        ``tool_id`` (zero runs, incl. declared-but-dataless tools) or
        ``tool_id=None``: the B-weighted population average with inflated
        epistemic (module docstring has the formula) — never a silent
        known-tool impersonation.
        """
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        Xs = (Xq - self._x_mean) / self._x_scale
        known = tool_id is not None and tool_id in self._tool_index

        means, epi = [], []
        if known:
            t = self._tool_index[tool_id]
            for gp, ym, ys in zip(self._gps, self._y_mean, self._y_scale, strict=True):
                mu, var = gp.predict_tool(Xs, t)
                means.append(ym + ys * mu)
                epi.append(ys * np.sqrt(var))
        else:
            for gp, ym, ys in zip(self._gps, self._y_mean, self._y_scale, strict=True):
                B = gp.hyper.B
                w = self._tool_weights(B)
                per_tool = [gp.predict_tool(Xs, t) for t in range(gp.T)]
                mus = np.stack([mu for mu, _ in per_tool])  # (T, nq)
                vars_ = np.stack([var for _, var in per_tool])
                mu_u = np.einsum("t,tq->q", w, mus)
                spread = np.einsum("t,tq->q", w, (mus - mu_u[None, :]) ** 2)
                rho = self._mean_pairwise_corr(B)
                var_u = vars_.max(axis=0) + spread + (1.0 - rho**2) * float(np.mean(np.diag(B)))
                means.append(ym + ys * mu_u)
                epi.append(ys * np.sqrt(var_u))
        mean = np.stack(means, axis=-1)  # (nq, m)
        epistemic = np.stack(epi, axis=-1)
        aleatoric = np.broadcast_to(self.noise_std_, mean.shape).copy()
        if single:
            mean, aleatoric, epistemic = mean[0], aleatoric[0], epistemic[0]
        return PredictiveDistribution(
            mean=mean,
            aleatoric_sigma=aleatoric,
            epistemic_sigma=epistemic,
            conformal_set=None,  # filled by the §5.6 calibration wrapper
        )

    def support_score(self, x: np.ndarray, tool_id: str | None = None) -> float | np.ndarray:
        """Negative Mahalanobis distance (higher = more in-distribution, max 0).

        Per-tool statistics when ``tool_id`` has >= d+2 training runs, else
        the global training cloud — same semantics as GPForwardModel."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        if tool_id is not None and tool_id in self._support_tool:
            center, cov_inv = self._support_tool[tool_id]
        else:
            center, cov_inv = np.zeros(Xs.shape[1]), self._support_global_inv
        Z = Xs - center
        d2 = np.einsum("ij,jk,ik->i", Z, cov_inv, Z)
        score = -np.sqrt(np.maximum(d2, 0.0))
        return float(score[0]) if single else score

    def jacobian(self, x: np.ndarray, tool_id: str | None = None) -> np.ndarray:
        """Analytic d(mean)/dx at a single (d,) point, raw units, (m, d).

        Known tool: that tool's posterior mean. Unknown/None: gradient of the
        population-average mean (the fallback ``predict`` mean)."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("jacobian(x) takes a single point of shape (d,)")
        xs = (x - self._x_mean) / self._x_scale
        rows = []
        known = tool_id is not None and tool_id in self._tool_index
        for gp, ys in zip(self._gps, self._y_scale, strict=True):
            if known:
                g = gp.mean_grad_tool(xs, self._tool_index[tool_id])
            else:
                w = self._tool_weights(gp.hyper.B)
                g = sum(w[t] * gp.mean_grad_tool(xs, t) for t in range(gp.T))
            rows.append(ys * g / self._x_scale)
        return np.stack(rows)

    def _unknown_tool_cov(
        self, gp: _ICMSingleOutputGP, Xs1: np.ndarray, Xs2: np.ndarray
    ) -> np.ndarray:
        """Unknown-tool joint latent cov (standardized units), (n1, n2).

        THE INVARIANT (audit 2026-07-17): its diagonal must equal ``predict``'s
        unknown-tool ``var_u`` EXACTLY, or EPIG is silently wrong — ``epig()``
        takes ``var_f_star`` from ``predict`` but ``Cov(f(x*), f(x))`` from here,
        so two different laws break Cauchy-Schwarz and collapse the log-ratio to
        ~0 nats (measured: EPIG 1.01 vs BALD 19.06, and exactly 0.0 across a
        4-candidate batch). That silently disables the prediction-targeted term
        precisely on the §10.4 chamber-onboarding path this method advertises,
        and precisely as λ anneals 0.2→0.9 to let EPIG dominate.

        Construction — the law of total covariance for the tool mixture:

            base(X1,X2) = Σ_t w_t·Cov_t(X1,X2)                    # within-tool
                        + Σ_t w_t·(μ_t(X1)−μ_u(X1))(μ_t(X2)−μ_u(X2))  # between-tool
                        + c,   c = (1−ρ̄²)·mean_t B[t,t]           # unknown-tool offset

        each term a valid PSD kernel (w ≥ 0; the offset is the constant kernel,
        c ≥ 0 since ρ̄ is clipped to [0,1]) — so ``base`` is PSD, and its diagonal
        is ``Σ_t w_t·var_t + spread + c``.

        ``predict`` (binding, BUILD_STATE WP-I) credits ``max_t var_t`` rather
        than ``Σ_t w_t·var_t``, so that unknown-tool epistemic DOMINATES every
        known tool elementwise (the §5.8 LOTO check holds by construction). We
        keep that and reconcile by a **congruence rescale**
        ``diag(s(X1))·base·diag(s(X2))``, ``s = sqrt(var_u / diag(base)) ≥ 1``:
        a congruence of a PSD matrix stays PSD, it preserves the mixture's
        CORRELATION structure, and it makes the diagonal equal ``var_u`` exactly.
        """
        B = gp.hyper.B
        w = self._tool_weights(B)
        c = (1.0 - self._mean_pairwise_corr(B) ** 2) * float(np.mean(np.diag(B)))

        def _parts(Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            per_tool = [gp.predict_tool(Xs, t) for t in range(gp.T)]
            mus = np.stack([mu for mu, _ in per_tool])  # (T, n)
            vars_ = np.stack([var for _, var in per_tool])
            mu_u = np.einsum("t,tq->q", w, mus)
            dev = mus - mu_u[None, :]  # (T, n)
            spread = np.einsum("t,tq->q", w, dev**2)
            # target == predict's var_u; have == diag(base) — both include spread + c
            target = vars_.max(axis=0) + spread + c
            have = np.einsum("t,tq->q", w, vars_) + spread + c
            return dev, np.sqrt(target / np.maximum(have, 1e-300))

        dev1, s1 = _parts(Xs1)
        dev2, s2 = _parts(Xs2)
        within = sum(w[t] * gp.cov_tool(Xs1, Xs2, t) for t in range(gp.T))
        between = np.einsum("t,ti,tj->ij", w, dev1, dev2)
        base = within + between + c
        return base * s1[:, None] * s2[None, :]

    def posterior_cov(
        self, X1: np.ndarray, X2: np.ndarray, tool_id: str | None = None
    ) -> np.ndarray:
        """Per-output posterior LATENT covariance Cov(f(X1), f(X2)) in raw output
        units, shape ``(m, n1, n2)``, conditioned on a tool — the EPIG/BatchBALD
        joint-covariance input (§9.4) for §10.4 chamber onboarding. KNOWN
        ``tool_id``: that tool's exact ICM joint cov. UNKNOWN/None: the tool
        mixture's total covariance, inflated to the fallback law (never a silent
        known-tool impersonation).

        In BOTH branches the diagonal equals ``predict``'s ``epistemic_sigma**2``
        for the same ``tool_id`` — the consistency EPIG depends on; see
        ``_unknown_tool_cov`` for why the unknown branch needs the rescale.
        """
        self._require_fitted()
        Xs1 = (np.atleast_2d(np.asarray(X1, dtype=float)) - self._x_mean) / self._x_scale
        Xs2 = (np.atleast_2d(np.asarray(X2, dtype=float)) - self._x_mean) / self._x_scale
        known = tool_id is not None and tool_id in self._tool_index
        out = []
        for gp, ys in zip(self._gps, self._y_scale, strict=True):
            if known:
                c = gp.cov_tool(Xs1, Xs2, self._tool_index[tool_id])
            else:
                c = self._unknown_tool_cov(gp, Xs1, Xs2)
            out.append(ys * ys * c)
        return np.stack(out)

    # -- tool-bound view ------------------------------------------------------------

    def for_tool(self, tool_id: str) -> ToolBoundForwardModel:
        """A ForwardModel-protocol view with ``tool_id`` bound, for tool-blind
        consumers (ConformalForwardModel, the WP-D inverse, ...)."""
        return ToolBoundForwardModel(self, tool_id)


class ToolBoundForwardModel:
    """Thin ForwardModel-conformant view: a MultiToolGPForwardModel with the
    tool_id bound. This is how the tool-aware model slots under tool-blind
    wrappers (e.g. ConformalForwardModel) without modifying them."""

    def __init__(self, base: MultiToolGPForwardModel, tool_id: str) -> None:
        self.base = base
        self.tool_id = str(tool_id)

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        return self.base.predict(x, tool_id=self.tool_id)

    def support_score(self, x: np.ndarray) -> float | np.ndarray:
        return self.base.support_score(x, tool_id=self.tool_id)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.base.jacobian(x, tool_id=self.tool_id)

    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.base.posterior_cov(X1, X2, tool_id=self.tool_id)

    def update(self, records: Iterable[Any]) -> None:
        self.base.update(records)  # records carry their own tool ids
