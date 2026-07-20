"""Deep-ensemble forward surrogate — the D3 large-data backbone (implementation-plan §5.4).

Backend B (>~300 runs, or 1-D/2-D fields where GPs scale poorly): a deep
ensemble (K=5 dev, K=10 final) of heteroscedastic **β-NLL** ResMLP surrogates,
each with a **spectral-normalized trunk + SNGP (RFF-GP Laplace) last layer** for
the mean (Liu et al. 2020). It exposes the SAME canonical
``PredictiveDistribution(mean, aleatoric_sigma, epistemic_sigma, conformal_set)``
provider as :class:`rig.forward.gp.GPForwardModel` (§3.2), so every downstream
consumer — the §5.6 conformal wrapper, the §8 inverse, the §9 AL loop — is
backend-agnostic and needs no change.

Design decisions (all binding, from the plan):

- **β-NLL, β=0.5** (§5.4): the aleatoric loss reweights the Gaussian NLL gradient
  by ``stopgrad(σ^{2β})`` (Seitzer et al. 2022) — recovers most of MSE's mean-fit
  quality while keeping a calibrated *input-dependent* variance, avoiding the
  variance-collapse of plain Gaussian NLL. Each net has a heteroscedastic head.
- **No bagging** (D3): ensemble diversity is independent inits + per-member input
  jitter (input-domain randomization) only. Bootstrap usually hurts on small data.
- **Predictive mixture** (§5.4): ``p(y|x)=(1/K)Σ N(μ_m,σ_m²)``; total variance
  splits into aleatoric ``E[σ_m²]`` and epistemic ``Var[μ_m]``. To that epistemic
  we ADD the SNGP Laplace variance (see below) — the distance-aware term.
- **OOD distance-awareness** (§5.4 honest qualification, §5.9 invariant 1): plain
  ensembles can agree confidently far OOD. The inflation comes from the
  spectral-normalized (≈bi-Lipschitz) trunk + the RFF-GP last layer whose Laplace
  posterior variance grows toward the prior as the trunk feature ``φ(x)`` moves
  away from the training features — NOT from ensembling per se. We combine both.
- **support_score** = negative Mahalanobis in the spectral-normalized latent
  ``φ`` (§8.2 / §11), the plan's own OOD gate space — an upgrade over the GP's
  input-space Mahalanobis.
- **Training** (§5.7): AdamW, lr 1e-3 cosine-decay, weight decay 1e-4, batch 128,
  early stop on validation β-NLL (patience 30). Standardization uses TRAIN
  statistics only (§5.3 leakage guard); ``predict`` speaks raw (SI-magnitude)
  units.

The torch nets run in float32 (GPU-friendly; the RTX 5050 sm_120 path is
exercised by ``device="cuda"``); the RFF-GP Laplace linear algebra is done in
float64 numpy/scipy for numerical safety, reusing the codebase's Cholesky
patterns. Determinism (§13.4): per-member ``torch.manual_seed`` + a seeded numpy
RNG for the fixed random features and input jitter; ``device="cpu"`` (default)
is bit-reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from rig.forward._gp_common import regularized_cov_inv as _regularized_cov_inv
from rig.forward._gp_common import standardize_stats as _standardize_stats
from rig.forward.data import records_to_arrays
from rig.interfaces import PredictiveDistribution

# β-NLL exponent (§5.4). β=0.5 is the plan default (Seitzer et al. 2022).
_BETA_NLL = 0.5
# aleatoric variance floor, standardized-output units (identifiability floor v0,
# §10.3; also keeps the heteroscedastic Laplace weights 1/σ² finite).
_SIGMA2_FLOOR = 1e-4


# ---------------------------------------------------------------------------
# torch trunk + heads (one per ensemble member)
# ---------------------------------------------------------------------------


class _SpectralResBlock(nn.Module):
    """Residual block with spectral-normalized linears (the bi-Lipschitz trunk).

    ``x + W2·act(W1·x)`` with each linear capped at spectral norm 1 (torch power
    iteration). Residual + capped-Lipschitz branches keep the map distance-aware
    (Liu et al. 2020): trunk features cannot collapse far-apart inputs together,
    which is what makes the downstream RFF-GP variance a valid OOD signal.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.lin1 = spectral_norm(nn.Linear(width, width))
        self.lin2 = spectral_norm(nn.Linear(width, width))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.lin2(self.act(self.lin1(x)))


class _Member(nn.Module):
    """One ensemble member: spectral trunk φ(x) + heteroscedastic β-NLL aleatoric
    head + a learnable RFF-GP mean layer (β on fixed random features).

    The mean is ``Φ(φ(x)) · βᵀ`` where ``Φ`` are fixed random Fourier features of
    the trunk output; ``β`` is the only trained part of the GP layer, weight-
    decayed (= a Gaussian prior). The Laplace posterior covariance over ``β`` is
    computed post-hoc in :class:`DeepEnsembleForwardModel` (numpy) and gives the
    distance-aware epistemic term.
    """

    def __init__(
        self,
        d_in: int,
        m_out: int,
        width: int,
        n_blocks: int,
        d_rff: int,
        rff_scale: float,
        seed: int,
    ) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.input_proj = spectral_norm(nn.Linear(d_in, width))
        self.blocks = nn.ModuleList(_SpectralResBlock(width) for _ in range(n_blocks))
        self.act = nn.GELU()
        # heteroscedastic aleatoric head: φ -> log σ² per output
        self.log_var_head = nn.Sequential(
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, m_out)
        )
        # fixed random Fourier features Φ(φ) = sqrt(2/D) cos(φ·W_rff + b_rff),
        # W_rff ~ N(0, 1/rff_scale²) so rff_scale is the feature lengthscale
        # (set by the median-distance heuristic on the trunk features).
        w = torch.randn(width, d_rff, generator=gen) / float(rff_scale)
        b = torch.rand(d_rff, generator=gen) * (2.0 * torch.pi)
        self.register_buffer("rff_w", w)
        self.register_buffer("rff_b", b)
        self.d_rff = d_rff
        # trained GP mean weights β (no bias; prior = weight decay -> Laplace ridge)
        self.beta = nn.Linear(d_rff, m_out, bias=False)
        # seed the trainable params for member diversity
        self._seed_params(seed)

    def _seed_params(self, seed: int) -> None:
        gen = torch.Generator().manual_seed(seed + 7919)
        for p in self.parameters():
            if p.requires_grad and p.dim() >= 2:
                nn.init.xavier_uniform_(p, generator=gen)
            elif p.requires_grad:
                nn.init.zeros_(p)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Spectral-normalized trunk features φ(x), shape (B, width)."""
        h = self.act(self.input_proj(x))
        for blk in self.blocks:
            h = blk(h)
        return h

    def rff(self, phi: torch.Tensor) -> torch.Tensor:
        """Random Fourier features Φ(φ), shape (B, d_rff)."""
        proj = phi @ self.rff_w + self.rff_b
        return torch.sqrt(torch.tensor(2.0 / self.d_rff)) * torch.cos(proj)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, sigma2, rff) in standardized-output units."""
        phi = self.features(x)
        rff = self.rff(phi)
        mean = self.beta(rff)
        log_var = self.log_var_head(phi)
        sigma2 = torch.nn.functional.softplus(log_var) + _SIGMA2_FLOOR
        return mean, sigma2, rff


def _beta_nll_loss(mean: torch.Tensor, sigma2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """β-NLL (β=0.5), summed over outputs, averaged over the batch (§5.4).

    Gaussian NLL reweighted per-element by ``stopgrad(σ^{2β})`` (Seitzer 2022):
    dampens the variance term's ability to starve the mean of gradient.
    """
    nll = 0.5 * ((y - mean) ** 2 / sigma2 + torch.log(sigma2))
    weight = sigma2.detach() ** _BETA_NLL
    return (weight * nll).sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# the ForwardModel
# ---------------------------------------------------------------------------


class DeepEnsembleForwardModel:
    """Deep-ensemble ForwardModel (implementation-plan §3.2 protocol; §5.4 large-data D3).

    Same canonical surface as :class:`rig.forward.gp.GPForwardModel`:

    - ``predict(x) -> PredictiveDistribution`` (``conformal_set=None``; the §5.6
      wrapper fills it). ``x`` is ``(d,)`` -> fields ``(m,)`` or ``(n,d)`` ->
      ``(n,m)``. ``aleatoric_sigma = sqrt(E_m[σ_m²])``; ``epistemic_sigma =
      sqrt(Var_m[μ_m] + E_m[SNGP-Laplace var])`` — the ensemble spread plus the
      distance-aware RFF-GP term (§5.4).
    - ``support_score(x)`` = negative Mahalanobis in the spectral-normalized
      latent φ of the reference member (§8.2, §11). Higher = more in-distribution.
    - ``jacobian(x)`` = autograd d(mixture-mean)/dx at a single point, ``(m,d)``,
      raw units.
    - ``update(records)`` refits on old + new data (needs ``input_keys`` /
      ``output_keys`` at construction).

    Defaults follow §5.7. ``n_members`` defaults to 5 (dev K); use 10 for finals.
    ``device="cpu"`` is deterministic; ``device="cuda"`` runs the sm_120 path.
    """

    def __init__(
        self,
        input_keys: Sequence[str] | None = None,
        output_keys: Sequence[str] | None = None,
        n_members: int = 5,
        width: int = 128,
        n_blocks: int = 2,
        d_rff: int = 256,
        max_epochs: int = 400,
        patience: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        ridge: float = 1.0,
        input_jitter: float = 0.05,
        val_fraction: float = 0.2,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.input_keys = list(input_keys) if input_keys is not None else None
        self.output_keys = list(output_keys) if output_keys is not None else None
        self.n_members = n_members
        self.width = width
        self.n_blocks = n_blocks
        self.d_rff = d_rff
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.ridge = ridge
        self.input_jitter = input_jitter
        self.val_fraction = val_fraction
        self.seed = seed
        self.device = torch.device(device)
        self._members: list[_Member] = []
        # per-member RFF-GP Laplace covariance (numpy, one (d_rff,d_rff) per output)
        self._sigma_cho: list[list[tuple[np.ndarray, bool]]] = []
        self._X_raw: np.ndarray | None = None
        self._Y_raw: np.ndarray | None = None

    # -- fitting ---------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return bool(self._members)

    @property
    def n_train_(self) -> int:
        return 0 if self._X_raw is None else int(self._X_raw.shape[0])

    def _standardize(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._x_mean, self._x_scale = _standardize_stats(X)
        self._y_mean, self._y_scale = _standardize_stats(Y)
        Xs = (X - self._x_mean) / self._x_scale
        Ys = (Y - self._y_mean) / self._y_scale
        return Xs, Ys

    def fit(self, X: np.ndarray, Y: np.ndarray) -> DeepEnsembleForwardModel:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d); got shape {X.shape}")
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
        self._X_raw, self._Y_raw = X, Y
        Xs, Ys = self._standardize(X, Y)
        d_in, m_out = Xs.shape[1], Ys.shape[1]

        # rff lengthscale via the median-distance heuristic on RAW standardized
        # inputs (a stable, member-independent scale; the trunk is ≈1-Lipschitz
        # so input-space median is a sound proxy for feature-space scale).
        rff_scale = self._median_distance(Xs)

        self._members = []
        self._sigma_cho = []
        for k in range(self.n_members):
            member = self._train_member(Xs, Ys, d_in, m_out, rff_scale, seed=self.seed + 101 * k)
            self._members.append(member)
            self._sigma_cho.append(self._laplace_cov(member, Xs, Ys))

        # support-score space: reference member's spectral-normalized latent φ
        phi_train = self._member_features(self._members[0], Xs)
        self._phi_mean = phi_train.mean(axis=0)
        self._support_cov_inv = _regularized_cov_inv(phi_train - self._phi_mean)
        return self

    @staticmethod
    def _median_distance(Xs: np.ndarray) -> float:
        n = Xs.shape[0]
        if n < 2:
            return 1.0
        # subsample for the pairwise-distance median (cheap, deterministic)
        idx = np.arange(n)
        D = np.sqrt(((Xs[idx, None, :] - Xs[None, idx, :]) ** 2).sum(-1))
        med = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
        return max(med, 1e-3)

    def _train_member(
        self,
        Xs: np.ndarray,
        Ys: np.ndarray,
        d_in: int,
        m_out: int,
        rff_scale: float,
        seed: int,
    ) -> _Member:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        member = _Member(d_in, m_out, self.width, self.n_blocks, self.d_rff, rff_scale, seed).to(
            self.device
        )

        n = Xs.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(self.val_fraction * n))) if n >= 5 else 0
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:] if n_val else perm
        Xtr = torch.as_tensor(Xs[tr_idx], dtype=torch.float32, device=self.device)
        Ytr = torch.as_tensor(Ys[tr_idx], dtype=torch.float32, device=self.device)
        Xval = torch.as_tensor(Xs[val_idx], dtype=torch.float32, device=self.device)
        Yval = torch.as_tensor(Ys[val_idx], dtype=torch.float32, device=self.device)

        opt = torch.optim.AdamW(member.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        best_val = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        stale = 0
        n_tr = Xtr.shape[0]
        for _epoch in range(self.max_epochs):
            member.train()
            order = torch.as_tensor(rng.permutation(n_tr), device=self.device)
            for start in range(0, n_tr, self.batch_size):
                bidx = order[start : start + self.batch_size]
                xb = Xtr[bidx]
                if self.input_jitter > 0:  # input-domain randomization (no bagging)
                    xb = xb + self.input_jitter * torch.randn_like(xb)
                mean, sigma2, _ = member(xb)
                loss = _beta_nll_loss(mean, sigma2, Ytr[bidx])
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            # early stop on validation β-NLL (patience); no val -> track train loss
            member.eval()
            with torch.no_grad():
                if n_val:
                    vm, vs, _ = member(Xval)
                    vloss = float(_beta_nll_loss(vm, vs, Yval))
                else:
                    tm, ts, _ = member(Xtr)
                    vloss = float(_beta_nll_loss(tm, ts, Ytr))
            if vloss < best_val - 1e-5:
                best_val = vloss
                best_state = {k2: v.detach().clone() for k2, v in member.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            member.load_state_dict(best_state)
        member.eval()
        return member

    def _member_features(self, member: _Member, Xs: np.ndarray) -> np.ndarray:
        member.eval()
        with torch.no_grad():
            x = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
            return member.features(x).cpu().numpy().astype(np.float64)

    def _member_rff(self, member: _Member, Xs: np.ndarray) -> np.ndarray:
        member.eval()
        with torch.no_grad():
            x = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
            phi = member.features(x)
            return member.rff(phi).cpu().numpy().astype(np.float64)

    def _member_sigma2(self, member: _Member, Xs: np.ndarray) -> np.ndarray:
        member.eval()
        with torch.no_grad():
            x = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
            _, sigma2, _ = member(x)
            return sigma2.cpu().numpy().astype(np.float64)

    def _laplace_cov(
        self, member: _Member, Xs: np.ndarray, Ys: np.ndarray
    ) -> list[tuple[np.ndarray, bool]]:
        """RFF-GP Laplace covariance per output (Liu et al. 2020, regression form).

        Precision ``S_j = ridge·I + Φᵀ diag(1/σ²_j) Φ`` (heteroscedastic ridge on
        the fixed random features); we store its Cholesky and evaluate the
        predictive epistemic variance ``Φ(x*)ᵀ S_j⁻¹ Φ(x*)`` at query time. Far
        from the training features Φ(x*) decorrelates from the trained Gram, so
        the variance rises toward the prior ``‖Φ‖²/ridge ≈ 1/ridge`` (§5.9 inv. 1).
        """
        Phi = self._member_rff(member, Xs)  # (n, D)
        sig2 = np.clip(self._member_sigma2(member, Xs), _SIGMA2_FLOOR, None)  # (n, m)
        d_rff = Phi.shape[1]
        chos: list[tuple[np.ndarray, bool]] = []
        for j in range(Ys.shape[1]):
            w = 1.0 / sig2[:, j]  # (n,)
            S = self.ridge * np.eye(d_rff) + (Phi * w[:, None]).T @ Phi
            chos.append(cho_factor(S, lower=True))
        return chos

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

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("DeepEnsembleForwardModel is not fitted; call fit(X, Y) first")

    # -- ForwardModel protocol ---------------------------------------------------

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        Xs = (Xq - self._x_mean) / self._x_scale  # (n, d)

        member_means = []  # (K, n, m) standardized
        member_ale = []  # (K, n, m) standardized aleatoric var
        member_epi = []  # (K, n, m) standardized SNGP Laplace var
        for member, chos in zip(self._members, self._sigma_cho, strict=True):
            member.eval()
            with torch.no_grad():
                xt = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
                mean, sigma2, rff = member(xt)
            member_means.append(mean.cpu().numpy().astype(np.float64))
            member_ale.append(sigma2.cpu().numpy().astype(np.float64))
            Phi = rff.cpu().numpy().astype(np.float64)  # (n, D)
            epi = np.empty((Phi.shape[0], len(chos)), dtype=np.float64)
            for j, cho in enumerate(chos):
                v = cho_solve(cho, Phi.T)  # S_j^{-1} Φᵀ, (D, n)
                epi[:, j] = np.einsum("ij,ji->i", Phi, v)  # Φ Σ Φᵀ diag, (n,)
            member_epi.append(np.maximum(epi, 0.0))

        mm = np.stack(member_means)  # (K, n, m)
        mean_std = mm.mean(axis=0)  # (n, m)
        aleatoric_var = np.stack(member_ale).mean(axis=0)  # (n, m) = E[σ_m²]
        epi_ens = mm.var(axis=0)  # (n, m) = Var[μ_m]
        epi_sngp = np.stack(member_epi).mean(axis=0)  # (n, m) = E[Laplace var]
        epistemic_var = epi_ens + epi_sngp

        # de-standardize to raw output units
        mean = self._y_mean + self._y_scale * mean_std
        aleatoric = self._y_scale * np.sqrt(aleatoric_var)
        epistemic = self._y_scale * np.sqrt(epistemic_var)
        if single:
            mean, aleatoric, epistemic = mean[0], aleatoric[0], epistemic[0]
        return PredictiveDistribution(
            mean=mean,
            aleatoric_sigma=aleatoric,
            epistemic_sigma=epistemic,
            conformal_set=None,  # filled by the §5.6 calibration wrapper
        )

    def support_score(self, x: np.ndarray) -> float | np.ndarray:
        """Negative Mahalanobis distance to the training set in the reference
        member's spectral-normalized latent φ (§8.2, §11). Float for a single
        (d,) point; (n,) array for a batch."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        phi = self._member_features(self._members[0], Xs) - self._phi_mean
        d2 = np.einsum("ij,jk,ik->i", phi, self._support_cov_inv, phi)
        score = -np.sqrt(np.maximum(d2, 0.0))
        return float(score[0]) if single else score

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Autograd d(mixture-mean)/dx at a single point x (d,), raw units,
        shape (m, d). Chains through both standardizations."""
        self._require_fitted()
        return self._jacobian(self._members, x)

    def _jacobian(self, members: list[_Member], x: np.ndarray) -> np.ndarray:
        """Autograd d(mean-over-``members``)/dx at a single point, raw (m, d)."""
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("jacobian(x) takes a single point of shape (d,)")
        xs = (x - self._x_mean) / self._x_scale
        xt = torch.as_tensor(xs, dtype=torch.float32, device=self.device).requires_grad_(True)
        means = []
        for member in members:
            member.eval()
            mean, _, _ = member(xt.unsqueeze(0))
            means.append(mean.squeeze(0))
        mix = torch.stack(means).mean(dim=0)  # (m,) standardized
        m = mix.shape[0]
        rows = np.empty((m, xs.shape[0]), dtype=np.float64)
        for j in range(m):
            grad = torch.autograd.grad(mix[j], xt, retain_graph=(j < m - 1))[0]
            # chain: y = y_mean + y_scale·mix(x_std), x_std = (x - x_mean)/x_scale
            rows[j] = self._y_scale[j] * grad.detach().cpu().numpy() / self._x_scale
        return rows

    # -- joint epistemic covariance (EPIG, §9.4) + fast inner-loop views (§5.7) --

    def _forward_np(
        self, member: _Member, Xs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One member's standardized (mean, sigma2, rff) as float64 numpy."""
        member.eval()
        with torch.no_grad():
            xt = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
            mean, sigma2, rff = member(xt)
        return (
            mean.cpu().numpy().astype(np.float64),
            sigma2.cpu().numpy().astype(np.float64),
            rff.cpu().numpy().astype(np.float64),
        )

    def _member_sngp_var(self, member_idx: int, Phi: np.ndarray) -> np.ndarray:
        """diag(Φ Σ_j Φᵀ) per output for one member — the SNGP-Laplace epistemic
        variance, standardized, shape (n, m)."""
        chos = self._sigma_cho[member_idx]
        epi = np.empty((Phi.shape[0], len(chos)), dtype=np.float64)
        for j, cho in enumerate(chos):
            v = cho_solve(cho, Phi.T)
            epi[:, j] = np.einsum("ij,ji->i", Phi, v)
        return np.maximum(epi, 0.0)

    def _member_sngp_cov(self, member_idx: int, Phi1: np.ndarray, Phi2: np.ndarray) -> np.ndarray:
        """Φ1 Σ_j Φ2ᵀ per output for one member — the SNGP-Laplace joint epistemic
        covariance, standardized, shape (m, n1, n2)."""
        chos = self._sigma_cho[member_idx]
        return np.stack([Phi1 @ cho_solve(cho, Phi2.T) for cho in chos])

    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Per-output joint epistemic covariance ``Cov(f(X1), f(X2))`` in raw
        output units, shape ``(m, n1, n2)`` — what the §9.4 EPIG acquisition needs.

        Consistent with :meth:`predict`'s ``epistemic_sigma``: it is the sum of
        (a) the ENSEMBLE-SPREAD covariance ``Cov_m(μ_m(X1), μ_m(X2))`` (population,
        matching ``Var[μ_m]``) and (b) the ensemble-averaged SNGP-Laplace joint
        covariance ``E_m[Φ_m(X1) Σ_m Φ_m(X2)ᵀ]``. Its diagonal therefore equals
        ``epistemic_sigma²`` exactly, so EPIG's ``σ²(x*|x)`` reduction is
        self-consistent (a mismatch would silently mis-scale the info gain)."""
        self._require_fitted()
        Xs1 = (np.atleast_2d(np.asarray(X1, dtype=float)) - self._x_mean) / self._x_scale
        Xs2 = (np.atleast_2d(np.asarray(X2, dtype=float)) - self._x_mean) / self._x_scale
        means1, means2, rffs1, rffs2 = [], [], [], []
        for member in self._members:
            m1, _, p1 = self._forward_np(member, Xs1)
            m2, _, p2 = self._forward_np(member, Xs2)
            means1.append(m1)
            means2.append(m2)
            rffs1.append(p1)
            rffs2.append(p2)
        K = len(self._members)
        M1 = np.stack(means1)  # (K, n1, m)
        M2 = np.stack(means2)  # (K, n2, m)
        c1 = M1 - M1.mean(axis=0)  # centered member means
        c2 = M2 - M2.mean(axis=0)
        spread = np.einsum("knj,koj->jno", c1, c2) / K  # (m, n1, n2), ddof=0
        sngp = np.mean(
            [self._member_sngp_cov(i, rffs1[i], rffs2[i]) for i in range(K)], axis=0
        )  # (m, n1, n2)
        cov_std = spread + sngp
        return cov_std * (self._y_scale**2)[:, None, None]

    def sngp_member_view(self, member: int = 0) -> _SNGPMemberView:
        """A fast single-member ForwardModel view for the §8 inner loop (§5.7
        option B — the SNGP single member, zero extra training).

        Reuses ``self._members[member]`` + its SNGP-Laplace Cholesky, so a
        ``predict`` is ONE forward pass (not the K-member mixture): ~K× cheaper,
        and it keeps a distance-aware epistemic (the member's SNGP-Laplace var),
        so the pessimistic search + support gate still behave. The full ensemble
        remains the arbiter — re-validate final candidates against it + conformal
        (the §8 inner-loop-budget division of labor). ``support_score`` delegates
        to the full model (the shared spectral-latent Mahalanobis)."""
        self._require_fitted()
        if not 0 <= member < len(self._members):
            raise IndexError(f"member {member} out of range (K={len(self._members)})")
        return _SNGPMemberView(self, member)

    def inner_loop_surrogate(
        self, *, mode: str = "sngp_member", member: int = 0
    ) -> _SNGPMemberView:
        """Select the fast inner-loop surrogate (§5.7). ``mode="sngp_member"`` is
        the free, zero-training default (option B). Ensemble-distribution
        distillation into a single distributional net (option A, Malinin 2020, the
        `/invert` serving path + the ≥20× target) is a follow-on WP-E slice."""
        if mode != "sngp_member":
            raise NotImplementedError(
                f"inner_loop_surrogate mode {mode!r}: only 'sngp_member' is wired "
                "(distilled option A is a follow-on WP-E slice, §5.7)"
            )
        return self.sngp_member_view(member)


class _SNGPMemberView:
    """Read-only fast ForwardModel view over ONE ensemble member (§5.7 option B).

    ForwardModel + _JointModel conformant (predict/support_score/jacobian/
    posterior_cov), so it drops into ``PessimisticInverseSolver`` and the §9
    acquisition unchanged. Epistemic = the member's SNGP-Laplace variance only
    (no ensemble-spread term — a single member has none), which is what makes it
    a screening surrogate whose proposals the full ensemble re-validates."""

    def __init__(self, parent: DeepEnsembleForwardModel, member_idx: int) -> None:
        self._parent = parent
        self._i = member_idx

    @property
    def n_train_(self) -> int:
        return self._parent.n_train_

    @property
    def is_fitted(self) -> bool:
        return self._parent.is_fitted

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        p = self._parent
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - p._x_mean) / p._x_scale
        mean_std, sigma2, Phi = p._forward_np(p._members[self._i], Xs)
        epi_var = p._member_sngp_var(self._i, Phi)  # SNGP-Laplace only (no spread)
        mean = p._y_mean + p._y_scale * mean_std
        aleatoric = p._y_scale * np.sqrt(sigma2)
        epistemic = p._y_scale * np.sqrt(epi_var)
        if single:
            mean, aleatoric, epistemic = mean[0], aleatoric[0], epistemic[0]
        return PredictiveDistribution(
            mean=mean,
            aleatoric_sigma=aleatoric,
            epistemic_sigma=epistemic,
            conformal_set=None,
        )

    def support_score(self, x: np.ndarray) -> float | np.ndarray:
        return self._parent.support_score(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        p = self._parent
        return p._jacobian([p._members[self._i]], x)

    def update(self, records: Iterable[Any]) -> None:
        """Refit the FULL parent ensemble (invariant 2d); this view then reflects
        member ``i`` of the refitted ensemble."""
        self._parent.update(records)

    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        p = self._parent
        Xs1 = (np.atleast_2d(np.asarray(X1, dtype=float)) - p._x_mean) / p._x_scale
        Xs2 = (np.atleast_2d(np.asarray(X2, dtype=float)) - p._x_mean) / p._x_scale
        _, _, p1 = p._forward_np(p._members[self._i], Xs1)
        _, _, p2 = p._forward_np(p._members[self._i], Xs2)
        cov = p._member_sngp_cov(self._i, p1, p2)  # (m, n1, n2), standardized
        return cov * (p._y_scale**2)[:, None, None]
