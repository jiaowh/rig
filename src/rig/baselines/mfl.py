"""Model Feedback Learning (MFL) — the amortized-inverse baseline to compare against.

Faithful re-implementation of **Algorithm 1** of

    Gu et al., "Few-Shot Test-Time Optimization Without Retraining for
    Semiconductor Recipe Generation and Beyond", arXiv:2505.16060v1 (2025),
    UC Berkeley / Virginia Tech / Lam Research / UCL.

MFL is the comparator for the pre-registered bake-off
(``docs/prereg-mfl-bakeoff-2026-07-17.md``). It is a *point-target* amortized
inverse: ONE reverse model ``R_θ`` conditioned on a target ``z'`` serves the whole
target distribution, trained by feedback through a learned emulator ``E`` (Loop A)
and then through the machine ``M`` itself (Loop B, their Eq. 4).

It pulls torch, so — like :class:`~rig.baselines.botorch_bo.BoTorchBO` — it is
imported lazily by ``rig.baselines.__init__`` and ``import rig`` stays torch-free.

Algorithm 1, transcribed (line references are to the paper; see the comments)
----------------------------------------------------------------------------
Inputs: machine ``M``; reverse model ``R_θ`` (θ⁰); emulator data ``{(x_i,z_i)}``;
targets ``{z'_j}``; learning rates ``α1 > α2``; periods ``T, T0, τ, τ0``; ``δ``.

- **Step 1** — train emulator ``E`` on ``{(x_i,z_i)}`` supervised, WITH domain
  randomization (zero-mean Gaussian noise added to the inputs during training).
- **Loop A** (``t = 0..T-1``) on ``E``:
  ``x'_{t,j} = R_θ(z'_j)``; ``y' = E(x')``; ``L = (1/n')Σ‖z'_j − y'_j‖²``;
  if ``t ≥ T0`` and ``mean_j s_E(x'_{t,j}) ≥ δ`` → ``lr = α2`` else ``α1``.
  The gradient flows THROUGH ``E`` (their Eq. 4:
  ``[∂R/∂θ]ᵀ[∂E/∂x]ᵀ(y'−z')``) — NOT a detached regression target.
- **Loop B** (``h = 0..τ-1``) on the MACHINE ``M``: identical, with ``∂M/∂x`` in
  place of ``∂E/∂x``, sensitivity ``s_M``, and onset ``τ0``.
- **Sensitivity** ``s_f(x)`` = the induced L2 (spectral) norm of ``∂f/∂x``.
- **Table 10 defaults** (all exposed as kwargs): ``α1=0.01``, ``α2=0.99·0.01``,
  hidden ``64``, emulator epochs ``700``, ``T=1200``, ``T0=1150``, ``τ=200``,
  ``τ0=150``, ``δ=0.9``. Input bounds: **clip** (the literal reading).

Deviations from the paper (documented, one line each — see README for rationale)
--------------------------------------------------------------------------------
1. **All NN I/O is standardized** (train-set mean/std of x and z). The paper's data
   is Gaussian-sampled ~unit-scale; MBE recipes/outputs span 20 orders of magnitude
   (K vs metres), so a net cannot train without it. Sensitivities ``s_E/s_M``, the
   ``δ`` gate, the FD step, and the clip bounds are ALL therefore in standardized
   units — self-consistent, and the only scale-free reading of ``δ=0.9``.
2. **``∂M/∂x`` by forward finite differences** (``d`` probes / point): the paper's
   ``M`` is a differentiable MLP; :class:`~rig_adapters.mbe.machine.InSilicoMachine`
   has no autograd. This is the crux of prereg §3 and the whole point of the
   charitable-vs-deployable ledger. Same for a real tool. NB the FD step ``fd_step``
   (standardized units) must EXCEED the machine's standardized metrology-noise floor,
   or the difference is noise-dominated and Loop B injects pure-noise gradients that
   destroy ``R`` (measured: ``fd_step=1e-3`` on the noisy MBE machine drives recovery
   error 0.03 → 2.4; ``0.05`` restores it). The default is therefore ``0.05`` — a
   steelman choice the paper gives no guidance on (their ``M`` is noiseless). A truly
   differentiable ``M`` would use autograd and none of this.
3. **MLP depth = 2 hidden layers of width 64.** The paper fixes width (64) but not
   depth; two layers is the smallest expressive choice.
4. **Optimizer = plain SGD** with the scheduled lr, so ``α1/α2`` act as literal
   learning rates (Adam would re-scale them).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

# Table 10 defaults (Gu et al. 2025), exposed as kwargs on the class below.
ALPHA1 = 0.01
ALPHA2 = 0.99 * 0.01
HIDDEN = 64
EMULATOR_EPOCHS = 700
T_LOOP_A = 1200
T0_LOOP_A = 1150
TAU_LOOP_B = 200
TAU0_LOOP_B = 150
DELTA = 0.9


@dataclass
class MFLLedger:
    """Machine-query ledger for one MFL training run (prereg §0 ``machine_queries``).

    ``seed_runs`` and ``revalidation_evals`` are set by the runner (they are shared
    accounting); MFL itself only ever increments ``loopB_evals`` (the per-target base
    value each Loop-B iteration, dual-use as the FD base point) and ``fd_probe_evals``
    (the ``d`` forward-difference probes per target per iteration). The two totals are
    the prereg §3 charitable-vs-deployable split: the charitable arm (their setting, a
    differentiable deployed ``M``) does NOT count FD probes; the deployable arm (a real
    tool) counts every machine touch.
    """

    seed_runs: int = 0
    loopB_evals: int = 0
    fd_probe_evals: int = 0
    revalidation_evals: int = 0

    @property
    def charitable_total(self) -> int:
        return self.seed_runs + self.loopB_evals + self.revalidation_evals

    @property
    def deployable_total(self) -> int:
        return self.seed_runs + self.loopB_evals + self.fd_probe_evals + self.revalidation_evals

    def as_dict(self) -> dict[str, int]:
        return {
            "seed_runs": self.seed_runs,
            "loopB_evals": self.loopB_evals,
            "fd_probe_evals": self.fd_probe_evals,
            "revalidation_evals": self.revalidation_evals,
            "charitable_total": self.charitable_total,
            "deployable_total": self.deployable_total,
        }


def fd_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    f0: np.ndarray,
    h: float = 1e-3,
) -> tuple[np.ndarray, int]:
    """Forward finite-difference Jacobian ``∂f/∂x`` at ``x``, plus the probe count.

    ``f`` maps ``(d,) -> (m,)``. ``f0 = f(x)`` is supplied by the caller (it is the
    per-iteration base VALUE that is also the FD base point, so it is counted once as
    a ``loopB_eval`` rather than a probe). This routine spends exactly ``d`` further
    evaluations — the ``fd_probe_evals`` charged only in the deployable ledger.
    Returns ``(J, n_probes)`` with ``J`` shape ``(m, d)`` and ``n_probes == d``.
    """
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f0, dtype=float)
    d = x.size
    m = f0.size
    J = np.empty((m, d), dtype=float)
    for i in range(d):
        xp = x.copy()
        xp[i] += h
        J[:, i] = (np.asarray(f(xp), dtype=float) - f0) / h
    return J, d


def spectral_norm(J: np.ndarray) -> float:
    """Induced L2 norm of a Jacobian (its largest singular value) — the paper's
    ``s_f(x)`` sensitivity (Alg. 1 line 20)."""
    J = np.atleast_2d(np.asarray(J, dtype=float))
    if J.size == 0:
        return 0.0
    return float(np.linalg.svd(J, compute_uv=False)[0])


class _MLP(nn.Module):
    """Two-hidden-layer width-``hidden`` MLP (deviation 3)."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ModelFeedbackLearning:
    """Faithful MFL (Gu et al. 2025, Algorithm 1). Point-target amortized inverse.

    Parameters
    ----------
    x_dim, z_dim
        Recipe (input) and target (output) dimensions.
    x_lower, x_upper
        Recipe box bounds in RAW units, per recipe variable — used both to
        standardize inputs and as the Alg. 1 ``clip`` on ``R``'s output.
    hidden, alpha1, alpha2, emulator_epochs, T, T0, tau, tau0, delta
        Table 10 hyperparameters (see module constants for the paper defaults).
    domain_randomization
        Std (standardized units) of the zero-mean Gaussian added to ``E``'s inputs
        during Step-1 training (Alg. 1 domain randomization). Default 0.05.
    fd_step
        Forward-difference step ``h`` for ``∂M/∂x`` (standardized units, deviation 2).
        Must exceed the machine's standardized metrology-noise floor (default 0.05).
    seed, device
        Determinism + torch device (CPU default: small nets, reproducible).
    """

    def __init__(
        self,
        *,
        x_dim: int,
        z_dim: int,
        x_lower: Sequence[float],
        x_upper: Sequence[float],
        hidden: int = HIDDEN,
        alpha1: float = ALPHA1,
        alpha2: float = ALPHA2,
        emulator_epochs: int = EMULATOR_EPOCHS,
        T: int = T_LOOP_A,
        T0: int = T0_LOOP_A,
        tau: int = TAU_LOOP_B,
        tau0: int = TAU0_LOOP_B,
        delta: float = DELTA,
        domain_randomization: float = 0.05,
        fd_step: float = 0.05,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.x_dim = int(x_dim)
        self.z_dim = int(z_dim)
        self.x_lower = np.asarray(x_lower, dtype=float)
        self.x_upper = np.asarray(x_upper, dtype=float)
        if self.x_lower.shape != (self.x_dim,) or self.x_upper.shape != (self.x_dim,):
            raise ValueError("x_lower/x_upper must each have length x_dim")
        self.hidden = int(hidden)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.emulator_epochs = int(emulator_epochs)
        self.T = int(T)
        self.T0 = int(T0)
        self.tau = int(tau)
        self.tau0 = int(tau0)
        self.delta = float(delta)
        self.domain_randomization = float(domain_randomization)
        self.fd_step = float(fd_step)
        self.seed = int(seed)
        self.device = torch.device(device)

        torch.manual_seed(self.seed)
        self.E = _MLP(self.x_dim, self.z_dim, self.hidden).to(self.device).double()
        self.R = _MLP(self.z_dim, self.x_dim, self.hidden).to(self.device).double()

        # standardization stats, filled by fit_emulator (train-set only, §5.3 leakage).
        self._x_mean = np.zeros(self.x_dim)
        self._x_scale = np.ones(self.x_dim)
        self._z_mean = np.zeros(self.z_dim)
        self._z_scale = np.ones(self.z_dim)
        self._fitted = False

        # conservative-LR gate counters (test 3 reads these).
        self.alpha2_count_loopA = 0
        self.alpha2_count_loopB = 0

    # -- standardization helpers ------------------------------------------------

    def _xs(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self._x_mean) / self._x_scale

    def _x_raw(self, Xs: np.ndarray) -> np.ndarray:
        # clip to the RAW box: standardized clamp + denorm can land a hair outside a
        # bound in float (e.g. 1.9999e-7 vs a 2e-7 lower edge), which a strict machine
        # rejects. The Alg. 1 clip is on the box, so enforcing it in raw units is exact.
        x = np.asarray(Xs, dtype=float) * self._x_scale + self._x_mean
        return np.clip(x, self.x_lower, self.x_upper)

    def _zs(self, Z: np.ndarray) -> np.ndarray:
        return (np.asarray(Z, dtype=float) - self._z_mean) / self._z_scale

    @property
    def _xs_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        lo = torch.as_tensor((self.x_lower - self._x_mean) / self._x_scale, dtype=torch.double)
        hi = torch.as_tensor((self.x_upper - self._x_mean) / self._x_scale, dtype=torch.double)
        return lo.to(self.device), hi.to(self.device)

    def _clip_xs(self, xs: torch.Tensor) -> torch.Tensor:
        """Alg. 1 input-bound clip on ``R``'s output (literal reading: torch.clamp,
        so the gradient is zeroed at a saturated bound)."""
        lo, hi = self._xs_bounds
        return torch.clamp(xs, lo, hi)

    # -- Step 1: emulator with domain randomization -----------------------------

    def fit_emulator(self, X: np.ndarray, Z: np.ndarray) -> ModelFeedbackLearning:
        """Train ``E`` on ``{(x_i, z_i)}`` supervised, WITH domain randomization
        (Alg. 1 Step 1). Standardizes both spaces on the train set first."""
        X = np.asarray(X, dtype=float)
        Z = np.asarray(Z, dtype=float)
        if X.ndim != 2 or Z.ndim != 2:
            raise ValueError("X and Z must be 2-D (n, d) / (n, m)")
        self._x_mean = X.mean(axis=0)
        self._x_scale = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        self._z_mean = Z.mean(axis=0)
        self._z_scale = np.where(Z.std(axis=0) > 1e-12, Z.std(axis=0), 1.0)

        Xs = torch.as_tensor(self._xs(X), dtype=torch.double, device=self.device)
        Zs = torch.as_tensor(self._zs(Z), dtype=torch.double, device=self.device)

        torch.manual_seed(self.seed + 1)
        opt = torch.optim.Adam(self.E.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()
        gen = torch.Generator(device=self.device).manual_seed(self.seed + 2)
        self.E.train()
        for _ in range(self.emulator_epochs):
            opt.zero_grad()
            # domain randomization: perturb the (standardized) inputs each epoch.
            noise = torch.randn(Xs.shape, dtype=torch.double, generator=gen, device=self.device)
            pred = self.E(Xs + self.domain_randomization * noise)
            loss = loss_fn(pred, Zs)
            loss.backward()
            opt.step()
        self.E.eval()
        self._fitted = True
        return self

    def emulator_predict(self, X: np.ndarray) -> np.ndarray:
        """``E(x)`` in RAW output units (test / diagnostics)."""
        self._require_fitted()
        Xs = torch.as_tensor(self._xs(X), dtype=torch.double, device=self.device)
        with torch.no_grad():
            Zs = self.E(Xs).cpu().numpy()
        return Zs * self._z_scale + self._z_mean

    # -- Loops A + B: train the reverse model -----------------------------------

    def train_reverse(
        self,
        targets_z: np.ndarray,
        machine: Callable[[np.ndarray], np.ndarray] | None = None,
        ledger: MFLLedger | None = None,
    ) -> ModelFeedbackLearning:
        """Run Loop A (on ``E``) then Loop B (on ``machine`` ``M``) to train the ONE
        reverse model ``R`` that serves the whole target set ``{z'_j}``.

        ``machine`` maps a batch of RAW recipe vectors ``(n', d) -> (n', m)`` RAW
        controlled outputs (the noisy tool). It is required whenever ``tau > 0``.
        ``ledger`` (an :class:`MFLLedger`) accrues the Loop-B machine touches.
        """
        self._require_fitted()
        targets_z = np.asarray(targets_z, dtype=float)
        if targets_z.ndim != 2 or targets_z.shape[1] != self.z_dim:
            raise ValueError(f"targets_z must be (n', {self.z_dim})")
        zt = torch.as_tensor(self._zs(targets_z), dtype=torch.double, device=self.device)

        # E is frozen throughout reverse training — the gradient flows THROUGH it to
        # R (Alg. 1 Eq. 4), but E's own parameters never update.
        for p in self.E.parameters():
            p.requires_grad_(False)
        opt = torch.optim.SGD(self.R.parameters(), lr=self.alpha1)

        self._loop_a(zt, opt)
        if self.tau > 0:
            if machine is None:
                raise ValueError("train_reverse needs `machine` when tau > 0 (Loop B on M)")
            self._loop_b(zt, targets_z, opt, machine, ledger if ledger is not None else MFLLedger())

        for p in self.E.parameters():
            p.requires_grad_(True)
        return self

    def _set_lr(self, opt: torch.optim.Optimizer, lr: float) -> None:
        for g in opt.param_groups:
            g["lr"] = lr

    def _loop_a(self, zt: torch.Tensor, opt: torch.optim.Optimizer) -> None:
        """Loop A on the emulator ``E`` (Alg. 1). Gradient flows through ``E``."""
        self.R.train()
        for t in range(self.T):
            opt.zero_grad()
            xs = self._clip_xs(self.R(zt))  # x'_{t,j} = R(z'_j), clipped
            ys = self.E(xs)  # y' = E(x'), E frozen
            # L = (1/n')Σ_j ‖z'_j − y'_j‖² (paper Eq. 4 scaling): SUM over the z-dim,
            # MEAN over targets — consistent with Loop B's 2/n' gradient below. (A plain
            # .mean() over all elements differs only by the constant 1/z_dim; steelman-
            # verified immaterial to the outcome, but this is the paper-faithful form.)
            loss = ((ys - zt) ** 2).sum(dim=1).mean()
            loss.backward()  # [∂R/∂θ]ᵀ[∂E/∂x]ᵀ(y'−z'), Eq. 4
            lr = self.alpha1
            if t >= self.T0:
                if self._mean_sensitivity_E(xs.detach()) >= self.delta:
                    lr = self.alpha2
                    self.alpha2_count_loopA += 1
            self._set_lr(opt, lr)
            opt.step()

    def _mean_sensitivity_E(self, xs: torch.Tensor) -> float:
        """``mean_j s_E(x'_j)`` — spectral norm of the emulator Jacobian, averaged
        over the target set (Alg. 1 line 20). Autograd, since ``E`` is differentiable."""
        vals = []
        for row in xs:
            J = torch.autograd.functional.jacobian(
                lambda u: self.E(u), row, vectorize=True, create_graph=False
            )
            vals.append(float(torch.linalg.matrix_norm(J.reshape(self.z_dim, self.x_dim), ord=2)))
        return float(np.mean(vals)) if vals else 0.0

    def _loop_b(
        self,
        zt: torch.Tensor,
        targets_raw: np.ndarray,
        opt: torch.optim.Optimizer,
        machine: Callable[[np.ndarray], np.ndarray],
        ledger: MFLLedger,
    ) -> None:
        """Loop B on the MACHINE ``M`` (Alg. 1). ``∂M/∂x`` by forward finite
        differences (deviation 2); the gradient is injected into ``R`` via
        ``xs.backward(gradient=...)`` so it is exactly Eq. 4 with ``M`` in place of
        ``E``. Every machine touch is charged to ``ledger``."""
        n = zt.shape[0]
        zt_np = self._zs(targets_raw)  # standardized targets
        self.R.train()
        for h in range(self.tau):
            opt.zero_grad()
            xs = self._clip_xs(self.R(zt))  # (n, d) standardized, requires grad
            xs_np = xs.detach().cpu().numpy()

            zs_hat = np.empty((n, self.z_dim), dtype=float)
            J_all = np.empty((n, self.z_dim, self.x_dim), dtype=float)
            # M in standardized coordinates: standardize(M(denorm(xs))).
            m_std = self._machine_std(machine)
            for j in range(n):
                base = m_std(xs_np[j])  # 1 machine eval (dual-use FD base)
                ledger.loopB_evals += 1
                Jj, n_probes = fd_jacobian(m_std, xs_np[j], base, h=self.fd_step)
                ledger.fd_probe_evals += n_probes
                zs_hat[j] = base
                J_all[j] = Jj

            resid = zs_hat - zt_np  # (n, m) = y' − z'
            # dL/dx'_j = (2/n')·[∂M/∂x]ᵀ(y'−z') (L = (1/n')Σ‖·‖²).
            grad = np.einsum("jmd,jm->jd", J_all, resid) * (2.0 / n)
            xs.backward(gradient=torch.as_tensor(grad, dtype=torch.double, device=self.device))

            lr = self.alpha1
            if h >= self.tau0:
                s_M = float(np.mean([spectral_norm(J_all[j]) for j in range(n)]))
                if s_M >= self.delta:
                    lr = self.alpha2
                    self.alpha2_count_loopB += 1
            self._set_lr(opt, lr)
            opt.step()

    def _machine_std(
        self, machine: Callable[[np.ndarray], np.ndarray]
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Wrap a RAW-space machine ``(d,)->(m,)`` as a STANDARDIZED-space map
        ``(d,)->(m,)``: denorm the recipe, query, standardize the outputs."""

        def f(xs_vec: np.ndarray) -> np.ndarray:
            x_raw = self._x_raw(np.clip(xs_vec, *self._xs_bounds_np))
            z_raw = np.asarray(machine(x_raw[None, :]), dtype=float).reshape(-1)
            return (z_raw - self._z_mean) / self._z_scale

        return f

    @property
    def _xs_bounds_np(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            (self.x_lower - self._x_mean) / self._x_scale,
            (self.x_upper - self._x_mean) / self._x_scale,
        )

    # -- inference: present a recipe for a target (MFL cannot abstain) -----------

    def propose(self, z: np.ndarray) -> np.ndarray:
        """``R(z')`` clipped to the box → RAW recipe vector(s). A single ``(m,)``
        target returns ``(d,)``; a batch ``(n, m)`` returns ``(n, d)``. MFL always
        emits an ``x`` (no abstention branch — prereg P4)."""
        self._require_fitted()
        z = np.asarray(z, dtype=float)
        single = z.ndim == 1
        Z = np.atleast_2d(z)
        zs = torch.as_tensor(self._zs(Z), dtype=torch.double, device=self.device)
        self.R.eval()
        with torch.no_grad():
            xs = self._clip_xs(self.R(zs)).cpu().numpy()
        X = self._x_raw(xs)
        return X[0] if single else X

    def propose_recipe(self, z: np.ndarray, recipe_keys: Sequence[str]) -> dict[str, float]:
        """:meth:`propose` for one target, keyed by recipe-variable name."""
        x = self.propose(np.asarray(z, dtype=float).reshape(-1))
        return {k: float(v) for k, v in zip(recipe_keys, x, strict=True)}

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("fit_emulator(X, Z) must run before reverse training / propose")
