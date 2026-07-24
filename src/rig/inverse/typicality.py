"""Normalizing-flow TYPICALITY screen — the §8.2 off-manifold upgrade (D2/WP-E).

The §8 solver's cheap, always-available manifold gate is a negative-Mahalanobis
``support_score`` with a hard floor (:mod:`rig.inverse.pessimistic`). §8.2 asks for a
STRONGER composite: a recipe is on-manifold only if it is BOTH *typical* under a
normalizing flow of the input marginal ``p(x)`` AND low-disagreement under the
epistemic score. This module is the flow half — an opt-in screen wired ALONGSIDE (never
replacing) the Mahalanobis floor.

Why typicality, NOT raw flow density (the load-bearing design choice)
--------------------------------------------------------------------
Raw flow log-density is UNSAFE as an OOD gate: deep generative models assign *higher*
likelihood to some out-of-distribution inputs than to in-distribution data
(Nalisnick et al. 2019, "Do Deep Generative Models Know What They Don't Know?"). The
mechanism is the typical-set / volume effect — for a standard Gaussian the density
PEAKS at the origin, yet almost no samples land there (they concentrate on the shell at
radius ``√d``); the origin has high density but is deeply atypical. So we do NOT
threshold ``log p_flow(x)``; we use the **typicality-set** statistic

    score(x) = −|log p_flow(x) − E_train[log p_flow]|                       (§8.2)

and reject a point whose log-density is atypical — too FAR from the training-set mean
log-density in EITHER direction (atypically high or low). Higher ``score`` = more
typical; the hard ``floor`` is the ``floor_percentile`` (default 5th) percentile of the
training scores, mirroring the Mahalanobis floor's construction so the two screens read
the same way.

What this closes that Mahalanobis cannot
----------------------------------------
A single Mahalanobis distance to the pooled training mean is a UNIMODAL gate: for a
multimodal training set it scores the empty gap BETWEEN the modes (which sits near the
pooled mean) as highly on-support, admitting a recipe no training point is anywhere
near. The flow, having learned the actual (multimodal) density, scores that gap as a
low-density → atypical → rejected point. This is the §8.2 hole the screen exists to
close; :mod:`tests.test_inverse_hardening` proves it does.

Scope / honesty
---------------
* This is a SCREEN, not a certificate: a flow can itself be miscalibrated, and §8.2
  keeps the Mahalanobis reject as the fail-closed fallback for exactly that reason — so
  the solver applies BOTH and never lets the flow REPLACE the floor.
* It is not folded into the solver's soft ``λ_m`` reward (the flow density is not
  differentiable through the analytic-gradient path, and a torch forward pass per
  L-BFGS-B step is too costly for the hot loop) — screen-only, by design.

Implementation. A small ``zuko`` unconditional neural-spline flow over the standardized
input marginal (TRAIN statistics only, §5.3). Budgets are deliberately modest: this is
a low-dimensional density screen, not a generative model — seconds to fit on CPU.
torch + zuko are the ``[torch]`` optional extra, imported lazily by
``rig.inverse.__init__`` so ``import rig`` stays torch-free. Determinism (§13.4):
seeded torch + numpy RNGs; ``device="cpu"`` (default) is reproducible.
"""

from __future__ import annotations

import numpy as np
import torch
import zuko

from rig.forward._gp_common import standardize_stats as _standardize_stats


class FlowTypicalityScore:
    """§8.2 normalizing-flow typicality screen over the input marginal ``p(x)``.

    Fit on the training recipe matrix ``X`` ``(n, d)`` (the SAME flat-key recipe space
    the §8 solver searches and the SAME data used for the Mahalanobis floor), then hand
    the fitted instance to :class:`~rig.inverse.PessimisticInverseSolver` as
    ``typicality=``. The solver reads two members:

    * ``score(x)`` — the typicality statistic ``−|log p(x) − E_train[log p]|`` (higher =
      more typical); ``float`` for a single ``(d,)`` point, ``(n,)`` for a batch.
    * ``floor`` — the hard reject threshold (``floor_percentile`` of the train scores).

    ``log_density(x)`` exposes the RAW flow log-density (the quantity §8.2 warns must
    NOT be thresholded directly) for inspection and for tests contrasting the two rules.
    """

    def __init__(
        self,
        *,
        transforms: int = 3,
        hidden: tuple[int, ...] = (64, 64),
        bins: int = 8,
        max_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        floor_percentile: float = 5.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.transforms = int(transforms)
        self.hidden = tuple(hidden)
        self.bins = int(bins)
        self.max_epochs = int(max_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.floor_percentile = float(floor_percentile)
        self.seed = int(seed)
        self.device = torch.device(device)
        self._flow: object | None = None
        self._floor: float | None = None
        self._e_train: float | None = None

    # -- fitting ----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._flow is not None

    def fit(self, X: np.ndarray) -> FlowTypicalityScore:
        """Train the flow on the input marginal of ``X`` ``(n, d)`` and derive the
        reference mean log-density ``E_train[log p]`` and the typicality ``floor``."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be (n, d); got shape {X.shape}")
        n, d = X.shape
        self.d = int(d)
        self._x_mean, self._x_scale = _standardize_stats(X)  # TRAIN stats only (§5.3)
        Xs = (X - self._x_mean) / self._x_scale

        torch.manual_seed(self.seed)
        flow = zuko.flows.NSF(
            features=d,
            context=0,  # unconditional density of the input marginal
            transforms=self.transforms,
            hidden_features=self.hidden,
            bins=self.bins,
        ).to(self.device)
        rng = np.random.default_rng(self.seed)
        xt = torch.as_tensor(Xs, dtype=torch.float32, device=self.device)
        opt = torch.optim.AdamW(flow.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        flow.train()
        for _epoch in range(self.max_epochs):
            order = torch.as_tensor(rng.permutation(n), device=self.device)
            for start in range(0, n, self.batch_size):
                bidx = order[start : start + self.batch_size]
                loss = -flow().log_prob(xt[bidx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        flow.eval()
        self._flow = flow

        lp_train = self._log_density_std(Xs)
        self._e_train = float(np.mean(lp_train))
        scores = -np.abs(lp_train - self._e_train)
        self._floor = float(np.percentile(scores, self.floor_percentile))
        return self

    def _require_fitted(self) -> None:
        if self._flow is None:
            raise RuntimeError(
                "FlowTypicalityScore is not fitted; call fit(X_train) before using it "
                "as a §8.2 screen (fail-closed: an unfitted screen has no calibrated "
                "floor and would silently admit everything)."
            )

    # -- scoring ----------------------------------------------------------------

    def _log_density_std(self, Xs: np.ndarray) -> np.ndarray:
        """Raw flow log-density at ALREADY-standardized inputs ``Xs`` ``(n, d)``."""
        assert self._flow is not None
        xt = torch.as_tensor(np.atleast_2d(Xs), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._flow().log_prob(xt).cpu().numpy().astype(np.float64)

    def log_density(self, x: np.ndarray) -> float | np.ndarray:
        """Raw flow log-density ``log p_flow(x)`` in the standardized input space.

        Exposed for inspection and for the Nalisnick contrast test; it is NOT the screen
        statistic. Thresholding this directly is the §8.2 failure mode — use
        :meth:`score` / :attr:`floor` instead. ``float`` for a single ``(d,)`` point,
        ``(n,)`` for a batch."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        lp = self._log_density_std(Xs)
        return float(lp[0]) if single else lp

    def score(self, x: np.ndarray) -> float | np.ndarray:
        """§8.2 typicality score ``−|log p_flow(x) − E_train[log p_flow]|`` (higher =
        more typical). ``float`` for a single ``(d,)`` point, ``(n,)`` for a batch.

        Same sign convention as ``ForwardModel.support_score``: a point is rejected when
        ``score(x) < floor``."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        lp = self._log_density_std(Xs)
        s = -np.abs(lp - self._e_train)
        return float(s[0]) if single else s

    @property
    def floor(self) -> float:
        """The hard reject threshold — the ``floor_percentile`` percentile of the
        training typicality scores (default 5th). Raises if unfitted (fail-closed)."""
        self._require_fitted()
        assert self._floor is not None
        return self._floor

    @property
    def e_train(self) -> float:
        """The reference mean log-density ``E_train[log p_flow]`` the score centres on."""
        self._require_fitted()
        assert self._e_train is not None
        return self._e_train
