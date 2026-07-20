"""Ensemble distribution distillation — the §5.7 option-A single-net serving path.

§5.7 states the inverse-loop cost problem and its two resolutions verbatim:
"the inverse inner loop runs thousands of surrogate passes, so K=10
heteroscedastic forwards/step is a real cost. Resolve it *not* by shrinking UQ
but by **ensemble distribution distillation** (Malinin et al. 2020) into a
single distributional net, or by using the **SNGP single member** inside the
loop, then **re-validating final candidates against the full ensemble +
conformal**." Option B (the SNGP single member) is
:meth:`rig.forward.ensemble.DeepEnsembleForwardModel.sngp_member_view`. This
module is option A: one net, trained offline against the K-member teacher, that
serves ``predict``/``support_score``/``jacobian`` in a single forward pass.

MEASURED ``predict`` speedup vs the teacher (``tests/test_distill.py``, CPU, two
runs): at the §5.2 production teacher size (K=10, ``d_rff``=256, m=3) **26.0x and
27.8x**; at the dev-small fixture teacher (K=3, ``d_rff``=128, m=1) 5.7x and 3.5x
(wall-clock on a shared box is noisy — the tests assert a loose bound and PRINT
the figure rather than pinning one). The win scales with K and with ``d_rff``·m:
the teacher pays K forward passes PLUS K·m Cholesky solves of the (``d_rff``,
``d_rff``) SNGP-Laplace factor per query, the student pays one forward pass.
Quote the ~26x WITH its K=10 — it is a function of the teacher, not a constant,
and it is not the number a K=3 dev ensemble gives you.

**The crux: the aleatoric/epistemic split must survive.** A student fitted to the
teacher's predictive MEAN and TOTAL variance is strictly cheaper and strictly
wrong for our consumers — total variance is all one distilled net would need to
match the teacher's *marginal predictive*, but it cannot recover which half is
which. Both of §8 and §9 are functions of the split, not of the total:

- §8 displaces the credited band by ``z_epi·σ_epi`` and divides the residual
  margin by ``σ_ale``. Collapse the split and pessimism becomes an arbitrary
  rescaling of the total — silently optimistic wherever the teacher was mostly
  epistemic, which is exactly off-manifold, which is exactly where §8 exists.
- §9's BALD is ``0.5·log(1+σ_epi²/σ_ale²)`` and EPIG is an epistemic-only info
  gain. Both are functions of the RATIO; a collapsed split makes them constants.

So this student regresses the teacher's two components as SEPARATE heads against
SEPARATE targets (``E_m[σ_m²]`` and ``Var_m[μ_m] + E_m[SNGP-Laplace var]``, read
off the teacher's public ``predict``), never their sum. That is the practical
form of Malinin et al.'s ensemble *distribution* distillation for this codebase's
contract: :class:`rig.interfaces.PredictiveDistribution` exposes exactly the
mean and the two σ's, so matching those three fields (plus ``support_score``) IS
matching the law the teacher ships. It is moment-matching, not the full
Normal-Wishart EnD² — see "Not distilled" below for what that costs.

**Transfer set (the reason OOD inflation survives).** Distillation is only as
faithful as the inputs it is scored on, and the teacher is queryable anywhere —
so the transfer set deliberately covers more than the training manifold: half
scrambled-Sobol over a DILATED box around the training data, half jittered
training rows. The dilated half is what teaches the epistemic head to rise past
the data's edge (§5.9 invariant 1). Distilling on the training rows alone would
leave the epistemic head unconstrained precisely where epistemic matters and
would quietly reproduce a flat, blind OOD gate.

``n_transfer`` is consequently the knob that matters, and by a wide margin —
measured on the ``tests/test_distill.py`` fixture, median relative error on
``epistemic_sigma`` vs the teacher: 4096 pts/300 epochs **21.7%**, 4096/400
12.3%, 8192/300 2.3%, **8192/400 1.0%**. Student width bought nothing (4096/300
at width 128: 20.4%). That is the whole thesis of distilling against a transfer
set: the teacher is queryable for free, so fidelity is bought with QUERIES, not
with student capacity or training time. Do not cut ``n_transfer`` to make
distillation faster — it is a ONE-OFF offline cost amortized over thousands of
inner-loop passes, and cutting it degrades precisely the epistemic channel that
§8 and §5.9 depend on, while leaving the mean (which every casual check looks at)
almost untouched.

**Outside the transfer box the student is extrapolating**, and an MLP's
extrapolation is not a distance-aware epistemic — it is arbitrary. Measured on
the test fixture, the raw heads at 400 (a box of [-6.2, 12.5]) returned
``σ_epi`` = 0.2x its IN-DISTRIBUTION value, ``σ_ale`` = 5e-14, and a mean of
26.7 on a sine: confidently, precisely wrong, which is the §8.2 far-OOD hole the
whole support gate exists to defend against. A bounded additive penalty cannot
rescue that (the head had already dived to ``log σ²_epi`` ≈ -29). So queries are
**clipped to the transfer box before the trunk**: the heads are only ever
evaluated where they were distilled, and the out-of-box part of the query is
carried by ``ood_inflation`` instead — ``+gain·d`` on the log-variance and
``−gain·d`` on the support score, ``d`` = the (capped) distance from ``x`` to the
box, which is exactly the distance to the clipped point.

The shipped law outside the box is therefore "**the nearest thing we actually
distilled, plus an honest, monotone ignorance flag proportional to how far we had
to travel to find it**" — no silent decay, no invented mean. It is EXACTLY ZERO
inside the box (so nothing in-domain is manufactured by it), and it fails in the
safe direction (more epistemic, less support ⇒ §8 abstains). It is a guard rail,
not a distance-aware posterior: the honest domain of this model is its transfer
box, which :meth:`DistilledForwardModel.in_transfer_box` reports.

**Not distilled** (loud, per §5.7's own "re-validate against the full ensemble"):

- ``posterior_cov`` is NOT implemented and NOT fakeable — a single net has no
  joint epistemic law over pairs of inputs. The distilled model is therefore NOT
  ``_JointModel``-conformant and CANNOT drive §9 EPIG. Use the ensemble or
  ``sngp_member_view`` for the active loop; this tier is the ``/invert`` inner
  loop and serving path only.
- The Jacobian is inherited implicitly through the mean fit; it is NOT
  Sobolev-supervised (§6.1 would match ``∂P/∂x`` explicitly). It tracks the
  teacher's well enough to drive the §8 δ box in the tests, but that is measured,
  not guaranteed — see ``tests/test_distill.py``.
- Higher moments / member-level disagreement structure are discarded: this is
  moment matching of the two components, not Normal-Wishart EnD².
- The student is a SCREENING surrogate by design. §5.7 requires final candidates
  to be re-validated against the full ensemble + conformal — the §8 solver's
  ``revalidation_model`` is that hook.

Determinism (§13.4): seeded transfer sampling + ``torch.manual_seed``;
``device="cpu"`` (default) is bit-reproducible.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from scipy.stats import qmc
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from rig.forward._gp_common import standardize_stats as _standardize_stats
from rig.forward.data import records_to_arrays
from rig.forward.ensemble import _SpectralResBlock
from rig.interfaces import PredictiveDistribution

# Variance floor in y_scale² units (targets are standardized before the log, so a
# single absolute floor is scale-free here — a raw-SI floor could not be, with
# σ's spanning nanometres to Kelvin across outputs).
_VAR_FLOOR = 1e-12
# Keep exp(log_var) finite for any query the §8 multi-start might throw at us.
_LOG_VAR_CLIP = 60.0
# Cap the out-of-box guard's distance so a wild query inflates hugely but finitely.
_MAX_BOX_EXCESS = 20.0


def _seed_params(module: nn.Module, seed: int) -> None:
    """Xavier-uniform every matrix param, zero every vector param, from ``seed``."""
    gen = torch.Generator().manual_seed(seed + 7919)
    for p in module.parameters():
        if p.requires_grad and p.dim() >= 2:
            nn.init.xavier_uniform_(p, generator=gen)
        elif p.requires_grad:
            nn.init.zeros_(p)


class _DistillTeacher(Protocol):
    """What distillation needs of a teacher: the canonical ForwardModel surface.

    Structural, so ANY §3.2-conformant model distills — the deep ensemble is the
    §5.7 motivation but the GP and the multi-task GP satisfy this too.
    """

    def predict(self, x: np.ndarray) -> PredictiveDistribution: ...

    def support_score(self, x: np.ndarray) -> float | np.ndarray: ...


class _StudentNet(nn.Module):
    """Spectral trunk + four heads: mean, log σ²_ale, log σ²_epi, support.

    The heads are separate all the way down to their targets — the split is a
    structural property of the net, not a post-hoc division of a total. The
    trunk keeps the ensemble tier's spectral normalization: distances in φ may
    not collapse, which is what lets the log-variance heads express a bump in an
    interior hole rather than smoothing across it.
    """

    def __init__(self, d_in: int, m_out: int, width: int, n_blocks: int, seed: int) -> None:
        super().__init__()
        self.input_proj = spectral_norm(nn.Linear(d_in, width))
        self.blocks = nn.ModuleList(_SpectralResBlock(width) for _ in range(n_blocks))
        self.act = nn.GELU()
        self.head_mean = self._head(width, m_out)
        self.head_log_ale = self._head(width, m_out)
        self.head_log_epi = self._head(width, m_out)
        self.head_support = self._head(width, 1)
        _seed_params(self, seed)

    @staticmethod
    def _head(width: int, out: int) -> nn.Module:
        return nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, out))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))
        for blk in self.blocks:
            h = blk(h)
        return h

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phi = self.features(x)
        return (
            self.head_mean(phi),
            self.head_log_ale(phi),
            self.head_log_epi(phi),
            self.head_support(phi),
        )


class DistilledForwardModel:
    """Single distributional net distilled from a ForwardModel teacher (§5.7 A).

    Canonical §3.2 surface, identical to the GP and ensemble tiers, so it is a
    drop-in behind the §5.6 conformal wrapper and the §8 solver:

    - ``predict(x) -> PredictiveDistribution`` (``conformal_set=None``; the §5.6
      wrapper fills it). ``(d,)`` -> fields ``(m,)``; ``(n,d)`` -> ``(n,m)``.
      ``aleatoric_sigma`` and ``epistemic_sigma`` come from two SEPARATE heads
      matched to the teacher's two SEPARATE components — never a split total.
    - ``support_score(x)`` — the teacher's own score, distilled onto a head, so
      it keeps the teacher's numeric scale and a §8.2 ``support_floor`` computed
      against either model means the same thing. Clipped at 0 (the contract:
      negative Mahalanobis, max 0).
    - ``jacobian(x)`` — autograd d(mean head)/dx at a single point, ``(m,d)``,
      raw units.
    - ``update(records)`` — updates the TEACHER and re-distills (needs
      ``input_keys``/``output_keys``); a distilled net cannot learn from data
      directly, its only supervision is the teacher.

    ``posterior_cov`` is deliberately absent (see the module docstring): this
    tier serves the inverse inner loop, not §9 EPIG.

    Construct-then-distill mirrors the fit idiom of the other tiers::

        student = DistilledForwardModel(seed=0).distill(teacher, X_train)

    or use :func:`distill_ensemble`. ``X_train`` is the teacher's training
    inputs: it anchors the transfer box and the on-manifold half of the transfer
    set. Defaults follow §5.7 (AdamW, lr 1e-3 cosine, wd 1e-4, early stop
    patience 30).
    """

    def __init__(
        self,
        *,
        n_transfer: int = 8192,
        box_fraction: float = 0.5,
        dilate: float = 1.0,
        jitter: float = 0.05,
        ood_inflation: float = 1.0,
        width: int = 128,
        n_blocks: int = 2,
        max_epochs: int = 400,
        patience: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        input_keys: Sequence[str] | None = None,
        output_keys: Sequence[str] | None = None,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if not 0.0 <= box_fraction <= 1.0:
            raise ValueError(f"box_fraction must be in [0, 1]; got {box_fraction}")
        if dilate < 0.0:
            raise ValueError(f"dilate must be >= 0; got {dilate}")
        if ood_inflation < 0.0:
            raise ValueError(f"ood_inflation must be >= 0; got {ood_inflation}")
        self.n_transfer = n_transfer
        self.box_fraction = box_fraction
        self.dilate = dilate
        self.jitter = jitter
        self.ood_inflation = ood_inflation
        self.width = width
        self.n_blocks = n_blocks
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.input_keys = list(input_keys) if input_keys is not None else None
        self.output_keys = list(output_keys) if output_keys is not None else None
        self.seed = seed
        self.device = torch.device(device)
        self._net: _StudentNet | None = None
        self._teacher: _DistillTeacher | None = None
        self._X_raw: np.ndarray | None = None

    # -- distillation -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._net is not None

    @property
    def n_train_(self) -> int:
        return 0 if self._X_raw is None else int(self._X_raw.shape[0])

    @property
    def transfer_box(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` of the transfer box in RAW input units — the honest
        domain of this model. Outside it, ``epistemic_sigma``/``support_score``
        are the ``ood_inflation`` guard rail, not distilled values."""
        self._require_fitted()
        lo = self._x_mean + self._x_scale * self._box_lo
        hi = self._x_mean + self._x_scale * self._box_hi
        return lo, hi

    def in_transfer_box(self, x: np.ndarray) -> bool | np.ndarray:
        """Is ``x`` inside the distilled domain? bool for ``(d,)``, ``(n,)`` else."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        inside = np.all((Xs >= self._box_lo) & (Xs <= self._box_hi), axis=1)
        return bool(inside[0]) if single else inside

    def _transfer_inputs(self, Xs: np.ndarray) -> np.ndarray:
        """Standardized transfer inputs: scrambled-Sobol over the dilated box +
        jittered training rows. The box half carries the OOD supervision; the
        jittered half keeps on-manifold density high in dims where a uniform box
        sample is sparse."""
        n, d = Xs.shape
        lo, hi = Xs.min(axis=0), Xs.max(axis=0)
        span = hi - lo
        span = np.where(span > 0.0, span, 1.0)  # a pinned input dim still needs a collar
        self._box_lo = lo - self.dilate * span
        self._box_hi = hi + self.dilate * span

        rng = np.random.default_rng(self.seed)
        n_box = int(round(self.box_fraction * self.n_transfer))
        n_near = self.n_transfer - n_box
        parts = []
        if n_box:
            sampler = qmc.Sobol(d=d, scramble=True, seed=self.seed)
            with warnings.catch_warnings():
                # non-power-of-2 n is a valid scrambled design; only BALANCE needs 2^k
                warnings.simplefilter("ignore", category=UserWarning)
                u = sampler.random(n_box)
            parts.append(qmc.scale(u, self._box_lo, self._box_hi))
        if n_near:
            idx = rng.integers(0, n, size=n_near)
            parts.append(Xs[idx] + self.jitter * rng.standard_normal((n_near, d)))
        return np.vstack(parts)

    def _teacher_targets(self, X_transfer_raw: np.ndarray) -> dict[str, np.ndarray]:
        """Query the teacher's PUBLIC surface and standardize its three fields.

        Aleatoric and epistemic are carried as log-variances in ``y_scale²``
        units: positive by construction on the way back, and log-space is the
        only space in which a σ spanning orders of magnitude is a fair
        regression target (an MSE on σ itself would be dominated by the largest
        values and would let the small, in-distribution epistemic — the
        denominator of every §9 ratio — drift by orders of magnitude for free).
        """
        teacher = self._teacher
        assert teacher is not None
        pred = teacher.predict(X_transfer_raw)
        support = np.asarray(teacher.support_score(X_transfer_raw), dtype=float).reshape(-1, 1)
        mean = np.atleast_2d(np.asarray(pred.mean, dtype=float))
        ale = np.atleast_2d(np.asarray(pred.aleatoric_sigma, dtype=float))
        epi = np.atleast_2d(np.asarray(pred.epistemic_sigma, dtype=float))

        self._y_mean, self._y_scale = _standardize_stats(mean)
        log_ale = np.log(np.maximum((ale / self._y_scale) ** 2, _VAR_FLOOR))
        log_epi = np.log(np.maximum((epi / self._y_scale) ** 2, _VAR_FLOOR))
        self._la_mean, self._la_scale = _standardize_stats(log_ale)
        self._le_mean, self._le_scale = _standardize_stats(log_epi)
        self._sup_mean, self._sup_scale = _standardize_stats(support)
        return {
            "mean": (mean - self._y_mean) / self._y_scale,
            "log_ale": (log_ale - self._la_mean) / self._la_scale,
            "log_epi": (log_epi - self._le_mean) / self._le_scale,
            "support": (support - self._sup_mean) / self._sup_scale,
        }

    def distill(self, teacher: _DistillTeacher, X_train: np.ndarray) -> DistilledForwardModel:
        """Distill ``teacher`` into this net. ``X_train`` = the teacher's training
        inputs (raw units), which anchor the transfer box."""
        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim != 2:
            raise ValueError(f"X_train must be (n, d); got shape {X_train.shape}")
        self._teacher = teacher
        self._X_raw = X_train
        self._x_mean, self._x_scale = _standardize_stats(X_train)
        Xs_train = (X_train - self._x_mean) / self._x_scale

        Xs_t = self._transfer_inputs(Xs_train)
        targets = self._teacher_targets(Xs_t * self._x_scale + self._x_mean)
        self._net = self._train_student(Xs_t, targets)
        return self

    def _train_student(self, Xs_t: np.ndarray, targets: dict[str, np.ndarray]) -> _StudentNet:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed + 1)
        d_in, m_out = Xs_t.shape[1], targets["mean"].shape[1]
        net = _StudentNet(d_in, m_out, self.width, self.n_blocks, self.seed).to(self.device)

        def _t(a: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        n = Xs_t.shape[0]
        perm = rng.permutation(n)
        n_val = max(1, int(round(self.val_fraction * n))) if n >= 10 else 0
        val_idx, tr_idx = perm[:n_val], (perm[n_val:] if n_val else perm)
        X = _t(Xs_t)
        T = {k: _t(v) for k, v in targets.items()}
        itr = torch.as_tensor(tr_idx, device=self.device)
        ival = torch.as_tensor(val_idx, device=self.device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        best_val, best_state, stale = float("inf"), None, 0
        n_tr = int(itr.shape[0])
        for _epoch in range(self.max_epochs):
            net.train()
            order = torch.as_tensor(rng.permutation(n_tr), device=self.device)
            for start in range(0, n_tr, self.batch_size):
                b = itr[order[start : start + self.batch_size]]
                loss = _distill_loss(net(X[b]), {k: v[b] for k, v in T.items()})
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                b = ival if n_val else itr
                vloss = float(_distill_loss(net(X[b]), {k: v[b] for k, v in T.items()}))
            if vloss < best_val - 1e-6:
                best_val, stale = vloss, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.distill_val_loss_ = best_val
        return net

    def update(self, records: Iterable[Any]) -> None:
        """Update the TEACHER on new records, then re-distill (invariant 2d).

        A distilled net has no likelihood of its own — the teacher is its only
        supervision, so "update" can only mean offline re-distillation against an
        updated teacher (§5.7: training stays full-ensemble; the inner loop is
        distilled). Needs ``input_keys``/``output_keys`` to read the new inputs.
        """
        self._require_fitted()
        if self.input_keys is None or self.output_keys is None:
            raise ValueError(
                "update(records) needs input_keys/output_keys at construction to map "
                "RunRecords to arrays; otherwise update the teacher yourself and call "
                "distill(teacher, X_train) again"
            )
        teacher = self._teacher
        if not hasattr(teacher, "update"):
            raise TypeError(
                f"teacher {type(teacher).__name__} has no update(); re-distill from a "
                "freshly-fitted teacher instead"
            )
        teacher.update(records)  # type: ignore[attr-defined]
        X_new, _ = records_to_arrays(records, self.input_keys, self.output_keys)
        self.distill(teacher, np.vstack([self._X_raw, X_new]))

    def _require_fitted(self) -> None:
        if self._net is None:
            raise RuntimeError(
                "DistilledForwardModel is not fitted; call distill(teacher, X_train) first"
            )

    # -- ForwardModel protocol ---------------------------------------------------

    def _box_excess(self, Xs: np.ndarray) -> np.ndarray:
        """L2 distance (standardized input units) from ``Xs`` to the transfer box,
        capped at ``_MAX_BOX_EXCESS``.

        Exactly 0 inside — the guard must never touch a distilled value, or an
        in-domain epistemic test would be measuring the guard, not the student.
        """
        excess = np.maximum(self._box_lo - Xs, 0.0) + np.maximum(Xs - self._box_hi, 0.0)
        return np.minimum(np.linalg.norm(excess, axis=1), _MAX_BOX_EXCESS)

    def _eval_inputs(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(clipped_inputs, guard_distance)`` — never evaluate the heads off the
        distilled domain (see the module docstring: the raw extrapolation is
        arbitrary and reads as CONFIDENT). Inside the box both are a no-op: the
        clip returns ``Xs`` unchanged and the distance is 0."""
        return np.clip(Xs, self._box_lo, self._box_hi), self._box_excess(Xs)

    def _heads(self, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            out = self._net(torch.as_tensor(Xs, dtype=torch.float32, device=self.device))
        return tuple(o.cpu().numpy().astype(np.float64) for o in out)  # type: ignore[return-value]

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        Xc, d_out = self._eval_inputs(Xs)
        mean_s, la_s, le_s, _ = self._heads(Xc)

        log_ale = self._la_mean + self._la_scale * la_s
        log_epi = self._le_mean + self._le_scale * le_s
        # conservative out-of-box guard (0 inside the box); see the module docstring
        log_epi = log_epi + self.ood_inflation * d_out[:, None]
        log_ale = np.clip(log_ale, -_LOG_VAR_CLIP, _LOG_VAR_CLIP)
        log_epi = np.clip(log_epi, -_LOG_VAR_CLIP, _LOG_VAR_CLIP)

        mean = self._y_mean + self._y_scale * mean_s
        aleatoric = self._y_scale * np.exp(0.5 * log_ale)
        epistemic = self._y_scale * np.exp(0.5 * log_epi)
        if single:
            mean, aleatoric, epistemic = mean[0], aleatoric[0], epistemic[0]
        return PredictiveDistribution(
            mean=mean,
            aleatoric_sigma=aleatoric,
            epistemic_sigma=epistemic,
            conformal_set=None,  # filled by the §5.6 calibration wrapper
        )

    def support_score(self, x: np.ndarray) -> float | np.ndarray:
        """The teacher's support score, distilled (same scale, so a §8.2 floor
        transfers between the two), penalized by the out-of-box guard and clipped
        at 0 per the negative-Mahalanobis contract."""
        self._require_fitted()
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        Xs = (np.atleast_2d(x) - self._x_mean) / self._x_scale
        Xc, d_out = self._eval_inputs(Xs)
        _, _, _, sup_s = self._heads(Xc)
        score = (self._sup_mean + self._sup_scale * sup_s)[:, 0]
        score = np.minimum(score - self.ood_inflation * d_out, 0.0)
        return float(score[0]) if single else score

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Autograd d(mean head)/dx at a single point ``(d,)``, raw units ``(m,d)``.

        NOT Sobolev-supervised (§6.1) — it is whatever the distilled mean's slope
        is. OUTSIDE the transfer box this is the slope at the nearest in-box
        point, NOT the derivative of the shipped mean (which the clip in
        :meth:`_eval_inputs` makes locally constant out there, i.e. exactly 0).
        Deliberate: §8.5 spends the Jacobian on a ``Σ_i|J_ji|·Δ_i`` sensitivity
        PENALTY, so handing it a true-but-vacuous 0 would understate the penalty
        and make the solver optimistic off-manifold — the one direction §8 must
        never fail in. The last known slope is the conservative answer.
        """
        self._require_fitted()
        assert self._net is not None
        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("jacobian(x) takes a single point of shape (d,)")
        xs = self._eval_inputs(((x - self._x_mean) / self._x_scale)[None, :])[0][0]
        xt = torch.as_tensor(xs, dtype=torch.float32, device=self.device).requires_grad_(True)
        self._net.eval()
        mean, _, _, _ = self._net(xt.unsqueeze(0))
        mean = mean.squeeze(0)
        m = int(mean.shape[0])
        rows = np.empty((m, xs.shape[0]), dtype=np.float64)
        for j in range(m):
            grad = torch.autograd.grad(mean[j], xt, retain_graph=(j < m - 1))[0]
            # chain: y = y_mean + y_scale·head(x_std), x_std = (x - x_mean)/x_scale
            rows[j] = self._y_scale[j] * grad.detach().cpu().numpy() / self._x_scale
        return rows


def _distill_loss(
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tgt: dict[str, torch.Tensor],
) -> torch.Tensor:
    """MSE against the teacher's standardized (mean, log σ²_ale, log σ²_epi, support).

    Four terms, equally weighted because every target is standardized to unit
    scale (the §5.7 Kendall-style learned head weighting buys nothing once the
    targets are commensurate, and it would give the optimizer a way to trade the
    epistemic head away). The two variance terms are SEPARATE — summing them into
    a total-variance term is the exact defect this module exists to avoid.
    """
    mean, log_ale, log_epi, support = out
    return (
        ((mean - tgt["mean"]) ** 2).sum(dim=-1).mean()
        + ((log_ale - tgt["log_ale"]) ** 2).sum(dim=-1).mean()
        + ((log_epi - tgt["log_epi"]) ** 2).sum(dim=-1).mean()
        + ((support - tgt["support"]) ** 2).sum(dim=-1).mean()
    )


def distill_ensemble(
    teacher: _DistillTeacher, X_train: np.ndarray, **kwargs: Any
) -> DistilledForwardModel:
    """Distill ``teacher`` into a single net (§5.7 option A). ``kwargs`` go to
    :class:`DistilledForwardModel`; ``X_train`` is the teacher's training inputs."""
    return DistilledForwardModel(**kwargs).distill(teacher, X_train)
