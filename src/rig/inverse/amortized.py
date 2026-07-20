"""Amortized inverse posterior — the M3 NPE generator (implementation-plan §14.3, D2).

Neural Posterior Estimation with **conditional neural spline flows** (Durkan et al.
2019), deep-ensembled 5-10 members for epistemic spread (§14.3), implemented on
`zuko` (pure-torch flows — the on-prem-clean, torch-only dependency). This is the
offline "instant-answer" proposal service of D2: it learns the amortized inverse
posterior ``q(recipe | target)`` from simulated ``(recipe, outcome)`` pairs, so a
new target is answered by a forward pass + sampling rather than a fresh
optimization. Per D2 every emitted proposal is then polished by ONE per-query
pessimistic §8 refinement, and **calibration attaches to this proposal (via the
§14.6 SBC/TARP gate) and to the conformally re-validated selected set — never to
the refined output**.

Design decisions (binding):

- **Constraint-by-construction (§14.4).** The flow is trained and sampled in the
  UNCONSTRAINED reparameterized ``u``-space of :class:`~rig.transforms.RecipeTransform`;
  every sample is mapped ``u → recipe`` through the box-sigmoid / simplex-softmax,
  so a proposal is ALWAYS a feasible recipe (box + mixture constraints hold exactly),
  and the flow enjoys full ``ℝ^d`` support (no boundary pathologies).
- **Region-augmented box conditioning (D2).** Ranged box targets are served by the
  region-augmentation trick: the flow conditions on the standardized spec BOX
  ``[lower, upper]`` (``2m`` context), and training draws a random box around each
  simulated outcome (half-widths sampled per example), so ``q(recipe | y ∈ box)`` is
  learned directly — a point target is just a tight box.
- **Deep ensemble (§14.3).** K independent flows (seed diversity); samples are the
  even mixture, ``log_prob`` is the mixture log-density (logsumexp − log K).
- **SBC/TARP is a BLOCKING gate (§14.6).** :meth:`validate` runs Simulation-Based
  Calibration + TARP (the WP-G :mod:`rig.eval.calibration_gates`) on the trained
  flow relative to a supplied generative model; ``no posterior ships until it
  passes``. The gate certifies the posterior only relative to the surrogate's own
  generative model (Caveat 1), not the real machine.

Standardization uses TRAIN statistics only (§5.3). Determinism (§13.4): seeded
per-member torch + numpy RNGs; ``device="cpu"`` (default) is reproducible. torch +
zuko are the ``[torch]`` optional extra, imported lazily by ``rig.inverse.__init__``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import zuko

from rig.eval.calibration_gates import sbc_ranks, sbc_test, tarp_test
from rig.forward._gp_common import standardize_stats as _standardize_stats
from rig.interfaces import CompositionalVariable, ContinuousVariable
from rig.inverse.pessimistic import parse_targets
from rig.transforms import RecipeTransform


@dataclass(frozen=True)
class CalibrationGate:
    """Result of the §14.6 blocking SBC/TARP gate. ``passed`` iff BOTH pass."""

    passed: bool
    sbc_passed: bool
    tarp_passed: bool
    sbc_p_values: list[float]  # per recipe dimension
    tarp_max_calibration_error: float


def _flat_keys(variables: Sequence[ContinuousVariable | CompositionalVariable]) -> list[str]:
    keys: list[str] = []
    for v in variables:
        if isinstance(v, ContinuousVariable):
            keys.append(v.name)
        else:
            keys.extend(f"{v.name}.{c}" for c in v.components)
    return keys


class AmortizedInverseGenerator:
    """Amortized NPE inverse-posterior generator (§14.3, D2 proposal service).

    - ``fit(X, Y)`` trains the ensemble on recipes ``X`` ``(n, d_recipe)`` (columns
      in flat-key order: continuous → ``name``, compositional → ``name.component``)
      and outcomes ``Y`` ``(n, m)``.
    - ``sample(spec, n)`` → ``n`` feasible candidate recipes (dicts) for the spec
      box; ``sample_array`` returns the ``(n, d_recipe)`` matrix.
    - ``log_prob(recipe, spec)`` → mixture log-density.
    - ``validate(simulator, ...)`` → the §14.6 SBC/TARP :class:`CalibrationGate`.

    NB (D2): this is the PROPOSAL. It does not certify feasibility — pass its
    samples to :class:`~rig.inverse.PessimisticInverseSolver` for the per-query
    refinement + the conformal re-validation before shipping a recipe.
    """

    def __init__(
        self,
        variables: Sequence[ContinuousVariable | CompositionalVariable],
        output_keys: Sequence[str],
        *,
        n_members: int = 5,
        transforms: int = 3,
        hidden: tuple[int, ...] = (128, 128),
        bins: int = 8,
        max_epochs: int = 300,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        region_hw: tuple[float, float] = (0.25, 2.0),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.variables = list(variables)
        self.output_keys = list(output_keys)
        self._rt = RecipeTransform(self.variables)
        self._flat_keys = _flat_keys(self.variables)
        self.d_u = self._rt.dim
        self.m = len(self.output_keys)
        self.n_members = n_members
        self.transforms = transforms
        self.hidden = tuple(hidden)
        self.bins = bins
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.region_hw = region_hw
        self.seed = int(seed)
        self.device = torch.device(device)
        self._flows: list[Any] = []
        # per-call draw counter: `_draw_recipes` advances the sampling stream with
        # it, so repeated calls give FRESH draws while a same-seeded fresh
        # generator still replays the identical sequence (§13.4). NB the §14.6
        # `validate` gate seeds itself per trial and does not touch this.
        self._draw_index = 0
        # (audit 2026-07-17: a dead `u_bound` kwarg was removed here — it was
        # assigned and never read. The flow's proposals are feasible-by-
        # construction through `RecipeTransform`, so there is no u-box to bound;
        # the knob only existed to look like the §8 solver's.)

    # -- data mapping -----------------------------------------------------------

    def _x_to_u(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.stack(
            [self._rt.inverse(dict(zip(self._flat_keys, row, strict=True))) for row in X]
        )

    def _u_to_recipes(self, U: np.ndarray) -> list[dict[str, float]]:
        return [self._rt.forward(u) for u in np.asarray(U, dtype=float)]

    def _make_flow(self, seed: int):
        torch.manual_seed(seed)
        return zuko.flows.NSF(
            features=self.d_u,
            context=2 * self.m,
            transforms=self.transforms,
            hidden_features=self.hidden,
            bins=self.bins,
        ).to(self.device)

    # -- fitting ----------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return bool(self._flows)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> AmortizedInverseGenerator:
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y[:, None]
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but Y has {Y.shape[0]}")
        U = self._x_to_u(X)  # (n, d_u), unconstrained
        self._draw_index = 0  # a refit restarts the sampling stream (§13.4 replay)
        self._U_train = U  # retained: the empirical SBC prior (validate default branch)
        self._u_mean, self._u_scale = _standardize_stats(U)
        self._y_mean, self._y_scale = _standardize_stats(Y)
        U_std = (U - self._u_mean) / self._u_scale
        Y_std = (Y - self._y_mean) / self._y_scale

        self._flows = [
            self._train_flow(U_std, Y_std, seed=self.seed + 131 * k) for k in range(self.n_members)
        ]
        return self

    def _train_flow(self, U_std: np.ndarray, Y_std: np.ndarray, seed: int):
        flow = self._make_flow(seed)
        rng = np.random.default_rng(seed)
        u = torch.as_tensor(U_std, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(Y_std, dtype=torch.float32, device=self.device)
        n = u.shape[0]
        opt = torch.optim.AdamW(flow.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.max_epochs)
        hw_lo, hw_hi = self.region_hw
        flow.train()
        for _epoch in range(self.max_epochs):
            order = torch.as_tensor(rng.permutation(n), device=self.device)
            for start in range(0, n, self.batch_size):
                bidx = order[start : start + self.batch_size]
                yb = y[bidx]
                # region augmentation: a random standardized box AROUND each outcome
                hw = torch.as_tensor(
                    rng.uniform(hw_lo, hw_hi, size=yb.shape),
                    dtype=torch.float32,
                    device=self.device,
                )
                ctx = torch.cat([yb - hw, yb + hw], dim=-1)  # (B, 2m)
                loss = -flow(ctx).log_prob(u[bidx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        flow.eval()
        return flow

    def _require_fitted(self) -> None:
        if not self._flows:
            raise RuntimeError("AmortizedInverseGenerator is not fitted; call fit(X, Y)")

    # -- conditioning context ---------------------------------------------------

    def _spec_context(self, spec: Mapping[str, Any]) -> np.ndarray:
        """Standardized box context ``[lower_std, upper_std]`` (2m,) for a spec.

        An open (±inf) side is closed to keep the served box on the
        region-augmentation manifold the flow was TRAINED on: training only ever
        sees symmetric boxes of standardized half-width ``hw ∈ region_hw`` around a
        real outcome (per-output width ``2·hw ≤ 2·region_hw[1]``). So a fully
        unconstrained output is served the widest trained box centered at the
        outcome mean (standardized ``[-hw_max, +hw_max]``), and a one-sided finite
        bound anchors its edge with the open side extended by the max trained WIDTH
        ``2·hw_max``. This keeps the context in-support (a 6·σ clamp gave width-12
        boxes, ~3× the widest trained box) and makes ``lower ≤ upper`` structural —
        an extreme finite bound can no longer invert the box."""
        box = parse_targets(spec["targets"], self.output_keys)
        lo_std = np.full(self.m, -np.inf)
        hi_std = np.full(self.m, np.inf)
        idx = [self.output_keys.index(nm) for nm in box.output_names]
        lo_std[idx] = (box.lower - self._y_mean[idx]) / self._y_scale[idx]
        hi_std[idx] = (box.upper - self._y_mean[idx]) / self._y_scale[idx]
        w = 2.0 * self.region_hw[1]  # max trained standardized box WIDTH
        lo_fin = np.isfinite(lo_std)
        hi_fin = np.isfinite(hi_std)
        both_inf = ~lo_fin & ~hi_fin  # unconstrained output -> widest box at the mean (0)
        lo_std = np.where(both_inf, -0.5 * w, lo_std)
        hi_std = np.where(both_inf, 0.5 * w, hi_std)
        open_hi = lo_fin & ~hi_fin  # lower bound only -> extend up by the max width
        open_lo = ~lo_fin & hi_fin  # upper bound only -> extend down by the max width
        hi_std = np.where(open_hi, lo_std + w, hi_std)
        lo_std = np.where(open_lo, hi_std - w, lo_std)
        return np.concatenate([lo_std, hi_std])  # (2m,)

    # -- sampling ---------------------------------------------------------------

    def _member_counts(self, n: int, rng: np.random.Generator) -> list[int]:
        """How many of ``n`` draws each member contributes — an EVEN mixture in law.

        Audit 2026-07-17 (the small-n mirror of the Session-6 HIGH gate bug): the
        old ``base + (1 if k < rem else 0)`` handed the ``divmod`` remainder to the
        LOW-INDEX members every time, so the shipped draw was a skewed — and for
        ``n < K`` a TRUNCATED — sub-mixture, while ``log_prob`` (logsumexp − log K)
        and the §14.6 gate (``n_posterior=100`` → an exact ``[20]*5``) both assume
        the even one::

            n=3, K=5 → [1,1,1,0,0]   members 3 and 4 could NEVER be drawn
            n=8, K=5 → [2,2,2,1,1]   weights .25/.25/.25/.125/.125, not .2 each

        ``AmortizedRefiner``'s default ``n_proposals=8`` sat exactly on that case:
        the gate certified one law and D2 shipped another. Fix: every member gets
        ``base``, and the ``rem`` leftovers go to ``rem`` DISTINCT members chosen
        uniformly at random from the seeded stream — so each member's expected
        weight is exactly ``1/K`` for every ``n``. When ``K | n`` this is unchanged
        (still exactly ``base`` each), so the gate's stratified ``[20]*5`` — and
        its lower sampling variance — are preserved bit-for-bit.
        """
        base, rem = divmod(n, self.n_members)
        counts = [base] * self.n_members
        if rem:
            for k in rng.choice(self.n_members, size=rem, replace=False):
                counts[int(k)] += 1
        return counts

    def _draw_u_std_mixture(self, ctx: torch.Tensor, n: int, base_seed: int) -> np.ndarray:
        """``(n, d_u)`` standardized u-samples from the even ensemble MIXTURE for a
        context tensor, splitting the draw across EVERY member (``_member_counts``).
        This is the shipped posterior law: drawing all ``n`` from one member would
        sample a single component — narrower than the mixture whenever members
        disagree (the §14.3 epistemic spread), which is exactly what the SBC/TARP
        gate must certify."""
        rng = np.random.default_rng(base_seed)
        parts: list[np.ndarray] = []
        for k, count in enumerate(self._member_counts(n, rng)):
            if count == 0:
                continue
            torch.manual_seed(base_seed + 977 * k + 1)
            with torch.no_grad():
                s = self._flows[k](ctx).sample((count,))  # (count, d_u), standardized
            parts.append(s.cpu().numpy().astype(np.float64))
        return np.concatenate(parts, axis=0)

    def _draw_recipes(
        self, spec: Mapping[str, Any], n: int
    ) -> tuple[list[dict[str, float]], np.ndarray]:
        """Shared draw path for :meth:`sample` / :meth:`sample_array`: one mixture
        draw → ``(recipes, matrix)`` (no redundant u↔recipe round-trip)."""
        self._require_fitted()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        d = len(self._flat_keys)
        if n == 0:  # degenerate 'draw nothing' -> clean empty, not an opaque numpy error
            return [], np.empty((0, d), dtype=float)
        ctx = torch.as_tensor(self._spec_context(spec), dtype=torch.float32, device=self.device)
        # Audit 2026-07-17: ADVANCE the stream per call. This used to pass
        # `self.seed` every time, so a posterior SAMPLER returned bit-identical
        # rows on every call — `2× sample(spec, 3)` gave 6 rows that were 3
        # duplicated pairs, silently collapsing any MC estimate or pooled
        # proposal set built by looping. §13.4 determinism is preserved in the
        # sense that actually matters: a fresh generator with the same seed
        # replays the identical SEQUENCE of draws (see the tests).
        U_std = self._draw_u_std_mixture(ctx, n, self.seed + 7919 * self._draw_index)
        self._draw_index += 1
        U = self._u_mean + self._u_scale * U_std
        recipes = self._u_to_recipes(U)
        return recipes, np.array([[r[k] for k in self._flat_keys] for r in recipes], dtype=float)

    def sample_array(self, spec: Mapping[str, Any], n: int) -> np.ndarray:
        """``(n, d_recipe)`` matrix of feasible recipe vectors (flat-key order)."""
        return self._draw_recipes(spec, n)[1]

    def sample(self, spec: Mapping[str, Any], n: int) -> list[dict[str, float]]:
        """``n`` feasible candidate recipes (dicts) for the spec box (D2 proposal)."""
        return self._draw_recipes(spec, n)[0]

    def log_prob(self, recipe: Mapping[str, float], spec: Mapping[str, Any]) -> float:
        """Mixture log-density ``log q(recipe | box)`` (logsumexp − log K), as a
        density over RECIPE space (per unit of each recipe variable).

        Audit 2026-07-17: this used to return the ``u``-space density and call it
        ``log q(recipe | box)``, excusing the gap as "a monotone reparam of recipe
        space". A monotone reparam preserves the ordering of the VARIABLE, not of
        the DENSITY — the change of variables is mandatory::

            log q(x) = log q_u_std(u_std) − Σ log u_scale + log|det du/dx|

        Without the last two terms the returned values did not integrate to 1
        (measured 3.36 over a 1-D box) and, because ``σ'(u)`` varies ~38× across a
        box, they RE-ORDERED the posterior: 3274 of 400×400 recipe pairs compared
        backwards against the true density. Anything ranking or importance-
        weighting proposals by ``log_prob`` would have been silently wrong.
        """
        self._require_fitted()
        u = self._rt.inverse(recipe)
        u_std = (u - self._u_mean) / self._u_scale
        ctx = torch.as_tensor(self._spec_context(spec), dtype=torch.float32, device=self.device)
        ut = torch.as_tensor(u_std, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lps = torch.stack([flow(ctx).log_prob(ut) for flow in self._flows])
        log_q_u_std = float(torch.logsumexp(lps, dim=0) - np.log(self.n_members))
        log_du_std_du = -float(np.sum(np.log(self._u_scale)))  # u_std = (u − mean)/scale
        return log_q_u_std + log_du_std_du + self._rt.log_abs_det_du_dx(recipe)

    # -- §14.6 SBC/TARP blocking gate -------------------------------------------

    def validate(
        self,
        simulator,
        *,
        prior_sampler=None,
        n_sim: int = 200,
        n_posterior: int = 100,
        hw_std: float = 1.0,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> CalibrationGate:
        """The §14.6 blocking gate: SBC (per-coordinate rank uniformity) + TARP
        (expected-coverage) on the trained flow, relative to ``simulator``
        (``recipe → outcome``). ``prior_sampler() → recipe`` should match the
        training generative process (else the gate is invalid, §14.6); the default
        BOOTSTRAP-resamples the empirical training-u rows — the exact prior the flow
        was trained under (a moment-matched Gaussian would mis-shape a non-Gaussian
        marginal, e.g. logistic-from-uniform-DoE or bimodal, and silently invalidate
        the test).

        The box conditioned on is ``[y − hw_std·σ_y, y + hw_std·σ_y]`` (standardized
        half-width ``hw_std``). SBC/TARP run in u-space (theta_true = the true u;
        posterior = flow u-samples drawn as the SHIPPED even mixture over all
        members, not a single component). A failed gate is a BLOCKING defect."""
        self._require_fitted()
        rng = np.random.default_rng(seed)
        theta_u = np.empty((n_sim, self.d_u))
        post = np.empty((n_sim, n_posterior, self.d_u))
        hw_vec = float(hw_std)
        n_train = self._U_train.shape[0]
        for i in range(n_sim):
            if prior_sampler is not None:
                recipe = prior_sampler()
                u_true = self._rt.inverse(recipe)
            else:  # default prior: bootstrap the empirical training-u marginal (§14.6)
                u_true = self._U_train[rng.integers(n_train)]
                recipe = self._rt.forward(u_true)
            theta_u[i] = u_true
            y = np.atleast_1d(np.asarray(simulator(recipe), dtype=float))
            y_std = (y - self._y_mean) / self._y_scale
            ctx = torch.as_tensor(
                np.concatenate([y_std - hw_vec, y_std + hw_vec]),
                dtype=torch.float32,
                device=self.device,
            )
            # posterior = the SHIPPED mixture (split across all members), not one member
            s = self._draw_u_std_mixture(ctx, n_posterior, base_seed=seed + 100003 * i)
            post[i] = self._u_mean + self._u_scale * s

        ranks = sbc_ranks(theta_u, post)
        sbc_res = sbc_test(ranks, n_posterior, confidence=confidence, seed=seed)
        tarp_res = tarp_test(theta_u, post, confidence=confidence, seed=seed)
        sbc_passed = all(r.passed for r in sbc_res)
        return CalibrationGate(
            passed=bool(sbc_passed and tarp_res.passed),
            sbc_passed=bool(sbc_passed),
            tarp_passed=bool(tarp_res.passed),
            sbc_p_values=[float(r.p_value) for r in sbc_res],
            tarp_max_calibration_error=float(tarp_res.max_calibration_error),
        )
