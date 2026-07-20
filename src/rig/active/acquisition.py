"""Active-learning acquisition — the §9.4 cost-cooled blend (numpy tier).

The load-bearing answer to MFL's "given enough (x,z) pairs": which real
experiment to run next. §9.4 specifies a TWO-PHASE schedule (not one linear
blend), because qLogNEHVI is an expected-hypervolume quantity (outcome-volume
units), NOT nats, and cannot be linearly added to the information terms:

  Phase I (explore):  α(x) = [ λ·EPIG_S(x) + (1−λ)·BALD(x) ] / cost(x)^β
  Phase II (exploit): a SEPARATE qLogNEHVI/cost^β acquisition, chosen by a hand-off

Both Phase-I families are in **nats**, hence linearly blendable; cost enters by
**division / cost-cooling** (CArBO; Lee et al. 2020), never by subtracting
dollars from nats.

- **BALD** (Houlsby et al. 2011), decomposed as ``H[total] − E[H[aleatoric]]``
  — never raw variance (that chases aleatoric noise). Closed form for the
  Gaussian GP predictive: ``0.5·log(σ_total²/σ_ale²)`` per output. Global
  epistemic reduction; dominates early.
- **EPIG** (Bickford Smith et al. 2023): prediction-targeted information about
  outcomes at ``p*`` = the inverse engine's candidate recipes for targets in
  ``S``. The real improvement over MFL — accuracy WHERE the inverse will
  propose. Needs the GP joint posterior covariance (``model.posterior_cov``).
- **β (cost-cooling)** annealed 1→0 (cost-frugal early, cost-agnostic late — the
  CArBO direction); **λ** annealed 0.2→0.9 (BALD→EPIG slide). Fixed ``c_batch``
  does NOT enter the per-recipe ratio (it enters the §11 stop/continue rule).

Phase II (:func:`qlognehvi_phase2`) is the torch/BoTorch WP-E tier: a
feasibility-weighted qLogNEHVI over the per-output *margins* to the spec box,
cost-cooled in log space. It is never blended into Phase I — see its docstring
for the reference-point choice (the one thing that silently distorts a
hypervolume) and for what it deliberately does not do. Phase I is numpy only;
torch/botorch are imported lazily inside Phase II so ``import rig`` stays
torch-free, and the §9.4 hand-off that would *select* Phase II is not yet wired
into ``rig.active.loop``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rig.interfaces import ForwardModel


class _JointModel(Protocol):
    """A ForwardModel that also exposes the joint posterior covariance (GP)."""

    def predict(self, x: np.ndarray) -> Any: ...
    def posterior_cov(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...


def _sigmas(model: ForwardModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(aleatoric, epistemic) sigma at X ``(n, d)`` → each ``(n, m)``."""
    dist = model.predict(np.atleast_2d(np.asarray(X, dtype=float)))
    ale = np.atleast_2d(np.asarray(dist.aleatoric_sigma, dtype=float))
    epi = np.atleast_2d(np.asarray(dist.epistemic_sigma, dtype=float))
    return ale, epi


def bald(model: ForwardModel, X: np.ndarray) -> np.ndarray:
    """BALD (nats), summed over outputs, at each row of ``X`` ``(n, d)`` → ``(n,)``.

    ``H[total] − E[H[aleatoric]] = 0.5·log(σ_total²/σ_ale²) = 0.5·log(1 +
    σ_epi²/σ_ale²)`` per output (the Gaussian closed form; strictly epistemic, it
    does not chase aleatoric noise). Requires a positive aleatoric floor."""
    ale, epi = _sigmas(model, X)
    ale2 = np.maximum(ale**2, 1e-300)
    per_output = 0.5 * np.log1p(epi**2 / ale2)  # (n, m)
    return per_output.sum(axis=1)


def epig(
    model: _JointModel,
    X: np.ndarray,
    X_star: np.ndarray,
) -> np.ndarray:
    """EPIG (nats): expected predictive information about outcomes at the target
    pre-image points ``X_star`` gained by observing each candidate in ``X``.

    For a GP, observing the noisy ``y(x)`` reduces the latent variance at a
    target point ``x*``:
      ``σ²(x*|x) = σ_f²(x*) − Cov(f(x*),f(x))² / (σ_f²(x) + σ_ale²(x))``
    and the information gained about ``f(x*)`` is ``0.5·log(σ_f²(x*)/σ²(x*|x))``.
    ``EPIG_S(x)`` averages this over ``x* ∈ X_star`` and sums over outputs — so
    it rewards runs that sharpen the surrogate exactly where the inverse
    proposes. ``X_star`` are the inverse engine's candidate recipes (§9.4).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_star = np.atleast_2d(np.asarray(X_star, dtype=float))
    ale_x, epi_x = _sigmas(model, X)  # (n, m) at candidates
    var_f_x = epi_x**2  # latent (epistemic) variance at candidates
    obs_var_x = var_f_x + ale_x**2  # observation variance at candidates
    # latent variance at the target points (epistemic only)
    _, epi_star = _sigmas(model, X_star)  # (s, m)
    var_f_star = epi_star**2  # (s, m)

    cov = model.posterior_cov(X, X_star)  # (m, n, s) per output
    m = cov.shape[0]
    n, s = X.shape[0], X_star.shape[0]
    info = np.zeros((n, s))
    for j in range(m):
        c = cov[j]  # (n, s) Cov(f_x, f_x*) for output j
        denom = obs_var_x[:, j][:, None]  # (n, 1)
        reduced = var_f_star[:, j][None, :] - c**2 / np.maximum(denom, 1e-300)  # (n, s)
        reduced = np.maximum(reduced, 1e-300)
        ratio = var_f_star[:, j][None, :] / reduced
        info += 0.5 * np.log(np.maximum(ratio, 1.0))  # info gain >= 0
    return info.mean(axis=1)  # average over target points → (n,)


def anneal(progress: float, start: float, end: float) -> float:
    """Linear anneal from ``start`` (progress 0) to ``end`` (progress 1),
    clamped. λ uses (0.2, 0.9); β uses (1.0, 0.0) — both §9.8 defaults."""
    p = float(np.clip(progress, 0.0, 1.0))
    return (1.0 - p) * start + p * end


def cost_cooled_acquisition(
    model: _JointModel,
    X: np.ndarray,
    X_star: np.ndarray,
    *,
    cost_fn: Callable[[Mapping[str, Any]], float] | None = None,
    recipes: list[Mapping[str, Any]] | None = None,
    lam: float = 0.2,
    beta: float = 1.0,
) -> np.ndarray:
    """Phase-I cost-cooled blend (§9.4): ``[λ·EPIG + (1−λ)·BALD] / cost^β``.

    ``lam`` (0.2→0.9 over the run) slides BALD→EPIG; ``beta`` (1→0) is the CArBO
    cost-cooling exponent. ``cost_fn``+``recipes`` supply the per-recipe variable
    cost (the fixed ``c_batch`` is NOT here — it enters the §11 stop rule); when
    omitted, cost is uniform (β has no effect). Returns per-candidate α ``(n,)``;
    pick argmax (single) or feed to the §9.5 batch selector.
    """
    b = bald(model, X)
    e = epig(model, X, X_star)
    info = lam * e + (1.0 - lam) * b  # nats, linearly blendable
    if cost_fn is not None and recipes is not None:
        cost = np.array([max(float(cost_fn(r)), 1e-300) for r in recipes])
        if cost.shape[0] != info.shape[0]:
            raise ValueError("recipes length must match X rows")
        return info / cost**beta
    return info


# --- Phase II: feasibility-weighted qLogNEHVI toward the spec box (§9.4/§11.3) ---

_MC_SAMPLES = 128  # §8.7 ("128 MC samples")
_REF_PAD_FRAC = 0.10  # §8.7 ("reference point = spec-box nadir + 10%")


@dataclass(frozen=True)
class _SpecSplit:
    """How the §8 spec box maps onto qLogNEHVI's two native slots.

    A **two-sided** target has a finite half-width, so its margin is expressible on
    a tolerance-normalized scale and can be a hypervolume coordinate. A **one-sided**
    target (``upper_spec``-only, e.g. MBE ``slip_max_ratio ≤ 1.0``) has NO finite
    half-width — its margin has no tolerance scale and no finite reference point, so
    it cannot be a hypervolume axis at all (hypervolume is scale-dependent; an
    un-normalized axis would arbitrarily dominate or vanish). It is therefore the
    §11.3 **native outcome constraint** (feasibility weight), which is where it
    belongs anyway. This split is forced by the geometry, not a taste call.
    """

    obj_idx: np.ndarray  # (k,) model-output indices that are HV coordinates
    center: np.ndarray  # (k,) box centers c_j
    halfwidth: np.ndarray  # (k,) box half-widths w_j (> 0, checked by parse_targets)
    con_idx: np.ndarray  # (r,) model-output indices that are outcome constraints
    con_sign: np.ndarray  # (r,) +1 if upper-bounded, -1 if lower-bounded
    con_limit: np.ndarray  # (r,) the finite bound


def _split_spec(box: Any, output_keys: Sequence[str]) -> _SpecSplit:
    keys = list(output_keys)
    obj_i, cen, half, con_i, sgn, lim = [], [], [], [], [], []
    for k, name in enumerate(box.output_names):
        j = keys.index(name)
        lo, hi = float(box.lower[k]), float(box.upper[k])
        if np.isfinite(lo) and np.isfinite(hi):
            obj_i.append(j)
            cen.append(0.5 * (lo + hi))
            half.append(0.5 * (hi - lo))
        elif np.isfinite(hi):
            con_i.append(j), sgn.append(1.0), lim.append(hi)
        else:
            con_i.append(j), sgn.append(-1.0), lim.append(lo)
    return _SpecSplit(
        np.array(obj_i, dtype=int),
        np.array(cen, dtype=float),
        np.array(half, dtype=float),
        np.array(con_i, dtype=int),
        np.array(sgn, dtype=float),
        np.array(lim, dtype=float),
    )


def _pessimistic_view(model: _JointModel, band_shift: np.ndarray, m: int) -> Any:
    """A BoTorch ``Model`` view of a RIG forward model: the joint LATENT posterior.

    Lazily defined (it subclasses ``botorch.models.model.Model``) so ``import rig``
    stays torch-free. ``posterior(X)`` honours BoTorch's contract — for a
    ``batch x q x d`` input it returns a law over ``batch x q x m`` whose covariance
    is **joint across the q points** (independent across outputs, which is exactly
    the GP tier's structure) and independent across t-batches. That joint is
    load-bearing: qLogNEHVI is the *noisy* form, so it samples candidates and
    ``X_baseline`` together and the cross-covariance is what makes the baseline
    Pareto front uncertain rather than a fixed set of points.

    The covariance is EPISTEMIC only (``posterior_cov`` is the latent covariance),
    which is deliberate: the epistemic law is what qLogNEHVI takes the *expectation*
    over — that is the "E" in EHVI. ``band_shift[j]·σ_ale,j(x)`` displaces the mean
    of the CONSTRAINT outputs toward their limit (§8.7 "feed μ−κσ-style estimates");
    objective outputs are left alone because the objective callable applies their
    credited band itself (it is the only one of the two that BoTorch hands ``X``).
    Every output is banded by exactly one mechanism.
    """
    import torch
    from botorch.models.model import Model
    from botorch.posteriors.gpytorch import GPyTorchPosterior
    from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

    class _RigLatentView(Model):
        _num_outputs = m

        @property
        def num_outputs(self) -> int:
            return m

        @property
        def batch_shape(self) -> Any:
            return torch.Size([])

        def posterior(
            self,
            X,
            output_indices=None,
            observation_noise=False,
            posterior_transform=None,
            **kwargs,
        ):  # noqa: ANN001, ANN003
            # qLogNEHVI's path never asks for these, and honouring none of them
            # silently would be a wrong answer rather than an error — in particular
            # observation_noise=True would hand back the LATENT law and quietly drop
            # the aleatoric term the caller asked for.
            if observation_noise or output_indices is not None or posterior_transform:
                raise NotImplementedError(
                    "_RigLatentView serves the plain joint latent posterior only "
                    "(observation_noise/output_indices/posterior_transform unsupported)"
                )
            Xn = np.asarray(X.detach().cpu().numpy(), dtype=float)
            q, d = Xn.shape[-2], Xn.shape[-1]
            batch = Xn.shape[:-2]
            flat = Xn.reshape(-1, d)
            n = flat.shape[0]
            dist = model.predict(flat)
            mean = np.atleast_2d(np.asarray(dist.mean, dtype=float))
            ale = np.atleast_2d(np.asarray(dist.aleatoric_sigma, dtype=float))
            mean = mean + band_shift[None, :] * ale
            # Per-t-batch blocks only: the cross-t-batch covariance is never used and
            # a single posterior_cov over all n points would be an (m, n, n) array —
            # gigabytes at a 256-candidate pool against a ~40-run baseline.
            cov = np.empty((m, n // q, q, q), dtype=float)
            for b in range(n // q):
                cov[:, b] = model.posterior_cov(
                    flat[b * q : (b + 1) * q], flat[b * q : (b + 1) * q]
                )
            cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
            dg = np.arange(q)
            for j in range(m):
                scale = max(float(cov[j][:, dg, dg].max()), 1e-30)
                cov[j][:, dg, dg] += 1e-8 * scale
            mean_t = torch.as_tensor(mean.reshape(*batch, q, m), dtype=torch.double)
            cov_t = torch.as_tensor(cov.reshape(m, *batch, q, q), dtype=torch.double)
            # from_independent_mvns hard-requires >= 2 MVNs. m >= 2 always holds here:
            # the caller rejects a spec with < 2 two-sided targets, and targets are a
            # subset of the model's outputs.
            mvns = [MultivariateNormal(mean_t[..., j], cov_t[j]) for j in range(m)]
            return GPyTorchPosterior(
                distribution=MultitaskMultivariateNormal.from_independent_mvns(mvns)
            )

    return _RigLatentView()


def qlognehvi_phase2(
    model: _JointModel,
    X: np.ndarray,
    spec: Mapping[str, Any],
    output_keys: Sequence[str],
    *,
    X_baseline: np.ndarray,
    cost_fn: Callable[[Mapping[str, Any]], float] | None = None,
    recipes: list[Mapping[str, Any]] | None = None,
    kappa: float = 2.0,
    beta: float = 0.0,
    mc_samples: int = _MC_SAMPLES,
    seed: int = 0,
) -> np.ndarray:
    """Phase-II exploit acquisition (§9.4/§11.3) — feasibility-weighted qLogNEHVI
    toward the spec box, cost-cooled. Returns a **log-scale** score per row of
    ``X`` ``(n, d)`` → ``(n,)``; rank by argmax.

    This is an expected-**hypervolume** quantity, NOT nats, so it is a SEPARATE
    acquisition selected by the §9.4 hand-off — never a term added into
    :func:`cost_cooled_acquisition`'s blend. The two are not commensurable and
    nothing here is meant to be.

    **The objectives are the per-output MARGINS, and that is the whole idea.** A
    RIG spec is a box, not a maximization: there is no "maximize etch rate" axis to
    build a Pareto front over. What competes (§8.7: etch depth vs CD vs uniformity)
    is how much *room to spec* each KPI has, so the hypervolume coordinates are

        m_j(x) = (w_j − |f_j(x) − c_j| − κ·σ_ale,j(x)) / w_j

    for each two-sided target (``c_j`` center, ``w_j`` half-width), maximized
    jointly. ``m_j = 1`` is dead-center with no credited band, ``m_j = 0`` is the
    §8 feasibility boundary (the κ·σ_ale credited band of §8.4 exactly touching a
    box edge), ``m_j < 0`` violates. Sampling ``f_j`` from the latent posterior and
    averaging is the "E" of EHVI.

    **Reference point = the spec-box nadir, padded 10% (§8.7/§11.3), i.e. −0.1 in
    every coordinate.** Three things make this the right choice and each of them is
    a way to get it wrong:

    - It is **absolute** — anchored to the spec, which is a fixed physical
      requirement, so hypervolume means the same thing in batch 1 and batch 9.
      BoTorch's usual ``infer_reference_point`` over observed data would slide the
      origin toward wherever the data happens to sit, silently redefining the
      quantity every refit. §11.3 says "from the spec-box nadir, **not a default**"
      for exactly this reason.
    - It is applied on the **tolerance-normalized** margin, so every coordinate
      shares one scale and a single scalar ref is coherent. On raw margins the
      hypervolume would be dominated by whichever KPI has the largest units — a
      silent, invisible re-weighting of the Pareto front. (Verified: scaling an
      output and its box by 1000× does not move a single rank.)
    - The **10% pad itself is the weakest of the three, and is inherited from the
      plan rather than earned here.** Measured: flipping the pad to 0.0 (ref exactly
      at the nadir) leaves the candidate ORDER unchanged and only compresses the
      score spread ~13.8→11.4 nats-of-log-HV. That is the Log form doing its job —
      with plain qNEHVI a point at the ref would have exactly-zero HVI and everything
      outside the box would tie, which is what the pad classically protects against;
      qLogNEHVI's smoothed relu already keeps an ordering below the ref. So the pad
      is cheap insurance and plan-conformance, NOT a correctness property of this
      implementation. A grossly wrong ref (say −5, or one inferred from data) *is*
      caught by the tests; 0.10-vs-0.0 is not.

    **The Log form is load-bearing, not cosmetic.** Every candidate far outside the
    box has HVI that underflows a float to exactly 0; plain qNEHVI would rank those
    arbitrarily. qLogNEHVI (Ament et al. 2023) keeps a usable ordering, so this
    returns a LOG score — typically negative and unbounded below.

    **Cost-cooling is therefore SUBTRACTIVE here.** §9.4 writes Phase II as
    ``qLogNEHVI(x)/cost(x)^β``, but taking that literally against a log-valued
    acquisition inverts the sign whenever ``log HVI < 0`` (the normal case): dividing
    a negative score by ``cost^β > 1`` *raises* it, i.e. cost-cooling would reward
    expensive runs. The CArBO quantity is a ratio of positives, ``HVI/cost^β``, and
    its log is ``log HVI − β·log cost`` — which is what we return, and which ranks
    identically to the §9.4 ratio because log is monotone.

    ``kappa`` is the §8.4 credited band (default 2.0, matching the §8 solver, so
    Phase II chases the same feasibility the §8 gate will later certify).

    NOT IMPLEMENTED / owed, do not read these in:

    - **The §9.4 hand-off trigger is not wired.** ``rig.active.loop`` still runs
      Phase I for the whole campaign; nothing calls this yet. Deciding *when* to
      switch (R's proposals stable AND ≥ φ of budget consumed) is separate work.
    - **No z_epi worst-case displacement.** §8's solver displaces by ``z_epi·σ_epi``;
      an acquisition must not. Fleeing epistemic uncertainty is the opposite of what
      picking an experiment is for — here the epistemic law is integrated over, and
      pessimism stays where it belongs, in the §8 certifier.
    - **No §8.5 δ-box term** (``Σ|J_ji|·Δ_i``) and **no §8.2 support gate**. This
      acquisition can therefore propose an off-support run — for an explore/exploit
      *proposal* that is a defensible thing to do (it gets queried on the real
      machine, never trusted), but it is NOT the §8 fail-closed guarantee.
    - **Discrete pool, q=1 scoring.** The RIG forward model is numpy, so there is no
      autograd path to ``x`` and no ``optimize_acqf`` (contrast
      ``rig.baselines.BoTorchBO``, which owns a real BoTorch GP and can). This scores
      a pool exactly like :func:`cost_cooled_acquisition` and leaves batching to the
      §9.5 selector — so qLogNEHVI's joint-q Pareto reasoning is unused. NB
      ``select_batch``'s ``w_div`` penalty is ADDITIVE on the score scale, and these
      are logs of order −10..−700, not nats of order 1: reusing the Phase-I
      ``w_div=0.5`` against these scores silently disables diversity entirely.
    """
    import torch
    from botorch.acquisition.multi_objective.logei import (
        qLogNoisyExpectedHypervolumeImprovement,
    )
    from botorch.acquisition.multi_objective.objective import GenericMCMultiOutputObjective
    from botorch.sampling.normal import SobolQMCNormalSampler

    from rig.inverse.pessimistic import parse_targets

    X = np.atleast_2d(np.asarray(X, dtype=float))
    X_baseline = np.atleast_2d(np.asarray(X_baseline, dtype=float))
    if X_baseline.shape[0] == 0:
        raise ValueError(
            "qlognehvi_phase2 needs X_baseline (the already-observed recipes): "
            "qLogNEHVI is the NOISY form and prices improvement against the "
            "posterior Pareto front at those points. Pass the loop's training X."
        )
    box = parse_targets(spec["targets"], output_keys)
    split = _split_spec(box, output_keys)
    if split.obj_idx.size < 2:
        raise ValueError(
            f"qLogNEHVI is a MULTI-objective acquisition, but spec['targets'] has "
            f"{split.obj_idx.size} two-sided target(s) — there is no Pareto trade-off "
            "to explore. One-sided targets are feasibility constraints, not "
            "hypervolume axes (they have no finite half-width to normalize by). "
            "Use Phase I (cost_cooled_acquisition) for a single-KPI spec."
        )

    m = int(np.atleast_2d(np.asarray(model.predict(X[:1]).mean, dtype=float)).shape[1])
    band_shift = np.zeros(m, dtype=float)
    band_shift[split.con_idx] = kappa * split.con_sign

    torch.manual_seed(seed)
    view = _pessimistic_view(model, band_shift, m)

    obj_idx_t = torch.as_tensor(split.obj_idx, dtype=torch.long)
    cen_t = torch.as_tensor(split.center, dtype=torch.double)
    half_t = torch.as_tensor(split.halfwidth, dtype=torch.double)

    # The parameter MUST be named `X`: BoTorch calls `objective(samples, X=X)` by
    # keyword, and it shadows the candidate array of the enclosing scope on purpose
    # — it is the baseline X on the cell-bounds pass and the candidate X on forward.
    def _margin(samples, X=None):  # noqa: ANN001
        if X is None:
            raise RuntimeError("margin objective needs X to evaluate σ_ale(x)")
        Xn = np.asarray(X.detach().cpu().numpy(), dtype=float)
        flat = Xn.reshape(-1, Xn.shape[-1])
        ale = np.atleast_2d(np.asarray(model.predict(flat).aleatoric_sigma, dtype=float))
        ale = ale[:, split.obj_idx].reshape(*Xn.shape[:-1], split.obj_idx.size)
        ale_t = torch.as_tensor(ale, dtype=samples.dtype, device=samples.device)
        f = samples[..., obj_idx_t]
        return (half_t - (f - cen_t).abs() - kappa * ale_t) / half_t

    constraints = None
    if split.con_idx.size:
        # A fixed per-output smoothing scale so `eta` means the same thing on every
        # constraint regardless of its raw units. This only sets how sharply the
        # feasibility weight falls off at the boundary — it does NOT move the
        # boundary, which lives in the band-shifted mean.
        ale_b = np.atleast_2d(np.asarray(model.predict(X_baseline).aleatoric_sigma, dtype=float))
        scales = np.maximum(np.median(ale_b[:, split.con_idx], axis=0), 1e-12)

        def _make_con(j: int, sign: float, limit: float, scale: float):
            def _con(samples):  # noqa: ANN001 — BoTorch contract: negative == feasible
                return sign * (samples[..., j] - limit) / scale

            return _con

        constraints = [
            _make_con(int(j), float(s), float(lim), float(sc))
            for j, s, lim, sc in zip(
                split.con_idx, split.con_sign, split.con_limit, scales, strict=True
            )
        ]

    acqf = qLogNoisyExpectedHypervolumeImprovement(
        model=view,
        ref_point=[-_REF_PAD_FRAC] * int(split.obj_idx.size),
        X_baseline=torch.as_tensor(X_baseline, dtype=torch.double),
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([int(mc_samples)]), seed=seed),
        objective=GenericMCMultiOutputObjective(_margin),
        constraints=constraints,
        prune_baseline=False,
        cache_root=False,  # the low-rank baseline cache assumes a real GPyTorch model
    )
    with torch.no_grad():
        scores = acqf(torch.as_tensor(X, dtype=torch.double).unsqueeze(-2))
    out = np.asarray(scores.detach().cpu().numpy(), dtype=float).reshape(-1)

    if cost_fn is not None and recipes is not None:
        cost = np.array([max(float(cost_fn(r)), 1e-300) for r in recipes])
        if cost.shape[0] != out.shape[0]:
            raise ValueError("recipes length must match X rows")
        return out - beta * np.log(cost)  # == log(HVI / cost^β); see docstring
    return out
