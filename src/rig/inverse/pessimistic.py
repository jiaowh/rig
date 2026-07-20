"""Per-query pessimistic inverse solver — the D2 canonical refiner (implementation-plan §8).

This is the GP-era (numpy/scipy) realization of the §8 objective and its
per-query solver loop (§8.6). It is risk-averse *by construction*: the
optimizer is never rewarded for driving recipes into regions where the
surrogate is confidently wrong.

Design note — how each §8 pessimism channel maps onto the GP backbone
---------------------------------------------------------------------
The plan's §8.1 objective is
``maximize_u  log P̂_lcb(Y ∈ Z* | x) + λ_m · log p̂(x)  s.t. x = g(u)``
with the inner worst case taken over ensemble members (EPISTEMIC) and an
input tolerance box δ ∈ Δ (process variation), crediting only a band
``q̂ = κ·σ`` (ALEATORIC). At the GP tier we realize it as a **robust
worst-case credited interval must lie inside the spec box** (the §8.5
min-max made concrete, its first-order Taylor recovering MFL's Jacobian
gate):

    s_j(x)   = z_epi · σ_epi,j(x)            # epistemic worst-member proxy
             + Σ_i |J_ji(x)| · Δ_i           # worst δ over the ℓ∞ box (§8.5)
    u_hi_j   = (U_j − μ_j(x) − s_j) / σ_ale,j(x)     # standardized room to upper
    u_lo_j   = (μ_j(x) − L_j − s_j) / σ_ale,j(x)     # standardized room to lower

- **Epistemic enters once** (§8.1): via ``s_j`` displacing the mean by
  ``z_epi·σ_epi`` toward each boundary — the GP-tier stand-in for the deep
  ensemble's worst-of-K (WP-E). We deliberately do NOT also subtract a
  standalone ``κ·U_epi`` (the double-count the plan removed).
- **Input tolerance δ** (§8.5): ``Σ_i |J_ji|·Δ_i`` is the exact first-order
  Taylor of ``max_{δ∈Δ} L(x+δ)`` over an anisotropic ℓ∞ box — the analytic
  sensitivity penalty the plan blesses as the fallback when PGD (torch, WP-E)
  is not available. We already have an analytic GP Jacobian, so we use it.
- **Aleatoric credited band** (§8.4): ``κ`` is the credited-*band* multiplier;
  a recipe is **feasible** iff after the worst-case displacement it is at
  least ``κ`` aleatoric-σ inside every boundary: ``min_j min(u_hi_j,u_lo_j) ≥ κ``.
  Betting where the surrogate is uncertain is penalized automatically (larger
  σ_ale ⇒ larger required raw margin).
- **Manifold / anti-reward-hacking** (§8.2): ``log p̂(x)`` is realized as the
  model ``support_score`` (negative Mahalanobis = quadratic log-density proxy,
  the plan's explicit cheap fallback for the GP tier; normalizing-flow
  typicality is WP-E). It enters twice: a soft ``λ_m·support_score`` reward in
  the objective AND a **hard reject** below ``support_floor`` (default = 5th
  percentile of train scores). The hard reject is the defense against the
  §8.2 failure mode where ``σ_epi`` is spuriously small in a far-OOD hole, so
  the solver FAILS CLOSED: a ``support_floor`` (or ``X_train`` to derive it)
  is required, never silently skipped.

Constraint-by-construction (§8.3): box + simplex hold for every ``u`` via
:class:`rig.transforms.RecipeTransform`. Hard-to-change / tool factors are
*conditioning*, never searched — bind the tool with ``model.for_tool(tool_id)``
BEFORE constructing the solver (see WP-I).

Declared couplings (§8.3) — what is enforced, and how
-----------------------------------------------------
The optional ``constraints`` argument is the process's
:class:`rig.constraints.ConstraintSet`. Enforcement is split strictly by what the
reparameterization can already express, and each half says exactly what it is:

- **Box + simplex — exact by construction, never re-enforced, never clipped.**
  ``RecipeTransform`` maps every ``u`` to an in-box, sums-to-one recipe, so these
  declarations cost nothing at solve time. The CONSTRUCTOR verifies each one is
  actually realized by the free variables rather than assuming it: a
  ``BoxConstraint`` narrower than its ``ContinuousVariable`` raises (two sources of
  truth for one interval — the search space is then wrong, and bolting a penalty on
  would hide that rather than fix it); a ``SimplexConstraint`` that no
  ``CompositionalVariable`` realizes raises ``NotImplementedError``, because a
  sum-to-total EQUALITY cannot ride the barrier + reject below — every restart
  would miss it and the solver would abstain unconditionally.
- **Linear couplings — soft barrier + HARD reject** (also a ``BoxConstraint`` on a
  simplex *component*: the transform pins the sum, not the individual shares, so
  that is a coupling too). The barrier is a ``log-sigmoid(τ_c·g)`` term on each
  finite side, ``g`` = slack as a fraction of the constraint's achievable span —
  the same idiom as the feasibility margins above, and it is an **optimization aid
  only**. The safety property is the reject: every survivor is filtered through
  ``ConstraintSet.is_satisfied``, so a violating recipe can never be returned with
  ``feasibility_flag=True`` — at ANY ``constraint_penalty``, including ``0.0``.
  When nothing admissible survives, the verdict is a first-class ``Infeasible``
  naming the violated constraint (see ``_constraint_blocked``), never a clipped
  point; attributing it correctly costs one extra coupling-free multi-start, on
  the abstention path only.
  Two honest limits. (i) A barrier is not a projection, so on a small admissible
  set the multi-start can fail to hold it and abstain — the same false-INFEASIBLE
  class §8.8 warns about, which is why the verdict says so rather than blaming the
  coupling. (ii) It is a smooth hinge, so its pull decays inward but never to zero,
  and the spec-feasible plateau is flat — so a satisfied coupling still COMPRESSES
  the returned pre-image toward its interior (measured: spread 1.40 → 0.28; see
  ``_CONSTRAINT_TAU``), costing §8.7 diversity. An exactly-feasible layer
  (cvxpylayers / DC3 projection) is the torch-era WP-E upgrade that removes both.
- **Monotone declarations — NOT IMPLEMENTED here, and not a solver-side property.**
  A ``MonotoneConstraint`` relates an OUTPUT to an INPUT, so it is not checkable on
  a recipe at all; ``ConstraintSet.validate`` skips it for the same reason. It is a
  shape declaration for the surrogate (§6.3) and NOTHING in this repo consumes it
  yet. A candidate passing ``is_satisfied`` therefore says **nothing** about
  monotonicity — do not read the wiring above as covering it.

``constraints=None`` (the default) is the historical path byte-for-byte: no barrier
term is added to the objective and no reject filter runs.

Non-injectivity (§8.7 cause b): one spec box has many pre-images. We return a
DIVERSE set of feasible recipes via greedy max-min selection in u-space (the
GP-tier stand-in for the k-DPP; qLogNEHVI Pareto handling of *competing* KPIs
is WP-E). An empty pessimistic-feasible set is a **reportable outcome**:
:class:`rig.interfaces.Infeasible` with the nearest achievable point, its
distance-to-feasible, and the per-output spec relaxation that would admit it —
never a clipped point.

Compute (§8.6): the plan's ``R=512 restarts × 300 Adam steps`` is the GPU
distilled-surrogate budget. At the GP tier (CPU, low input dim, smooth
posterior) we use a Sobol multi-start + L-BFGS-B on the smooth objective with
far fewer restarts; restart/step counts are the first knobs to cut.

Dimensionality — measured, not assumed (audit 2026-07-17, finding F9)
---------------------------------------------------------------------
Nothing in this repo had ever run the inverse above **2 input dimensions**, so
"it is dimension-agnostic" was an untested claim. It was then measured against
GROUND TRUTH (solve, then evaluate the TRUE function at the returned recipe —
not the model's own opinion) on a smooth 2-output process with every input dim
active, 12·d Sobol training runs:

    d=2,4,6,8,10,15 → FEASIBLE, and 3/3 returned recipes genuinely in spec.

So the machinery does generalize; what was missing was evidence. Two REAL
dimensional weaknesses did surface, and both are handled here:

1. **The restart budget was fixed at 48** regardless of ``dim`` — dense in 2-D,
   vanishing in 20-D. That degrades into a FALSE ``INFEASIBLE``: we fail to
   *find* a recipe and report that none *exists*, the exact confusion §8.8 is
   built to prevent. Now ``n_restarts=None`` ⇒ ``24·dim`` (== 48 at dim=2, so
   2-D results are unchanged).
2. **Cost grows ~O(d²) and it is the GRADIENT, not the model.** By default
   ``minimize`` is called WITHOUT ``jac``, so SciPy finite-differences the
   objective: ``d+1`` evaluations per gradient step, each running ``predict`` AND
   ``jacobian``. Combined with the (correctly) growing restart budget, measured
   solve time went 2.6 s at d=2 → ~150 s at d=20. ``analytic_grad=True`` replaces
   that with a closed-form gradient — see below. It is OPT-IN, so the default path
   is untouched.

Analytic objective gradient (``analytic_grad=True``, opt-in) — 2026-07-17
------------------------------------------------------------------------
``analytic_grad=True`` differentiates the objective in closed form and passes
``jac=True`` to L-BFGS-B, so a gradient costs ONE model evaluation instead of
``d+1``. The chain is ``u → x`` (the ``RecipeTransform`` sigmoid/softmax
derivative) → ``μ, σ_ale, σ_epi, J, support`` → margins → objective. What each
link needs, and where it comes from:

- ``∂μ/∂x`` — the model's analytic Jacobian ``J`` (already used by the δ term).
- ``∂σ_epi/∂x`` — the derivative of the GP posterior std. NOT on the
  ``ForwardModel`` protocol, so it is computed here from the fitted GP's own
  state: ``∂var/∂x = −2·(∂k_*/∂x)ᵀ K⁻¹ k_*``.
- ``∂J/∂x`` — the HESSIAN of the GP posterior mean, needed only because the §8.5
  δ term carries ``Σ_i |J_ji|·Δ_i``. Closed form for Matérn-5/2 (smooth at r=0;
  the ``A'(r)/r`` factor cancels the apparent 1/r singularity). Skipped entirely
  when ``delta_frac == 0``.
- ``∂support/∂x`` — the negative-Mahalanobis derivative, ``−C⁻¹x_s/‖x_s‖_C``.
- ``∂σ_ale/∂x = 0`` — the GP tier's aleatoric noise is a fitted CONSTANT per
  output (§10.3 floor v0). This is a property of the GP backend, not of the
  objective; it is asserted here rather than assumed.

**Why it is GP-only, and why it fails loud rather than falling back.** Those
derivatives are not expressible through the ``ForwardModel`` protocol, so the
provider reads the fitted GP directly (unwrapping ``.base`` chains such as the
conformal wrapper, whose μ/σ pass through unchanged). Any other backend — the
deep ensemble, the multi-task ICM model — RAISES at construction instead of
silently reverting to finite differences: a silent revert would turn a 10× cost
claim into a lie that nothing could detect. The torch tier should use autograd,
not this.

**Non-smooth points are real and are NOT hidden.** ``|J|`` kinks at ``J=0``,
``clip(margin, None, 10)`` kinks at 10, and ``support`` kinks at the training
mean. The objective ALREADY had these kinks — finite differences merely smeared
them. The analytic gradient returns a valid one-sided subgradient there; L-BFGS-B
tolerates that as it did before. This is why the verification below samples random
points and reports a max relative error rather than asserting smoothness.

Verified against central finite differences at 1e-6 relative tolerance across
d ∈ {1..20}, with and without the δ/Hessian term, with and without couplings, and
on the ``ignore_epi`` probe objective — see
``tests/test_inverse.py::test_analytic_gradient_matches_finite_differences``.
Measured max relative error and speedup are recorded there.

Beyond ~20-D at ~12·d runs the solver returns ``INFEASIBLE`` with the
**epistemic-limited** diagnosis (§8.8 cause b) — "the target is reachable, you
have not earned it yet; collect runs". That is the honest verdict for a sparse
high-dim design, not a failure — but see the note in (1): always check the
restart budget before believing a high-dim abstention.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm, qmc

from rig.constraints import ConstraintSet
from rig.forward._gp_common import SQRT5 as _SQRT5
from rig.forward.gp import GPForwardModel
from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
    ForwardModel,
    Infeasible,
    InverseResult,
    RecipeCandidate,
)
from rig.transforms import _U_CLIP as _RT_U_CLIP
from rig.transforms import RecipeTransform

# absolute tolerance on the "≥ κ σ_ale margin" feasibility comparison, so a
# recipe sitting EXACTLY on the κ bar is not flipped to infeasible by float
# round-off (a measure-zero knife-edge, never a real spec).
_FEAS_TOL = 1e-9

# two recipes closer than this in normalized recipe space (fraction of each
# variable's range) are treated as the same pre-image point by the diversity
# selection — so the returned set never pads with near-duplicates.
_DEDUP_RADIUS = 0.02

# §8.6 multi-start budget, per unit of SEARCH dimension (audit 2026-07-17, F9).
# The budget used to be a FIXED 48 regardless of `dim`, which is dense in 2-D and
# vanishing in 20-D — a silent search-quality cliff, and one that surfaces as a
# FALSE `INFEASIBLE` (we failed to FIND the recipe, and report that no recipe
# exists). 24/dim reproduces the historical 48 EXACTLY at dim=2, so every
# existing 2-D result stands unchanged, while higher-dim problems get a budget
# that grows with the space. `_MIN_RESTARTS` keeps 1-D honest.
_RESTARTS_PER_DIM = 24
_MIN_RESTARTS = 48

# §8.3 coupling-barrier sharpness. Deliberately NOT `self.tau`: that τ acts on margins
# measured in aleatoric σ, while a coupling's slack is measured as a fraction of the
# constraint's own achievable span — different units, so sharing a scale would be a
# coincidence, not a design.
#
# It is also a §8.7 DIVERSITY knob, and the cost is measured, not hypothetical. A
# log-sigmoid is a SMOOTH hinge, so its pull decays exponentially inward but never to
# exactly zero — and the spec-feasible plateau is FLAT (the margin reward saturates past
# κ), so that residual pull is the only gradient along it and slides the whole pre-image
# toward the constraint's interior. On a line pre-image whose admissible segment spans
# a≈[3.5,5] (`test_constraint_barrier_does_not_collapse_preimage_diversity`):
#   τ_c=20 → the returned set COLLAPSES to a single corner point (4 candidates → 1);
#   τ_c=60 → 4 candidates, spread 0.28 (the unconstrained pre-image spreads 1.40).
# 60 is therefore the measured floor for keeping the set diverse — NOT a bias-free
# choice: the returned pre-image is still compressed toward the interior. Only a
# projection (cvxpylayers / DC3, WP-E) removes that bias rather than shrinking it.
_CONSTRAINT_TAU = 60.0

# tolerance on the constructor's "the transform already implies this box" check —
# an agreement test on declared bounds, not a runtime feasibility decision.
_BOX_AGREE_TOL = 1e-9

# ---------------------------------------------------------------------------
# spec box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpecBox:
    """Parsed per-output target box (implementation-plan §8.1: Z* is a box, not a point).

    ``lower``/``upper`` are aligned to ``output_names`` (a subset of the
    model's outputs); an unconstrained side is ``-inf`` / ``+inf``.
    """

    output_names: tuple[str, ...]
    lower: np.ndarray  # (m_spec,)
    upper: np.ndarray  # (m_spec,)


def parse_targets(targets: Mapping[str, Any], output_keys: Sequence[str]) -> SpecBox:
    """Parse a ``spec['targets']`` mapping into a :class:`SpecBox`.

    Each value may be:
      * ``(lower, upper)`` tuple/list (either may be ``None``/``±inf``),
      * ``{"lower": L, "upper": U}`` (either key optional),
      * ``{"target": t, "tol": d}`` → box ``[t-d, t+d]`` (``d`` MUST be > 0).
    A zero-width box (``lower == upper``, e.g. a bare ``{"target": t}`` with no
    ``tol``) is REJECTED: no recipe can hold a κ·σ robustness margin around a
    single point, so it would be reported infeasible unconditionally — we fail
    loud with an actionable message instead of silently returning INFEASIBLE.
    Output names must be a subset of ``output_keys`` (the model's output order).
    """
    known = list(output_keys)
    names: list[str] = []
    los: list[float] = []
    his: list[float] = []
    for name, val in targets.items():
        if name not in known:
            raise KeyError(f"spec target {name!r} is not a declared output (known: {known})")
        lo, hi = _parse_one_target(name, val)
        if not lo <= hi:
            raise ValueError(f"target {name!r}: lower {lo} must be <= upper {hi}")
        if not np.isfinite(lo) and not np.isfinite(hi):
            raise ValueError(f"target {name!r} constrains neither bound")
        if lo == hi:
            raise ValueError(
                f"target {name!r}: zero-width spec box [{lo}, {hi}] can never "
                "hold a κ·σ robustness margin. Give a tolerance, e.g. "
                f"{{'target': {lo}, 'tol': d}} with d > 0, or a "
                "[lower, upper] with lower < upper."
            )
        names.append(name)
        los.append(lo)
        his.append(hi)
    if not names:
        raise ValueError("spec['targets'] is empty; nothing to solve for")
    return SpecBox(tuple(names), np.asarray(los, float), np.asarray(his, float))


def _parse_one_target(name: str, val: Any) -> tuple[float, float]:
    if isinstance(val, Mapping):
        if "target" in val:
            t = float(val["target"])
            d = float(val.get("tol", 0.0))
            if d < 0:
                raise ValueError(f"target {name!r}: tol must be >= 0")
            return t - d, t + d
        lo = val.get("lower", None)
        hi = val.get("upper", None)
        return (
            -np.inf if lo is None else float(lo),
            np.inf if hi is None else float(hi),
        )
    if isinstance(val, (tuple, list)) and len(val) == 2:
        lo, hi = val
        return (
            -np.inf if lo is None else float(lo),
            np.inf if hi is None else float(hi),
        )
    raise TypeError(
        f"target {name!r}: expected (lower, upper), "
        "{'lower':..,'upper':..} or {'target':..,'tol':..}, "
        f"got {val!r}"
    )


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Terms:
    """The forward model's predictive terms AND their first derivatives w.r.t. the
    raw recipe vector ``x``, at one point (internal; the analytic-gradient path).

    The values agree with what ``ForwardModel.predict`` / ``.jacobian`` /
    ``.support_score`` return for the same point to ~1e-12 relative — NOT bitwise:
    the same quantities are formed by an equivalent but differently-associated set
    of floating-point operations (e.g. ``A·(D/ell²)`` here vs ``(A·D)/ell²`` in
    ``matern52_grad_x``). That gap is ~1e4 times smaller than the FD noise the
    default path already carries, but it is why the two solver paths are equal in
    verdict rather than bitwise. A provider that disagreed by MORE would move the
    ANSWER and not just the speed, so the agreement is pinned by
    ``test_analytic_grad_terms_agree_with_the_models_public_methods``.
    """

    mu: np.ndarray  # (m,)
    sig_ale: np.ndarray  # (m,)
    sig_epi: np.ndarray  # (m,)
    jac: np.ndarray  # (m, d)  = ∂μ/∂x
    support: float
    d_sig_epi: np.ndarray  # (m, d)
    d_sig_ale: np.ndarray  # (m, d)
    d_support: np.ndarray  # (d,)
    hess: np.ndarray | None  # (m, d, d) = ∂²μ/∂x∂x, None when δ is disabled


def _unwrap_gp(model: ForwardModel) -> GPForwardModel | None:
    """Find the fitted :class:`~rig.forward.gp.GPForwardModel` behind ``model``, or
    ``None``. Wrappers that leave μ/σ untouched and only ADD to the
    ``PredictiveDistribution`` (the §5.6 conformal wrapper) expose the wrapped model
    as ``.base``, and the §8 margins read only μ/σ_ale/σ_epi — so differentiating the
    base is differentiating what the margins actually consume. A ``.base`` chain that
    ends anywhere else (the ICM multi-task model, the deep ensemble) yields ``None``
    and the caller must fail loud."""
    seen = 0
    while model is not None and seen < 8:
        if isinstance(model, GPForwardModel):
            return model if model.is_fitted else None
        model = getattr(model, "base", None)
        seen += 1
    return None


class _GPTermProvider:
    """Analytic ∂/∂x of a fitted GP's predictive terms (the ``analytic_grad`` path).

    Lives here rather than on :class:`~rig.forward.gp.GPForwardModel` because these
    derivatives are not part of the §3.2 ``ForwardModel`` protocol and only the §8
    objective needs them. That means it reads the GP's fitted state directly; the
    coupling is deliberate and narrow, and every formula below is verified against
    central finite differences of the GP's OWN public methods.
    """

    def __init__(self, gp: GPForwardModel, *, need_hessian: bool) -> None:
        self.gp = gp
        self.need_hessian = bool(need_hessian)
        # §10.3 floor v0: one fitted noise scalar per output, so ∂σ_ale/∂x ≡ 0. The
        # analytic gradient DEPENDS on that; if the backend ever learns an
        # input-dependent noise this must grow a term rather than silently drop one.
        self._sig_ale = np.asarray(gp.noise_std_, dtype=float)

    def terms(self, x: np.ndarray) -> _Terms:
        gp = self.gp
        xs = (np.asarray(x, dtype=float) - gp._x_mean) / gp._x_scale
        d = xs.size
        m = len(gp._gps)
        mu = np.empty(m)
        sig_epi = np.empty(m)
        jac = np.empty((m, d))
        d_sig_epi = np.zeros((m, d))
        hess = np.empty((m, d, d)) if self.need_hessian else None

        for j, (g, ym, ys) in enumerate(zip(gp._gps, gp._y_mean, gp._y_scale, strict=True)):
            h = g.hyper
            diff = xs[None, :] - g.X  # (n, d)
            r2 = np.sum((diff / h.ell) ** 2, axis=1)
            r = np.sqrt(np.maximum(r2, 0.0))
            E = np.exp(-_SQRT5 * r)
            Ks = h.sf2 * (1.0 + _SQRT5 * r + (5.0 / 3.0) * r2) * E  # (n,)
            # A(r) is the scalar factor of ∂k/∂x: ∂k/∂x_a = A(r)·diff_a/ell_a².
            A = -(5.0 / 3.0) * h.sf2 * (1.0 + _SQRT5 * r) * E  # (n,)
            P = diff / (h.ell**2)[None, :]  # (n, d)
            dKs = A[:, None] * P  # (n, d)

            mu[j] = ym + ys * float(Ks @ g._alpha)
            jac[j] = ys * (g._alpha @ dKs) / gp._x_scale

            V = solve_triangular(g._L, Ks[:, None], lower=True)  # (n, 1)
            var = max(float(h.sf2 - np.sum(V * V)), 0.0)
            sig_epi[j] = ys * np.sqrt(var)
            if var > 0.0:
                # var = sf2 − k_*ᵀK⁻¹k_*  ⇒  ∂var/∂x = −2·(∂k_*/∂x)ᵀ K⁻¹k_*.
                w = cho_solve((g._L, True), Ks)  # (n,)
                d_var = -2.0 * (dKs.T @ w)  # (d,)
                d_sig_epi[j] = ys * d_var / (2.0 * np.sqrt(var)) / gp._x_scale
            # var == 0 is the σ_epi=0 cusp of √var (unreachable with a fitted noise
            # floor). Leave the row at 0: a subgradient, not a NaN.

            if hess is not None:
                # ∂²k/∂x_a∂x_b = (25/3)sf2·e^{−√5 r}·P_a·P_b + A(r)·δ_ab/ell_a².
                # The A'(r)·∂r/∂x_b product carries a 1/r that A'(r) ∝ r cancels, so
                # this is smooth at r=0 — no singularity at a training point.
                c = (25.0 / 3.0) * h.sf2 * E * g._alpha  # (n,)
                Hs = P.T @ (c[:, None] * P)
                Hs[np.diag_indices(d)] += float(g._alpha @ A) / (h.ell**2)
                hess[j] = ys * Hs / np.outer(gp._x_scale, gp._x_scale)

        d2 = float(xs @ gp._support_cov_inv @ xs)
        support = -np.sqrt(max(d2, 0.0))
        if d2 > 0.0:
            d_support = -(gp._support_cov_inv @ xs) / (np.sqrt(d2) * gp._x_scale)
        else:
            # the Mahalanobis cone tip (x at the training mean) — support_score's
            # maximum and a genuine kink; 0 is the valid subgradient.
            d_support = np.zeros(d)

        return _Terms(
            mu=mu,
            sig_ale=self._sig_ale.copy(),
            sig_epi=sig_epi,
            jac=jac,
            support=float(support),
            d_sig_epi=d_sig_epi,
            d_sig_ale=np.zeros((m, d)),
            d_support=d_support,
            hess=hess,
        )


@dataclass(frozen=True)
class _LinRow:
    """One declared coupling the §8.3 reparameterization cannot express, normalized
    (internal): ``lower <= coef @ x[idx] <= upper`` on the flat recipe vector.

    ``scale`` is the row's achievable span over the search box, so slacks from rows
    of wildly different magnitude share one barrier sharpness.
    """

    idx: np.ndarray
    coef: np.ndarray
    lower: float
    upper: float
    scale: float


@dataclass
class _Restart:
    """One converged multi-start result (internal)."""

    u: np.ndarray
    x: np.ndarray
    recipe: dict[str, float]
    u_hi: np.ndarray  # (m_spec,) standardized room to upper (worst-cased)
    u_lo: np.ndarray  # (m_spec,) standardized room to lower (worst-cased)
    margin: float  # min_j min(u_hi_j, u_lo_j)
    confidence: float  # Π_j pessimistic per-output spec-hit probability
    support: float
    mean: np.ndarray  # (m,) predicted mean, raw units
    sig_ale: np.ndarray  # (m,) aleatoric sigma, raw units
    scale_spec: np.ndarray  # (m_spec,) floored σ scale the margins used
    interval: np.ndarray  # (m_spec, 2) worst-case credited outcome interval
    margin_no_epi: float  # robust margin with the epistemic displacement removed


class PessimisticInverseSolver:
    """Per-query pessimistic inverse (implementation-plan §8). Implements
    :class:`rig.interfaces.InverseSolver`.

    Parameters
    ----------
    model
        A fitted :class:`~rig.interfaces.ForwardModel`. For a tool-aware model
        bind the tool FIRST via ``model.for_tool(tool_id)`` (§8.3: recipes are
        generated *given* a tool; never search over tool_id).
    variables
        The free RECIPE variables (continuous / compositional), in the SAME
        order the model's ``predict`` expects its input vector. Categorical /
        hard-to-change factors are conditioning and are excluded (§8.3).
    output_keys
        The model's output names, in column order — used to map spec targets to
        output indices.
    kappa
        Credited-band multiplier (§8.4). Feasible iff every boundary has ≥ κ
        aleatoric-σ of worst-case margin. Default 2.0 (≈ one-sided 97.5%).
    z_epi
        Epistemic worst-member multiplier (mean displaced by ``z_epi·σ_epi``).
        Default 2.0.
    delta_frac
        Input-tolerance box half-width as a fraction of each variable's range
        (Gage R&R / tool repeatability, §8.5). Default 0.02. Set 0 to disable
        the ``‖J‖`` term (skips the Jacobian call).
    lambda_m
        Soft manifold-reward weight on ``support_score`` (§8.1). Default 0.3.
    support_floor / X_train
        Hard manifold reject threshold (§8.2). Provide the float directly, OR
        ``X_train`` to derive it as the 5th percentile of train support scores.
        One of the two is REQUIRED (fail-closed anti-reward-hacking).
    constraints
        The process's declared :class:`~rig.constraints.ConstraintSet` (§8.3), or
        ``None`` (default) for the historical, byte-for-byte-unchanged path: no
        barrier term and no reject filter. When given, box/simplex declarations are
        VERIFIED here against the free variables (they are exact by construction and
        are never re-enforced or clipped — a disagreement raises); linear couplings,
        and a box on a simplex component, get the barrier + hard reject. Monotone
        declarations are NOT enforced — see the module docstring.
    constraint_penalty
        Weight of the §8.3 coupling barrier. Default 10.0. This is an OPTIMIZATION
        AID ONLY: the hard reject is what guarantees every returned candidate
        satisfies ``ConstraintSet.is_satisfied``, and that holds at
        ``constraint_penalty=0.0``. Turning it down costs search quality (more
        restarts converge outside the admissible set and are thrown away), never
        safety.
    analytic_grad
        OPT-IN (default ``False``). ``False`` is the historical path byte-for-byte:
        ``minimize`` runs without ``jac`` and SciPy finite-differences the objective
        at ``d+1`` model evaluations per gradient step. ``True`` supplies the
        closed-form gradient instead (ONE evaluation per step) — see the module
        docstring for the derivation, its GP-only scope, and its verification. It
        changes the SEARCH PATH, not the objective: L-BFGS-B takes different steps
        with an exact gradient than with a finite-difference approximation of it, so
        results are equal in VERDICT and recipe to FD tolerance, not bitwise. That is
        why it is opt-in and why the default stays FD — every published 2-D number
        (M2, the AL loop) was produced on the FD path and must keep reproducing there.
        Raises at construction when the model is not a (possibly wrapper-wrapped)
        fitted ``GPForwardModel``: silently reverting to FD would make the speedup
        claim unfalsifiable.
    n_restarts, max_iter, u_bound, seed
        Multi-start budget and reparameterization bounds (§8.6).
        ``n_restarts=None`` (the default) scales the budget with the search
        dimension: ``_RESTARTS_PER_DIM * dim``, floored at ``_MIN_RESTARTS``.
        See the class docstring's dimensionality note for why a FIXED budget is
        a silent quality cliff as ``dim`` grows. At ``dim=2`` this evaluates to
        exactly 48 — the historical default — so every 2-D result (M2, the AL
        loop) is bit-for-bit unchanged. Pass an int to override.
    """

    def __init__(
        self,
        model: ForwardModel,
        variables: Sequence[ContinuousVariable | CompositionalVariable],
        output_keys: Sequence[str],
        *,
        kappa: float = 2.0,
        z_epi: float = 2.0,
        delta_frac: float = 0.02,
        lambda_m: float = 0.3,
        support_floor: float | None = None,
        X_train: np.ndarray | None = None,
        constraints: ConstraintSet | None = None,
        constraint_penalty: float = 10.0,
        revalidation_model: ForwardModel | None = None,
        analytic_grad: bool = False,
        n_restarts: int | None = None,
        max_iter: int = 100,
        u_bound: float = 8.0,
        tau: float = 4.0,
        seed: int = 0,
    ) -> None:
        self.model = model
        # §5.7 / §8 inner-loop budget: `model` may be a FAST screening surrogate
        # (e.g. a deep-ensemble SNGP single member). When `revalidation_model` is
        # set (the full ensemble, optionally conformal-wrapped), solve() re-scores
        # the selected candidates against it and drops any the full model does not
        # certify — "the inner loop is distilled; final candidates re-validated on
        # the full ensemble + conformal". Default None ⇒ no re-validation, byte-for
        # -byte identical to the single-model path (the GP tier + the M2 harness).
        self.revalidation_model = revalidation_model
        self.variables = list(variables)
        self.output_keys = list(output_keys)
        self.kappa = float(kappa)
        self.z_epi = float(z_epi)
        self.delta_frac = float(delta_frac)
        self.lambda_m = float(lambda_m)
        self.max_iter = int(max_iter)
        self.u_bound = float(u_bound)
        self.tau = float(tau)
        self.seed = int(seed)

        self._rt = RecipeTransform(self.variables)
        # Multi-start budget scales with the SEARCH dimension (see the class
        # docstring). `_rt.dim` is the u-space dimension — the number of free
        # coordinates the optimizer actually explores (K-1 per simplex), which is
        # what governs how sparse a fixed Sobol budget becomes.
        self.n_restarts = (
            int(n_restarts)
            if n_restarts is not None
            else max(_MIN_RESTARTS, _RESTARTS_PER_DIM * self._rt.dim)
        )
        self._flat_keys = self._build_flat_keys()
        self._delta_raw = self._build_delta_raw()
        # per-flat-key divisor so recipe-space distances are comparable across
        # variables of different scale (continuous → box range; compositional
        # components already live on [0,1]). Used by the diversity selection.
        self._flat_scale = self._build_flat_scale()
        self._flat_bounds = self._build_flat_bounds()

        # §8.3 declared couplings. `_penalty_rows` is EMPTY unless a declaration
        # exists that the reparameterization cannot express, so the default path
        # adds no term to the objective at all.
        self.constraints = constraints
        self.constraint_penalty = float(constraint_penalty)
        self._penalty_rows = self._bind_constraints(constraints)

        # opt-in analytic objective gradient. `None` ⇒ the FD path, untouched.
        self.analytic_grad = bool(analytic_grad)
        self._terms = self._bind_gradients() if self.analytic_grad else None

        # hard manifold floor (§8.2) — fail closed.
        self._X_train = None if X_train is None else np.asarray(X_train, dtype=float)
        if support_floor is None:
            if self._X_train is None:
                raise ValueError(
                    "PessimisticInverseSolver requires support_floor or X_train "
                    "to set the §8.2 manifold reject threshold (fail-closed "
                    "anti-reward-hacking); refusing to run without it."
                )
            support_floor = self._support_floor_of(self.model)
        self.support_floor = float(support_floor)

    def _support_floor_of(self, model: ForwardModel) -> float:
        """5th-percentile support-score floor of ``model`` over the train set. Used
        both for ``self.model`` (the fast surrogate) and, per-model, for the
        re-validation model — support_score is a per-model quantity, so the floor
        must be derived on the SAME model it gates (else a differently-scaled model
        is judged against the wrong threshold)."""
        scores = np.atleast_1d(model.support_score(self._X_train))
        return float(np.percentile(scores, 5.0))

    # -- construction helpers ---------------------------------------------------

    def _build_flat_keys(self) -> list[str]:
        keys: list[str] = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                keys.append(v.name)
            elif isinstance(v, CompositionalVariable):
                keys.extend(f"{v.name}.{c}" for c in v.components)
            else:  # pragma: no cover - RecipeTransform already rejects these
                raise TypeError(f"unsupported free variable {v!r}")
        return keys

    def _build_delta_raw(self) -> np.ndarray:
        """Per-input tolerance half-widths Δ_i in raw units (§8.5)."""
        delta: list[float] = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                delta.append(self.delta_frac * (v.upper - v.lower))
            else:  # compositional components live on the simplex, range ~1
                delta.extend([self.delta_frac] * len(v.components))
        return np.asarray(delta, float)

    def _build_flat_scale(self) -> np.ndarray:
        """Per-flat-key divisor for comparable recipe-space distances."""
        scale: list[float] = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                scale.append(v.upper - v.lower)
            else:
                scale.extend([1.0] * len(v.components))
        return np.asarray(scale, float)

    def _build_flat_bounds(self) -> list[tuple[float, float]]:
        """Per-flat-key (lower, upper) reachable range — the IMAGE of
        :class:`~rig.transforms.RecipeTransform`, not a declaration. Simplex shares
        live on [0, 1] whatever the coupling says."""
        bounds: list[tuple[float, float]] = []
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                bounds.append((float(v.lower), float(v.upper)))
            else:
                bounds.extend([(0.0, 1.0)] * len(v.components))
        return bounds

    def _bind_gradients(self) -> _GPTermProvider:
        """Resolve the analytic-gradient provider, or refuse. The refusal is the
        point: a fallback to finite differences here would be invisible — same
        answers, silently none of the speedup — so an unsupported backend is a
        construction-time error, not a runtime shrug."""
        gp = _unwrap_gp(self.model)
        if gp is None:
            raise ValueError(
                "analytic_grad=True needs the objective's ∂σ_epi/∂x and (when "
                "delta_frac > 0) the Hessian ∂J/∂x, which are not on the §3.2 "
                f"ForwardModel protocol; {type(self.model).__name__} is not a fitted "
                "GPForwardModel (nor a wrapper around one), so they cannot be formed. "
                "Use analytic_grad=False (finite differences), or the torch tier's "
                "autograd for a torch backend — do NOT read this as 'the gradient is "
                "unavailable so the solve is wrong': the FD path is correct, only "
                "slower."
            )
        return _GPTermProvider(gp, need_hessian=self.delta_frac > 0.0)

    def _norm_x(self, x: np.ndarray) -> np.ndarray:
        """Recipe vector scaled to comparable per-dim units (for diversity)."""
        return np.asarray(x, float) / self._flat_scale

    def _u_to_x(self, u: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        recipe = self._rt.forward(np.asarray(u, float))
        x = np.array([recipe[k] for k in self._flat_keys], dtype=float)
        return x, recipe

    # -- declared couplings (§8.3) ----------------------------------------------

    def _bind_constraints(self, constraints: ConstraintSet | None) -> tuple[_LinRow, ...]:
        """Split the declared :class:`~rig.constraints.ConstraintSet` into what the
        transform already guarantees (verified, then dropped) and what needs the
        barrier + reject (returned as normalized rows). Monotone declarations relate
        an output to an input and are not recipe-checkable at all — they are the
        surrogate's business (§6.3) and are untouched here."""
        if constraints is None:
            return ()
        cont = {v.name: v for v in self.variables if isinstance(v, ContinuousVariable)}
        comp = [v for v in self.variables if isinstance(v, CompositionalVariable)]
        rows: list[_LinRow] = []

        for b in constraints.box:
            self._require_free(b.name, f"box constraint on {b.name!r}")
            v = cont.get(b.name)
            if v is None:
                # a box on a simplex SHARE is a coupling: the transform pins the
                # sum, not the individual components.
                rows.append(self._make_row({b.name: 1.0}, b.lower, b.upper))
                continue
            if b.lower - _BOX_AGREE_TOL > v.lower or v.upper > b.upper + _BOX_AGREE_TOL:
                raise ValueError(
                    f"box constraint on {b.name!r} declares [{b.lower}, {b.upper}] but the "
                    f"free variable spans [{v.lower}, {v.upper}]: the §8.3 reparameterization "
                    "makes the VARIABLE's range exact by construction, so the solver would "
                    "search outside the declared box and the hard reject would throw the "
                    "excess away. Two sources of truth for one interval is a spec bug — "
                    "narrow the ContinuousVariable instead of constraining it twice."
                )

        for s in constraints.simplex:
            for c in s.components:
                self._require_free(c, f"simplex constraint over {list(s.components)}")
            if not self._simplex_is_exact(s, comp):
                raise NotImplementedError(
                    f"simplex constraint over {list(s.components)} (total={s.total}) is not "
                    "realized by any CompositionalVariable in `variables`, so it is NOT "
                    "exact by construction (§8.3) — and a sum-to-total EQUALITY cannot be "
                    "enforced by the barrier + reject used for linear couplings: every "
                    "restart would miss it and the solver would abstain unconditionally. "
                    "Declare the components as one CompositionalVariable, or drop the "
                    "constraint from the set handed to this solver."
                )

        for lin in constraints.linear:
            for n in lin.coefficients:
                self._require_free(n, f"linear constraint {dict(lin.coefficients)}")
            rows.append(self._make_row(lin.coefficients, lin.lower, lin.upper))

        return tuple(rows)

    def _require_free(self, name: str, what: str) -> None:
        if name not in self._flat_keys:
            raise ValueError(
                f"{what} references {name!r}, which is not one of this solver's free "
                f"variables {self._flat_keys}. Hard-to-change / categorical factors are "
                "CONDITIONING, not searched (§8.3), so a coupling that touches one cannot "
                "be checked on a candidate recipe — substitute the fixed value and declare "
                "the reduced constraint, rather than have it silently skipped."
            )

    def _simplex_is_exact(self, s: Any, comp: list[CompositionalVariable]) -> bool:
        """True iff some free CompositionalVariable's flat keys ARE this simplex —
        i.e. :class:`~rig.transforms.SimplexTransform` already makes it hold for every
        ``u``. Its softmax always sums to exactly 1, so any other total is unreachable."""
        if abs(float(s.total) - 1.0) > 1e-12:
            return False
        want = set(s.components)
        return any({f"{v.name}.{c}" for c in v.components} == want for v in comp)

    def _make_row(self, coefficients: Mapping[str, float], lower: float, upper: float) -> _LinRow:
        idx = np.array([self._flat_keys.index(n) for n in coefficients], dtype=int)
        coef = np.array([float(c) for c in coefficients.values()], dtype=float)
        # span of `coef @ x` over the transform's image, so the barrier's slack is a
        # fraction of what the constraint can actually vary by.
        lo = np.array([self._flat_bounds[i][0] for i in idx], dtype=float)
        hi = np.array([self._flat_bounds[i][1] for i in idx], dtype=float)
        ends = np.stack([coef * lo, coef * hi])
        span = float(np.sum(ends.max(axis=0) - ends.min(axis=0)))
        return _LinRow(
            idx=idx,
            coef=coef,
            lower=float(lower),
            upper=float(upper),
            scale=span if span > 0.0 else 1.0,
        )

    def _constraint_reward(self, x: np.ndarray) -> float:
        """Smooth log-sigmoid barrier on the coupling rows — the §8.3 SOFT half.

        Same idiom as the feasibility margins (``-logaddexp(0, -τ·g)`` = log-sigmoid),
        with the bar at slack 0 rather than κ: a coupling is a property of the
        SETPOINT the operator dials in, so no κ·σ robustness band is credited against
        it and none is claimed. Returns a reward (≤ 0); the caller negates it."""
        r = 0.0
        for row in self._penalty_rows:
            v = float(row.coef @ x[row.idx])
            for g in ((row.upper - v) / row.scale, (v - row.lower) / row.scale):
                if np.isfinite(g):  # an unbounded side has no barrier
                    r += float(-np.logaddexp(0.0, -_CONSTRAINT_TAU * g))
        return r

    def _constraint_excursion(self, x: np.ndarray) -> tuple[float, float]:
        """(max RAW excursion beyond any coupling, total SPAN-NORMALIZED excursion).

        Raw is the operator-facing distance-to-feasible (the constraint's own units);
        normalized is the ranking key, since rows of different magnitude are otherwise
        incomparable. Both are 0.0 at an admissible point."""
        raw = 0.0
        total = 0.0
        for row in self._penalty_rows:
            v = float(row.coef @ x[row.idx])
            e = max(0.0, row.lower - v, v - row.upper)  # ±inf sides drop out
            raw = max(raw, e)
            total += e / row.scale
        return raw, total

    def _admissible(self, restarts: list[_Restart]) -> list[_Restart]:
        """The §8.3 HARD reject. This — not the barrier — is the safety property: it
        holds at any ``constraint_penalty``, including 0. Identity when no
        ConstraintSet was given, so the default path is untouched."""
        if self.constraints is None:
            return restarts
        return [r for r in restarts if self.constraints.is_satisfied(r.recipe)]

    # -- core: worst-case feasibility margins -----------------------------------

    def _margins(
        self, x: np.ndarray, box: SpecBox, out_idx: np.ndarray, model: ForwardModel | None = None
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray
    ]:
        """Return (u_hi, u_lo, mu_all, sig_ale_all, sig_epi_all, s_all, support, sc).

        ``u_hi``/``u_lo`` are the standardized worst-cased margins to the upper
        and lower spec boundaries for the constrained outputs (§8.1). ``s_all``
        is the per-output worst-case mean displacement (epistemic + δ); ``sc``
        is the floored per-output σ scale the margins were divided by (returned
        so callers reuse the exact same scale). A single ``predict`` call per
        point (the solver's hot path). ``model`` defaults to ``self.model``; the
        re-validation pass passes the full ensemble instead (§5.7)."""
        model = model if model is not None else self.model
        dist = model.predict(x)
        mu = np.atleast_1d(np.asarray(dist.mean, dtype=float))
        sig_ale = np.atleast_1d(np.asarray(dist.aleatoric_sigma, dtype=float))
        sig_epi = np.atleast_1d(np.asarray(dist.epistemic_sigma, dtype=float))
        s = self.z_epi * sig_epi
        if self.delta_frac > 0.0:
            J = np.atleast_2d(np.asarray(model.jacobian(x), dtype=float))  # (m,d)
            s = s + np.abs(J) @ self._delta_raw
        support = float(model.support_score(x))

        # numerically safe credited band: floor the aleatoric σ used as the
        # standardization scale so a (near-)deterministic output does not blow
        # the margins up to ±inf. Floor is relative to the output magnitude.
        scale = np.maximum(sig_ale, 1e-9 * (np.abs(mu) + 1.0))

        mu_s = mu[out_idx]
        s_s = s[out_idx]
        sc = scale[out_idx]
        u_hi = (box.upper - mu_s - s_s) / sc  # +inf where upper is +inf
        u_lo = (mu_s - box.lower - s_s) / sc  # +inf where lower is -inf
        return u_hi, u_lo, mu, sig_ale, sig_epi, s, support, sc

    def _neg_objective(
        self,
        u: np.ndarray,
        box: SpecBox,
        out_idx: np.ndarray,
        ignore_epi: bool = False,
        ignore_constraints: bool = False,
    ) -> float:
        x, _ = self._u_to_x(u)
        u_hi, u_lo, _, _, sig_epi, _, support, sc = self._margins(x, box, out_idx)
        if ignore_epi:
            # the epistemic-free (nominal) objective: add the epistemic share
            # back onto both margins so the optimizer answers "is the spec
            # reachable if the model were CERTAIN?" (the §8.8 diagnostic probe).
            e = self.z_epi * sig_epi[out_idx] / sc
            u_hi = u_hi + e
            u_lo = u_lo + e
        r = 0.0
        for m in (u_hi, u_lo):
            fin = m[np.isfinite(m)]
            # log-sigmoid(τ(margin − κ)): drives every margin to ≥ κ then
            # saturates (no reward for over-conservatism). Stable form.
            r += float(np.sum(-np.logaddexp(0.0, -self.tau * (fin - self.kappa))))
            # gentle centering to break the feasible-plateau ties toward the
            # interior (bounded so it never dominates the feasibility reward).
            r += 1e-3 * float(np.sum(np.clip(fin, None, 10.0)))
        r += self.lambda_m * support
        if self._penalty_rows and not ignore_constraints:
            # the coupling-free objective: drop the barrier so the probe can answer
            # "would the spec be reachable if the couplings were not declared?"
            r += self.constraint_penalty * self._constraint_reward(x)
        return -r

    # -- analytic objective gradient (opt-in) -----------------------------------

    def _dx_du(self, u: np.ndarray) -> np.ndarray:
        """``∂x/∂u`` of :class:`~rig.transforms.RecipeTransform` at ``u``, shape
        ``(n_flat, dim)``, in ``_flat_keys`` order (block-diagonal: a variable's
        recipe entries depend only on its own u-coordinates)."""
        M = np.zeros((len(self._flat_keys), self._rt.dim))
        iu = ix = 0
        for bi, v in enumerate(self.variables):
            if isinstance(v, ContinuousVariable):
                # x = lo + (hi−lo)·sigmoid(u) ⇒ ∂x/∂u = (hi−lo)·s(1−s). The clip
                # mirrors RecipeTransform's, so the saturated tail agrees with the
                # forward map rather than reporting a slope the map does not have.
                s = expit(np.clip(u[iu], -_RT_U_CLIP, _RT_U_CLIP))
                M[ix, iu] = (v.upper - v.lower) * s * (1.0 - s)
                iu += 1
                ix += 1
            else:
                # x = softmax([u, 0]) ⇒ ∂x_a/∂u_b = x_a(δ_ab − x_b) over the K−1 FREE
                # coords; the pinned last logit is exactly why the block is (K, K−1).
                k = len(v.components)
                xb = self._rt._blocks[bi][1].forward(u[iu : iu + k - 1])
                M[ix : ix + k, iu : iu + k - 1] = xb[:, None] * (
                    np.eye(k)[:, : k - 1] - xb[None, : k - 1]
                )
                iu += k - 1
                ix += k
        return M

    def _neg_objective_grad(
        self,
        u: np.ndarray,
        box: SpecBox,
        out_idx: np.ndarray,
        ignore_epi: bool = False,
        ignore_constraints: bool = False,
    ) -> tuple[float, np.ndarray]:
        """``(value, ∂value/∂u)`` of :meth:`_neg_objective` in closed form.

        The value MUST equal ``_neg_objective(u, ...)`` — it is the same objective,
        only differentiated — so the arithmetic below deliberately mirrors that
        method line for line rather than being re-derived into a tidier but
        differently-rounded form. Guarded by
        ``test_analytic_objective_value_matches_the_finite_difference_objective``.
        """
        x, _ = self._u_to_x(u)
        t = self._terms.terms(x)

        s = self.z_epi * t.sig_epi
        d_s = self.z_epi * t.d_sig_epi
        if self.delta_frac > 0.0:
            s = s + np.abs(t.jac) @ self._delta_raw
            # ∂|J_ji|/∂x_k = sign(J_ji)·H_jik. sign(0)=0 is the subgradient at the
            # kink |J|=0 — the objective was already non-smooth there.
            d_s = d_s + np.einsum("ji,jik->jk", np.sign(t.jac) * self._delta_raw[None, :], t.hess)

        # the §8 σ floor, and its derivative. For the GP tier `sig_ale` is a fitted
        # constant that dominates the floor, so `d_scale` is 0 — but the branch is
        # written out because the floor IS active for a (near-)deterministic output,
        # and then the scale genuinely moves with μ.
        floor = 1e-9 * (np.abs(t.mu) + 1.0)
        use_ale = t.sig_ale >= floor
        scale = np.maximum(t.sig_ale, floor)
        d_scale = np.where(use_ale[:, None], t.d_sig_ale, 1e-9 * np.sign(t.mu)[:, None] * t.jac)

        sc = scale[out_idx]
        d_sc = d_scale[out_idx]
        u_hi = (box.upper - t.mu[out_idx] - s[out_idx]) / sc
        u_lo = (t.mu[out_idx] - box.lower - s[out_idx]) / sc
        inv = (1.0 / sc)[:, None]
        d_hi = (-t.jac[out_idx] - d_s[out_idx]) * inv - u_hi[:, None] * d_sc * inv
        d_lo = (t.jac[out_idx] - d_s[out_idx]) * inv - u_lo[:, None] * d_sc * inv

        if ignore_epi:
            e = self.z_epi * t.sig_epi[out_idx] / sc
            d_e = self.z_epi * t.d_sig_epi[out_idx] * inv - e[:, None] * d_sc * inv
            u_hi = u_hi + e
            u_lo = u_lo + e
            d_hi = d_hi + d_e
            d_lo = d_lo + d_e

        r = 0.0
        d_r = np.zeros(x.size)
        for margin, d_margin in ((u_hi, d_hi), (u_lo, d_lo)):
            fin = np.isfinite(margin)
            mf = margin[fin]
            dmf = d_margin[fin]
            z = -self.tau * (mf - self.kappa)
            r += float(np.sum(-np.logaddexp(0.0, z)))
            d_r += (self.tau * expit(z)) @ dmf
            r += 1e-3 * float(np.sum(np.clip(mf, None, 10.0)))
            d_r += (1e-3 * (mf < 10.0)) @ dmf

        r += self.lambda_m * t.support
        d_r += self.lambda_m * t.d_support

        if self._penalty_rows and not ignore_constraints:
            r += self.constraint_penalty * self._constraint_reward(x)
            d_r += self.constraint_penalty * self._constraint_reward_grad(x)

        return -r, -(d_r @ self._dx_du(u))

    def _constraint_reward_grad(self, x: np.ndarray) -> np.ndarray:
        """``∂``:meth:`_constraint_reward```/∂x`` — same rows, same finite-side rule."""
        g = np.zeros(x.size)
        for row in self._penalty_rows:
            v = float(row.coef @ x[row.idx])
            for slack, sign in (
                ((row.upper - v) / row.scale, -1.0),
                ((v - row.lower) / row.scale, +1.0),
            ):
                if np.isfinite(slack):
                    w = _CONSTRAINT_TAU * expit(-_CONSTRAINT_TAU * slack)
                    g[row.idx] += w * sign * row.coef / row.scale
        return g

    # -- confidence + interval at a converged point -----------------------------

    def _evaluate(
        self, u: np.ndarray, box: SpecBox, out_idx: np.ndarray, model: ForwardModel | None = None
    ) -> _Restart:
        x, recipe = self._u_to_x(u)
        u_hi, u_lo, mu, sig_ale, sig_epi, s, support, scale_spec = self._margins(
            x, box, out_idx, model
        )
        margin = float(min(np.min(u_hi), np.min(u_lo)))
        # pessimistic per-output spec-hit probability P_j = Φ(u_hi)+Φ(u_lo)−1,
        # clipped to a valid probability; product under the per-output GP
        # independence approximation (§8.1 fast path; joint-residual-covariance
        # MC is WP-E).
        p_hi = norm.cdf(np.clip(u_hi, -37.0, 37.0))
        p_lo = norm.cdf(np.clip(u_lo, -37.0, 37.0))
        p_j = np.clip(p_hi + p_lo - 1.0, 0.0, 1.0)
        confidence = float(np.prod(p_j))
        # worst-case credited outcome interval per constrained output (§8.6):
        # [μ − s − κσ_ale, μ + s + κσ_ale].
        band = self.kappa * sig_ale[out_idx]
        lo = mu[out_idx] - s[out_idx] - band
        hi = mu[out_idx] + s[out_idx] + band
        interval = np.stack([lo, hi], axis=-1)  # (m_spec, 2)
        # robust margin with the epistemic displacement stripped (δ kept): this
        # is what the §8.8 diagnostic (in _infeasible) uses to tell an
        # epistemic-limited-but-mean-reachable spec ("collect more runs") apart
        # from a hard spec conflict ("relax the spec"). We remove ONLY the
        # epistemic share z_epi·σ_epi so a δ/Jacobian-driven miss is not
        # mislabeled.
        e = self.z_epi * sig_epi[out_idx] / scale_spec  # epistemic share / σ
        margin_no_epi = float(min(np.min(u_hi + e), np.min(u_lo + e)))
        return _Restart(
            u=np.asarray(u, float),
            x=x,
            recipe=recipe,
            u_hi=u_hi,
            u_lo=u_lo,
            margin=margin,
            confidence=confidence,
            support=support,
            mean=mu,
            sig_ale=sig_ale,
            scale_spec=scale_spec,
            interval=interval,
            margin_no_epi=margin_no_epi,
        )

    # -- multi-start ------------------------------------------------------------

    def _seed_starts(self, dim: int) -> np.ndarray:
        """Sobol multi-start in u-space over [-u_bound, u_bound]^dim, plus the
        box-centre start u=0 (§8.6 step 2)."""
        starts = [np.zeros(dim)]
        n = max(0, self.n_restarts - 1)
        if n > 0:
            sampler = qmc.Sobol(d=dim, scramble=True, seed=self.seed)
            with warnings.catch_warnings():
                # an arbitrary (non-power-of-2) restart count is intentional;
                # the balance-property warning is not actionable here.
                warnings.simplefilter("ignore", category=UserWarning)
                u = sampler.random(n)  # (n, dim) in [0,1)
            starts.extend((2.0 * u - 1.0) * self.u_bound)
        return np.asarray(starts, float)

    def _warm_start_u(self, recipes: Any) -> np.ndarray | None:
        """Convert optional warm-start recipe dicts (D2: amortized §14.3 proposals)
        into u-space multi-start seeds, clipped to the search box. Returns ``None``
        when absent, so the cold Sobol path stays byte-for-byte unchanged. Each
        recipe is the generator's flat dict (same flat-key layout as this solver's
        :class:`~rig.transforms.RecipeTransform`)."""
        if not recipes:
            return None
        u = np.stack([np.asarray(self._rt.inverse(r), float) for r in recipes])
        return np.clip(u, -self.u_bound, self.u_bound)

    def _run_multistart(
        self,
        box: SpecBox,
        out_idx: np.ndarray,
        *,
        ignore_epi: bool,
        warm_u: np.ndarray | None = None,
        ignore_constraints: bool = False,
    ) -> list[_Restart]:
        """Sobol multi-start + L-BFGS-B on the (pessimistic or, when
        ``ignore_epi``, epistemic-free; or, when ``ignore_constraints``,
        coupling-free) objective (§8.6 steps 2-3). Optional ``warm_u`` seeds (D2
        amortized proposals) are refined FIRST, then the cold Sobol starts — the
        single canonical refiner polishes both alike."""
        dim = self._rt.dim
        bounds = [(-self.u_bound, self.u_bound)] * dim
        seeds = self._seed_starts(dim)
        if warm_u is not None and len(warm_u):
            seeds = np.concatenate([np.atleast_2d(warm_u), seeds], axis=0)
        out: list[_Restart] = []
        # `jac=True` costs ONE model evaluation per gradient; the default (no `jac`)
        # costs SciPy `dim+1` of them. Same objective either way — see the ctor's
        # `analytic_grad` note on why the two paths are not bitwise equal.
        # `jac` is passed only on the analytic path, so the FD call is the historical
        # one argument-for-argument rather than relying on scipy mapping a falsy `jac`
        # back to None.
        fun = self._neg_objective
        kw: dict[str, Any] = {}
        if self._terms is not None:
            fun, kw = self._neg_objective_grad, {"jac": True}
        for u0 in seeds:
            res = minimize(
                fun,
                u0,
                args=(box, out_idx, ignore_epi, ignore_constraints),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": self.max_iter},
                **kw,
            )
            out.append(self._evaluate(np.clip(res.x, -self.u_bound, self.u_bound), box, out_idx))
        return out

    # -- InverseSolver protocol -------------------------------------------------

    def solve(self, spec: Mapping[str, Any]) -> InverseResult:
        """Solve one inverse query (implementation-plan §3.3 / §8.6).

        ``spec`` keys:
          * ``targets`` (required): per-output box, see :func:`parse_targets`.
          * ``max_candidates`` (optional, default 4): q, the returned set size.
          * ``tool_id`` (optional): informational only — enriches the
            "collect more runs" message; the tool must ALREADY be bound into
            ``model`` via ``for_tool`` (this solver never searches over tools).
          * ``warm_start_recipes`` (optional): a list of recipe dicts refined
            FIRST as extra multi-start seeds (D2: the §14.3 amortized proposals;
            see :class:`~rig.inverse.AmortizedRefiner`). Absent ⇒ cold Sobol only.
        Returns a ranked, diversity-selected list of feasible
        :class:`~rig.interfaces.RecipeCandidate`, or an explicit
        :class:`~rig.interfaces.Infeasible` verdict.
        """
        if "targets" not in spec:
            raise KeyError("spec must contain 'targets' (implementation-plan §8.1: Z* is a box)")
        box = parse_targets(spec["targets"], self.output_keys)
        out_idx = np.array([self.output_keys.index(n) for n in box.output_names], dtype=int)
        q = int(spec.get("max_candidates", 4))
        if q < 1:
            raise ValueError(
                f"max_candidates must be >= 1, got {q} (an empty request is not a "
                "feasibility verdict — fail loud rather than return [] a caller "
                "would misread as INFEASIBLE)"
            )
        tool_id = spec.get("tool_id", None)
        warm_u = self._warm_start_u(spec.get("warm_start_recipes"))

        dim = self._rt.dim
        if dim == 0:
            raise ValueError("no free variables to optimize over")

        restarts = self._run_multistart(box, out_idx, ignore_epi=False, warm_u=warm_u)

        # hard coupling reject (§8.3): whatever the barrier did, a recipe violating a
        # declared constraint is never a survivor. Identity when constraints is None.
        adm = self._admissible(restarts)

        # hard manifold reject (§8.2): a survivor must be on-support.
        on_support = [r for r in adm if r.support >= self.support_floor]
        feasible = [
            r for r in on_support if r.margin >= self.kappa - _FEAS_TOL and r.confidence > 0.0
        ]

        nominal: list[_Restart] | None = None
        nominal_adm: list[_Restart] = []
        if not feasible:
            # The §8.8 nominal-feasibility probe (epistemic-free multi-start).
            # It serves two ends: (a) surface an on-support point the pessimistic
            # search MISSED that is nonetheless pessimistically feasible — never
            # return INFEASIBLE when a feasible recipe exists; (b) diagnose the
            # infeasibility cause. Computed once, reused for both.
            nominal = self._run_multistart(box, out_idx, ignore_epi=True, warm_u=warm_u)
            nominal_adm = self._admissible(nominal)
            feasible = [
                r
                for r in nominal_adm
                if r.support >= self.support_floor
                and r.margin >= self.kappa - _FEAS_TOL
                and r.confidence > 0.0
            ]
            if not feasible:
                if self.constraints is not None:
                    # Ask FIRST whether the declared couplings are what made this
                    # abstention: the §8.8 taxonomy speaks about the model and the
                    # target, and would name the wrong remedy for a constraint block.
                    blocked = self._constraint_blocked(restarts + nominal, box, out_idx, warm_u)
                    if blocked is not None:
                        return blocked
                # `x or y`: whichever survived the reject. Both are the full restart
                # lists when constraints is None, and _constraint_blocked has already
                # returned above when NEITHER has an admissible member, so the
                # taxonomy never sees an empty pool.
                return self._infeasible(
                    adm or nominal_adm, nominal_adm or adm, box, out_idx, tool_id
                )

        # rank by pessimistic spec-hit probability, then greedy diversity
        # selection (§8.7 non-injectivity): a diverse pre-image set, not q
        # copies of one optimum.
        feasible.sort(key=lambda r: r.confidence, reverse=True)
        chosen = self._greedy_diverse(feasible, q)

        # §5.7 / §13.2 re-validation: when `model` was a fast screening surrogate,
        # re-score the selected set on the full ensemble (+ conformal) and keep
        # only what it certifies. Skipped entirely when revalidation_model is None.
        if self.revalidation_model is not None:
            revalidated = self._revalidate(chosen, box, out_idx)
            if not revalidated:
                # Audit 2026-07-17: do NOT declare INFEASIBLE having tested only the
                # q DIVERSE picks. `feasible` routinely holds tens of survivors and
                # `_greedy_diverse` cuts to q=4 by SPREAD, not by revalidation
                # merit — so the q that failed the full-ensemble gate say nothing
                # about the rest of the pre-image, and returning INFEASIBLE while a
                # certifiable recipe sat unexamined in the pool is exactly the
                # false-abstention the §8 contract forbids. Fall back to sweeping
                # the remainder (highest-confidence first) and re-select diversely
                # from whatever the full model actually certifies. The fast path is
                # unchanged: this costs extra ONLY when we were about to abstain.
                # identity, not `in`: these records carry numpy arrays, so `==`
                # would raise on the ambiguous truth value of an array.
                picked = {id(r) for r in chosen}
                rest = [r for r in feasible if id(r) not in picked]
                survivors = self._revalidate(rest, box, out_idx) if rest else []
                if not survivors:
                    return self._reval_infeasible(chosen[0], box, out_idx)
                survivors.sort(key=lambda r: r.confidence, reverse=True)
                revalidated = self._greedy_diverse(survivors, q)
            return [self._to_candidate(r, box) for r in revalidated]
        return [self._to_candidate(r, box) for r in chosen]

    def _reval_support_floor(self) -> float:
        """The §8.2 floor for the RE-VALIDATION model. support_score is per-model,
        so gate the full model against ITS OWN train-set floor (not the fast
        surrogate's) — else a differently-scaled model is judged wrongly. Falls
        back to ``self.support_floor`` when no X_train was given (the caller then
        owns the fact that the explicit floor was tuned for ``self.model``)."""
        if self._X_train is None:
            return self.support_floor
        return self._support_floor_of(self.revalidation_model)

    def _revalidate(
        self, chosen: list[_Restart], box: SpecBox, out_idx: np.ndarray
    ) -> list[_Restart]:
        """Re-score the selected candidates on ``self.revalidation_model`` (the
        full ensemble, optionally conformal-wrapped) and keep only those it still
        certifies pessimistically feasible + on-support + inside the §13.2
        conformal acceptance gate. The surviving ``_Restart``s carry the FULL
        model's confidence/interval (D2: calibration attaches to the re-validated
        set, never the fast surrogate's refined output)."""
        reval_floor = self._reval_support_floor()
        survivors: list[_Restart] = []
        for r in chosen:
            rr = self._evaluate(r.u, box, out_idx, model=self.revalidation_model)
            if rr.support < reval_floor:
                continue
            if rr.margin < self.kappa - _FEAS_TOL or rr.confidence <= 0.0:
                continue
            if not self._conformal_in_box(rr.x, box, out_idx):
                continue
            survivors.append(rr)
        survivors.sort(key=lambda r: r.confidence, reverse=True)
        return survivors

    def _reval_infeasible(self, top: _Restart, box: SpecBox, out_idx: np.ndarray) -> Infeasible:
        """Diagnose WHY re-validation rejected everything, so the verdict names the
        right remedy (§8.8-style). The failure modes are distinct: an off-support /
        margin-limited rejection is epistemic ("collect runs"), whereas a
        conformal-only rejection means the calibrated band is wider than the spec
        box — an aleatoric/coverage issue more runs alone will not fix."""
        best = self._evaluate(top.u, box, out_idx, model=self.revalidation_model)
        on_support = best.support >= self._reval_support_floor()
        margin_ok = best.margin >= self.kappa - _FEAS_TOL and best.confidence > 0.0
        if on_support and margin_ok and not self._conformal_in_box(best.x, box, out_idx):
            return Infeasible(
                nearest_achievable=dict(best.recipe),
                distance_to_feasible=self._conformal_spill(best.x, box, out_idx),
                reason=(
                    "full-ensemble re-validation: the mean reaches spec but the "
                    "calibrated conformal interval is wider than the spec box "
                    "(C(x') ⊄ Z*, §13.2) — reduce process variation, relax κ, or "
                    "widen the spec tolerance; more runs alone will not shrink an "
                    "irreducible-aleatoric band"
                ),
            )
        return Infeasible(
            nearest_achievable=dict(best.recipe),
            distance_to_feasible=float(max(self.kappa - best.margin, 0.0)),
            reason=(
                "selected recipes failed full-ensemble re-validation (§5.7): the "
                "fast inner-loop surrogate proposed points the K-member ensemble "
                "does not certify (off-support or margin-limited) — collect runs to "
                "sharpen the surrogate, or widen the search"
            ),
        )

    def _conformal_set(self, x: np.ndarray, out_idx: np.ndarray) -> np.ndarray | None:
        cs = self.revalidation_model.predict(x).conformal_set
        if cs is None:
            return None
        return np.atleast_2d(np.asarray(cs, dtype=float))[out_idx]  # (m_spec, 2)

    def _conformal_in_box(self, x: np.ndarray, box: SpecBox, out_idx: np.ndarray) -> bool:
        """§13.2 gate C(x') ⊆ Z*: the full model's conformal interval for every
        constrained output must sit inside the spec box. Inactive (returns True)
        when the re-validation model is not conformal-wrapped (conformal_set None)."""
        cs = self._conformal_set(x, out_idx)
        if cs is None:
            return True
        return bool(
            np.all(cs[:, 0] >= box.lower - _FEAS_TOL) and np.all(cs[:, 1] <= box.upper + _FEAS_TOL)
        )

    def _conformal_spill(self, x: np.ndarray, box: SpecBox, out_idx: np.ndarray) -> float:
        """Max raw excursion of the conformal interval beyond the spec box over the
        constrained outputs (the honest distance-to-feasible for a §13.2 gate
        rejection). 0.0 when inside; +inf shouldn't arise (one-sided ∞ bounds
        never spill)."""
        cs = self._conformal_set(x, out_idx)
        if cs is None:
            return 0.0
        below = np.where(np.isfinite(box.lower), box.lower - cs[:, 0], 0.0)
        above = np.where(np.isfinite(box.upper), cs[:, 1] - box.upper, 0.0)
        return float(np.max(np.maximum(np.maximum(below, above), 0.0)))

    # -- result assembly --------------------------------------------------------

    def _to_candidate(self, r: _Restart, box: SpecBox) -> RecipeCandidate:
        return RecipeCandidate(
            recipe=dict(r.recipe),
            confidence=r.confidence,
            predicted_outcome_interval={
                name: (float(r.interval[j, 0]), float(r.interval[j, 1]))
                for j, name in enumerate(box.output_names)
            },
            feasibility_flag=True,
            support_score=r.support,
        )

    def _greedy_diverse(self, ranked: list[_Restart], q: int) -> list[_Restart]:
        """Farthest-point (max-min) selection in normalized recipe space —
        the §8.7 diversity stand-in for the k-DPP (that, and the amortized
        generator sampling the pre-image, are the torch-era WP-E). Anchors on
        the highest-confidence recipe, then repeatedly adds the candidate that
        MAXIMIZES the minimum distance to the already-chosen set, so the
        returned set genuinely spreads the pre-image manifold rather than
        returning q copies of one optimum. Stops early (returns fewer than q)
        when the remaining pool is within ``_DEDUP_RADIUS`` of the chosen set —
        i.e. the pre-image really is a single point; we never pad with
        near-duplicates."""
        if q <= 0 or not ranked:
            return []
        coords = [self._norm_x(r.x) for r in ranked]
        chosen_idx = [0]
        while len(chosen_idx) < q:
            best_i, best_d = -1, -1.0
            for i in range(len(ranked)):
                if i in chosen_idx:
                    continue
                dmin = min(float(np.linalg.norm(coords[i] - coords[j])) for j in chosen_idx)
                if dmin > best_d:
                    best_d, best_i = dmin, i
            if best_i < 0 or best_d < _DEDUP_RADIUS:
                break  # nothing distinct left — the pre-image is (near) a point
            chosen_idx.append(best_i)
        return [ranked[i] for i in chosen_idx]

    def _spec_reaching_violators(self, restarts: list[_Restart]) -> list[_Restart]:
        """Restarts that WOULD be certified — robust at κ and on-support — and are
        stopped by the §8.3 reject alone."""
        return [
            r
            for r in restarts
            if not self.constraints.is_satisfied(r.recipe)
            and r.support >= self.support_floor
            and r.margin >= self.kappa - _FEAS_TOL
            and r.confidence > 0.0
        ]

    def _constraint_blocked(
        self,
        restarts: list[_Restart],
        box: SpecBox,
        out_idx: np.ndarray,
        warm_u: np.ndarray | None,
    ) -> Infeasible | None:
        """Attribute an abstention to the declared couplings — but ONLY on evidence.

        Returns ``None`` unless one of two patterns holds, so a spec that would have
        failed anyway still gets the §8.8 taxonomy and its (correct) remedy:

        (a) a recipe reaches the spec robustly and on-support, and the §8.3 reject is
            the only thing standing between it and certification;
        (b) not one restart is admissible.

        Both verdicts report a CONSTRAINT-VIOLATING recipe as ``nearest_achievable``
        (there is no admissible point to report, and a clipped one is exactly what §8.7
        forbids), so the reason says so in as many words. Unlike the §8.8 path,
        ``distance_to_feasible`` here is the raw constraint excursion in the
        constraint's own units, not a σ-margin deficit — the two answer different
        questions.

        The COUPLING-FREE PROBE is what makes (a) sound. The barrier holds the search
        INSIDE the couplings, so a spec-reaching violator is often never sampled — and
        then "no evidence the coupling is at fault" is absence of evidence, not evidence
        of absence, and §8.8 would confidently answer "genuinely unreachable, relax the
        target" about a target that is reachable and a coupling that is what blocks it.
        So we ask directly, with the idiom §8.8 already uses for epistemic: re-run the
        multi-start with the term REMOVED and see whether the spec becomes reachable.
        Like the nominal probe, it runs ONLY on the abstention path, and only when the
        cheap evidence is inconclusive.

        The claims stay inside what a multi-start supports. "Every spec-reaching recipe
        found violates the coupling" is evidence; "the spec is reachable ONLY outside
        the coupling" is not — a thin admissible set the search never held produces the
        same symptom. The verdict therefore carries the search caveat whenever no
        admissible restart survived (the §8.8 false-INFEASIBLE mode), and drops it only
        when admissible restarts WERE explored and none reached the spec.
        """
        admissible = [r for r in restarts if self.constraints.is_satisfied(r.recipe)]
        blocked = self._spec_reaching_violators(restarts)
        if not blocked and admissible:
            probe = self._run_multistart(
                box, out_idx, ignore_epi=False, warm_u=warm_u, ignore_constraints=True
            )
            blocked = self._spec_reaching_violators(probe)
            if not blocked:
                # unreachable with the couplings AND without them — they are not the
                # story, so let §8.8 name the real cause.
                return None
        # the search never held the admissible set, so a coupling verdict cannot be
        # separated from a starved multi-start; say so rather than blame the coupling.
        caveat = (
            ""
            if admissible
            else (
                " NB not one restart stayed inside the constraints, so this may equally "
                "be a SEARCH artifact rather than the coupling's fault: the barrier is a "
                "soft guide, not a projection (§8.8 false-INFEASIBLE). Raise "
                "`constraint_penalty` or warm-start from a known admissible recipe "
                "before concluding the declaration is what blocks you."
            )
        )
        if blocked:
            best = max(blocked, key=lambda r: r.confidence)
            raw, _ = self._constraint_excursion(best.x)
            verdict = (
                "constraint-blocked (§8.3): every recipe the search found that reaches "
                "the spec robustly and on-support VIOLATES a declared constraint, so "
                "none can be certified. The reported point is one of them — it meets "
                f"the spec and violates {self._violation_text(best)}, excursion "
                f"{raw:.3g} in the constraint's own units — and is the nearest "
                "achievable point, NOT a usable recipe."
            )
            if admissible:
                verdict += (
                    " Admissible restarts WERE explored and none reached the spec: the "
                    "coupling and the target are in conflict, and more data will not "
                    "resolve it — relax one, or change the process."
                )
            return Infeasible(
                nearest_achievable=dict(best.recipe),
                distance_to_feasible=raw,
                reason=verdict + caveat,
            )
        best = min(restarts, key=lambda r: self._constraint_excursion(r.x)[1])
        raw, _ = self._constraint_excursion(best.x)
        return Infeasible(
            nearest_achievable=dict(best.recipe),
            distance_to_feasible=raw,
            reason=(
                "constraint-infeasible (§8.3): not one restart landed inside the declared "
                f"constraints — {self._violation_text(best)} at the least-violating point, "
                f"excursion {raw:.3g}. The reported recipe VIOLATES the declaration and is "
                "not usable. No violating restart reached the spec either, so the target "
                "may be out of reach independently of the coupling." + caveat
            ),
        )

    def _violation_text(self, r: _Restart) -> str:
        return "; ".join(self.constraints.validate(r.recipe))

    def _infeasible(
        self,
        restarts: list[_Restart],
        nominal: list[_Restart],
        box: SpecBox,
        out_idx: np.ndarray,
        tool_id: Any,
    ) -> Infeasible:
        """Assemble the §8.7 infeasibility verdict: nearest achievable point,
        distance-to-feasible, and the per-output spec relaxation that admits it.

        Distinguishes the §8.8 causes so the operator gets the RIGHT action:
        off-manifold (expand support via §9), epistemic-limited but mean-reachable
        (collect more runs), a partly-epistemic tight box (data helps + relax),
        or a hard spec conflict (relax the spec / lower κ). ``nominal`` is the
        already-computed epistemic-free probe (the pessimistic search AVOIDS
        high-epistemic regions, so a mean-feasible point there is never in
        ``restarts``); reused here so the probe runs only once per solve.
        """
        on_support = [r for r in restarts if r.support >= self.support_floor]
        if not on_support:
            best = max(restarts, key=lambda r: r.margin)
            return Infeasible(
                nearest_achievable=dict(best.recipe),
                distance_to_feasible=float(max(0.0, self.kappa - best.margin)),
                reason=(
                    "no on-manifold recipe found: every restart fell below the "
                    f"§8.2 support floor ({self.support_floor:.3g}). The spec box "
                    "likely lies outside the trained recipe region — expand "
                    "support via active learning (§9) rather than lowering the "
                    "floor."
                ),
            )

        # §8.8: is the spec reachable if the model were CERTAIN? Use the probe's
        # best on-support point; if mean-feasible (margin_no_epi ≥ κ), EPISTEMIC.
        nominal_os = [r for r in nominal if r.support >= self.support_floor]
        best_nominal = max(nominal_os or nominal, key=lambda r: r.margin_no_epi)
        epi_limited = (
            best_nominal.support >= self.support_floor
            and best_nominal.margin_no_epi >= self.kappa - _FEAS_TOL
        )

        if epi_limited:
            nearest = best_nominal
        else:
            nearest = max(on_support, key=lambda r: r.margin)
        deficit = float(max(0.0, self.kappa - nearest.margin))

        # per-output relaxation (raw units) at `nearest`: how far each violated
        # boundary must move to admit it at the κ bar. The margin is in
        # FLOORED-σ units, so convert with the SAME floored scale the margins
        # used (raw σ_ale would collapse relaxation to 0 for a deterministic
        # output whose raw σ_ale is below the floor).
        relax: dict[str, float] = {}
        for j, name in enumerate(box.output_names):
            m_j = float(min(nearest.u_hi[j], nearest.u_lo[j]))
            if m_j < self.kappa - _FEAS_TOL:
                relax[name] = float((self.kappa - m_j) * nearest.scale_spec[j])

        # Decompose the deficit at `nearest` for honest guidance (§8.8). Three
        # facts drive the message: (i) is the NOMINAL predicted mean actually
        # inside the box for every output? (ii) would removing epistemic make it
        # fully robust (epi_limited)? (iii) does epistemic contribute to the
        # deficit at all (so more data helps even if it does not fully fix it)?
        mean_spec = nearest.mean[out_idx]
        means_out = [
            name
            for j, name in enumerate(box.output_names)
            if not (box.lower[j] <= mean_spec[j] <= box.upper[j])
        ]
        epi_helps = (nearest.margin_no_epi - nearest.margin) > _FEAS_TOL
        residual = float(max(0.0, self.kappa - nearest.margin_no_epi))  # σ, sans epi

        if epi_limited:
            where = f"tool {tool_id!r}" if tool_id is not None else "this region"
            reason = (
                "pessimistic-infeasible, but the binding term is EPISTEMIC "
                "(model uncertainty), not a hard spec conflict: the predicted "
                "mean reaches the spec, only the uncertainty is too large — "
                f"collect more runs in {where} near the reported recipe and "
                f"re-solve (WP-I handoff); robust-margin deficit {deficit:.3g}σ."
            )
        elif means_out:
            reason = (
                "genuinely unreachable: the predicted mean itself is outside the "
                f"spec box for {', '.join(means_out)} (margin {nearest.margin:.3g}"
                "σ). More data will not move it in — relax the target (not just "
                "the tolerance) or change the process."
            )
        elif epi_helps:
            reason = (
                f"in-spec on the mean but not robustly at κ={self.kappa}: PARTLY "
                "epistemic — collecting runs near the reported recipe cuts the "
                f"required relaxation from {deficit:.3g}σ toward {residual:.3g}σ, "
                "and relaxing the spec or lowering κ closes the rest."
            )
        else:
            reason = (
                f"in-spec on the mean but not robustly at κ={self.kappa}: the spec "
                "box is tighter than the ±κσ credited band (irreducible aleatoric "
                "noise / process variation δ), so more data will not help — relax "
                f"the spec, lower κ, or reduce variation. Deficit {deficit:.3g}σ."
            )
        if relax:
            terms = ", ".join(f"{k} by {v:.3g}" for k, v in relax.items())
            reason += f" Required per-output relaxation: {terms}."

        return Infeasible(
            nearest_achievable=dict(nearest.recipe),
            distance_to_feasible=deficit,
            reason=reason,
        )
