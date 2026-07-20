"""Constraint-by-construction reparameterizations (implementation-plan §8.3).

Default to constraint-by-construction; never clip. Box ranges use
``x = lo + (hi - lo) * sigmoid(u)`` - feasible for ALL u. Compositions use
a fixed-gauge softmax (additive-log-ratio, ALR): ``x = softmax([u, 0])``
with inverse ``u_i = log(x_i / x_K)``.

Design choice (documented per WP-A brief): ALR / fixed-gauge softmax was
chosen over ILR. Both make non-negativity and sum-to-1 exact; ILR's
orthonormal (Helmert) basis matters for statistical analysis of
compositions, but for an optimizer's unconstrained parameterization the
ALR gauge is simpler, has a closed-form inverse, and pins the softmax
translation-invariance so the map is a true bijection on K-1 dims.

Numerical stability: sigmoid inputs are clipped to +/-40 (sigmoid saturates
in float64 well before that); inverses clip interior coordinates away from
the boundary by ``eps`` before taking logs/logits. Because sigmoid underflows
to exactly 1.0 for u >~ 37, ``forward`` can return the CLOSED boundary lo/hi
(not strictly interior) in the saturated tail — harmless for the optimizer,
which consumes the recipe as plain floats. ``inverse`` raises on inputs
genuinely outside [lo, hi] (Box) or off the simplex (negative share / sum far
from 1, Simplex) rather than silently clamping them, modulo a small float-drift
tolerance (``1e-9``, same magnitude both transforms use for their own edge
tolerance) — range/unit drift beyond that is an ingest-time contract
violation, implementation-plan §3.5.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
)

_U_CLIP = 40.0
_EPS = 1e-12


def _sigmoid(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, -_U_CLIP, _U_CLIP)
    return 1.0 / (1.0 + np.exp(-u))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


class BoxTransform:
    """Bijection R^d -> (lo, hi)^d: x = lo + (hi - lo) * sigmoid(u) (§8.3)."""

    def __init__(self, lower: np.ndarray | list[float], upper: np.ndarray | list[float]):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower/upper shape mismatch")
        if not np.all(self.lower < self.upper):
            raise ValueError("require lower < upper elementwise")

    @property
    def dim(self) -> int:
        return int(self.lower.size)

    def forward(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return self.lower + (self.upper - self.lower) * _sigmoid(u)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p = (x - self.lower) / (self.upper - self.lower)
        # Fail loud on genuine out-of-bounds inputs instead of letting _logit's
        # p-clip fabricate a finite u (audit D1). Exact bounds (p in {0,1}) are
        # allowed — they are the saturated image of forward(). NaN must also
        # raise here: it compares False against BOTH `p < -tol` and
        # `p > 1+tol`, so without an explicit isfinite check it fell through
        # the guard and _logit silently produced a NaN u (finding A).
        tol = 1e-9
        if np.any(~np.isfinite(p)) or np.any(p < -tol) or np.any(p > 1.0 + tol):
            raise ValueError(
                "BoxTransform.inverse: value outside [lower, upper] or non-finite "
                f"(normalized p={p.tolist()}); range/unit drift must be caught "
                "at ingest, not silently clamped (implementation-plan §3.5)"
            )
        return _logit(p)

    def log_abs_det_du_dx(self, x: np.ndarray) -> float:
        """``log|det du/dx|`` at ``x`` — the change-of-variables term needed to turn
        a density over ``u`` into a density over ``x`` (§8.3).

        ``u = logit(p)``, ``p = (x−lo)/(hi−lo)``, so ``du/dx`` is diagonal with
        entries ``1/(p(1−p)(hi−lo))`` and
        ``log|det| = Σ_i [−log p_i − log(1−p_i) − log(hi_i−lo_i)]``.
        This is NOT a constant: it diverges at the box edges, so omitting it does
        not merely offset a log-density — it reorders it.
        """
        x = np.asarray(x, dtype=float)
        p = np.clip((x - self.lower) / (self.upper - self.lower), _EPS, 1.0 - _EPS)
        return float(np.sum(-np.log(p) - np.log1p(-p) - np.log(self.upper - self.lower)))


class SimplexTransform:
    """Bijection R^(K-1) -> interior of the (K-1)-simplex (§8.3).

    Forward: x = softmax([u_1, ..., u_{K-1}, 0]) (fixed gauge: last logit
    pinned to 0). Inverse (ALR): u_i = log(x_i / x_K). Non-negativity and
    sum-to-1 are exact by construction. See module docstring for the
    ALR-over-ILR rationale.
    """

    def __init__(self, n_components: int):
        if n_components < 2:
            raise ValueError("a simplex needs >= 2 components")
        self.n_components = int(n_components)

    @property
    def dim(self) -> int:
        """Unconstrained dimension: K - 1."""
        return self.n_components - 1

    def forward(self, u: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(u, dtype=float), -_U_CLIP, _U_CLIP)
        if u.shape[-1] != self.dim:
            raise ValueError(f"expected {self.dim} unconstrained coords, got {u.shape[-1]}")
        z = np.concatenate([u, np.zeros(u.shape[:-1] + (1,))], axis=-1)
        z = z - z.max(axis=-1, keepdims=True)  # log-sum-exp stabilization
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    def inverse(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1] != self.n_components:
            raise ValueError(f"expected {self.n_components} components, got {x.shape[-1]}")
        # Fail loud on genuinely out-of-simplex input instead of silently
        # clamping/renormalizing it into a fabricated finite u (finding B) --
        # matches BoxTransform.inverse's D1 contract. `tol` is the same
        # magnitude as BoxTransform's own edge tolerance on normalized p
        # (§8.3): components and the sum both live on a [0, 1]-scale, so a
        # deviation is judged against the same absolute bar. This is loose
        # enough to tolerate the float64-precision drift (~1e-12 or tighter)
        # that RecipeTransform.forward()'s softmax normalization and its own
        # downstream round-trips (amortized.py, pessimistic.py warm starts)
        # produce, while still catching genuine violations (negative shares,
        # sums far from 1).
        tol = 1e-9
        s = x.sum(axis=-1, keepdims=True)
        if np.any(~np.isfinite(x)) or np.any(x < -tol) or np.any(np.abs(s - 1.0) > tol):
            raise ValueError(
                "SimplexTransform.inverse: value outside the simplex or non-finite "
                f"(x={x.tolist()}, sum={s.ravel().tolist()}); range/unit drift must "
                "be caught at ingest, not silently clamped (implementation-plan §3.5)"
            )
        x = np.clip(x, _EPS, 1.0)
        x = x / s
        return np.log(x[..., :-1]) - np.log(x[..., -1:])

    def log_abs_det_du_dx(self, x: np.ndarray) -> float:
        """``log|det du/dx|`` at ``x`` over the K−1 FREE coords ``x_1..x_{K-1}``.

        For ALR ``u_i = log(x_i/x_K)`` with ``x_K = 1 − Σ_{j<K} x_j``:
        ``∂u_i/∂x_j = δ_ij/x_i + 1/x_K``, i.e. ``J = diag(1/x_i) + (1/x_K)·11ᵀ``.
        By the matrix-determinant lemma
        ``det J = (Π_{i<K} 1/x_i)·(1 + (1/x_K)Σ_{i<K} x_i) = 1/Π_{i=1}^{K} x_i``,
        hence ``log|det J| = −Σ_{i=1}^{K} log x_i`` (over ALL K components).
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, _EPS, 1.0)
        x = x / x.sum(axis=-1, keepdims=True)
        return float(-np.sum(np.log(x)))


class RecipeTransform:
    """Composable map: unconstrained u-vector <-> typed recipe dict (§8.3).

    Continuous variables get a BoxTransform block (1 dim each);
    compositional variables get a SimplexTransform block (K-1 dims each).
    Categorical / hard-to-change factors are conditioning, not free
    variables (§8.3) - they are not part of u and must be handled by the
    caller as fixed context.

    Recipe dict format: {name: float} for continuous variables and
    {"comp.name": float} per component ("<variable>.<component>") for
    compositional variables - matching RecipeRecord key flattening.
    """

    def __init__(self, variables: list[ContinuousVariable | CompositionalVariable]):
        self.variables = list(variables)
        self._blocks: list[tuple[Any, Any, int]] = []  # (spec, transform, u_dim)
        for v in self.variables:
            if isinstance(v, ContinuousVariable):
                self._blocks.append((v, BoxTransform([v.lower], [v.upper]), 1))
            elif isinstance(v, CompositionalVariable):
                self._blocks.append((v, SimplexTransform(len(v.components)), len(v.components) - 1))
            else:
                raise TypeError(
                    f"{v!r}: only continuous/compositional variables are free "
                    "dimensions; categoricals are conditioning (implementation-plan §8.3)"
                )
        self.dim = sum(d for _, _, d in self._blocks)

    def forward(self, u: np.ndarray) -> dict[str, float]:
        u = np.asarray(u, dtype=float)
        if u.shape != (self.dim,):
            raise ValueError(f"expected u of shape ({self.dim},), got {u.shape}")
        out: dict[str, float] = {}
        i = 0
        for spec, t, d in self._blocks:
            block = t.forward(u[i : i + d])
            if isinstance(spec, ContinuousVariable):
                out[spec.name] = float(block[0])
            else:
                for comp, val in zip(spec.components, block, strict=True):
                    out[f"{spec.name}.{comp}"] = float(val)
            i += d
        return out

    def inverse(self, recipe: Mapping[str, float]) -> np.ndarray:
        parts: list[np.ndarray] = []
        for spec, t, _ in self._blocks:
            if isinstance(spec, ContinuousVariable):
                parts.append(t.inverse(np.array([recipe[spec.name]], dtype=float)))
            else:
                x = np.array([recipe[f"{spec.name}.{c}"] for c in spec.components], dtype=float)
                parts.append(t.inverse(x))
        return np.concatenate(parts) if parts else np.empty(0)

    def log_abs_det_du_dx(self, recipe: Mapping[str, float]) -> float:
        """``log|det du/dx|`` at ``recipe`` — block-diagonal, so the blocks' terms add.

        Anything holding a density in ``u``-space (e.g. the §14.3 amortized flow)
        MUST add this to report a density over recipes:
        ``log q(x) = log q_u(u) + log|det du/dx|``.
        """
        total = 0.0
        for spec, t, _ in self._blocks:
            if isinstance(spec, ContinuousVariable):
                total += t.log_abs_det_du_dx(np.array([recipe[spec.name]], dtype=float))
            else:
                x = np.array([recipe[f"{spec.name}.{c}"] for c in spec.components], dtype=float)
                total += t.log_abs_det_du_dx(x)
        return float(total)
