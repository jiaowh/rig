"""Declarative constraint set (implementation-plan §3.1, §8.3).

Pure declarations + a pointwise ``validate(x)`` checker. Enforcement is
constraint-by-construction wherever possible (see :mod:`rig.transforms`);
penalties/projections only for what cannot be parameterized away (§8.3).
Monotone constraints relate an OUTPUT to an input, so they cannot be checked
pointwise on a recipe - they are declarations consumed by the surrogate
(§6) and are skipped by ``validate``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from rig.interfaces import ChangeCost

_ATOL = 1e-9


@dataclass(frozen=True)
class BoxConstraint:
    """lower <= x[name] <= upper (in the variable's canonical unit)."""

    name: str
    lower: float
    upper: float


@dataclass(frozen=True)
class SimplexConstraint:
    """Mixture/compositional constraint: named components are non-negative
    and sum to ``total`` (default 1). TRUE compositions only - independent
    MFC flows are NOT a simplex (implementation-plan §3.1)."""

    components: tuple[str, ...]
    total: float = 1.0


@dataclass(frozen=True)
class LinearConstraint:
    """Linear coupling: lower <= sum_i coefficients[i] * x[i] <= upper."""

    coefficients: Mapping[str, float]
    lower: float = float("-inf")
    upper: float = float("inf")


@dataclass(frozen=True)
class MonotoneConstraint:
    """Declared monotonicity of an output w.r.t. an input (implementation-plan §6.3).

    Not pointwise-checkable on a recipe; consumed by the surrogate/shape
    layer. Kept here so the adapter declares it in one place."""

    output: str
    wrt_input: str
    direction: Literal["increasing", "decreasing"]


@dataclass(frozen=True)
class ConstraintSet:
    """Declarative constraint bundle for one process (implementation-plan §3.1).

    ``change_cost`` tags each variable hard_to_change vs easy (split-plot
    structure, §8.3: hard-to-change factors are conditioning, not free).
    """

    box: tuple[BoxConstraint, ...] = ()
    simplex: tuple[SimplexConstraint, ...] = ()
    linear: tuple[LinearConstraint, ...] = ()
    monotone: tuple[MonotoneConstraint, ...] = ()
    change_cost: Mapping[str, ChangeCost] = field(default_factory=dict)

    def validate(self, x: Mapping[str, float]) -> list[str]:
        """Check a flat numeric recipe pointwise. Returns violation messages
        (empty list = satisfied). Monotone declarations are skipped (not
        pointwise-checkable)."""
        violations: list[str] = []
        for b in self.box:
            if b.name not in x:
                violations.append(f"box: missing variable {b.name!r}")
            elif not (b.lower - _ATOL <= x[b.name] <= b.upper + _ATOL):
                violations.append(f"box: {b.name}={x[b.name]} outside [{b.lower}, {b.upper}]")
        for s in self.simplex:
            missing = [c for c in s.components if c not in x]
            if missing:
                violations.append(f"simplex: missing components {missing}")
                continue
            vals = [x[c] for c in s.components]
            if any(v < -_ATOL for v in vals):
                violations.append(f"simplex: negative component in {s.components}")
            if abs(sum(vals) - s.total) > 1e-6:
                violations.append(
                    f"simplex: components {s.components} sum to {sum(vals)}, expected {s.total}"
                )
        for lin in self.linear:
            missing = [n for n in lin.coefficients if n not in x]
            if missing:
                violations.append(f"linear: missing variables {missing}")
                continue
            val = sum(c * x[n] for n, c in lin.coefficients.items())
            if not (lin.lower - _ATOL <= val <= lin.upper + _ATOL):
                violations.append(
                    f"linear: {dict(lin.coefficients)} -> {val} outside [{lin.lower}, {lin.upper}]"
                )
        return violations

    def is_satisfied(self, x: Mapping[str, float]) -> bool:
        return not self.validate(x)
