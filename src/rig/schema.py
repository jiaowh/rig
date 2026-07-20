"""The data record (implementation-plan §3.5): RunRecord / RecipeRecord / OutcomeRecord / Provenance.

Every row is a RunRecord, validated on read AND write (Pydantic v2 + Pint).
Units are canonicalized to SI base units at validation time - this kills the
sccm-vs-slm / degC-vs-K silent-unit bug class. ``Provenance.source`` is
load-bearing: all headline metrics are computed on ``source == "real_tool"``;
the physics sim is only bootstrap/prior.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

import pint
from pydantic import BaseModel, ConfigDict, Field, model_validator

from rig.interfaces import (
    CategoricalVariable,
    CompositionalVariable,
    ContinuousVariable,
    Modality,
    VariableSpec,
)

# One shared unit registry for the whole core. Pint's default registry does
# not define sccm/slm (standard-condition flow units); declare them as
# volumetric flows so they canonicalize to m^3/s.
ureg = pint.UnitRegistry()
ureg.define("sccm = cm ** 3 / min = standard_cubic_centimeter_per_minute")
ureg.define("slm = liter / min = standard_liter_per_minute")


class Quantity(BaseModel):
    """A quantity-like numeric recipe value {magnitude, unit}, canonicalized
    to SI base units at validation time (implementation-plan §3.5).

    Examples: Quantity(magnitude=100, unit="degC") -> 373.15 kelvin;
    Quantity(magnitude=10, unit="sccm") -> 1.667e-7 m^3/s.
    """

    model_config = ConfigDict(frozen=True)

    magnitude: float
    unit: str = "dimensionless"

    @model_validator(mode="after")
    def _canonicalize_to_si(self) -> Quantity:
        try:
            q = ureg.Quantity(self.magnitude, self.unit).to_base_units()
        except pint.errors.UndefinedUnitError as e:
            raise ValueError(f"unknown unit {self.unit!r}: {e}") from e
        # frozen model: bypass immutability for canonicalization-in-place
        object.__setattr__(self, "magnitude", float(q.magnitude))
        object.__setattr__(self, "unit", f"{q.units:~}" or "dimensionless")
        return self

    def as_pint(self) -> pint.Quantity:
        return ureg.Quantity(self.magnitude, self.unit)


class CategoricalValue(BaseModel):
    """A categorical recipe value, validated against its declared levels."""

    model_config = ConfigDict(frozen=True)

    value: str
    levels: tuple[str, ...]

    @model_validator(mode="after")
    def _check_level(self) -> CategoricalValue:
        if self.value not in self.levels:
            raise ValueError(
                f"categorical value {self.value!r} not in declared levels {self.levels}"
            )
        return self


class Fraction(BaseModel):
    """A member of a compositional/simplex factor (implementation-plan §3.1, §3.5).

    Deliberately a DISTINCT type from a bare :class:`Quantity`: a Fraction is
    a dimensionless simplex coordinate in [0, 1], never an independent flow.
    """

    model_config = ConfigDict(frozen=True)

    value: Annotated[float, Field(ge=0.0, le=1.0)]


class ArrayRef(BaseModel):
    """Placeholder reference to an array payload by content hash + path.

    Profiles/fields (curve_1d, field_2d) are stored out-of-band and
    referenced here; DVC tracking lands later (implementation-plan §13.1) - this type
    carries the hash/path contract without a DVC dependency.
    """

    model_config = ConfigDict(frozen=True)

    hash: str
    path: str


RecipeValue = Quantity | CategoricalValue | Fraction


class RecipeRecord(BaseModel):
    """Typed recipe: name -> Quantity | CategoricalValue | Fraction (§3.5)."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, RecipeValue]

    def validate_against(self, input_schema: list[VariableSpec] | tuple[VariableSpec, ...]) -> None:
        """Validate values against an adapter's declared input schema.

        Numeric values are compared in SI: declared bounds (in the variable's
        declared unit) are canonicalized the same way as the value. Raises
        ValueError on any violation.
        """
        by_name: dict[str, VariableSpec] = {}
        for spec in input_schema:
            if isinstance(spec, CompositionalVariable):
                for comp in spec.components:
                    by_name[f"{spec.name}.{comp}"] = spec
            else:
                by_name[spec.name] = spec

        for name, value in self.values.items():
            spec = by_name.get(name)
            if spec is None:
                raise ValueError(f"recipe value {name!r} not declared in adapter input schema")
            if isinstance(spec, ContinuousVariable):
                if not isinstance(value, Quantity):
                    raise ValueError(f"{name!r}: continuous variable requires a Quantity")
                lo = Quantity(magnitude=spec.lower, unit=spec.unit)
                hi = Quantity(magnitude=spec.upper, unit=spec.unit)
                if value.unit != lo.unit:
                    raise ValueError(
                        f"{name!r}: unit {value.unit!r} incompatible with declared {spec.unit!r}"
                    )
                if not (lo.magnitude <= value.magnitude <= hi.magnitude):
                    raise ValueError(
                        f"{name!r}: {value.magnitude} {value.unit} outside declared range "
                        f"[{lo.magnitude}, {hi.magnitude}] {lo.unit}"
                    )
            elif isinstance(spec, CategoricalVariable):
                if not isinstance(value, CategoricalValue):
                    raise ValueError(f"{name!r}: categorical variable requires a CategoricalValue")
                if value.value not in spec.levels:
                    raise ValueError(
                        f"{name!r}: level {value.value!r} not in adapter levels {spec.levels}"
                    )
            elif isinstance(spec, CompositionalVariable):
                if not isinstance(value, Fraction):
                    raise ValueError(
                        f"{name!r}: compositional component requires a Fraction "
                        "(NOT a bare Quantity - implementation-plan §3.1)"
                    )

        # Simplex sum-to-1 + component completeness (implementation-plan §3.1: compositional
        # factors live on the simplex). Enforced for any factor that APPEARS in
        # the recipe — a partial or !=1 composition is a data-contract violation
        # (the per-component [0,1] Fraction bound alone does not imply sum-to-1).
        for spec in input_schema:
            if not isinstance(spec, CompositionalVariable):
                continue
            comp_keys = [f"{spec.name}.{c}" for c in spec.components]
            present = [k for k in comp_keys if k in self.values]
            if not present:
                continue  # factor absent entirely: presence is the encoder's gate
            missing = [k for k in comp_keys if k not in self.values]
            if missing:
                raise ValueError(
                    f"{spec.name!r}: compositional factor missing components "
                    f"{missing} (implementation-plan §3.1: all simplex components required)"
                )
            total = math.fsum(float(self.values[k].value) for k in comp_keys)
            if not math.isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError(
                    f"{spec.name!r}: compositional components sum to {total:.6g}, "
                    "must be 1 (implementation-plan §3.1: simplex, sum-to-1)"
                )


class OutcomeRecord(BaseModel):
    """One modality-tagged measured outcome (implementation-plan §3.5).

    ``scalar_vector`` outcomes carry an inline SI-canonical Quantity;
    ``curve_1d`` / ``field_2d`` payloads must be ArrayRefs.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    modality: Modality
    value: Quantity | ArrayRef

    @model_validator(mode="after")
    def _check_modality_payload(self) -> OutcomeRecord:
        if self.modality in ("curve_1d", "field_2d") and not isinstance(self.value, ArrayRef):
            raise ValueError(
                f"{self.name!r}: modality {self.modality!r} requires an ArrayRef payload"
            )
        if self.modality == "scalar_vector" and not isinstance(self.value, Quantity):
            raise ValueError(f"{self.name!r}: scalar_vector outcome requires an inline Quantity")
        return self


class Provenance(BaseModel):
    """Provenance of a run (implementation-plan §3.5). ``source`` is load-bearing:
    headline metrics only ever on ``real_tool``; ``physics_sim`` is
    bootstrap/prior only."""

    model_config = ConfigDict(frozen=True)

    source: Literal["physics_sim", "real_tool"]
    operator: str | None = None
    calibration_state: str | None = None
    data_hash: str | None = None
    git_sha: str | None = None


class RunRecord(BaseModel):
    """One validated run row (implementation-plan §3.5)."""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    process_id: str
    tool_id: str  # chamber/tool identity -> leave-tool-out splits & drift
    timestamp: datetime  # -> temporal splits & drift monitoring
    recipe: RecipeRecord
    outcomes: list[OutcomeRecord]
    provenance: Provenance
    extra: dict[str, Any] = Field(default_factory=dict)
