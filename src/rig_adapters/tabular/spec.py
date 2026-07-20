"""Declarative process-spec format for the generic tabular adapter (WP-H).

This is the schema-elicitation intake of implementation-plan §15.6 E5: a process engineer
describes their process (variables, bounds, units, change-cost classes, the
genuine mixture-vs-independent-MFC tags, outputs, cost model) in a small
declarative file, and everything downstream — adapter, ingestion, DoE seeds —
is derived from it. Validation is deliberately strict and loud: a typo in a
spec must fail at load time with a message that names the offending key, not
surface later as a silent unit bug.

Format choice: **TOML is the primary format** (loaded with stdlib
``tomllib`` — zero new dependencies, comment-friendly for annotated specs,
and the same dialect as ``pyproject.toml``). JSON is accepted as a secondary
format for machine-generated specs (``load_spec`` dispatches on the file
suffix; ``parse_spec`` takes an already-parsed mapping).

Example (see ``examples/pecvd_example.toml`` for a fully annotated spec)::

    process_id = "pecvd_sin"

    [inputs.temperature]
    kind = "continuous"
    unit = "degC"
    lower = 200.0
    upper = 400.0

    [inputs.precursor_blend]
    kind = "compositional"          # TRUE sum-to-1 fractions (implementation-plan §3.1)
    components = ["silane", "ammonia", "nitrogen"]

    [outputs.thickness]
    unit = "nm"

    [cost]
    c_batch = 1000.0
    c_recipe = 1000.0
    batch_size = 4

The E5-mandated compositional tag is enforced here: a ``compositional`` block
whose declared unit is not dimensionless (e.g. ``sccm``) is REJECTED —
independent MFC gas flows are NOT a simplex (implementation-plan §3.1); declare each flow
as its own ``continuous`` variable instead.
"""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pint

from rig.interfaces import (
    CategoricalVariable,
    ChangeCost,
    CompositionalVariable,
    ContinuousVariable,
    OutputSpec,
    VariableSpec,
)
from rig.schema import ureg  # the ONE shared pint registry — never a second one

DEFAULT_C_BATCH = 1000.0  # Kanarik defaults (implementation-plan §11.1), matching the MBE adapter
DEFAULT_C_RECIPE = 1000.0
DEFAULT_BATCH_SIZE = 4

_ALLOWED_TOP_KEYS = {"process_id", "description", "inputs", "outputs", "cost"}
_ALLOWED_INPUT_KEYS = {
    "continuous": {"kind", "unit", "lower", "upper", "change_cost"},
    "categorical": {"kind", "levels", "change_cost"},
    # "unit" is syntactically accepted for compositional blocks purely so we
    # can reject non-dimensionless declarations with a targeted message.
    "compositional": {"kind", "components", "change_cost", "unit"},
}
_ALLOWED_OUTPUT_KEYS = {"unit", "modality", "target", "lower_spec", "upper_spec"}
_ALLOWED_COST_KEYS = {"c_batch", "c_recipe", "batch_size"}
_CHANGE_COST_MAP = {
    "easy": ChangeCost.EASY,
    "hard": ChangeCost.HARD_TO_CHANGE,
    "hard_to_change": ChangeCost.HARD_TO_CHANGE,
}


class SpecError(ValueError):
    """A process spec failed validation. The message names the offending key."""


def _si_bound(value: float, unit: str) -> float:
    """Convert one declared-unit VALUE to its SI base magnitude, offset-safe.

    Via pint Quantity arithmetic — never a multiplicative scale factor — so an
    OFFSET unit like ``degC`` converts by +273.15, not by a bogus scale. Bounds
    and spec limits are absolute POINTS, not deltas: ``200 degC`` is an absolute
    temperature ``473.15 K`` (a WIDTH would convert by factor 1, but these fields
    are points, not widths).
    """
    return float(ureg.Quantity(value, unit).to_base_units().magnitude)


def _si_unit(unit: str) -> str:
    """The SI base-unit symbol for ``unit`` (``''`` -> ``'dimensionless'``)."""
    return f"{ureg.Quantity(1.0, unit).to_base_units().units:~}" or "dimensionless"


@dataclass(frozen=True)
class ProcessSpec:
    """A fully validated declarative process description (implementation-plan §3.1 fields)."""

    process_id: str
    variables: tuple[VariableSpec, ...]
    outputs: tuple[OutputSpec, ...]
    description: str = ""
    c_batch: float = DEFAULT_C_BATCH
    c_recipe: float = DEFAULT_C_RECIPE
    batch_size: int = DEFAULT_BATCH_SIZE
    source: str = "<dict>"  # where the spec was loaded from (diagnostics only)

    # derived lookups (populated in __post_init__)
    continuous: tuple[ContinuousVariable, ...] = field(init=False)
    categorical: tuple[CategoricalVariable, ...] = field(init=False)
    compositional: tuple[CompositionalVariable, ...] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "continuous",
            tuple(v for v in self.variables if isinstance(v, ContinuousVariable)),
        )
        object.__setattr__(
            self,
            "categorical",
            tuple(v for v in self.variables if isinstance(v, CategoricalVariable)),
        )
        object.__setattr__(
            self,
            "compositional",
            tuple(v for v in self.variables if isinstance(v, CompositionalVariable)),
        )

    @property
    def continuous_si(self) -> tuple[ContinuousVariable, ...]:
        """``continuous``, with bounds converted to SI base units (§3.5).

        **Use THIS — not ``continuous`` — for anything that touches ingested data:
        fitting, the §8 inverse's ``variables``, support scores, plots.**

        The trap this closes (audit 2026-07-17, a live defect in the sputtering
        example): ``continuous`` carries the bounds in the SPEC-DECLARED units,
        because ``ingest`` must read CSV cells in those units. But ingest then
        SI-canonicalizes every value, so every array downstream is SI. Pairing
        declared-unit BOUNDS with SI-canonical DATA silently searches the wrong
        space, and the mismatch is invisible when the declared unit happens to be
        SI already (W stays W) — it only bites on the scaled ones. In the real
        sputtering spec, pressure is declared ``1..43 mtorr``; the data lands in
        ``0.133..5.73 Pa``; the inverse was handed ``1..43`` and searched
        ``1..43 Pa ≈ 7.5..322 mTorr`` — a range whose LOWER bound sits above the
        data's maximum. Only the §8.2 fail-closed support floor kept the returned
        recipes on-support; nothing errored.
        """
        out = []
        for v in self.continuous:
            # Convert each bound as an absolute quantity (offset-safe, finding A) —
            # NOT via a shared scale factor, which mis-converts degC and other
            # offset units.
            out.append(
                ContinuousVariable(
                    v.name,
                    _si_bound(v.lower, v.unit),
                    _si_bound(v.upper, v.unit),
                    unit=_si_unit(v.unit),
                    change_cost=v.change_cost,
                )
            )
        return tuple(out)

    @property
    def outputs_si(self) -> tuple[OutputSpec, ...]:
        """``outputs``, with ``target``/``lower_spec``/``upper_spec`` converted to
        SI base units (§3.5) — the output-side twin of :attr:`continuous_si`.

        **Use THIS — not ``outputs`` — for anything that compares a spec limit to
        ingested OUTCOME data**, which ``ingest`` SI-canonicalizes. The same latent
        trap ``continuous_si`` closes on the input side (declared-unit limits paired
        with SI data silently compare the wrong space; invisible when the declared
        unit is already SI) applies here; see the CLAUDE.md SI-trap note. Conversion
        is offset-safe (finding A), so a ``degC`` target/spec converts by +273.15.
        Unset limits (``None``) stay ``None``.
        """
        out = []
        for o in self.outputs:
            out.append(
                OutputSpec(
                    o.name,
                    o.modality,
                    unit=_si_unit(o.unit),
                    target=None if o.target is None else _si_bound(o.target, o.unit),
                    lower_spec=None if o.lower_spec is None else _si_bound(o.lower_spec, o.unit),
                    upper_spec=None if o.upper_spec is None else _si_bound(o.upper_spec, o.unit),
                )
            )
        return tuple(out)

    @property
    def flat_input_names(self) -> tuple[str, ...]:
        """All input column names in spec order, with compositional components
        flattened as ``"<variable>.<component>"`` (WP-A standing decision)."""
        names: list[str] = []
        for var in self.variables:
            if isinstance(var, CompositionalVariable):
                names.extend(f"{var.name}.{comp}" for comp in var.components)
            else:
                names.append(var.name)
        return tuple(names)

    @property
    def numeric_input_names(self) -> tuple[str, ...]:
        """Flat input names excluding categoricals (which are conditioning,
        implementation-plan §8.3) — the encoder/DoE dimension ordering."""
        names: list[str] = []
        for var in self.variables:
            if isinstance(var, CompositionalVariable):
                names.extend(f"{var.name}.{comp}" for comp in var.components)
            elif isinstance(var, ContinuousVariable):
                names.append(var.name)
        return tuple(names)

    @property
    def gp_input_keys(self) -> tuple[str, ...]:
        """``numeric_input_names`` with exactly ONE component dropped per
        compositional variable — the rank-safe GP design (audit B4).

        A compositional factor sums to 1, so keeping all its components makes the
        design matrix exactly collinear (a rank-deficient GP with a phantom DOF).
        This drops the reference component of EACH compositional variable
        regardless of declaration order — unlike ``numeric_input_names[:-1]``,
        which is correct only when a single compositional block is declared last.
        """
        dropped = {f"{v.name}.{v.components[0]}" for v in self.compositional}
        return tuple(k for k in self.numeric_input_names if k not in dropped)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.outputs)


def _err(where: str, msg: str) -> SpecError:
    return SpecError(f"process spec: {where}: {msg}")


def _require_number(value: Any, where: str, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _err(where, f"{key!r} must be a number, got {value!r}")
    return float(value)


def _check_unit(unit: Any, where: str) -> str:
    if not isinstance(unit, str) or not unit:
        raise _err(where, f"'unit' must be a non-empty string, got {unit!r}")
    try:
        ureg.Unit(unit)
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError, ValueError) as e:
        raise _err(where, f"unit {unit!r} is not pint-parseable: {e}") from e
    return unit


def _check_name(name: str, where: str) -> None:
    if not name:
        raise _err(where, "names must be non-empty")
    if "." in name:
        raise _err(
            where,
            f"name {name!r} may not contain '.': dots are reserved for the "
            "'<variable>.<component>' compositional flattening convention",
        )


def _parse_change_cost(raw: Any, where: str) -> ChangeCost:
    if raw is None:
        return ChangeCost.EASY
    if isinstance(raw, str) and raw.lower() in _CHANGE_COST_MAP:
        return _CHANGE_COST_MAP[raw.lower()]
    raise _err(where, f"change_cost must be 'easy' or 'hard', got {raw!r}")


def _parse_input(name: str, block: Any) -> VariableSpec:
    where = f"inputs.{name}"
    _check_name(name, where)
    if not isinstance(block, Mapping):
        raise _err(where, f"must be a table/object, got {type(block).__name__}")
    kind = block.get("kind")
    if kind not in _ALLOWED_INPUT_KEYS:
        raise _err(
            where,
            f"'kind' must be one of {sorted(_ALLOWED_INPUT_KEYS)}, got {kind!r}",
        )
    unknown = set(block) - _ALLOWED_INPUT_KEYS[kind]
    if unknown:
        raise _err(
            where,
            f"unknown key(s) {sorted(unknown)} for kind {kind!r} "
            f"(allowed: {sorted(_ALLOWED_INPUT_KEYS[kind])})",
        )
    change_cost = _parse_change_cost(block.get("change_cost"), where)

    if kind == "continuous":
        for req in ("lower", "upper"):
            if req not in block:
                raise _err(where, f"continuous variable requires {req!r}")
        lower = _require_number(block["lower"], where, "lower")
        upper = _require_number(block["upper"], where, "upper")
        if not lower < upper:
            raise _err(where, f"lower ({lower}) must be < upper ({upper})")
        unit = _check_unit(block.get("unit", "dimensionless"), where)
        return ContinuousVariable(name, lower, upper, unit=unit, change_cost=change_cost)

    if kind == "categorical":
        levels = block.get("levels")
        if (
            not isinstance(levels, (list, tuple))
            or len(levels) < 2
            or not all(isinstance(lv, str) and lv for lv in levels)
            or len(set(levels)) != len(levels)
        ):
            raise _err(where, f"'levels' must be >=2 unique non-empty strings, got {levels!r}")
        return CategoricalVariable(name, tuple(levels), change_cost=change_cost)

    # compositional
    components = block.get("components")
    if (
        not isinstance(components, (list, tuple))
        or len(components) < 2
        or not all(isinstance(c, str) and c for c in components)
        or len(set(components)) != len(components)
    ):
        raise _err(where, f"'components' must be >=2 unique non-empty strings, got {components!r}")
    for comp in components:
        _check_name(comp, f"{where}.components")
    if "unit" in block:
        unit = _check_unit(block["unit"], where)
        # Compare the UNIT itself, not just its dimensionality (audit D10):
        # 'percent'/'ppm' are dimensionless-DIMENSIONALITY but scaled, and ingest
        # ignores the unit and reads raw values as fractions — so a 'percent' block
        # would silently mis-scale (or crash the [0,1] Fraction bound). Only a
        # literally-dimensionless unit (or no unit at all) is a true fraction.
        if ureg.Unit(unit) != ureg.Unit("dimensionless"):
            raise _err(
                where,
                f"compositional variables are TRUE sum-to-1 fractions and must be "
                f"literally dimensionless, but declared unit {unit!r}. Scaled units "
                "like percent/ppm are ignored at ingest (values are read as raw "
                "fractions), and independent MFC gas flows (e.g. sccm) are NOT a "
                "simplex (implementation-plan §3.1). Drop the unit (or use 'dimensionless') for "
                "genuine fractions, or declare each flow as its own 'continuous' "
                "variable.",
            )
    return CompositionalVariable(name, tuple(components), change_cost=change_cost)


def _parse_output(name: str, block: Any) -> OutputSpec:
    where = f"outputs.{name}"
    _check_name(name, where)
    if not isinstance(block, Mapping):
        raise _err(where, f"must be a table/object, got {type(block).__name__}")
    unknown = set(block) - _ALLOWED_OUTPUT_KEYS
    if unknown:
        raise _err(
            where, f"unknown key(s) {sorted(unknown)} (allowed: {sorted(_ALLOWED_OUTPUT_KEYS)})"
        )
    modality = block.get("modality", "scalar_vector")
    if modality in ("curve_1d", "field_2d"):
        raise _err(
            where,
            f"modality {modality!r} is not yet supported by the tabular adapter "
            "(v0 handles 'scalar_vector' only; curve/field payloads need the "
            "ArrayRef/DVC pipeline, implementation-plan §13.1)",
        )
    if modality != "scalar_vector":
        raise _err(where, f"unknown modality {modality!r} (v0 supports 'scalar_vector')")
    unit = _check_unit(block.get("unit", "dimensionless"), where)
    target = lower_spec = upper_spec = None
    if "target" in block:
        target = _require_number(block["target"], where, "target")
    if "lower_spec" in block:
        lower_spec = _require_number(block["lower_spec"], where, "lower_spec")
    if "upper_spec" in block:
        upper_spec = _require_number(block["upper_spec"], where, "upper_spec")
    if lower_spec is not None and upper_spec is not None and not lower_spec < upper_spec:
        raise _err(where, f"lower_spec ({lower_spec}) must be < upper_spec ({upper_spec})")
    if target is not None:
        if lower_spec is not None and target < lower_spec:
            raise _err(where, f"target ({target}) below lower_spec ({lower_spec})")
        if upper_spec is not None and target > upper_spec:
            raise _err(where, f"target ({target}) above upper_spec ({upper_spec})")
    return OutputSpec(
        name,
        "scalar_vector",
        unit=unit,
        target=target,
        lower_spec=lower_spec,
        upper_spec=upper_spec,
    )


def parse_spec(data: Mapping[str, Any], source: str = "<dict>") -> ProcessSpec:
    """Validate an already-parsed mapping into a :class:`ProcessSpec`.

    Raises :class:`SpecError` with a message naming the offending key on any
    violation (the E5 schema-elicitation intake must fail loudly at load).
    """
    if not isinstance(data, Mapping):
        raise SpecError(f"process spec ({source}): top level must be a table/object")
    unknown = set(data) - _ALLOWED_TOP_KEYS
    if unknown:
        raise _err(
            source,
            f"unknown top-level key(s) {sorted(unknown)} (allowed: {sorted(_ALLOWED_TOP_KEYS)})",
        )

    process_id = data.get("process_id")
    if not isinstance(process_id, str) or not process_id:
        raise _err(source, f"'process_id' must be a non-empty string, got {process_id!r}")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise _err(source, f"'description' must be a string, got {description!r}")

    inputs = data.get("inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise _err(source, "'inputs' must be a non-empty table of variable blocks")
    variables = tuple(_parse_input(name, block) for name, block in inputs.items())

    outputs = data.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise _err(source, "'outputs' must be a non-empty table of output blocks")
    output_specs = tuple(_parse_output(name, block) for name, block in outputs.items())

    cost = data.get("cost", {})
    if not isinstance(cost, Mapping):
        raise _err(source, "'cost' must be a table/object")
    unknown = set(cost) - _ALLOWED_COST_KEYS
    if unknown:
        raise _err(
            source,
            f"cost: unknown key(s) {sorted(unknown)} (allowed: {sorted(_ALLOWED_COST_KEYS)})",
        )
    c_batch = _require_number(cost.get("c_batch", DEFAULT_C_BATCH), "cost", "c_batch")
    c_recipe = _require_number(cost.get("c_recipe", DEFAULT_C_RECIPE), "cost", "c_recipe")
    if c_batch < 0 or c_recipe < 0:
        raise _err("cost", "c_batch and c_recipe must be >= 0")
    batch_size_raw = cost.get("batch_size", DEFAULT_BATCH_SIZE)
    if (
        isinstance(batch_size_raw, bool)
        or not isinstance(batch_size_raw, int)
        or batch_size_raw < 1
    ):
        raise _err("cost", f"batch_size must be an integer >= 1, got {batch_size_raw!r}")

    spec = ProcessSpec(
        process_id=process_id,
        description=description,
        variables=variables,
        outputs=output_specs,
        c_batch=c_batch,
        c_recipe=c_recipe,
        batch_size=batch_size_raw,
        source=source,
    )

    # cross-cutting name checks on the FLATTENED namespace
    flat = spec.flat_input_names
    dupes = sorted({n for n in flat if list(flat).count(n) > 1})
    if dupes:
        raise _err(source, f"flattened input names collide: {dupes}")
    out_dupes = sorted({n for n in spec.output_names if spec.output_names.count(n) > 1})
    if out_dupes:  # pragma: no cover - TOML/JSON object keys are already unique
        raise _err(source, f"output names collide: {out_dupes}")
    overlap = sorted(set(flat) & set(spec.output_names))
    if overlap:
        raise _err(source, f"name(s) declared as both input and output: {overlap}")
    return spec


def load_spec(path: str | Path) -> ProcessSpec:
    """Load and validate a process spec from a ``.toml`` (primary) or
    ``.json`` file. Raises :class:`SpecError` on any problem."""
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"process spec file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".toml":
        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except tomllib.TOMLDecodeError as e:
            raise SpecError(f"process spec ({path}): invalid TOML: {e}") from e
    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SpecError(f"process spec ({path}): invalid JSON: {e}") from e
    else:
        raise SpecError(
            f"process spec ({path}): unsupported suffix {suffix!r} (use .toml or .json)"
        )
    return parse_spec(data, source=str(path))
