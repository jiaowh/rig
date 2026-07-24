"""Frame validation for the tabular adapter's CSV ingest path (implementation-plan §15.6
E1 — deferred 2026-07-17 when pandera was DROPPED because no real data contract existed
yet to validate against, see CLAUDE.md; built now against the ACTUAL contract that landed
with M0 / Empa HiPIMS, 2026-07-23).

Validates a tabular "frame" — CSV header + row dicts, exactly the shape
``csv.DictReader`` already hands :func:`rig_adapters.tabular.ingest.ingest_csv` — against
a :class:`~rig_adapters.tabular.spec.ProcessSpec`, BEFORE row-wise ``RunRecord``
construction. Produces a typed, machine-readable :class:`ValidationReport` (never print
statements) so a caller can see exactly which rows/columns tripped which check.

**No new dependency.** rig has no pandas/pandera in ``pyproject.toml`` (pydantic + pint +
numpy + scipy only) — pandera was dropped precisely for implying a guarantee that did not
exist. "Frame" here is deliberately just what the ingest path already reads: a header
tuple + a tuple of row mappings. Introducing a heavy dataframe dependency for this would
repeat the same mistake pandera was dropped for (declaring machinery beyond what the
actual contract needs).

**The SI trap (CLAUDE.md "SI is the contract, and it is a live trap"):** ingest
canonicalizes VALUES to SI, but a spec's declared BOUNDS stay in the declared unit. Every
check below reads the RAW CSV cell — still in the declared unit, never ingested/converted
— and compares it against ``spec.continuous`` / ``spec.categorical`` / ``spec.compositional``
(declared-unit bounds/levels). **Never** ``spec.continuous_si`` and never an SI-canonical
value. Because both sides already share the same declared unit, the bounds check needs NO
pint conversion at all — which also means it reproduces ingest's own accept/reject boundary
bit-for-bit (no extra floating-point noise from a round-trip through SI); this is pinned by
the Empa-agreement test in ``tests/test_frame_validation.py``.

Checks (name -> what it enforces vs. what it only reports):

- ``missing_columns`` (BLOCKING) — every declared input/output (+ optional order/tool/
  timestamp column) must be present in the header. ``ingest_csv`` itself hard-errors on
  this unconditionally, for every ``on_error`` policy; this check catches the same thing
  before a single row is touched.
- ``dtype`` (BLOCKING) — every numeric column (continuous inputs, compositional
  components, outputs) must parse as a float in every row. A stray string is the "silent
  object-dtype" failure a pandas-like frame would hide; ingest's own ``_parse_float``
  already turns this into a per-row ``ValueError`` — this check makes it explicit and
  frame-wide instead of discovered one row at a time.
- ``nan_inf`` (BLOCKING) — no NaN/inf in any numeric column. Ingest enforces this
  explicitly for outcomes (``math.isfinite``) and only implicitly for continuous inputs (a
  NaN silently fails the bounds compare and is reported as a bounds violation instead) —
  this check makes the enforcement explicit and uniform for both input and output columns.
- ``bounds`` (ADVISORY) — declared-unit bounds / categorical-level / simplex membership
  for every knob column. Ingest already enforces this PER ROW (raises under
  ``on_error="raise"``, skips under ``on_error="skip"``) — this check REPORTS exactly the
  same violating rows without failing the frame as a whole, mirroring the documented
  "skip" behavior (the Ti-120W 3e-11 rounding edge is real data, not a corrupt frame).
- ``order_key`` (ADVISORY) — when an order-key column is named, its values must be
  monotone non-decreasing; an all-equal (degenerate) key carries NO run-order information
  and is flagged ``UNVERIFIED-ORDER`` rather than silently treated as valid ordering (the
  Ti-120W BatchNr==1-everywhere case).
- ``duplicate_rows`` (ADVISORY) — exact recipe+outcome duplicate rows are reported (row
  groups), never rejected — ingest has no de-duplication logic today.

``ValidationReport.passed`` is True iff there is no BLOCKING violation. ``strict=True``
raises :class:`FrameValidationError` when ``passed`` is False — a genuine "stop before
touching a corrupt frame" gate, not a "refuse every known real-data quirk" gate (bounds /
order-key / duplicate-row violations stay advisory, matching the pre-existing
skip-reject behavior of ``ingest_csv``).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from rig_adapters.tabular.spec import ProcessSpec

# Mirrors rig_adapters.tabular.ingest.SUM_TO_ONE_ATOL (kept independent on purpose:
# this module must not import ingest.py, which imports THIS module — see the
# "wiring" section of ingest.py).
SUM_TO_ONE_ATOL = 1e-6

Severity = Literal["blocking", "advisory"]


class FrameValidationError(ValueError):
    """Raised by :func:`validate_frame` (or ``ingest_csv(..., strict=True)``) when a
    BLOCKING violation is present."""


@dataclass(frozen=True)
class Frame:
    """A tabular frame: CSV header + row dicts — exactly what ``csv.DictReader``
    already produces for :func:`rig_adapters.tabular.ingest.ingest_csv`."""

    header: tuple[str, ...]
    rows: tuple[Mapping[str, Any], ...]


def frame_from_csv(path: str | Path) -> Frame:
    """Read a CSV into a :class:`Frame` the same way ``ingest_csv`` reads it
    (``utf-8-sig``, ``csv.DictReader``) — so validating the result and ingesting it
    see identical cells."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(h for h in (reader.fieldnames or []) if h is not None)
        rows = tuple(dict(row) for row in reader)
    return Frame(header=header, rows=rows)


@dataclass(frozen=True)
class Violation:
    """One check's finding. ``row_indices`` are 0-based data-row indices (header
    excluded) — the same indexing ``RejectedRow.row_index`` uses."""

    check: str
    severity: Severity
    message: str
    row_indices: tuple[int, ...] = ()
    columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """Machine-readable validation output: never print statements. ``passed`` is a
    computed property (not a settable field) so it can never drift out of sync with
    ``violations``."""

    n_rows: int
    violations: tuple[Violation, ...] = ()

    @property
    def passed(self) -> bool:
        return not any(v.severity == "blocking" for v in self.violations)

    def by_check(self, check: str) -> tuple[Violation, ...]:
        return tuple(v for v in self.violations if v.check == check)

    def rows_for(self, check: str) -> tuple[int, ...]:
        """All row indices flagged by any violation of ``check``, sorted+deduped."""
        rows: set[int] = set()
        for v in self.by_check(check):
            rows.update(v.row_indices)
        return tuple(sorted(rows))

    def summary(self, max_rows_shown: int = 8) -> str:
        """One-line-per-violation human summary (for CLI/console printing)."""
        lines = [
            f"validation: {self.n_rows} row(s), {len(self.violations)} violation(s), "
            f"passed={self.passed}"
        ]
        for v in self.violations:
            if v.row_indices:
                shown = list(v.row_indices[:max_rows_shown])
                more = len(v.row_indices) - len(shown)
                rows_str = f" rows={shown}" + (f" (+{more} more)" if more > 0 else "")
            else:
                rows_str = ""
            lines.append(f"  [{v.severity}] {v.check}: {v.message}{rows_str}")
        return "\n".join(lines)


def _parse_numeric(raw: Any) -> float | None:
    """``None`` on anything that is not a finite-or-not-but-parseable float:
    missing cell, blank string, or a stray non-numeric string. NaN/inf DO parse here
    (they are valid floats) -- that distinction is what separates the ``dtype`` check
    from the ``nan_inf`` check."""
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _check_order_key(frame: Frame, order_key: str | None) -> list[Violation]:
    if not order_key or order_key not in frame.header:
        return []
    parsed: list[float | None] = [_parse_numeric(row.get(order_key)) for row in frame.rows]
    bad_rows = [i for i, v in enumerate(parsed) if v is None]
    if bad_rows:
        return [
            Violation(
                check="order_key",
                severity="advisory",
                message=(
                    f"order key {order_key!r}: {len(bad_rows)} non-numeric value(s); "
                    "cannot assess monotonicity"
                ),
                row_indices=tuple(bad_rows),
                columns=(order_key,),
            )
        ]
    values = [v for v in parsed if v is not None]
    if len(set(values)) == 1:
        return [
            Violation(
                check="order_key",
                severity="advisory",
                message=(
                    f"UNVERIFIED-ORDER: order key {order_key!r} is constant "
                    f"({values[0]!r}) across all {len(values)} row(s) -- carries NO "
                    "run-order information; downstream temporal splits over this key "
                    "are meaningless (implementation-plan §12.4)"
                ),
                row_indices=tuple(range(len(values))),
                columns=(order_key,),
            )
        ]
    out_of_order = [i for i in range(1, len(values)) if values[i] < values[i - 1]]
    if out_of_order:
        return [
            Violation(
                check="order_key",
                severity="advisory",
                message=(
                    f"order key {order_key!r} is not monotone non-decreasing at "
                    f"row(s) {out_of_order}"
                ),
                row_indices=tuple(out_of_order),
                columns=(order_key,),
            )
        ]
    return []


def _check_duplicate_rows(frame: Frame, recipe_outcome_cols: Sequence[str]) -> list[Violation]:
    if not recipe_outcome_cols:
        return []
    seen: dict[tuple[str, ...], list[int]] = {}
    for i, row in enumerate(frame.rows):
        key = tuple(str(row.get(c, "")).strip() for c in recipe_outcome_cols)
        seen.setdefault(key, []).append(i)
    violations = []
    for key, idxs in seen.items():
        if len(idxs) > 1:
            values = dict(zip(recipe_outcome_cols, key, strict=True))
            violations.append(
                Violation(
                    check="duplicate_rows",
                    severity="advisory",
                    message=(f"{len(idxs)} row(s) share an identical recipe+outcome: {values}"),
                    row_indices=tuple(idxs),
                    columns=tuple(recipe_outcome_cols),
                )
            )
    return violations


def validate_frame(
    frame: Frame,
    spec: ProcessSpec,
    *,
    order_key: str | None = None,
    tool_column: str | None = None,
    timestamp_column: str | None = None,
    strict: bool = False,
) -> ValidationReport:
    """Validate ``frame`` against ``spec`` (declared-unit checks only — see module
    docstring on the SI trap). Returns a :class:`ValidationReport`; with
    ``strict=True`` raises :class:`FrameValidationError` when the report has any
    BLOCKING violation."""
    violations: list[Violation] = []
    header_set = set(frame.header)

    input_names = list(spec.flat_input_names)
    output_names = list(spec.output_names)
    required: list[str] = []
    for c in (*input_names, *output_names, order_key, tool_column, timestamp_column):
        if c and c not in required:
            required.append(c)

    missing = [c for c in required if c not in header_set]
    if missing:
        violations.append(
            Violation(
                check="missing_columns",
                severity="blocking",
                message=(
                    f"required column(s) missing from header: {missing}; "
                    f"header is {list(frame.header)}"
                ),
                columns=tuple(missing),
            )
        )

    continuous_by_name = {v.name: v for v in spec.continuous}
    categorical_by_name = {v.name: v for v in spec.categorical}
    compositional_component_of = {
        f"{v.name}.{comp}": v for v in spec.compositional for comp in v.components
    }

    numeric_cols = [
        c
        for c in (*continuous_by_name, *compositional_component_of, *output_names)
        if c in header_set
    ]
    # Parse every numeric cell exactly once; every subsequent check reuses this cache
    # instead of re-parsing (and instead of re-deriving "is this a dtype problem?").
    cells: dict[str, list[float | None]] = {
        col: [_parse_numeric(row.get(col)) for row in frame.rows] for col in numeric_cols
    }

    for col, vals in cells.items():
        bad = [i for i, v in enumerate(vals) if v is None]
        if bad:
            violations.append(
                Violation(
                    check="dtype",
                    severity="blocking",
                    message=(
                        f"column {col!r}: {len(bad)} row(s) are not numeric "
                        "(missing, blank, or a stray non-numeric string)"
                    ),
                    row_indices=tuple(bad),
                    columns=(col,),
                )
            )

    for col, vals in cells.items():
        bad = [i for i, v in enumerate(vals) if v is not None and not math.isfinite(v)]
        if bad:
            violations.append(
                Violation(
                    check="nan_inf",
                    severity="blocking",
                    message=f"column {col!r}: {len(bad)} row(s) are NaN/inf",
                    row_indices=tuple(bad),
                    columns=(col,),
                )
            )

    # -- bounds: continuous knobs, DECLARED unit (never continuous_si; see module docstring) --
    for name, var in continuous_by_name.items():
        vals = cells.get(name)
        if vals is None:
            continue
        bad = [
            i
            for i, v in enumerate(vals)
            if v is not None and math.isfinite(v) and not (var.lower <= v <= var.upper)
        ]
        if bad:
            violations.append(
                Violation(
                    check="bounds",
                    severity="advisory",
                    message=(
                        f"column {name!r}: {len(bad)} row(s) outside declared range "
                        f"[{var.lower}, {var.upper}] {var.unit} (declared unit -- "
                        "ProcessSpec.continuous, never continuous_si)"
                    ),
                    row_indices=tuple(bad),
                    columns=(name,),
                )
            )

    # -- bounds: compositional components, per-component [0, 1] --
    for flat_name in compositional_component_of:
        vals = cells.get(flat_name)
        if vals is None:
            continue
        bad = [
            i
            for i, v in enumerate(vals)
            if v is not None and math.isfinite(v) and not (0.0 <= v <= 1.0)
        ]
        if bad:
            violations.append(
                Violation(
                    check="bounds",
                    severity="advisory",
                    message=(
                        f"column {flat_name!r}: {len(bad)} row(s) outside the "
                        "[0, 1] simplex-component range"
                    ),
                    row_indices=tuple(bad),
                    columns=(flat_name,),
                )
            )

    # -- bounds: compositional sum-to-1 --
    for comp_var in spec.compositional:
        comp_cols = [f"{comp_var.name}.{c}" for c in comp_var.components]
        if not all(c in cells for c in comp_cols):
            continue
        bad = []
        for i in range(len(frame.rows)):
            comp_vals = [cells[c][i] for c in comp_cols]
            if any(v is None for v in comp_vals):
                continue  # already flagged by dtype
            total = math.fsum(v for v in comp_vals if v is not None)
            if not math.isfinite(total) or abs(total - 1.0) > SUM_TO_ONE_ATOL:
                bad.append(i)
        if bad:
            violations.append(
                Violation(
                    check="bounds",
                    severity="advisory",
                    message=(
                        f"compositional variable {comp_var.name!r}: {len(bad)} row(s) "
                        f"sum to != 1 (atol {SUM_TO_ONE_ATOL})"
                    ),
                    row_indices=tuple(bad),
                    columns=tuple(comp_cols),
                )
            )

    # -- bounds: categorical level membership --
    for name, var in categorical_by_name.items():
        if name not in header_set:
            continue
        bad = []
        for i, row in enumerate(frame.rows):
            raw = row.get(name)
            level = None if raw is None else str(raw).strip()
            if level not in var.levels:
                bad.append(i)
        if bad:
            violations.append(
                Violation(
                    check="bounds",
                    severity="advisory",
                    message=(
                        f"column {name!r}: {len(bad)} row(s) have a level outside "
                        f"declared {list(var.levels)}"
                    ),
                    row_indices=tuple(bad),
                    columns=(name,),
                )
            )

    violations.extend(_check_order_key(frame, order_key))
    recipe_outcome_cols = [c for c in (*input_names, *output_names) if c in header_set]
    violations.extend(_check_duplicate_rows(frame, recipe_outcome_cols))

    report = ValidationReport(n_rows=len(frame.rows), violations=tuple(violations))
    if strict and not report.passed:
        blocking = [v for v in report.violations if v.severity == "blocking"]
        raise FrameValidationError(
            f"frame validation failed ({len(blocking)} blocking violation(s)): "
            + "; ".join(f"{v.check} ({v.columns}): {v.message}" for v in blocking)
        )
    return report
