"""File ingestion for the tabular adapter: CSV / JSONL -> validated RunRecords.

This is the WP-H slice of implementation-plan §15.6 E1: the user hands us a flat
recipe->outcome file from ANY process plus its declarative spec, and gets
back schema-validated, SI-canonical :class:`rig.schema.RunRecord` rows.

Column matching (CSV):

- input columns are matched to the spec's flattened input names —
  compositional components as ``"<variable>.<component>"`` (WP-A standing
  decision), e.g. ``precursor_blend.silane``;
- output columns are matched to the spec's output names;
- MISSING required columns are a hard error listing them;
- unmatched extra columns are collected into ``RunRecord.extra
  ["unmatched_columns"]`` with a single warning listing them.

Values are interpreted in the SPEC-DECLARED units and canonicalized to SI by
the shared ``rig.schema`` validators on the way into each record (the one
``rig.schema.ureg`` registry — never a second one).

Timestamps: if ``timestamp_column`` is given, cells must be ISO-8601;
otherwise a deterministic monotone sequence is synthesized and every record
is flagged with ``extra["synthetic_timestamp"] = True`` — temporal splits and
drift monitoring over synthetic timestamps are MEANINGLESS (implementation-plan §12.4),
and downstream split logic must check this flag.

CLI::

    python -m rig_adapters.tabular.ingest --spec examples/pecvd_example.toml \
        --csv myruns.csv --out runs.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import uuid
import warnings
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from rig.interfaces import (
    CategoricalVariable,
    CompositionalVariable,
    ContinuousVariable,
)
from rig.schema import (
    CategoricalValue,
    Fraction,
    OutcomeRecord,
    Provenance,
    Quantity,
    RecipeRecord,
    RunRecord,
)
from rig_adapters.tabular.spec import ProcessSpec
from rig_adapters.tabular.validation import Frame, ValidationReport, validate_frame

SUM_TO_ONE_ATOL = 1e-6
_RUN_ID_NAMESPACE = uuid.UUID("6f9c2a1e-6f0e-4bda-9f3b-6c5df0a5b9d1")
# Fixed epoch for synthesized timestamps — deterministic, obviously synthetic.
_SYNTHETIC_BASE = datetime(2000, 1, 1, tzinfo=UTC)
_SYNTHETIC_STEP = timedelta(hours=1)

type OnError = Literal["raise", "skip"]
type Source = Literal["real_tool", "physics_sim"]


class IngestError(ValueError):
    """A file- or row-level ingestion failure."""


@dataclass(frozen=True)
class RejectedRow:
    """One row that failed validation under ``on_error='skip'``."""

    row_index: int  # 0-based data-row index (header excluded)
    reason: str


@dataclass
class IngestResult:
    """Ingestion output: accepted records + the rejects report.

    Iterates over ``records`` so it can be passed directly to
    ``records_to_arrays`` / ``write_jsonl``.
    """

    records: list[RunRecord] = field(default_factory=list)
    rejects: list[RejectedRow] = field(default_factory=list)
    unmatched_columns: tuple[str, ...] = ()
    synthetic_timestamps: bool = False
    # E1 frame-validation pre-pass report (implementation-plan §15.6). ``None`` only for
    # ``ingest_jsonl`` (which has no CSV frame to validate); ``ingest_csv`` always
    # attaches one. Additive field -- existing callers that construct/compare
    # IngestResult by keyword are unaffected.
    frame_report: ValidationReport | None = None

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)


def _parse_float(raw: str, column: str) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError) as e:
        raise ValueError(f"column {column!r}: {raw!r} is not a number") from e


def _validate_recipe(recipe: RecipeRecord, spec: ProcessSpec) -> None:
    """Spec conformance beyond Pydantic: presence + bounds/levels/types + sum-to-1."""
    # Completeness first (audit B3): every declared input must be present. Raise a
    # catchable ValueError listing the missing keys — NOT a KeyError that would
    # escape ingest_jsonl's `except ValueError` and abort the whole batch under
    # on_error='skip', and NOT a silently-accepted incomplete recipe (E1/E5).
    missing = [k for k in spec.flat_input_names if k not in recipe.values]
    if missing:
        raise ValueError(f"recipe is missing required input(s): {missing}")
    recipe.validate_against(list(spec.variables))
    for var in spec.compositional:
        total = 0.0
        for comp in var.components:
            value = recipe.values[f"{var.name}.{comp}"]
            assert isinstance(value, Fraction)
            total += value.value
        if abs(total - 1.0) > SUM_TO_ONE_ATOL:
            raise ValueError(
                f"compositional variable {var.name!r}: components "
                f"{list(var.components)} sum to {total!r}, not 1 "
                f"(atol {SUM_TO_ONE_ATOL})"
            )


def _row_to_record(
    row: dict[str, Any],
    row_index: int,
    spec: ProcessSpec,
    *,
    tool_column: str | None,
    timestamp_column: str | None,
    source: Source,
    default_tool_id: str,
    unmatched: Sequence[str],
    data_hash: str | None,
) -> RunRecord:
    def cell(column: str) -> str:
        raw = row.get(column)
        if raw is None or str(raw).strip() == "":
            raise ValueError(f"column {column!r}: empty value")
        return str(raw).strip()

    values: dict[str, Any] = {}
    for var in spec.variables:
        if isinstance(var, ContinuousVariable):
            values[var.name] = Quantity(
                magnitude=_parse_float(cell(var.name), var.name), unit=var.unit
            )
        elif isinstance(var, CategoricalVariable):
            level = cell(var.name)
            if level not in var.levels:
                raise ValueError(
                    f"column {var.name!r}: level {level!r} not in declared levels "
                    f"{list(var.levels)}"
                )
            values[var.name] = CategoricalValue(value=level, levels=var.levels)
        elif isinstance(var, CompositionalVariable):
            for comp in var.components:
                key = f"{var.name}.{comp}"
                values[key] = Fraction(value=_parse_float(cell(key), key))
    recipe = RecipeRecord(values=values)
    _validate_recipe(recipe, spec)

    outcomes = []
    for out in spec.outputs:
        magnitude = _parse_float(cell(out.name), out.name)
        # Fail closed on non-finite outcomes (NaN/inf): they pass Pydantic but
        # serialize to JSON null in write_jsonl, silently breaking the round-trip.
        if not math.isfinite(magnitude):
            raise ValueError(f"column {out.name!r}: {magnitude!r} is not finite")
        outcomes.append(
            OutcomeRecord(
                name=out.name,
                modality="scalar_vector",
                value=Quantity(magnitude=magnitude, unit=out.unit),
            )
        )

    tool_id = cell(tool_column) if tool_column else default_tool_id

    extra: dict[str, Any] = {}
    if timestamp_column:
        raw_ts = cell(timestamp_column)
        try:
            timestamp = datetime.fromisoformat(raw_ts)
        except ValueError as e:
            raise ValueError(f"column {timestamp_column!r}: {raw_ts!r} is not ISO-8601") from e
    else:
        timestamp = _SYNTHETIC_BASE + row_index * _SYNTHETIC_STEP
        # implementation-plan §12.4: temporal splits over synthetic order are meaningless.
        extra["synthetic_timestamp"] = True
    if unmatched:
        extra["unmatched_columns"] = {c: row.get(c) for c in unmatched}

    run_id = uuid.uuid5(
        _RUN_ID_NAMESPACE,
        f"{spec.process_id}|{row_index}|"
        + "|".join(f"{k}={row.get(k)}" for k in sorted(k for k in row if k is not None)),
    )
    return RunRecord(
        run_id=run_id,
        process_id=spec.process_id,
        tool_id=tool_id,
        timestamp=timestamp,
        recipe=recipe,
        outcomes=outcomes,
        provenance=Provenance(source=source, data_hash=data_hash),
        extra=extra,
    )


def ingest_csv(
    path: str | Path,
    spec: ProcessSpec,
    *,
    tool_column: str | None = None,
    timestamp_column: str | None = None,
    order_key: str | None = None,
    source: Source = "real_tool",
    default_tool_id: str = "unknown",
    on_error: OnError = "raise",
    strict: bool = False,
) -> IngestResult:
    """Ingest a flat CSV of recipe->outcome rows into validated RunRecords.

    ``on_error="raise"`` (default) aborts on the first bad row;
    ``on_error="skip"`` drops bad rows and reports them in
    :attr:`IngestResult.rejects`. Missing REQUIRED columns are always a hard
    :class:`IngestError` regardless of policy.

    Before any row is touched, a frame-level validation pre-pass (E1,
    implementation-plan §15.6) runs over the whole CSV via
    :func:`rig_adapters.tabular.validation.validate_frame` and is attached as
    :attr:`IngestResult.frame_report` -- this is purely additive and does not change
    accept/reject behavior for any existing caller. ``order_key`` names an optional
    column (e.g. a run-order/batch column, possibly UNDECLARED in ``spec`` -- it need
    only be present in the CSV) whose monotonicity the pre-pass checks; pass
    ``strict=True`` to raise :class:`~rig_adapters.tabular.validation.FrameValidationError`
    up front when the pre-pass finds a BLOCKING violation (missing columns, non-numeric
    cells, or NaN/inf) -- default ``False`` keeps today's behavior unchanged.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"CSV file not found: {path}")
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        header = [h for h in (reader.fieldnames or []) if h is not None]
        required = list(spec.flat_input_names) + list(spec.output_names)
        for special in (tool_column, timestamp_column):
            if special:
                required.append(special)
        missing = [c for c in required if c not in header]
        if missing:
            raise IngestError(
                f"{path}: missing required column(s) {missing}; CSV header is {header}"
            )
        unmatched = tuple(c for c in header if c not in required)
        if unmatched:
            warnings.warn(
                f"{path}: column(s) {list(unmatched)} match nothing in spec "
                f"{spec.process_id!r}; collected into RunRecord.extra['unmatched_columns']",
                stacklevel=2,
            )

        # Materialize once: the frame-level pre-pass and the existing per-row loop both
        # need every row, and this iterates the exact same DictReader either way -- not
        # a behavior change, just no longer lazy.
        rows = list(reader)
        frame_report = validate_frame(
            Frame(header=tuple(header), rows=tuple(rows)),
            spec,
            order_key=order_key,
            tool_column=tool_column,
            timestamp_column=timestamp_column,
            strict=strict,
        )

        result = IngestResult(
            unmatched_columns=unmatched,
            synthetic_timestamps=timestamp_column is None,
            frame_report=frame_report,
        )
        for row_index, row in enumerate(rows):
            try:
                record = _row_to_record(
                    row,
                    row_index,
                    spec,
                    tool_column=tool_column,
                    timestamp_column=timestamp_column,
                    source=source,
                    default_tool_id=default_tool_id,
                    unmatched=unmatched,
                    data_hash=data_hash,
                )
            except (ValueError, TypeError) as e:
                if on_error == "raise":
                    raise IngestError(f"{path}: row {row_index}: {e}") from e
                result.rejects.append(RejectedRow(row_index=row_index, reason=str(e)))
                continue
            result.records.append(record)

    # Fail closed on a MIX of naive and tz-aware timestamps: their datetimes cannot
    # be compared (ordering raises TypeError), so temporal splits/sorts would break.
    # All-naive (assumed local per fromisoformat) and all-aware are both fine.
    if timestamp_column and result.records:
        aware = {r.timestamp.tzinfo is not None for r in result.records}
        if len(aware) > 1:
            raise IngestError(
                f"{path}: column {timestamp_column!r} mixes naive and tz-aware "
                "ISO-8601 timestamps; such records cannot be ordered. Use all-naive "
                "or all-aware timestamps."
            )
    return result


def ingest_jsonl(
    path: str | Path,
    spec: ProcessSpec | None = None,
    *,
    on_error: OnError = "raise",
) -> IngestResult:
    """Load already-structured JSONL rows (one serialized RunRecord per line).

    Each line is re-validated by the Pydantic schema; when ``spec`` is given,
    recipes are additionally checked against it (bounds, levels, sum-to-1).
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    path = Path(path)
    if not path.is_file():
        raise IngestError(f"JSONL file not found: {path}")
    result = IngestResult()
    with path.open("r", encoding="utf-8") as fh:
        for line_index, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                record = RunRecord.model_validate_json(line)
                if spec is not None:
                    if record.process_id != spec.process_id:
                        raise ValueError(
                            f"process_id {record.process_id!r} != spec {spec.process_id!r}"
                        )
                    _validate_recipe(record.recipe, spec)
            except ValueError as e:
                if on_error == "raise":
                    raise IngestError(f"{path}: line {line_index}: {e}") from e
                result.rejects.append(RejectedRow(row_index=line_index, reason=str(e)))
                continue
            result.records.append(record)
    if any(r.extra.get("synthetic_timestamp") for r in result.records):
        result.synthetic_timestamps = True
    return result


def write_jsonl(records: Iterable[RunRecord], path: str | Path) -> Path:
    """Serialize RunRecords to JSONL (one ``model_dump_json`` per line).

    Round-trip guarantee (tested): ``ingest_jsonl(write_jsonl(records))``
    reloads records equal to the originals.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig_adapters.tabular.ingest",
        description="Ingest a flat recipe->outcome CSV into validated RunRecords (JSONL).",
    )
    parser.add_argument("--spec", required=True, help="process spec (.toml or .json)")
    parser.add_argument("--csv", required=True, help="input CSV path")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--tool-column", default=None, help="CSV column holding tool_id")
    parser.add_argument(
        "--timestamp-column", default=None, help="CSV column holding ISO-8601 timestamps"
    )
    parser.add_argument(
        "--source",
        default="real_tool",
        choices=("real_tool", "physics_sim"),
        help="provenance source tag (default real_tool)",
    )
    parser.add_argument(
        "--default-tool-id", default="unknown", help="tool_id when --tool-column is absent"
    )
    parser.add_argument(
        "--on-error",
        default="raise",
        choices=("raise", "skip"),
        help="row-failure policy (default raise)",
    )
    args = parser.parse_args(argv)

    from rig_adapters.tabular.spec import load_spec

    spec = load_spec(args.spec)
    result = ingest_csv(
        args.csv,
        spec,
        tool_column=args.tool_column,
        timestamp_column=args.timestamp_column,
        source=args.source,
        default_tool_id=args.default_tool_id,
        on_error=args.on_error,
    )
    out = write_jsonl(result.records, args.out)
    print(f"wrote {len(result.records)} RunRecords -> {out}")
    if result.synthetic_timestamps:
        print(
            "NOTE: timestamps are SYNTHETIC (no --timestamp-column); temporal splits are meaningless"
        )
    if result.unmatched_columns:
        print(f"unmatched columns (kept in extra): {list(result.unmatched_columns)}")
    if result.rejects:
        print(f"rejected {len(result.rejects)} row(s):")
        for reject in result.rejects:
            print(f"  row {reject.row_index}: {reject.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
