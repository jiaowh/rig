"""E1 frame-validation tests (implementation-plan §15.6): every check in
``rig_adapters.tabular.validation`` gets a passing case + a violating case, plus the
Empa-agreement pin (the report's rejected rows == ingest's actual skips on the real
Ti-120W data), the ti_120w degenerate-order-key flag, strict-mode raise/no-raise
(enforce vs report), and determinism.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rig_adapters.tabular.ingest import ingest_csv
from rig_adapters.tabular.spec import load_spec, parse_spec
from rig_adapters.tabular.validation import (
    Frame,
    FrameValidationError,
    ValidationReport,
    Violation,
    frame_from_csv,
    validate_frame,
)

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "real_data" / "empa_hipims"

DEGENERATE_SLUG = "ti_120w_short_pw"
DEGENERATE_N_REJECTS = 5


def write_csv(path: Path, columns: list[str], rows: list[list]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


# -- a small multi-kind spec (continuous + categorical + compositional + 1 output) --


def base_spec():
    return parse_spec(
        {
            "process_id": "frame_validation_unit_test",
            "inputs": {
                "temp": {"kind": "continuous", "unit": "degC", "lower": 100.0, "upper": 200.0},
                "mode": {"kind": "categorical", "levels": ["a", "b"]},
                "blend": {"kind": "compositional", "components": ["x", "y"]},
            },
            "outputs": {"thick": {"unit": "nm"}},
        }
    )


BASE_COLUMNS = ["temp", "mode", "blend.x", "blend.y", "thick"]
BASE_ROWS = [
    [150.0, "a", 0.3, 0.7, 500.0],
    [180.0, "b", 0.5, 0.5, 510.0],
    [120.0, "a", 0.2, 0.8, 490.0],
]


def base_frame(rows=None, columns=None) -> Frame:
    columns = columns if columns is not None else BASE_COLUMNS
    rows = rows if rows is not None else BASE_ROWS
    return Frame(
        header=tuple(columns),
        rows=tuple(dict(zip(columns, [str(c) for c in row], strict=True)) for row in rows),
    )


# ============================================================================
# (a) required columns present + no unexpected dtype
# ============================================================================


def test_missing_columns_passing():
    report = validate_frame(base_frame(), base_spec())
    assert report.by_check("missing_columns") == ()
    assert report.passed


def test_missing_columns_violating():
    frame = Frame(
        header=("temp", "mode", "blend.x", "blend.y"),  # "thick" (output) dropped
        rows=tuple(
            {"temp": "150", "mode": "a", "blend.x": "0.3", "blend.y": "0.7"} for _ in range(2)
        ),
    )
    report = validate_frame(frame, base_spec())
    violations = report.by_check("missing_columns")
    assert len(violations) == 1
    assert violations[0].severity == "blocking"
    assert violations[0].columns == ("thick",)
    assert not report.passed


def test_dtype_passing():
    report = validate_frame(base_frame(), base_spec())
    assert report.by_check("dtype") == ()


def test_dtype_violating_stray_string():
    rows = [list(r) for r in BASE_ROWS]
    rows[1][0] = "warm"  # temp column gets a non-numeric string
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("dtype")
    assert len(violations) == 1
    assert violations[0].severity == "blocking"
    assert violations[0].columns == ("temp",)
    assert violations[0].row_indices == (1,)
    assert not report.passed


def test_dtype_violating_blank_cell():
    rows = [list(r) for r in BASE_ROWS]
    rows[2][4] = ""  # thick (output) blank
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("dtype")
    assert len(violations) == 1
    assert violations[0].columns == ("thick",)
    assert violations[0].row_indices == (2,)


# ============================================================================
# (b) declared-unit bounds membership
# ============================================================================


def test_bounds_passing():
    report = validate_frame(base_frame(), base_spec())
    assert report.by_check("bounds") == ()
    assert report.passed


def test_bounds_violating_continuous_out_of_range():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = 999.0  # temp way above declared upper=200 degC
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("bounds")
    assert len(violations) == 1
    assert violations[0].severity == "advisory"  # reported, not a blocking failure
    assert violations[0].columns == ("temp",)
    assert violations[0].row_indices == (0,)
    # advisory-only violation: report still "passes" (no BLOCKING violation)
    assert report.passed


def test_bounds_respects_declared_unit_not_si():
    # 199 degC is INSIDE the declared [100, 200] degC range. If this check ever
    # compared against continuous_si (bounds converted to SI/Kelvin) while leaving
    # the raw cell in degC, 199 would be wrongly flagged (472.15 K vs a declared
    # [373.15, 473.15] K bound is still inside -- so pick a value that would
    # ONLY misbehave if the unit conversion were applied to just one side).
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = 199.0
    report = validate_frame(base_frame(rows=rows), base_spec())
    assert report.by_check("bounds") == ()


def test_bounds_violating_categorical_level():
    rows = [list(r) for r in BASE_ROWS]
    rows[1][1] = "sideways"  # "mode" is not in declared levels {a, b}
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("bounds")
    assert len(violations) == 1
    assert violations[0].columns == ("mode",)
    assert violations[0].row_indices == (1,)


def test_bounds_violating_compositional_sum():
    rows = [list(r) for r in BASE_ROWS]
    rows[2][2:4] = [0.5, 0.7]  # sums to 1.2, not 1
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = [v for v in report.by_check("bounds") if "sum" in v.message]
    assert len(violations) == 1
    assert violations[0].row_indices == (2,)


def test_bounds_violating_compositional_component_range():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][2:4] = [-0.2, 1.2]  # each component individually outside [0, 1]
    report = validate_frame(base_frame(rows=rows), base_spec())
    range_violations = [v for v in report.by_check("bounds") if "0, 1" in v.message]
    assert any(v.row_indices == (0,) for v in range_violations)


# ============================================================================
# (c) no NaN/inf in knob or outcome columns
# ============================================================================


def test_nan_inf_passing():
    report = validate_frame(base_frame(), base_spec())
    assert report.by_check("nan_inf") == ()


def test_nan_inf_violating_nan_input():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = "nan"
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("nan_inf")
    assert len(violations) == 1
    assert violations[0].severity == "blocking"
    assert violations[0].columns == ("temp",)
    assert violations[0].row_indices == (0,)
    assert not report.passed
    # a NaN input is NOT double-reported as a bounds violation (nan_inf catches it first)
    assert report.by_check("bounds") == ()


def test_nan_inf_violating_inf_output():
    rows = [list(r) for r in BASE_ROWS]
    rows[1][4] = "inf"
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("nan_inf")
    assert len(violations) == 1
    assert violations[0].columns == ("thick",)
    assert violations[0].row_indices == (1,)


# ============================================================================
# (d) order-key sanity
# ============================================================================


def test_order_key_passing_monotone():
    columns = [*BASE_COLUMNS, "seq"]
    rows = [[*r, seq] for r, seq in zip(BASE_ROWS, [1, 2, 2], strict=True)]  # ties allowed
    frame = base_frame(rows=rows, columns=columns)
    report = validate_frame(frame, base_spec(), order_key="seq")
    assert report.by_check("order_key") == ()


def test_order_key_violating_not_monotone():
    columns = [*BASE_COLUMNS, "seq"]
    rows = [[*r, seq] for r, seq in zip(BASE_ROWS, [1, 3, 2], strict=True)]  # decreases at row 2
    frame = base_frame(rows=rows, columns=columns)
    report = validate_frame(frame, base_spec(), order_key="seq")
    violations = report.by_check("order_key")
    assert len(violations) == 1
    assert violations[0].severity == "advisory"
    assert "not monotone" in violations[0].message
    assert violations[0].row_indices == (2,)
    assert report.passed  # advisory only


def test_order_key_violating_degenerate_all_equal():
    columns = [*BASE_COLUMNS, "seq"]
    rows = [[*r, 1] for r in BASE_ROWS]  # every row has seq == 1
    frame = base_frame(rows=rows, columns=columns)
    report = validate_frame(frame, base_spec(), order_key="seq")
    violations = report.by_check("order_key")
    assert len(violations) == 1
    assert "UNVERIFIED-ORDER" in violations[0].message
    assert violations[0].row_indices == (0, 1, 2)


def test_order_key_absent_column_is_a_no_op():
    report = validate_frame(base_frame(), base_spec(), order_key="does_not_exist")
    assert report.by_check("order_key") == ()


# ============================================================================
# (e) duplicate-row detection (reported, never rejected)
# ============================================================================


def test_duplicate_rows_passing():
    report = validate_frame(base_frame(), base_spec())
    assert report.by_check("duplicate_rows") == ()


def test_duplicate_rows_violating():
    rows = [list(r) for r in BASE_ROWS]
    rows[2] = list(rows[0])  # row 2 becomes an exact duplicate of row 0
    report = validate_frame(base_frame(rows=rows), base_spec())
    violations = report.by_check("duplicate_rows")
    assert len(violations) == 1
    assert violations[0].severity == "advisory"
    assert violations[0].row_indices == (0, 2)
    assert report.passed  # duplicates are reported, never rejected


# ============================================================================
# (f) outcome-column presence + numeric (covered structurally by (a)/(c); pinned
# explicitly here so the requirement is independently testable)
# ============================================================================


def test_outcome_column_missing_is_blocking():
    frame = Frame(
        header=("temp", "mode", "blend.x", "blend.y"),
        rows=tuple(
            {"temp": "150", "mode": "a", "blend.x": "0.3", "blend.y": "0.7"} for _ in range(1)
        ),
    )
    report = validate_frame(frame, base_spec())
    assert any(v.check == "missing_columns" and "thick" in v.columns for v in report.violations)
    assert not report.passed


def test_outcome_column_non_numeric_is_blocking():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][4] = "high"
    report = validate_frame(base_frame(rows=rows), base_spec())
    dtype_violations = report.by_check("dtype")
    assert any(v.columns == ("thick",) for v in dtype_violations)
    assert not report.passed


# ============================================================================
# strict mode: raises on BLOCKING, does NOT raise on advisory-only
# ============================================================================


def test_strict_raises_on_blocking_violation():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = "warm"  # dtype: blocking
    with pytest.raises(FrameValidationError, match="dtype"):
        validate_frame(base_frame(rows=rows), base_spec(), strict=True)


def test_strict_does_not_raise_on_advisory_only_violation():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = 999.0  # bounds: advisory only
    report = validate_frame(base_frame(rows=rows), base_spec(), strict=True)
    assert report.by_check("bounds")
    assert report.passed


# ============================================================================
# determinism
# ============================================================================


def test_determinism_same_frame_identical_report():
    frame = base_frame()
    spec = base_spec()
    r1 = validate_frame(frame, spec, order_key="temp")
    r2 = validate_frame(frame, spec, order_key="temp")
    assert r1 == r2


def test_determinism_violating_frame_identical_report():
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = 999.0
    rows[1] = list(rows[0])
    frame = base_frame(rows=rows)
    spec = base_spec()
    reports = [validate_frame(frame, spec) for _ in range(3)]
    assert reports[0] == reports[1] == reports[2]


# ============================================================================
# ingest_csv wiring: non-breaking (frame_report attached, existing behavior kept)
# ============================================================================


def test_ingest_csv_attaches_frame_report_without_changing_default_behavior(tmp_path):
    path = write_csv(tmp_path / "runs.csv", BASE_COLUMNS, BASE_ROWS)
    result = ingest_csv(path, base_spec())
    assert isinstance(result.frame_report, ValidationReport)
    assert result.frame_report.passed
    assert len(result.records) == 3 and not result.rejects


def test_ingest_csv_strict_raises_before_any_row_processing(tmp_path):
    rows = [list(r) for r in BASE_ROWS]
    rows[0][0] = "warm"  # would normally just reject row 0 under on_error="skip"
    path = write_csv(tmp_path / "runs.csv", BASE_COLUMNS, rows)
    with pytest.raises(FrameValidationError, match="dtype"):
        ingest_csv(path, base_spec(), on_error="skip", strict=True)


# ============================================================================
# Empa-agreement + ti_120w degenerate order key (real local data, sim-free)
# ============================================================================


def _empa_spec_and_frame(slug: str):
    spec = load_spec(EX / "specs" / f"{slug}.toml")
    frame = frame_from_csv(EX / "csv" / f"{slug}.csv")
    return spec, frame


@pytest.mark.skipif(
    not (EX / "csv" / f"{DEGENERATE_SLUG}.csv").is_file(),
    reason="Empa HiPIMS local CSVs not present",
)
def test_empa_agreement_bounds_report_matches_actual_ingest_skips():
    """The report's bounds-violating rows for Ti-120W must be EXACTLY the rows
    ingest_csv(on_error='skip') actually drops -- not just the same count."""
    spec, frame = _empa_spec_and_frame(DEGENERATE_SLUG)
    report = validate_frame(frame, spec, order_key="BatchNr")

    knob_names = set(spec.flat_input_names)
    bounds_rows: set[int] = set()
    for v in report.by_check("bounds"):
        if set(v.columns) & knob_names:
            bounds_rows.update(v.row_indices)

    assert len(bounds_rows) == DEGENERATE_N_REJECTS

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skip_result = ingest_csv(
            EX / "csv" / f"{DEGENERATE_SLUG}.csv", spec, source="real_tool", on_error="skip"
        )
    actual_reject_rows = {r.row_index for r in skip_result.rejects}
    assert actual_reject_rows == {i for i in bounds_rows}
    assert len(actual_reject_rows) == DEGENERATE_N_REJECTS
    # and the pre-pass report attached to that same ingest call agrees with itself
    assert skip_result.frame_report is not None
    assert set(skip_result.frame_report.rows_for("bounds")) >= actual_reject_rows


@pytest.mark.skipif(
    not (EX / "csv" / f"{DEGENERATE_SLUG}.csv").is_file(),
    reason="Empa HiPIMS local CSVs not present",
)
def test_empa_ti_120w_degenerate_order_key_flagged():
    spec, frame = _empa_spec_and_frame(DEGENERATE_SLUG)
    report = validate_frame(frame, spec, order_key="BatchNr")
    order_violations = report.by_check("order_key")
    assert len(order_violations) == 1
    assert "UNVERIFIED-ORDER" in order_violations[0].message
    assert len(order_violations[0].row_indices) == len(frame.rows) == 495


@pytest.mark.skipif(
    not (EX / "csv" / "al_120w_short_pw.csv").is_file(),
    reason="Empa HiPIMS local CSVs not present",
)
def test_empa_non_degenerate_campaign_order_key_clean():
    """A control: a campaign whose BatchNr really is 1..n must NOT get an
    order_key violation (rules out a check that fires unconditionally)."""
    spec, frame = _empa_spec_and_frame("al_120w_short_pw")
    report = validate_frame(frame, spec, order_key="BatchNr")
    assert report.by_check("order_key") == ()


@pytest.mark.skipif(
    not (EX / "csv" / f"{DEGENERATE_SLUG}.csv").is_file(),
    reason="Empa HiPIMS local CSVs not present",
)
def test_empa_determinism_real_data():
    spec, frame = _empa_spec_and_frame(DEGENERATE_SLUG)
    r1 = validate_frame(frame, spec, order_key="BatchNr")
    r2 = validate_frame(frame, spec, order_key="BatchNr")
    assert r1 == r2


# ============================================================================
# Frame / frame_from_csv plumbing
# ============================================================================


def test_frame_from_csv_matches_ingest_header(tmp_path):
    path = write_csv(tmp_path / "runs.csv", BASE_COLUMNS, BASE_ROWS)
    frame = frame_from_csv(path)
    assert frame.header == tuple(BASE_COLUMNS)
    assert len(frame.rows) == 3
    assert frame.rows[0]["temp"] == "150.0"


def test_violation_and_report_are_frozen_and_equatable():
    v1 = Violation(check="bounds", severity="advisory", message="x", row_indices=(1,))
    v2 = Violation(check="bounds", severity="advisory", message="x", row_indices=(1,))
    assert v1 == v2
    with pytest.raises(AttributeError):
        v1.message = "y"  # frozen dataclass
