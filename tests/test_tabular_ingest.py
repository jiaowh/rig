"""WP-H ingestion tests: CSV -> RunRecord with SI canonicalization, column
matching, on_error policies, sum-to-1 enforcement, timestamps, JSONL
round-trip, the CLI, and the end-to-end "not MBE-specific" integration proof
(synthetic PECVD-ish CSV -> records_to_arrays -> GPForwardModel)."""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from rig.forward import GPForwardModel, records_to_arrays
from rig.interfaces import PredictiveDistribution
from rig.schema import CategoricalValue, ureg
from rig_adapters.tabular.ingest import (
    IngestError,
    IngestResult,
    ingest_csv,
    ingest_jsonl,
    main,
    write_jsonl,
)
from rig_adapters.tabular.spec import load_spec, parse_spec

REPO = Path(__file__).resolve().parents[1]
PECVD_SPEC_PATH = REPO / "examples" / "pecvd_example.toml"

PECVD_COLUMNS = [
    "temperature",
    "pressure",
    "rf_power",
    "precursor_blend.silane",
    "precursor_blend.ammonia",
    "precursor_blend.nitrogen",
    "thickness",
    "nonuniformity",
    "stress",
]
# (temperature degC, pressure torr, rf W, silane, ammonia, nitrogen,
#  thickness nm, nonuniformity %, stress MPa)
PECVD_ROWS = [
    [300.0, 2.0, 100.0, 0.2, 0.3, 0.5, 500.0, 3.0, -50.0],
    [250.0, 1.0, 300.0, 0.1, 0.4, 0.5, 480.0, 4.0, 20.0],
    [380.0, 4.5, 450.0, 0.3, 0.3, 0.4, 545.0, 2.0, 150.0],
]


@pytest.fixture(scope="module")
def pecvd_spec():
    return load_spec(PECVD_SPEC_PATH)


def write_csv(path: Path, columns: list[str], rows: list[list]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)
    return path


def pecvd_csv(tmp_path: Path, rows=None, columns=None) -> Path:
    return write_csv(
        tmp_path / "runs.csv",
        columns if columns is not None else PECVD_COLUMNS,
        rows if rows is not None else PECVD_ROWS,
    )


# -- happy path + SI canonicalization -------------------------------------------


def test_csv_happy_path_si_canonicalization(tmp_path, pecvd_spec):
    result = ingest_csv(pecvd_csv(tmp_path), pecvd_spec)
    assert isinstance(result, IngestResult)
    assert len(result) == 3 and not result.rejects
    rec = result.records[0]
    assert rec.process_id == "pecvd_sin_demo"
    assert rec.tool_id == "unknown"
    # degC -> K
    temp = rec.recipe.values["temperature"]
    assert temp.magnitude == pytest.approx(573.15)
    assert temp.unit == "K"
    # torr -> SI base (Pa in base units)
    pressure = rec.recipe.values["pressure"]
    assert pressure.magnitude == pytest.approx(ureg.Quantity(2.0, "torr").to_base_units().magnitude)
    # fractions stay fractions
    assert rec.recipe.values["precursor_blend.silane"].value == pytest.approx(0.2)
    # outcomes: nm -> m, percent -> dimensionless fraction
    outcome = {o.name: o for o in rec.outcomes}
    assert outcome["thickness"].value.magnitude == pytest.approx(5e-7)
    assert outcome["thickness"].value.unit == "m"
    assert outcome["nonuniformity"].value.magnitude == pytest.approx(0.03)
    assert rec.provenance.source == "real_tool"
    assert rec.provenance.data_hash  # sha256 of the CSV file


def flow_spec():
    """Non-MBE spec with sccm flows as INDEPENDENT continuous vars + a categorical."""
    return parse_spec(
        {
            "process_id": "etch_demo",
            "inputs": {
                "temp": {"kind": "continuous", "unit": "degC", "lower": 20.0, "upper": 120.0},
                "cf4_flow": {"kind": "continuous", "unit": "sccm", "lower": 5.0, "upper": 100.0},
                "chuck": {"kind": "categorical", "levels": ["low", "high"]},
            },
            "outputs": {"etch_rate": {"unit": "nm/min"}},
        }
    )


def test_non_si_units_degc_and_sccm(tmp_path):
    spec = flow_spec()
    path = write_csv(
        tmp_path / "etch.csv",
        ["temp", "cf4_flow", "chuck", "etch_rate"],
        [[60.0, 10.0, "high", 120.0]],
    )
    rec = ingest_csv(path, spec).records[0]
    assert rec.recipe.values["temp"].magnitude == pytest.approx(333.15)  # degC -> K
    # sccm -> m^3/s
    assert rec.recipe.values["cf4_flow"].magnitude == pytest.approx(
        ureg.Quantity(10.0, "sccm").to_base_units().magnitude
    )
    assert rec.recipe.values["cf4_flow"].magnitude == pytest.approx(1.6667e-7, rel=1e-3)
    assert isinstance(rec.recipe.values["chuck"], CategoricalValue)
    # nm/min -> m/s
    assert rec.outcomes[0].value.magnitude == pytest.approx(2e-9)


# -- column matching ---------------------------------------------------------------


def test_unmatched_columns_warn_and_land_in_extra(tmp_path, pecvd_spec):
    path = pecvd_csv(
        tmp_path,
        columns=PECVD_COLUMNS + ["operator_note"],
        rows=[row + ["looked fine"] for row in PECVD_ROWS],
    )
    with pytest.warns(UserWarning, match="operator_note"):
        result = ingest_csv(path, pecvd_spec)
    assert result.unmatched_columns == ("operator_note",)
    assert result.records[0].extra["unmatched_columns"] == {"operator_note": "looked fine"}


def test_missing_required_columns_hard_error(tmp_path, pecvd_spec):
    columns = [c for c in PECVD_COLUMNS if c not in ("stress", "rf_power")]
    idx = [PECVD_COLUMNS.index(c) for c in columns]
    path = pecvd_csv(tmp_path, columns=columns, rows=[[r[i] for i in idx] for r in PECVD_ROWS])
    with pytest.raises(IngestError) as excinfo:
        ingest_csv(path, pecvd_spec)
    assert "rf_power" in str(excinfo.value) and "stress" in str(excinfo.value)


def test_tool_and_timestamp_columns(tmp_path, pecvd_spec):
    path = pecvd_csv(
        tmp_path,
        columns=PECVD_COLUMNS + ["tool", "run_time"],
        rows=[
            row + [f"chamber_{i}", f"2026-07-{10 + i}T08:00:00"] for i, row in enumerate(PECVD_ROWS)
        ],
    )
    result = ingest_csv(path, pecvd_spec, tool_column="tool", timestamp_column="run_time")
    assert not result.synthetic_timestamps
    assert [r.tool_id for r in result.records] == ["chamber_0", "chamber_1", "chamber_2"]
    assert result.records[1].timestamp == datetime(2026, 7, 11, 8, 0, 0)
    assert "synthetic_timestamp" not in result.records[0].extra


def test_bad_timestamp_rejected(tmp_path, pecvd_spec):
    path = pecvd_csv(
        tmp_path,
        columns=PECVD_COLUMNS + ["run_time"],
        rows=[PECVD_ROWS[0] + ["last tuesday"]],
    )
    with pytest.raises(IngestError, match="ISO-8601"):
        ingest_csv(path, pecvd_spec, timestamp_column="run_time")


# -- row validation + on_error policy ---------------------------------------------


def bad_rows():
    rows = [list(r) for r in PECVD_ROWS]
    rows[1][0] = 500.0  # temperature 500 degC > declared upper 400 degC
    return rows


def test_out_of_bounds_row_raises_by_default(tmp_path, pecvd_spec):
    with pytest.raises(IngestError, match=r"row 1.*temperature"):
        ingest_csv(pecvd_csv(tmp_path, rows=bad_rows()), pecvd_spec)


def test_on_error_skip_collects_rejects(tmp_path, pecvd_spec):
    result = ingest_csv(pecvd_csv(tmp_path, rows=bad_rows()), pecvd_spec, on_error="skip")
    assert len(result.records) == 2
    assert len(result.rejects) == 1
    assert result.rejects[0].row_index == 1
    assert "temperature" in result.rejects[0].reason


def test_bad_categorical_level_rejected(tmp_path):
    spec = flow_spec()
    path = write_csv(
        tmp_path / "etch.csv",
        ["temp", "cf4_flow", "chuck", "etch_rate"],
        [[60.0, 10.0, "sideways", 120.0]],
    )
    with pytest.raises(IngestError, match="sideways"):
        ingest_csv(path, spec)


def test_sum_to_one_enforced(tmp_path, pecvd_spec):
    rows = [list(r) for r in PECVD_ROWS]
    rows[0][3:6] = [0.5, 0.4, 0.2]  # sums to 1.1
    with pytest.raises(IngestError, match="sum"):
        ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)


def test_sum_to_one_within_atol_accepted(tmp_path, pecvd_spec):
    rows = [list(r) for r in PECVD_ROWS]
    rows[0][3:6] = [0.2, 0.3, 0.5 + 5e-7]  # inside atol 1e-6
    assert len(ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)) == 3


def test_empty_cell_rejected(tmp_path, pecvd_spec):
    rows = [list(r) for r in PECVD_ROWS]
    rows[2][8] = ""  # stress empty
    with pytest.raises(IngestError, match=r"row 2.*stress"):
        ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)


def test_non_numeric_cell_rejected(tmp_path, pecvd_spec):
    rows = [list(r) for r in PECVD_ROWS]
    rows[0][1] = "plenty"
    with pytest.raises(IngestError, match="plenty"):
        ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)


def test_invalid_on_error_value(tmp_path, pecvd_spec):
    with pytest.raises(ValueError, match="on_error"):
        ingest_csv(pecvd_csv(tmp_path), pecvd_spec, on_error="ignore")


def test_nan_outcome_cell_rejected(tmp_path, pecvd_spec):
    # Finding B: a non-finite outcome (NaN) must be REJECTED at ingest, not
    # ingested into a "validated" RunRecord that then serializes to null on
    # write_jsonl (breaking the round-trip). Fail closed, naming row + column.
    rows = [list(r) for r in PECVD_ROWS]
    rows[0][6] = "nan"  # thickness
    with pytest.raises(IngestError, match=r"row 0.*thickness"):
        ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)


def test_inf_outcome_cell_rejected(tmp_path, pecvd_spec):
    # Finding B: inf is non-finite too.
    rows = [list(r) for r in PECVD_ROWS]
    rows[1][7] = "inf"  # nonuniformity
    with pytest.raises(IngestError, match=r"row 1.*nonuniformity"):
        ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)


def test_nan_outcome_skipped_under_skip_policy(tmp_path, pecvd_spec):
    # Finding B: under on_error='skip' the bad row is dropped, not silently kept.
    rows = [list(r) for r in PECVD_ROWS]
    rows[2][6] = "nan"  # thickness
    result = ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec, on_error="skip")
    assert len(result.records) == 2
    assert result.rejects[0].row_index == 2 and "thickness" in result.rejects[0].reason


def test_mixed_naive_and_aware_timestamps_rejected(tmp_path, pecvd_spec):
    # Finding C: mixing naive and tz-aware ISO-8601 timestamps yields RunRecords
    # whose timestamps cannot be ORDERED (comparison raises TypeError). Fail closed.
    path = pecvd_csv(
        tmp_path,
        columns=PECVD_COLUMNS + ["run_time"],
        rows=[
            PECVD_ROWS[0] + ["2026-07-10T08:00:00"],  # naive
            PECVD_ROWS[1] + ["2026-07-11T08:00:00+00:00"],  # tz-aware
        ],
    )
    with pytest.raises(IngestError, match="naive"):
        ingest_csv(path, pecvd_spec, timestamp_column="run_time")


def test_all_aware_timestamps_accepted(tmp_path, pecvd_spec):
    # Finding C: uniformly tz-aware is fine (orderable).
    path = pecvd_csv(
        tmp_path,
        columns=PECVD_COLUMNS + ["run_time"],
        rows=[
            PECVD_ROWS[0] + ["2026-07-10T08:00:00+00:00"],
            PECVD_ROWS[1] + ["2026-07-11T08:00:00+00:00"],
        ],
    )
    result = ingest_csv(path, pecvd_spec, timestamp_column="run_time")
    assert len(result.records) == 2
    stamps = [r.timestamp for r in result.records]
    assert stamps == sorted(stamps)  # orderable


# -- timestamps --------------------------------------------------------------------


def test_synthetic_timestamps_monotone_flagged_deterministic(tmp_path, pecvd_spec):
    path = pecvd_csv(tmp_path)
    result = ingest_csv(path, pecvd_spec)
    assert result.synthetic_timestamps
    stamps = [r.timestamp for r in result.records]
    assert stamps == sorted(stamps) and len(set(stamps)) == len(stamps)  # strictly monotone
    assert all(r.extra["synthetic_timestamp"] is True for r in result.records)
    # deterministic: a second ingest gives bit-identical records
    again = ingest_csv(path, pecvd_spec)
    assert [r.model_dump_json() for r in again.records] == [
        r.model_dump_json() for r in result.records
    ]


# -- JSONL round-trip + CLI ----------------------------------------------------------


def test_jsonl_round_trip(tmp_path, pecvd_spec):
    result = ingest_csv(pecvd_csv(tmp_path), pecvd_spec)
    out = write_jsonl(result.records, tmp_path / "runs.jsonl")
    reloaded = ingest_jsonl(out, pecvd_spec)
    assert reloaded.records == result.records
    assert reloaded.synthetic_timestamps  # flag survives the round trip


def test_ingest_jsonl_spec_mismatch(tmp_path, pecvd_spec):
    result = ingest_csv(pecvd_csv(tmp_path), pecvd_spec)
    out = write_jsonl(result.records, tmp_path / "runs.jsonl")
    other = load_spec(REPO / "examples" / "tabular_minimal.toml")
    with pytest.raises(IngestError, match="process_id"):
        ingest_jsonl(out, other)


def _drop_recipe_key(path: Path, key: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    obj = json.loads(lines[0])
    del obj["recipe"]["values"][key]
    lines[0] = json.dumps(obj)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_jsonl_missing_compositional_component_is_catchable_skip(tmp_path, pecvd_spec):
    # audit B3: a JSONL row missing a compositional component must raise a
    # CATCHABLE ValueError so on_error='skip' drops just that row — previously a
    # bare KeyError escaped the handler and aborted the whole batch.
    result = ingest_csv(pecvd_csv(tmp_path), pecvd_spec)  # 3 rows
    out = write_jsonl(result.records, tmp_path / "runs.jsonl")
    _drop_recipe_key(out, "precursor_blend.nitrogen")
    res = ingest_jsonl(out, pecvd_spec, on_error="skip")
    assert len(res.records) == 2  # the corrupted row skipped, the other two kept
    assert res.rejects and "nitrogen" in res.rejects[0].reason


def test_jsonl_missing_continuous_input_rejected(tmp_path, pecvd_spec):
    # audit B3: a JSONL row missing a whole continuous input must be rejected,
    # not silently accepted as a valid (incomplete) record.
    result = ingest_csv(pecvd_csv(tmp_path), pecvd_spec)
    out = write_jsonl(result.records, tmp_path / "runs.jsonl")
    _drop_recipe_key(out, "temperature")
    with pytest.raises(IngestError, match="missing required input"):
        ingest_jsonl(out, pecvd_spec, on_error="raise")
    res = ingest_jsonl(out, pecvd_spec, on_error="skip")
    assert len(res.records) == 2 and res.rejects


def test_cli_end_to_end(tmp_path, capsys):
    csv_path = pecvd_csv(tmp_path)
    out_path = tmp_path / "runs.jsonl"
    rc = main(["--spec", str(PECVD_SPEC_PATH), "--csv", str(csv_path), "--out", str(out_path)])
    assert rc == 0
    assert len(ingest_jsonl(out_path).records) == 3
    printed = capsys.readouterr().out
    assert "wrote 3 RunRecords" in printed
    assert "SYNTHETIC" in printed  # the temporal-order warning is surfaced


# -- integration: the executable "not MBE-specific" proof ----------------------------


def test_end_to_end_generic_process_csv_to_calibrated_forward_model(tmp_path, pecvd_spec):
    """30-row synthetic PECVD-ish CSV (non-MBE variables, non-SI units) ->
    ingest -> records_to_arrays -> GPForwardModel -> canonical
    PredictiveDistribution + sane support_score ordering."""
    rng = np.random.default_rng(42)
    n = 30
    temp = rng.uniform(220.0, 390.0, n)  # degC
    pressure = rng.uniform(0.8, 4.5, n)  # torr
    rf = rng.uniform(80.0, 450.0, n)  # W
    blend = rng.dirichlet((4.0, 4.0, 8.0), n)  # silane, ammonia, nitrogen

    # seeded analytic ground truth in DECLARED units (smooth, invented)
    thickness = (
        250.0
        + 0.6 * temp
        + 15.0 * pressure
        + 0.08 * rf
        + 120.0 * blend[:, 0]
        + rng.normal(0.0, 2.0, n)
    )  # nm
    nonuniformity = (
        1.5
        + 3.0 * np.exp(-((temp - 320.0) ** 2) / 5000.0)
        + 0.4 * pressure
        + rng.normal(0.0, 0.05, n)
    )  # percent
    stress = -180.0 + 0.5 * rf + 90.0 * blend[:, 1] + rng.normal(0.0, 3.0, n)  # MPa

    rows = [
        [
            temp[i],
            pressure[i],
            rf[i],
            blend[i, 0],
            blend[i, 1],
            blend[i, 2],
            thickness[i],
            nonuniformity[i],
            stress[i],
        ]
        for i in range(n)
    ]
    result = ingest_csv(pecvd_csv(tmp_path, rows=rows), pecvd_spec)
    assert len(result.records) == n

    # drop the last blend component: sum-to-1 makes the full set collinear
    input_keys = [
        "temperature",
        "pressure",
        "rf_power",
        "precursor_blend.silane",
        "precursor_blend.ammonia",
    ]
    output_keys = ["thickness", "nonuniformity", "stress"]
    X, Y = records_to_arrays(result.records, input_keys, output_keys)
    assert X.shape == (n, 5) and Y.shape == (n, 3)
    # ingestion really canonicalized: temperatures are Kelvin now
    assert X[:, 0].min() > 400.0

    model = GPForwardModel(
        input_keys=input_keys, output_keys=output_keys, n_restarts=2, seed=0
    ).fit(X, Y)
    x_in = X.mean(axis=0)
    dist = model.predict(x_in)
    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (3,)
    assert dist.aleatoric_sigma.shape == (3,)
    assert dist.epistemic_sigma.shape == (3,)

    x_ood = x_in + 50.0 * (X.std(axis=0) + 1e-9)
    assert model.support_score(x_in) > model.support_score(x_ood)
