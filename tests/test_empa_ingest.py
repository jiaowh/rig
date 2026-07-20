"""Empa bipolar-HiPIMS data-prep slice (examples/real_data/empa_hipims): the
converter is deterministic and faithful (row counts, calibration, BatchNr
sort), every TOML spec's bounds exactly match its own Campaign.json, and the
tidy CSVs ingest through the generic WP-H tabular adapter with correct SI
canonicalization -- including the continuous_si trap (CLAUDE.md) and the
Ti-120W float-rounding edge (5 rows ~3e-11 outside their declared bounds)."""

import importlib.util
import json
import sys
import warnings
from csv import DictReader
from pathlib import Path

import pytest

from rig_adapters.tabular.ingest import IngestError, ingest_csv
from rig_adapters.tabular.spec import load_spec

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "real_data" / "empa_hipims"
DATA = REPO / "data" / "m0-candidates" / "empa-hipims" / "extracted_sample"

# (slug, expected rows) in campaign order -- the MANIFEST.md section 1 counts.
EXPECTED_ROWS = {
    "al_120w_short_pw": 601,
    "al_200w_high_pw": 651,
    "al_250w_low_duty": 401,
    "ti_120w_short_pw": 495,
    "ti_200w_high_pw": 601,
    "ti_250w_low_duty": 401,
}
TOTAL_ROWS = 3150

# The one campaign whose BatchNr column is degenerate (all 1, FitNr all null)
# and whose full-precision bounds clip 5 df-rounded rows (see its spec header).
DEGENERATE_SLUG = "ti_120w_short_pw"
DEGENERATE_N_REJECTS = 5


def _load_prepare_empa():
    """Import the converter by path (examples/ is not a package) so its
    manifest constants stay the single source of truth."""
    module_spec = importlib.util.spec_from_file_location("prepare_empa", EX / "prepare_empa.py")
    module = importlib.util.module_from_spec(module_spec)
    # dataclasses resolves the module through sys.modules at class-creation
    # time (PEP 563 annotations) -- register it before exec, or @dataclass
    # crashes with "'NoneType' object has no attribute '__dict__'".
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


prep = _load_prepare_empa()
BY_SLUG = {c.slug: c for c in prep.CAMPAIGNS}


def independent_factor(stem: str) -> float:
    """Read the calibration factor straight from calibration.txt, on purpose
    NOT via prepare_empa.read_calibration (independent check)."""
    lines = (DATA / f"{stem}__calibration.txt").read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].strip() == "us-window, dep-rate calibration"
    return float(lines[1].split(",")[1])


def raw_rows(campaign) -> list[dict]:
    return json.loads(prep.df_path(campaign).read_text(encoding="utf-8"))


def campaign_json_bounds(stem: str) -> dict[str, tuple[float, float]]:
    """Authoritative bounds from the campaign's OWN Campaign.json (a
    JSON-encoded STRING of BayBE state -> json.loads twice)."""
    outer = json.loads((DATA / f"{stem}__Campaign.json").read_text(encoding="utf-8"))
    state = json.loads(outer)
    return {
        p["name"]: (p["bounds"]["lower"], p["bounds"]["upper"])
        for p in state["searchspace"]["continuous"]["parameters"]
    }


def csv_rows(slug: str) -> list[dict]:
    with (EX / "csv" / f"{slug}.csv").open("r", encoding="utf-8", newline="") as fh:
        return list(DictReader(fh))


def ingest(slug: str, **kwargs):
    spec = load_spec(EX / "specs" / f"{slug}.toml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unmatched y1/BatchNr/FitNr warning is expected
        return spec, ingest_csv(
            EX / "csv" / f"{slug}.csv", spec, source="real_tool", **kwargs
        )


# -- converter ---------------------------------------------------------------


def test_converter_deterministic_and_current(tmp_path):
    """Two fresh runs are byte-identical, and both match the checked-in CSVs
    (i.e. the repo copies are exactly what the converter produces today)."""
    prep.main(tmp_path / "a")
    prep.main(tmp_path / "b")
    for slug in EXPECTED_ROWS:
        a = (tmp_path / "a" / f"{slug}.csv").read_bytes()
        b = (tmp_path / "b" / f"{slug}.csv").read_bytes()
        checked_in = (EX / "csv" / f"{slug}.csv").read_bytes()
        assert a == b, f"{slug}: converter output is not deterministic"
        assert a == checked_in, f"{slug}: checked-in CSV is stale vs the converter"


def test_row_counts_exact():
    counts = {slug: len(csv_rows(slug)) for slug in EXPECTED_ROWS}
    assert counts == EXPECTED_ROWS
    assert sum(counts.values()) == TOTAL_ROWS


def test_calibrated_column_is_y1_times_file_factor():
    for slug, campaign in BY_SLUG.items():
        factor = independent_factor(campaign.stem)
        # same stable sort the converter uses; for the degenerate campaign
        # (all BatchNr==1) stability makes this the raw file order too.
        expected = sorted(raw_rows(campaign), key=lambda r: int(r["BatchNr"]))
        rows = csv_rows(slug)
        assert len(rows) == len(expected)
        for got, raw in zip(rows, expected, strict=True):
            y1 = float(raw["y1"])
            assert float(got["y1"]) == y1
            assert float(got["dep_rate_A_per_s"]) == y1 * factor
        assert 0.01 <= min(float(r["dep_rate_A_per_s"]) for r in rows)
        assert max(float(r["dep_rate_A_per_s"]) for r in rows) <= 1000.0
    # the factors really vary by material -- a single hardcoded constant
    # could not have produced both.
    assert independent_factor(BY_SLUG["al_120w_short_pw"].stem) != independent_factor(
        BY_SLUG[DEGENERATE_SLUG].stem
    )


def test_csvs_sorted_by_batchnr():
    for slug in EXPECTED_ROWS:
        batch = [int(r["BatchNr"]) for r in csv_rows(slug)]
        assert batch == sorted(batch), f"{slug}: CSV not BatchNr-sorted"
        if slug == DEGENERATE_SLUG:
            assert set(batch) == {1}  # verified degenerate: no order info at all
        else:
            assert batch == list(range(1, len(batch) + 1))


# -- TOML specs vs Campaign.json --------------------------------------------


def test_spec_bounds_exactly_match_own_campaign_json():
    for slug, campaign in BY_SLUG.items():
        spec = load_spec(EX / "specs" / f"{slug}.toml")
        declared = {v.name: (v.lower, v.upper) for v in spec.continuous}
        authoritative = {
            prep.RENAMES.get(name, name): bounds
            for name, bounds in campaign_json_bounds(campaign.stem).items()
        }
        assert declared == authoritative, f"{slug}: spec bounds != Campaign.json bounds"
        # parameterization sanity: PRR campaigns declare PRR, duty ones Duty Cycle
        knob = "PRR (Hz)" if campaign.parameterization == "prr" else "Duty Cycle (ratio)"
        assert knob in declared
        assert spec.output_names == ("dep_rate_A_per_s", "Ipk (A)")


def test_continuous_si_respects_the_trap():
    """ProcessSpec.continuous_si (never .continuous) is what pairs with the
    SI-canonicalized ingested data -- the CLAUDE.md SI-vs-declared-bounds trap."""
    spec = load_spec(EX / "specs" / "al_120w_short_pw.toml")
    si = {v.name: v for v in spec.continuous_si}
    assert si["PW (us)"].upper == pytest.approx(1e-4, rel=1e-12)  # 100 us -> 1e-4 s
    assert si["PW (us)"].lower == pytest.approx(5e-6, rel=1e-12)
    assert si["PW (us)"].unit == "s"
    assert si["pos Delay (us)"].upper == pytest.approx(4e-5, rel=1e-12)
    assert (si["PRR (Hz)"].lower, si["PRR (Hz)"].upper) == (500.0, 5000.0)  # Hz -> 1/s, x1
    assert (si["pos Setpoint (V)"].lower, si["pos Setpoint (V)"].upper) == (0.0, 100.0)

    duty_spec = load_spec(EX / "specs" / "ti_250w_low_duty.toml")
    duty_si = {v.name: v for v in duty_spec.continuous_si}
    assert (duty_si["Duty Cycle (ratio)"].lower, duty_si["Duty Cycle (ratio)"].upper) == (
        0.012,
        0.0375,
    )


# -- ingest through the tabular adapter --------------------------------------


def test_ingest_prr_campaign():
    slug = "al_120w_short_pw"
    _, result = ingest(slug, on_error="raise")
    assert len(result) == EXPECTED_ROWS[slug] and not result.rejects
    assert set(result.unmatched_columns) == {"y1", "BatchNr", "FitNr"}

    rec = result.records[0]  # BatchNr 1 == raw file row 0
    raw = raw_rows(BY_SLUG[slug])[0]
    assert raw["BatchNr"] == 1
    assert rec.provenance.source == "real_tool"  # headline-metrics eligibility (§3.5)

    # SI canonicalization spot-checks: PW (us) value x1e-6 ...
    pw = rec.recipe.values["PW (us)"]
    assert pw.unit == "s"
    assert pw.magnitude == pytest.approx(raw["PW (us)"] * 1e-6, rel=1e-12)
    # ... and the angstrom/second dep rate x1e-10.
    factor = independent_factor(BY_SLUG[slug].stem)
    dep = next(o for o in rec.outcomes if o.name == "dep_rate_A_per_s")
    assert dep.value.unit == "m / s"
    assert dep.value.magnitude == pytest.approx(raw["y1"] * factor * 1e-10, rel=1e-12)

    # provenance columns rode along in extra (undeclared by design).
    assert rec.extra["unmatched_columns"]["BatchNr"] == "1"
    assert "FitNr" in rec.extra["unmatched_columns"]
    assert float(rec.extra["unmatched_columns"]["y1"]) == raw["y1"]


def test_ingest_duty_campaign():
    slug = "ti_250w_low_duty"
    _, result = ingest(slug, on_error="raise")
    assert len(result) == EXPECTED_ROWS[slug] and not result.rejects

    rec = result.records[0]
    raw = raw_rows(BY_SLUG[slug])[0]
    assert raw["BatchNr"] == 1
    assert rec.provenance.source == "real_tool"

    duty = rec.recipe.values["Duty Cycle (ratio)"]
    assert duty.unit == "dimensionless"
    assert duty.magnitude == raw["Duty Cycle (ratio)"]  # x1.0: bit-identical
    ipk = next(o for o in rec.outcomes if o.name == "Ipk (A)")
    assert ipk.value.unit == "A"
    assert ipk.value.magnitude == raw["Ipk (A)"]
    assert rec.extra["unmatched_columns"]["BatchNr"] == "1"


def test_ti_120w_bounds_rounding_edge_is_understood():
    """The documented Ti-120W quirk, pinned down: its Campaign.json bounds are
    full-precision data extents while df values are rounded to 10 decimals, so
    exactly 5 rows sit ~3e-11 outside and the exact-inclusive bounds check
    rejects them. This is data-as-is, not clamped -- fail loudly on 'raise',
    drop exactly those 5 on 'skip'."""
    with pytest.raises(IngestError, match="outside declared range"):
        ingest(DEGENERATE_SLUG, on_error="raise")

    _, result = ingest(DEGENERATE_SLUG, on_error="skip")
    assert len(result) == EXPECTED_ROWS[DEGENERATE_SLUG] - DEGENERATE_N_REJECTS
    assert len(result.rejects) == DEGENERATE_N_REJECTS
    assert all("outside declared range" in r.reason for r in result.rejects)
