"""Empa bipolar-HiPIMS data prep: raw Zenodo campaign JSONs -> tidy per-campaign CSVs.

Dataset : "Clean Datasets Used for Publication" from Zenodo record
          10.5281/zenodo.18495402 (concept DOI 10.5281/zenodo.18495401),
          license CC-BY-4.0. Six BayBE-driven campaigns on a real Empa bipolar
          HiPIMS sputter tool (Wieczorek et al., Digital Discovery 2026,
          DOI 10.1039/D6DD00063K). Raw files live under
          data/m0-candidates/empa-hipims/extracted_sample (see
          data/m0-candidates/MANIFEST.md section 1 for download verification).

What this script does (no RNG anywhere -- the pipeline is deterministic by
construction, and the tests assert byte-identical output across runs):

1. reads each campaign's ``<stem>__df_campaign_*.json`` (a JSON LIST of row
   dicts) and its ``<stem>__calibration.txt``;
2. applies the per-campaign calibration scalar read FROM THE FILE (never
   hardcoded -- see the cross-file sanity assertions in ``main``):
   ``dep_rate_A_per_s = y1 * factor``, in angstrom/second (the paper: "SHAP
   values retain the units of the model output (A s-1)"). BOTH the raw ``y1``
   and the calibrated column are kept in the output;
3. writes one tidy CSV per campaign to ``csv/<slug>.csv``, rows stable-sorted
   by ``BatchNr`` ascending, keeping ``BatchNr``/``FitNr`` (undeclared in the
   specs, so the tabular adapter parks them in ``RunRecord.extra``).

Column renames (recorded in ``RENAMES``): the tabular adapter reserves ``.``
in variable names for its ``<variable>.<component>`` compositional-flattening
convention, so the three raw ``pos. *`` headers are renamed by dropping the
dot (``pos. Delay (us)`` -> ``pos Delay (us)``, etc.). Everything else keeps
the raw header verbatim.

HONEST CAVEATS (repeat these wherever results are framed):

- **BO-driven sampling, not space-filling**: every campaign was sampled by a
  BayBE Bayesian-optimization loop chasing high deposition rate, so rows
  CLUSTER near optima. Coverage of the declared box is uneven, and any
  i.i.d./exchangeability assumption (e.g. split-conformal guarantees) is an
  approximation on this data.
- **Ipk (A) is measured, not set**: peak current was "not controlled in the
  Bayesian optimization (only measured)" (paper) -- it is an OUTPUT here,
  never an input knob.
- **Ti - 120 W - short PW is degenerate on provenance columns**: every row
  has ``BatchNr == 1`` and ``FitNr == null`` (verified), so BatchNr carries NO
  run-order information there and the stable sort preserves raw file order.
  Its Campaign.json bounds equal the observed data extents at full float
  precision, while the df_campaign values are rounded to 10 decimals -- 5 rows
  therefore sit 3e-11 to 4e-11 OUTSIDE the declared bounds (see the spec header and
  tests/test_empa_ingest.py).

Run:  PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/prepare_empa.py
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

# Force UTF-8 so a cp1252 Windows console cannot crash the summary print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DATA_DIR = REPO / "data" / "m0-candidates" / "empa-hipims" / "extracted_sample"
CSV_DIR = HERE / "csv"

# calibration.txt contract: exactly this header line, then "window, factor".
CAL_HEADER = "us-window, dep-rate calibration"

# Physical plausibility band for a sputter deposition rate in angstrom/second.
DEP_RATE_MIN_A_PER_S = 0.01
DEP_RATE_MAX_A_PER_S = 1000.0

# The tabular adapter reserves '.' in variable names (compositional
# flattening, spec._check_name), so the raw 'pos. *' headers are renamed by
# dropping the dot. Keys = raw df_campaign names, values = tidy CSV names.
RENAMES = {
    "pos. Delay (us)": "pos Delay (us)",
    "pos. PW (us)": "pos PW (us)",
    "pos. Setpoint (V)": "pos Setpoint (V)",
}

DEP_RATE_COLUMN = "dep_rate_A_per_s"  # calibrated deposition rate, angstrom/s


@dataclass(frozen=True)
class Campaign:
    """One raw Empa campaign (file stems verbatim -- NB the DOUBLE space in
    the Al short-PW stem and the missing dashes in the Ti low-duty stem)."""

    slug: str  # csv/<slug>.csv + specs/<slug>.toml
    stem: str  # raw file-name stem in DATA_DIR
    df_key: str  # df JSON name is "<stem>__df_campaign_<df_key>.json" (see df_path)
    material: str  # "Al" | "Ti"
    parameterization: str  # "prr" (PRR knob) | "duty" (Duty Cycle knob)
    n_rows: int  # expected row count (MANIFEST.md section 1, verified)


CAMPAIGNS: tuple[Campaign, ...] = (
    Campaign("al_120w_short_pw", "Al - 120 W  - short PW", "Al_shortPW", "Al", "prr", 601),
    Campaign("al_200w_high_pw", "Al - 200 W - high PW", "Al_highPW", "Al", "prr", 651),
    Campaign("al_250w_low_duty", "Al - 250 W - duty cycle series", "Al_lowDuty", "Al", "duty", 401),
    Campaign("ti_120w_short_pw", "Ti - 120 W - short PW", "Ti_lowPW", "Ti", "prr", 495),
    Campaign("ti_200w_high_pw", "Ti - 200 W - high PW", "Ti_highPW", "Ti", "prr", 601),
    Campaign("ti_250w_low_duty", "Ti 250 W low duty cycle", "Ti_lowDuty", "Ti", "duty", 401),
)


def df_path(campaign: Campaign) -> Path:
    """Full path of a campaign's df_campaign JSON (raw outcome table)."""
    return DATA_DIR / f"{campaign.stem}__df_campaign_{campaign.df_key}.json"


# The 5 continuous knobs per parameterization, in tidy (renamed) CSV spelling.
PRR_INPUTS = ("PRR (Hz)", "PW (us)", "pos Delay (us)", "pos PW (us)", "pos Setpoint (V)")
DUTY_INPUTS = ("Duty Cycle (ratio)", "PW (us)", "pos Delay (us)", "pos PW (us)", "pos Setpoint (V)")


def input_columns(campaign: Campaign) -> tuple[str, ...]:
    return PRR_INPUTS if campaign.parameterization == "prr" else DUTY_INPUTS


def csv_columns(campaign: Campaign) -> tuple[str, ...]:
    """Tidy CSV header: 5 knobs, raw y1, calibrated rate, Ipk, provenance."""
    return (*input_columns(campaign), "y1", DEP_RATE_COLUMN, "Ipk (A)", "BatchNr", "FitNr")


def read_calibration(path: Path) -> tuple[float, float]:
    """Parse ``calibration.txt`` -> (us_window, dep-rate factor).

    Loud on any drift: the header line must match ``CAL_HEADER`` verbatim and
    the factor must be a positive number -- this is the "actually read from
    the file, not hardcoded" guarantee (plus the cross-file checks in main).
    """
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != 2 or lines[0] != CAL_HEADER:
        raise ValueError(f"{path}: expected header {CAL_HEADER!r} + one data line, got {lines!r}")
    window_s, factor_s = (tok.strip() for tok in lines[1].split(","))
    window, factor = float(window_s), float(factor_s)
    if not (window > 0 and factor > 0):
        raise ValueError(f"{path}: non-positive calibration values {lines[1]!r}")
    return window, factor


def _fmt(value: object) -> str:
    """Deterministic cell formatting: shortest round-trip float repr; ints as
    ints; JSON null (FitNr) -> empty cell."""
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    return repr(float(value))


def convert_campaign(campaign: Campaign, out_dir: Path) -> dict[str, object]:
    """Read one campaign, calibrate, sort, write ``<out_dir>/<slug>.csv``."""
    source = df_path(campaign)
    rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != campaign.n_rows:
        raise ValueError(f"{source}: expected a list of {campaign.n_rows} rows, got {len(rows)}")

    # Loud schema check: raw key set must be exactly the known one.
    tidy_to_raw = {tidy: raw for raw, tidy in RENAMES.items()}
    raw_inputs = tuple(tidy_to_raw.get(c, c) for c in input_columns(campaign))
    expected_keys = {*raw_inputs, "y1", "BatchNr", "FitNr", "Ipk (A)"}
    for i, row in enumerate(rows):
        if set(row) != expected_keys:
            raise ValueError(f"{source}: row {i} keys {sorted(row)} != {sorted(expected_keys)}")

    window_us, factor = read_calibration(DATA_DIR / f"{campaign.stem}__calibration.txt")

    # Stable sort by BatchNr ascending. In 5 of 6 campaigns BatchNr is already
    # 1..n (sort is a no-op); in Ti - 120 W - short PW EVERY BatchNr is 1, so
    # stability preserves the raw file order (the only order we have there).
    rows = sorted(rows, key=lambda r: int(r["BatchNr"]))

    dep_rates: list[float] = []
    out_rows: list[list[str]] = []
    for row in rows:
        y1 = float(row["y1"])
        dep = y1 * factor
        if not (DEP_RATE_MIN_A_PER_S <= dep <= DEP_RATE_MAX_A_PER_S):
            raise ValueError(
                f"{campaign.slug}: calibrated rate {dep!r} A/s (y1={y1!r} * factor={factor!r}) "
                f"outside plausibility band [{DEP_RATE_MIN_A_PER_S}, {DEP_RATE_MAX_A_PER_S}]"
            )
        dep_rates.append(dep)
        cells = [_fmt(row[raw]) for raw in raw_inputs]
        cells += [_fmt(y1), _fmt(dep), _fmt(row["Ipk (A)"])]
        cells += [_fmt(row["BatchNr"]), _fmt(row["FitNr"])]
        out_rows.append(cells)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{campaign.slug}.csv"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(csv_columns(campaign))
        writer.writerows(out_rows)

    return {
        "slug": campaign.slug,
        "n": len(out_rows),
        "window_us": window_us,
        "factor": factor,
        "dep_min": min(dep_rates),
        "dep_median": statistics.median(dep_rates),
        "dep_max": max(dep_rates),
        "path": out_path,
    }


def main(out_dir: Path | None = None) -> int:
    out_dir = CSV_DIR if out_dir is None else Path(out_dir)
    summaries = []
    by_material: dict[str, set[float]] = {}
    for campaign in CAMPAIGNS:
        summary = convert_campaign(campaign, out_dir)
        summaries.append(summary)
        by_material.setdefault(campaign.material, set()).add(float(summary["factor"]))

    # Calibration factors must have come from the files, not one constant:
    # each material uses ONE factor across its 3 campaigns, and Al != Ti.
    per_material = {m: sorted(v) for m, v in by_material.items()}
    if any(len(v) != 1 for v in per_material.values()):
        raise ValueError(f"calibration factors vary within a material: {per_material}")
    if by_material["Al"] == by_material["Ti"]:
        raise ValueError(f"Al and Ti share one calibration factor (hardcoded?): {per_material}")

    total = sum(int(s["n"]) for s in summaries)
    print(f"Empa HiPIMS data prep -> {out_dir}  (total rows: {total})")
    hdr = f"{'campaign':<20}{'rows':>6}{'us-win':>8}{'factor':>10}{'dep min':>9}{'median':>9}{'max':>9}  [A/s]"
    print(hdr)
    print("-" * len(hdr))
    for s in summaries:
        print(
            f"{s['slug']:<20}{s['n']:>6}{s['window_us']:>8.0f}{s['factor']:>10.6g}"
            f"{s['dep_min']:>9.4f}{s['dep_median']:>9.4f}{s['dep_max']:>9.4f}"
        )
    print(
        "\nNB: BO-sampled (BayBE) data -- rows cluster near optima, NOT space-filling; "
        "Ipk is measured, never set; ti_120w_short_pw has BatchNr==1 everywhere "
        "(file order kept) and 5 rows 3e-11 to 4e-11 outside its Campaign.json bounds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
