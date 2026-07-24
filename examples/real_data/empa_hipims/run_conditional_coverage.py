"""CONDITIONAL / PER-REGION conformal coverage on the RECORDED Empa M1 results.

The recorded M1 gate (run_m1_empa.py -> results/m1_empa.json) reports POOLED
PICP per campaign/split (with per-output detail). Pooled/marginal coverage can
hide REGIONAL under-coverage: a calibrator can over-cover in the data-dense
centre and under-cover at the edges and still hit ~0.90 marginally. Section-20
conformal practice favours group-conditional / Mondrian checks. This script
measures the CONDITIONAL coverage of the EXISTING recorded static / ACI / PID
paths, honestly, WITHOUT touching the recorded artifact.

--------------------------------------------------------------------------------
PRE-STATED GROUPS  (fixed BEFORE any coverage number is computed -- choosing
groups after seeing coverage is p-hacking; there is no post-hoc slicing here and
no group other than the four below).  All group-defining statistics are functions
of the inputs / true outcomes / stream order ONLY -- never of the coverage
outcome -- so grouping cannot be tuned to a result.  Tertile edges are the 1/3
and 2/3 empirical quantiles of the grouping statistic over the TEST rows, applied
identically to every path (static/ACI/PID share the same test rows in the same
order, so one set of masks serves all three).  A group with < MIN_GROUP_N (=20)
test points is flagged UNDERPOWERED and its point estimate is reported only
alongside that flag.

  (1) KNOB-SPACE DENSITY tertiles -- near / mid / far FROM DATA.
      Statistic = Euclidean distance from a test recipe to its KNN_K-th (=5)
      nearest TRAINING neighbour in STANDARDIZED knob space (standardized by the
      TRAIN slice's own mean/std -- the same leakage-safe standardization the GP
      uses).  The neighbour reference set is the TRAINING (fit) recipes ONLY: no
      test point is ever in the reference, so a test point cannot be its own
      neighbour (no leakage).  Small distance = dense/near; large = sparse/far.
      Tertiles: near (low third), mid, far (high third).  Shared across outputs
      (the statistic is on the knob vector); coverage is measured per output.

  (2) OUTCOME-MAGNITUDE tertiles -- low / mid / high, PER OUTPUT.
      Statistic = the TRUE observed outcome value of that output on the test row
      (dep_rate_A_per_s and Ipk (A) tertiled separately).  Tertiles: low / mid /
      high third of the observed values.  A magnitude-dependent coverage tilt is
      the classic marginal-hiding failure (e.g. under-cover the high tail).

  (3) STREAM-POSITION tertiles -- early / mid / late (DRIFT PHASE), TEMPORAL
      SPLIT ONLY.  The temporal test slice is the last 20% of the campaign in
      BatchNr run order; position within that stream is split into three
      contiguous thirds: early / mid / late.  Late = most BO-drifted
      (exploitation) recipes -- the phase a static calibrator is least able to
      track and the phase the online (ACI/PID) paths exist to repair.  Not
      defined for the RANDOM split (shuffled order has no drift phase); reported
      as null there.

  (4) MONDRIAN PER-OUTPUT -- coverage of dep_rate_A_per_s and Ipk (A) separately
      over the whole test slice.  This is already implicit in the recorded
      per-output blocks; included for completeness and as the group-of-one
      reference the tertile groups refine.

For every (campaign x split x path x output x group x tertile) cell: covered k /
n, PICP, exact (Clopper-Pearson) binomial 95% CI, a directional flag
(nominal 0.90 inside the CI -- the SAME rule the recorded gate uses), the
under/over/ok direction, and the UNDERPOWERED flag.

--------------------------------------------------------------------------------
FIDELITY GATE (must pass for all 12 campaign/split cells before any conditional
number is trusted).  The per-step covered/not-covered indicator sequences are
REPRODUCED by re-running run_m1_empa.py's OWN protocol on IMPORTED primitives
(its ingest_campaign / split_indices / fit_and_eval fit the identical GP on the
identical splits; the static conformal band and the ACI/PID online streams are
built from the same rig.calibration objects the runner uses, and the online
per-step indicator is the controller's own observe() return -- no forked logic).
The reproduced pooled AND per-output k_covered for static/ACI/PID are asserted
byte-equal to results/m1_empa.json (and, for the online paths, cross-checked
against the runner's own aci_eval / pid_eval).  If any cell diverges the script
RAISES and writes nothing -- divergent replication is never carried forward.

Outputs: results/m1_empa_conditional.json (NEW file; results/m1_empa.json is
never touched).  Deterministic: no RNG of its own; the reused fits are seeded;
a second run is byte-identical.

Run (Windows cp1252 console -> force UTF-8):

    PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_conditional_coverage.py
        [--campaign <slug>] [--out <path.json>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
BASELINE_JSON = RESULTS_DIR / "m1_empa.json"

# -- pre-stated analysis constants (fixed before any coverage is computed) -----
KNN_K = 5  # k-th nearest TRAINING neighbour for the density statistic (group 1)
MIN_GROUP_N = 20  # groups smaller than this are flagged UNDERPOWERED
N_TERTILES = 3
DENSITY_LABELS = ("near", "mid", "far")  # low / mid / high kNN-distance
MAGNITUDE_LABELS = ("low", "mid", "high")  # low / mid / high true outcome
STREAM_LABELS = ("early", "mid", "late")  # temporal stream position thirds


def _load_runner():
    """Import run_m1_empa.py by PATH (examples/ is not a package), reusing ALL of
    its protocol primitives so the reproduction cannot fork the runner's logic.

    Mirrors tests/test_empa_ingest.py's by-path loader (register in sys.modules
    before exec so prepare_empa's PEP-563 dataclass resolution does not trip).
    prepare_empa.py is being edited by a concurrent agent; retry a transient
    import failure a few times rather than aborting the whole study.
    """
    last: Exception | None = None
    for _ in range(5):
        try:
            module_spec = importlib.util.spec_from_file_location(
                "run_m1_empa", HERE / "run_m1_empa.py"
            )
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            return module
        except Exception as exc:  # noqa: BLE001 -- transient mid-edit import, retry
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"could not import run_m1_empa.py (concurrent edit?): {last!r}")


runner = _load_runner()

# Reused verbatim from the runner (single source of truth for the protocol):
SEED = runner.SEED
ALPHA = runner.ALPHA
NOMINAL = runner.NOMINAL
CI_LEVEL = runner.CI_LEVEL
CAMPAIGNS = runner.CAMPAIGNS
binom_ci = runner.binom_ci  # the exact Clopper-Pearson CI the recorded gate uses


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ============================================================================
# Split reconstruction + the recorded per-step indicator sequences
# ============================================================================


def campaign_arrays(campaign):
    """Ingest one campaign EXACTLY as run_m1_empa.run_campaign does, returning
    (X SI, Y readable, input_keys, output_keys, units, n, degenerate).

    Uses the runner's own ingest_campaign / records_to_arrays / ureg so the
    array construction is byte-identical to the recorded run (the de-SI back to
    readable units mirrors run_campaign line-for-line)."""
    spec, result, degenerate = runner.ingest_campaign(campaign)
    records = result.records
    input_keys = list(spec.gp_input_keys)
    output_keys = list(spec.output_names)
    units = [o.unit for o in spec.outputs]
    if [v.name for v in spec.continuous_si] != input_keys:
        raise RuntimeError(f"{campaign.slug}: continuous_si order != gp_input_keys")
    X, Y_si = runner.records_to_arrays(records, input_keys, output_keys)
    si_per_raw = np.array(
        [float(runner.ureg.Quantity(1.0, u).to_base_units().magnitude) for u in units]
    )
    Y = Y_si / si_per_raw
    return X, Y, input_keys, output_keys, units, len(records), degenerate


def reconstruct_splits(n: int, campaign_index: int) -> dict:
    """Temporal (contiguous run order) + random (seeded permutation) splits,
    reconstructed with the runner's split arithmetic and the exact seed contract
    (random uses default_rng(SEED + campaign_index), the CAMPAIGNS enumerate
    index -- identical to run_m1_empa)."""
    n_train, n_cal = runner.split_indices(n)
    idx = np.arange(n)
    splits = {
        "temporal": (idx[:n_train], idx[n_train : n_train + n_cal], idx[n_train + n_cal :]),
    }
    perm = np.random.default_rng(SEED + campaign_index).permutation(n)
    splits["random"] = (perm[:n_train], perm[n_train : n_train + n_cal], perm[n_train + n_cal :])
    return splits


def static_indicators(model, Xc, Yc, Xt, Yt):
    """Static split-conformal per-step covered indicator (n_test, m).

    Reproduces fit_and_eval's `inside` computation with the identical primitives:
    fresh SplitConformalCalibrator on the calibration slice, ConformalForwardModel
    band, membership = (Yt >= lo) & (Yt <= hi)."""
    cal = runner.SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    conf = runner.ConformalForwardModel(model, cal)
    dist = conf.predict(Xt)
    cset = np.asarray(dist.conformal_set)  # (n_test, m, 2)
    lo, hi = cset[..., 0], cset[..., 1]
    return (Yt >= lo) & (Yt <= hi)


def stream_indicators(controller, Xt, Yt):
    """Online per-step covered indicator (n_test, m) + interval widths (n_test, m).

    Reproduces the aci_eval / pid_eval streaming protocol verbatim: score the
    band at the CURRENT (pre-update) state, THEN observe -- the per-step
    miscoverage indicator is the controller's OWN observe() return (the runner
    scores exactly this and asserts it equals its own miss computation), so this
    is the runner's logic, not a fork. widths[t] is taken from the same
    pre-update band and lets us reproduce n_infinite_width for ACI."""
    n_test, m = Yt.shape
    hits = np.zeros((n_test, m), dtype=bool)
    widths = np.empty((n_test, m))
    for t in range(n_test):
        itv = controller.interval(Xt[t])  # pre-update band; interval() does not mutate state
        lo, hi = itv[..., 0].reshape(-1), itv[..., 1].reshape(-1)
        widths[t] = hi - lo
        err = controller.observe(Xt[t], Yt[t])  # scores the SAME pre-update band
        hits[t] = err == 0.0
    return hits, widths


def path_indicators(model, Xc, Yc, Xt, Yt, output_keys, units):
    """All three recorded paths' per-step covered indicators on one fitted model.

    Returns {path: {"hits": (n_test, m) bool, "n_infinite_width": (m,) int,
    "aggregate": <runner metrics dict or None>}}. The online aggregates come from
    the runner's OWN aci_eval / pid_eval (imported) so the fidelity gate can
    cross-check the reproduced indicators against the runner function too."""
    out: dict[str, dict] = {}

    # static
    s_hits = static_indicators(model, Xc, Yc, Xt, Yt)
    out["static"] = {
        "hits": s_hits,
        "n_infinite_width": np.zeros(s_hits.shape[1], dtype=int),
        "aggregate": None,
    }

    # ACI (fresh calibrator + controller on the SAME model/cal slice, as aci_eval)
    cal_a = runner.SplitConformalCalibrator(alpha=ALPHA)
    cal_a.fit(model, Xc, Yc)
    aci = runner.ACIController(cal_a, alpha_target=ALPHA)
    a_hits, a_w = stream_indicators(aci, Xt, Yt)
    out["aci"] = {
        "hits": a_hits,
        "n_infinite_width": (~np.isfinite(a_w)).sum(axis=0).astype(int),
        "aggregate": runner.aci_eval(model, Xc, Yc, Xt, Yt, output_keys, units),
    }

    # conformal-PID (fresh calibrator + controller, as pid_eval)
    cal_p = runner.SplitConformalCalibrator(alpha=ALPHA)
    cal_p.fit(model, Xc, Yc)
    pid = runner.ConformalPIDController(cal_p, alpha_target=ALPHA)
    p_hits, p_w = stream_indicators(pid, Xt, Yt)
    out["pid"] = {
        "hits": p_hits,
        "n_infinite_width": (~np.isfinite(p_w)).sum(axis=0).astype(int),
        "aggregate": runner.pid_eval(model, Xc, Yc, Xt, Yt, output_keys, units),
    }
    return out


# ============================================================================
# Fidelity gate: reproduced coverage == recorded results/m1_empa.json
# ============================================================================


def fidelity_check(slug, split_name, paths, output_keys, recorded_split) -> dict:
    """Assert every reproduced k_covered (per output + pooled) equals the
    recorded numbers for static/ACI/PID; cross-check the online paths against the
    runner's own aci_eval/pid_eval; verify ACI n_infinite_width. RAISE on any
    mismatch (never proceed on divergent replication)."""
    report: dict[str, dict] = {}
    for path in ("static", "aci", "pid"):
        hits = paths[path]["hits"]
        rec = recorded_split if path == "static" else recorded_split[path]
        per_output = {}
        for j, key in enumerate(output_keys):
            repro_k = int(hits[:, j].sum())
            rec_k = int(rec["per_output"][key]["k_covered"])
            rec_n = int(rec["per_output"][key]["n_test"])
            if hits.shape[0] != rec_n:
                raise RuntimeError(
                    f"FIDELITY FAIL {slug}/{split_name}/{path}/{key}: n_test "
                    f"{hits.shape[0]} != recorded {rec_n}"
                )
            if repro_k != rec_k:
                raise RuntimeError(
                    f"FIDELITY FAIL {slug}/{split_name}/{path}/{key}: k_covered "
                    f"{repro_k} != recorded {rec_k} -- replication diverged, STOP"
                )
            # ACI n_infinite_width byte-check (0 for static/PID by construction)
            if path == "aci":
                repro_niw = int(paths[path]["n_infinite_width"][j])
                rec_niw = int(rec["per_output"][key].get("n_infinite_width", 0))
                if repro_niw != rec_niw:
                    raise RuntimeError(
                        f"FIDELITY FAIL {slug}/{split_name}/aci/{key}: n_infinite_width "
                        f"{repro_niw} != recorded {rec_niw}"
                    )
            # cross-check the reproduced indicators against the runner's OWN eval
            if paths[path]["aggregate"] is not None:
                fn_k = int(paths[path]["aggregate"]["per_output"][key]["k_covered"])
                if repro_k != fn_k:
                    raise RuntimeError(
                        f"FIDELITY FAIL {slug}/{split_name}/{path}/{key}: reproduced "
                        f"k {repro_k} != runner {path}_eval k {fn_k} -- forked logic"
                    )
            per_output[key] = {"k_covered": repro_k, "n_test": rec_n, "matches_recorded": True}
        # pooled
        repro_pool = int(hits.sum())
        rec_pool = int(rec["pooled"]["k_covered"])
        if repro_pool != rec_pool:
            raise RuntimeError(
                f"FIDELITY FAIL {slug}/{split_name}/{path}: pooled k_covered "
                f"{repro_pool} != recorded {rec_pool} -- STOP"
            )
        report[path] = {
            "per_output": per_output,
            "pooled": {
                "reproduced_picp": repro_pool / int(hits.size),
                "recorded_picp": rec["pooled"]["picp"],
                "k_covered": repro_pool,
                "n_trials": int(hits.size),
                "matches_recorded": True,
            },
        }
    return report


# ============================================================================
# Group assignment (pre-stated; deterministic; leakage-safe)
# ============================================================================


def standardize_by_train(X_test, X_train):
    """Standardize both by the TRAIN slice's mean/std (zero-variance -> scale 1),
    the same leakage-safe transform the GP uses. Returns (X_test_std, X_train_std)."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    scale = np.where(std > 0.0, std, 1.0)
    return (X_test - mean) / scale, (X_train - mean) / scale


def knn_distance_to_train(X_test, X_train, k=KNN_K):
    """Group-1 statistic: distance from each test recipe to its k-th nearest
    TRAINING neighbour in standardized knob space. Reference = training points
    ONLY (no test point in the neighbour set -> no leakage). Returns (n_test,)."""
    Xt_std, Xtr_std = standardize_by_train(X_test, X_train)
    k = min(k, Xtr_std.shape[0])  # guard tiny train sets (never triggers on Empa)
    d = cdist(Xt_std, Xtr_std)  # (n_test, n_train)
    d.sort(axis=1)
    return d[:, k - 1]  # k-th nearest (1-based)


def tertile_labels(values, labels):
    """Assign each value to a tertile by the 1/3, 2/3 empirical quantiles of
    `values`. Deterministic edges; np.digitize with the two interior edges.
    Returns (labels_array, (edge_lo, edge_hi))."""
    values = np.asarray(values, dtype=float)
    e_lo, e_hi = (float(q) for q in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0]))
    bins = np.digitize(values, [e_lo, e_hi])  # 0 (< e_lo), 1 ([e_lo,e_hi)), 2 (>= e_hi)
    return np.array([labels[b] for b in bins]), (e_lo, e_hi)


def stream_position_labels(n_test):
    """Group-3 labels: contiguous early/mid/late thirds of the stream (temporal
    only). np.array_split gives deterministic near-equal thirds."""
    labels = np.empty(n_test, dtype=object)
    parts = np.array_split(np.arange(n_test), N_TERTILES)
    for lab, part in zip(STREAM_LABELS, parts, strict=True):
        labels[part] = lab
    return labels


# ============================================================================
# Per-group coverage
# ============================================================================


def group_coverage(hits_col, mask):
    """Coverage of one output/path within a boolean group mask: k/n, PICP, exact
    binomial 95% CI, directional flag (nominal in CI), under/over/ok direction,
    UNDERPOWERED flag (n < MIN_GROUP_N)."""
    n = int(mask.sum())
    k = int(hits_col[mask].sum())
    if n == 0:
        return {
            "n": 0,
            "k_covered": 0,
            "picp": None,
            "ci95": None,
            "nominal_in_ci": None,
            "direction": None,
            "underpowered": True,
        }
    lo, hi = binom_ci(k, n)
    if hi < NOMINAL:
        direction = "under"
    elif lo > NOMINAL:
        direction = "over"
    else:
        direction = "ok"
    return {
        "n": n,
        "k_covered": k,
        "picp": k / n,
        "ci95": [lo, hi],
        "nominal_in_ci": bool(lo <= NOMINAL <= hi),
        "direction": direction,
        "underpowered": bool(n < MIN_GROUP_N),
    }


def coverage_by_paths(hits_by_path, mask, output_keys):
    """{path: {output: group_coverage}} for one group mask across static/ACI/PID."""
    return {
        path: {
            key: group_coverage(hits_by_path[path]["hits"][:, j], mask)
            for j, key in enumerate(output_keys)
        }
        for path in ("static", "aci", "pid")
    }


def tertile_block(hits_by_path, label_array, labels, output_keys):
    """{tertile: {"n": .., "by_path": coverage_by_paths}} over the labelled test rows."""
    block = {}
    for lab in labels:
        mask = label_array == lab
        block[lab] = {
            "n": int(mask.sum()),
            "by_path": coverage_by_paths(hits_by_path, mask, output_keys),
        }
    return block


# ============================================================================
# One campaign x split cell: fidelity + all four pre-stated groups
# ============================================================================


def analyze_cell(slug, split_name, hits_by_path, X, Y, fit_idx, test_idx, output_keys, degenerate):
    X_train, X_test = X[fit_idx], X[test_idx]
    Y_test = Y[test_idx]
    n_test = X_test.shape[0]

    # (1) knob-space density tertiles (near/mid/far) -- shared across outputs
    dens = knn_distance_to_train(X_test, X_train)
    dens_labels, dens_edges = tertile_labels(dens, DENSITY_LABELS)
    density = {
        "statistic": f"distance to {KNN_K}-th nearest TRAINING neighbour, standardized knobs",
        "edges_lo_hi": list(dens_edges),
        "tertiles": tertile_block(hits_by_path, dens_labels, DENSITY_LABELS, output_keys),
    }

    # (2) outcome-magnitude tertiles (low/mid/high) -- PER OUTPUT
    magnitude = {}
    for j, key in enumerate(output_keys):
        mag_labels, mag_edges = tertile_labels(Y_test[:, j], MAGNITUDE_LABELS)
        mag_block = {}
        for lab in MAGNITUDE_LABELS:
            mask = mag_labels == lab
            mag_block[lab] = {
                "n": int(mask.sum()),
                # magnitude is per-output; still report all three paths on THIS output
                "by_path": {
                    path: group_coverage(hits_by_path[path]["hits"][:, j], mask)
                    for path in ("static", "aci", "pid")
                },
            }
        magnitude[key] = {
            "statistic": "true observed outcome value on the test row",
            "edges_lo_hi": list(mag_edges),
            "tertiles": mag_block,
        }

    # (3) stream-position tertiles (early/mid/late) -- TEMPORAL split only
    if split_name == "temporal":
        pos_labels = stream_position_labels(n_test)
        stream = {
            "note": (
                "contiguous thirds of the BatchNr-order test stream; late = most BO-drifted"
                + (" (UNVERIFIED file order -- BatchNr degenerate)" if degenerate else "")
            ),
            "temporal_split_meaningful": not degenerate,
            "tertiles": tertile_block(hits_by_path, pos_labels, STREAM_LABELS, output_keys),
        }
    else:
        stream = {
            "note": "not defined for the random split (shuffled order has no drift phase)",
            "tertiles": None,
        }

    # (4) Mondrian per-output over the whole test slice
    full_mask = np.ones(n_test, dtype=bool)
    mondrian = coverage_by_paths(hits_by_path, full_mask, output_keys)

    return {
        "n_test": n_test,
        "groups": {
            "density_near_mid_far": density,
            "magnitude_low_mid_high": magnitude,
            "stream_position_early_mid_late": stream,
            "mondrian_per_output": mondrian,
        },
    }


# ============================================================================
# Headline synthesis
# ============================================================================


def synthesize_headline(campaigns_out):
    """Structured answer to the study's headline questions, derived from the
    computed (never hand-picked) cells."""
    hidden_under = []  # marginal (Mondrian) PASS but a POWERED group under-covers
    late_static_under = []  # temporal late tertile: static under-covers
    far_under = []  # far-from-data tertile under-covers (any path)
    repairs = []  # a group static-fails that ACI and/or PID repair

    for slug, c in campaigns_out.items():
        for split_name, cell in c["splits"].items():
            groups = cell["analysis"]["groups"]
            mond = groups["mondrian_per_output"]

            # scan density + stream-position tertiles for hidden under-coverage
            scan = []
            dens_t = groups["density_near_mid_far"]["tertiles"]
            for lab, blk in dens_t.items():
                scan.append(("density", lab, blk["by_path"]))
            if groups["stream_position_early_mid_late"]["tertiles"] is not None:
                for lab, blk in groups["stream_position_early_mid_late"]["tertiles"].items():
                    scan.append(("stream", lab, blk["by_path"]))
            for key_out in mond["static"].keys():
                for lab, blk in groups["magnitude_low_mid_high"][key_out]["tertiles"].items():
                    scan.append(
                        (
                            f"magnitude:{key_out}",
                            lab,
                            {p: {key_out: blk["by_path"][p]} for p in ("static", "aci", "pid")},
                        )
                    )

            for gname, lab, by_path in scan:
                for key in by_path["static"]:
                    st = by_path["static"][key]
                    if st["picp"] is None or st["underpowered"]:
                        continue
                    if st["direction"] == "under":
                        rec = f"{slug}/{split_name}/{gname}/{lab}/{key} static PICP {st['picp']:.3f} CI {st['ci95']}"
                        if bool(mond["static"][key]["nominal_in_ci"]):
                            hidden_under.append(rec + "  (marginal static PASSES)")
                        if gname == "stream" and lab == "late" and split_name == "temporal":
                            late_static_under.append(rec)
                        if gname == "density" and lab == "far":
                            far_under.append(rec)
                        # repair?
                        aci = by_path["aci"][key]
                        pid = by_path["pid"][key]
                        fixed_by = [
                            name
                            for name, g in (("ACI", aci), ("PID", pid))
                            if g["picp"] is not None
                            and not g["underpowered"]
                            and g["direction"] != "under"
                        ]
                        if fixed_by:
                            repairs.append(
                                f"{slug}/{split_name}/{gname}/{lab}/{key}: static under "
                                f"{st['picp']:.3f} -> repaired by {'+'.join(fixed_by)}"
                            )
    return {
        "question_1_hidden_regional_undercoverage": {
            "count": len(hidden_under),
            "cells": hidden_under,
        },
        "question_2_far_from_data_undercoverage": {"count": len(far_under), "cells": far_under},
        "question_3_late_drift_phase_static_undercoverage": {
            "count": len(late_static_under),
            "cells": late_static_under,
        },
        "question_4_online_repairs_of_static_regional_failures": {
            "count": len(repairs),
            "cells": repairs,
        },
    }


# ============================================================================
# main
# ============================================================================


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=[c.slug for c in CAMPAIGNS], default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if not BASELINE_JSON.exists():
        raise RuntimeError(f"recorded baseline {BASELINE_JSON} missing -- run run_m1_empa.py first")
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    gp_restarts = int(baseline["meta"]["gp_restarts"])  # 5 for the full recorded run

    np.random.seed(SEED)  # mirror run_m1_empa.main (harmless: fit RNG is local)
    out_path = args.out or (
        RESULTS_DIR / f"m1_empa_conditional{('.' + args.campaign) if args.campaign else ''}.json"
    )

    banner("CONDITIONAL / PER-REGION conformal coverage on the RECORDED Empa M1 results")
    print(
        f"baseline : {BASELINE_JSON.name}  (gp_restarts={gp_restarts}, seed={SEED}, alpha={ALPHA})"
    )
    print("groups   : (1) knob-density near/mid/far  (2) outcome-magnitude low/mid/high")
    print("           (3) stream-position early/mid/late [temporal only]  (4) Mondrian per-output")
    print(f"discipline: groups pre-stated in the docstring; UNDERPOWERED if n < {MIN_GROUP_N}")

    t_start = time.perf_counter()
    campaigns_out: dict[str, dict] = {}
    fidelity_cells = 0
    for campaign_index, campaign in enumerate(CAMPAIGNS):
        if args.campaign not in (None, campaign.slug):
            continue
        if campaign.slug not in baseline["campaigns"]:
            continue
        X, Y, input_keys, output_keys, units, n, degenerate = campaign_arrays(campaign)
        rec_campaign = baseline["campaigns"][campaign.slug]
        # split-size fidelity (mirror run_m1_empa_pooled's discipline)
        n_train, n_cal = runner.split_indices(n)
        recon_sizes = {"train": n_train, "cal": n_cal, "test": n - n_train - n_cal}
        if recon_sizes != rec_campaign["split_sizes"]:
            raise RuntimeError(
                f"{campaign.slug}: reconstructed split sizes {recon_sizes} != recorded "
                f"{rec_campaign['split_sizes']} -- split reconstruction drifted"
            )
        splits = reconstruct_splits(n, campaign_index)

        banner(
            f"CAMPAIGN {campaign.slug}  ({rec_campaign['process_id']})  n={n}  sizes={recon_sizes}"
        )
        campaign_out = {
            "process_id": rec_campaign["process_id"],
            "material": rec_campaign["material"],
            "parameterization": rec_campaign["parameterization"],
            "temporal_split_meaningful": rec_campaign["temporal_split_meaningful"],
            "split_sizes": recon_sizes,
            "splits": {},
        }
        for split_name, (fit_idx, cal_idx, test_idx) in splits.items():
            # reuse the runner's fit_and_eval for the identical seeded GP fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, model, _ = runner.fit_and_eval(
                    X[fit_idx],
                    Y[fit_idx],
                    X[cal_idx],
                    Y[cal_idx],
                    X[test_idx],
                    Y[test_idx],
                    input_keys,
                    output_keys,
                    units,
                    gp_restarts,
                )
            hits_by_path = path_indicators(
                model, X[cal_idx], Y[cal_idx], X[test_idx], Y[test_idx], output_keys, units
            )
            fid = fidelity_check(
                campaign.slug,
                split_name,
                hits_by_path,
                output_keys,
                rec_campaign["splits"][split_name],
            )
            fidelity_cells += 1
            analysis = analyze_cell(
                campaign.slug,
                split_name,
                hits_by_path,
                X,
                Y,
                fit_idx,
                test_idx,
                output_keys,
                degenerate,
            )
            campaign_out["splits"][split_name] = {"fidelity": fid, "analysis": analysis}
            _print_cell(campaign.slug, split_name, output_keys, analysis)
        campaigns_out[campaign.slug] = campaign_out

    headline = synthesize_headline(campaigns_out)
    _print_headline(headline)

    payload = {
        "meta": {
            "seed": SEED,
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL,
            "ci": f"exact (Clopper-Pearson) binomial, level {CI_LEVEL}",
            "gp_restarts": gp_restarts,
            "baseline": BASELINE_JSON.name,
            "knn_k": KNN_K,
            "min_group_n_powered": MIN_GROUP_N,
            "directional_flag": "nominal 0.90 inside the exact binomial 95% CI (same rule as the recorded gate)",
            "prestated_groups": {
                "density_near_mid_far": (
                    f"distance to the {KNN_K}-th nearest TRAINING neighbour in standardized knob "
                    "space (reference = training recipes only; no test-point leakage); tertiled near/mid/far"
                ),
                "magnitude_low_mid_high": "per-output true observed outcome tertiled low/mid/high",
                "stream_position_early_mid_late": "TEMPORAL split only: contiguous thirds of the run-order test stream (late = most BO-drifted)",
                "mondrian_per_output": "coverage per output over the whole test slice (completeness reference)",
            },
            "fidelity_gate": {
                "cells_checked": fidelity_cells,
                "status": "PASS -- every reproduced static/ACI/PID pooled AND per-output k_covered is byte-equal to results/m1_empa.json (ACI n_infinite_width verified; online paths cross-checked against runner aci_eval/pid_eval)",
            },
            "caveats": [
                "READ-ONLY re-analysis of the recorded results/m1_empa.json paths; that file is never modified",
                "REAL Empa tool data (real_tool) but NOT the signed M1 program gate (M0 venue is a user/PI decision; not the MBE target process)",
                "BO-driven (BayBE) sampling clusters rows near optima -- group sizes are uneven and the far/late tertiles are the sparse edges by construction",
                "per-tertile n ~ n_test/3 (~27-43); groups < 20 are flagged UNDERPOWERED, point estimate reported only with the flag",
                "conditional trials within a tertile still share the two outputs' test rows where pooled; per-output rows are the honest unit and are reported",
                "ti_120w_short_pw temporal order key is unverified file order (BatchNr degenerate) -- its stream-position tertiles inherit that caveat",
            ],
        },
        "campaigns": campaigns_out,
        "headline": headline,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nfidelity gate: PASS on all {fidelity_cells} campaign/split cells")
    print(f"results written -> {out_path}")
    print(f"total wall time : {time.perf_counter() - t_start:.1f} s")
    banner("DONE (conditional coverage on real Empa HiPIMS -- recorded artifact untouched)")
    return 0


# ============================================================================
# console printing
# ============================================================================


def _fmt_cov(g):
    if g["picp"] is None:
        return "   n=0"
    flag = {"under": "UNDER", "over": "over", "ok": "ok"}[g["direction"]]
    up = "*" if g["underpowered"] else " "
    return f"{g['picp']:.3f}[{g['ci95'][0]:.2f},{g['ci95'][1]:.2f}] {flag:<5}{up} n={g['n']}"


def _print_cell(slug, split_name, output_keys, analysis):
    g = analysis["groups"]
    print(
        f"\n-- {slug} / {split_name} (n_test={analysis['n_test']}) --  (* = UNDERPOWERED, n<{MIN_GROUP_N})"
    )
    for key in output_keys:
        print(f"  output: {key}")
        # density: by_path is {path: {key: cov}}
        dens = g["density_near_mid_far"]["tertiles"]
        row = "    " + f"{'density near/mid/far':<24}"
        for lab in DENSITY_LABELS:
            row += f"{lab}:{_fmt_cov(dens[lab]['by_path']['static'][key])}   "
        print(row)
        # magnitude: by_path is {path: cov} (output already fixed by `key`)
        mag = g["magnitude_low_mid_high"][key]["tertiles"]
        row = "    " + f"{'magnitude low/mid/high':<24}"
        for lab in MAGNITUDE_LABELS:
            row += f"{lab}:{_fmt_cov(mag[lab]['by_path']['static'])}   "
        print(row)
        # stream position (temporal only): all three paths, by_path is {path: {key: cov}}
        if g["stream_position_early_mid_late"]["tertiles"] is not None:
            block = g["stream_position_early_mid_late"]["tertiles"]
            for path in ("static", "aci", "pid"):
                row = "    " + f"{'stream ' + path:<24}"
                for lab in STREAM_LABELS:
                    row += f"{lab}:{_fmt_cov(block[lab]['by_path'][path][key])}   "
                print(row)


def _print_headline(h):
    banner("HEADLINE -- is a marginal PASS hiding a regional failure? where? repaired?")
    q1 = h["question_1_hidden_regional_undercoverage"]
    print(f"Q1 hidden regional under-coverage (marginal PASS, powered group UNDER): {q1['count']}")
    for c in q1["cells"]:
        print("   -", c)
    q3 = h["question_3_late_drift_phase_static_undercoverage"]
    print(f"Q3 late drift-phase static under-coverage (temporal): {q3['count']}")
    for c in q3["cells"]:
        print("   -", c)
    q2 = h["question_2_far_from_data_undercoverage"]
    print(f"Q2 far-from-data static under-coverage: {q2['count']}")
    for c in q2["cells"]:
        print("   -", c)
    q4 = h["question_4_online_repairs_of_static_regional_failures"]
    print(f"Q4 online (ACI/PID) repairs of static regional failures: {q4['count']}")
    for c in q4["cells"]:
        print("   -", c)


if __name__ == "__main__":
    raise SystemExit(main())
