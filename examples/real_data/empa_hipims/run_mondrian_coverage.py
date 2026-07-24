"""MONDRIAN (group-conditional) conformal on REAL Empa HiPIMS data -- does grouping
by PREDICTED magnitude move the high-outcome-tail under-coverage back to nominal?

The recorded static split-conformal gate (run_m1_empa.py -> results/m1_empa.json)
PASSES marginally per campaign/split, but the conditional-coverage study
(run_conditional_coverage.py -> results/m1_empa_conditional.json) showed that
POOLED PASS hides HIGH-OBSERVED-MAGNITUDE-TERTILE under-coverage: 8 of the 24
high-tertile (campaign x split x output) cells under-cover, and ACI/PID cannot fix
it (they adapt over TIME, not MAGNITUDE). This study evaluates a
MondrianConformalCalibrator that takes a SEPARATE conformal quantile per
predicted-magnitude tertile against that exact failure.

--------------------------------------------------------------------------------
PRE-STATED QUESTION (fixed BEFORE any coverage number is computed):

  Grouping the split-conformal quantile by the tertile of the PREDICTED MEAN
  (tertile edges frozen from the CALIBRATION slice's predicted means -- leakage
  free), does the HIGH-OBSERVED-MAGNITUDE-tertile conditional coverage move to
  NOMINAL (the 8 failing cells from m1_empa_conditional.json) -- and

    (a) WITHOUT breaking the marginal pooled PASS (marginal Mondrian PICP must
        still hold the gate's directional CI -- nominal 0.90 inside the exact
        binomial 95% CI, the SAME rule the recorded gate uses); and
    (b) at what BAND-WIDTH cost (Mondrian will widen the high group -- MPIW per
        observed tertile, static vs Mondrian, is reported).

HONEST DESIGN NOTE, stated up front: the Mondrian GROUP is assigned on the
PREDICTED mean (it must be -- the true outcome does not exist at predict time),
but the EVALUATION tertile is defined on the OBSERVED outcome (the same masks the
conditional study used, so the 8 failing cells are identical). Grouping-by-
predicted vs evaluating-by-observed is leakage-free but creates IMPERFECT group
alignment: a point whose observed magnitude is high but predicted mid is
calibrated as a mid point. We MEASURE that misalignment (assignment agreement
rate, per output) and report it -- Mondrian tightens the tail toward nominal, it
does not deliver oracle per-true-magnitude coverage.

--------------------------------------------------------------------------------
FIDELITY: the STATIC baseline reproduced here (fresh SplitConformalCalibrator +
ConformalForwardModel on the imported runner's identical seeded GP fit and splits)
has its per-output AND pooled k_covered asserted byte-equal to the recorded
results/m1_empa.json before any Mondrian number is trusted. The Mondrian path
reuses the SAME fitted GP and the SAME calibration slice, so only the conformal
quantile differs. All grouping / tertile / coverage machinery is IMPORTED from
run_conditional_coverage.py (tertile_labels, group_coverage, MAGNITUDE_LABELS,
MIN_GROUP_N, campaign_arrays, reconstruct_splits, static_indicators) -- not forked.

Outputs: results/m1_empa_mondrian.json (NEW file; the recorded artifacts are never
touched). Deterministic: no RNG of its own; the reused fits are seeded; a second
run is byte-identical.

Run (Windows cp1252 console -> force UTF-8):

    PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_mondrian_coverage.py
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

from rig.calibration.mondrian import (
    MondrianConformalCalibrator,
    MondrianConformalForwardModel,
    predicted_magnitude_group_fn,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
BASELINE_JSON = RESULTS_DIR / "m1_empa.json"
CONDITIONAL_JSON = RESULTS_DIR / "m1_empa_conditional.json"


def _load_by_path(name: str, filename: str):
    """Import an examples/ script by PATH (examples/ is not a package), retrying a
    transient mid-edit import a few times (a concurrent agent may be editing a
    sibling). Mirrors run_conditional_coverage's own by-path loader."""
    last: Exception | None = None
    for _ in range(5):
        try:
            spec = importlib.util.spec_from_file_location(name, HERE / filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
        except Exception as exc:  # noqa: BLE001 -- transient mid-edit import, retry
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"could not import {filename} (concurrent edit?): {last!r}")


# run_conditional_coverage imports run_m1_empa itself; reuse ITS primitives so the
# protocol has a single source of truth and cannot fork.
cond = _load_by_path("run_conditional_coverage", "run_conditional_coverage.py")
runner = cond.runner

# reused verbatim from the runner / conditional study (single source of truth):
SEED = runner.SEED
ALPHA = runner.ALPHA
NOMINAL = runner.NOMINAL
CI_LEVEL = runner.CI_LEVEL
CAMPAIGNS = runner.CAMPAIGNS
binom_ci = runner.binom_ci
MAGNITUDE_LABELS = cond.MAGNITUDE_LABELS
MIN_GROUP_N = cond.MIN_GROUP_N
tertile_labels = cond.tertile_labels
group_coverage = cond.group_coverage
campaign_arrays = cond.campaign_arrays
reconstruct_splits = cond.reconstruct_splits


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ============================================================================
# static + Mondrian bands on one fitted model
# ============================================================================


def _band_hits_widths(cset, Yt):
    """(hits (n,m) bool, widths (n,m)) from a conformal band cset (n,m,2)."""
    cset = np.asarray(cset, dtype=float)
    lo, hi = cset[..., 0], cset[..., 1]
    return (Yt >= lo) & (Yt <= hi), (hi - lo)


def static_band(model, Xc, Yc, Xt, Yt):
    """Static split-conformal band -- the recorded baseline. Fresh
    SplitConformalCalibrator + ConformalForwardModel, identical to fit_and_eval /
    run_conditional_coverage.static_indicators (byte-equal by the fidelity gate)."""
    cal = runner.SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    conf = runner.ConformalForwardModel(model, cal)
    cset = np.asarray(conf.predict(Xt).conformal_set)  # (n_test, m, 2)
    return _band_hits_widths(cset, Yt)


def mondrian_band(model, Xc, Yc, Xt, Yt, output_keys):
    """Mondrian band grouped by PREDICTED-magnitude tertile.

    Tertile edges are frozen from the CALIBRATION slice's PREDICTED means (per
    output; leakage-free -- never the test outcome). Returns (hits, widths,
    predicted-group-labels on the test slice, per-output edges, per-output group
    sizes on the calibration slice)."""
    mu_cal = np.atleast_2d(np.asarray(model.predict(Xc).mean, dtype=float))  # (n_cal, m)
    m = mu_cal.shape[1]
    edges = np.empty((m, 2))
    for j in range(m):
        # reuse the conditional study's tertile edge logic on the PREDICTED means.
        _, (e_lo, e_hi) = tertile_labels(mu_cal[:, j], MAGNITUDE_LABELS)
        edges[j] = (e_lo, e_hi)
    gf = predicted_magnitude_group_fn(edges, MAGNITUDE_LABELS)
    cal = MondrianConformalCalibrator(gf, alpha=ALPHA, min_group_n=MIN_GROUP_N)
    cal.fit(model, Xc, Yc)
    wrap = MondrianConformalForwardModel(model, cal)
    cset = np.asarray(wrap.predict(Xt).conformal_set)  # (n_test, m, 2)
    hits, widths = _band_hits_widths(cset, Yt)
    # predicted-group labels on the TEST slice (for the assignment-agreement rate)
    mu_test = np.atleast_2d(np.asarray(model.predict(Xt).mean, dtype=float))
    pred_groups = np.atleast_2d(gf(Xt, mu_test))  # (n_test, m) object
    cal_group_sizes = {
        key: {lab: int((cal.groups_[:, j] == lab).sum()) for lab in MAGNITUDE_LABELS}
        for j, key in enumerate(output_keys)
    }
    return hits, widths, pred_groups, edges, cal_group_sizes


# ============================================================================
# fidelity gate: reproduced static coverage == recorded results/m1_empa.json
# ============================================================================


def static_fidelity(slug, split_name, hits_static, output_keys, recorded_split):
    """Assert the reproduced static per-output + pooled k_covered equals the
    recorded numbers; RAISE on any mismatch (never carry divergent replication)."""
    per_output = {}
    for j, key in enumerate(output_keys):
        repro_k = int(hits_static[:, j].sum())
        rec_k = int(recorded_split["per_output"][key]["k_covered"])
        rec_n = int(recorded_split["per_output"][key]["n_test"])
        if hits_static.shape[0] != rec_n or repro_k != rec_k:
            raise RuntimeError(
                f"FIDELITY FAIL {slug}/{split_name}/static/{key}: reproduced "
                f"(k={repro_k}, n={hits_static.shape[0]}) != recorded (k={rec_k}, n={rec_n}) "
                "-- static replication diverged, STOP"
            )
        per_output[key] = {"k_covered": repro_k, "n_test": rec_n, "matches_recorded": True}
    repro_pool = int(hits_static.sum())
    rec_pool = int(recorded_split["pooled"]["k_covered"])
    if repro_pool != rec_pool:
        raise RuntimeError(
            f"FIDELITY FAIL {slug}/{split_name}/static pooled k_covered "
            f"{repro_pool} != recorded {rec_pool} -- STOP"
        )
    return {"per_output": per_output, "pooled": {"k_covered": repro_pool, "matches_recorded": True}}


# ============================================================================
# per-cell before/after within OBSERVED-magnitude tertiles
# ============================================================================


def mpiw_in_mask(widths_col, mask):
    finite = np.isfinite(widths_col) & mask
    if not finite.any():
        return None
    return float(widths_col[finite].mean())


def magnitude_analysis(hits_s, w_s, hits_m, w_m, pred_groups, Yt, output_keys):
    """For each output, tertile the OBSERVED test outcomes (the SAME masks the
    conditional study uses), then report static-vs-Mondrian coverage + MPIW per
    tertile and the predicted/observed assignment-agreement rate."""
    out: dict[str, dict] = {}
    for j, key in enumerate(output_keys):
        obs_labels, obs_edges = tertile_labels(Yt[:, j], MAGNITUDE_LABELS)
        # assignment agreement: predicted tertile == observed tertile, this output.
        agree = float(np.mean(pred_groups[:, j] == obs_labels))
        tertiles = {}
        for lab in MAGNITUDE_LABELS:
            mask = obs_labels == lab
            cov_s = group_coverage(hits_s[:, j], mask)
            cov_m = group_coverage(hits_m[:, j], mask)
            tertiles[lab] = {
                "n": int(mask.sum()),
                "static": cov_s,
                "mondrian": cov_m,
                "mpiw_static": mpiw_in_mask(w_s[:, j], mask),
                "mpiw_mondrian": mpiw_in_mask(w_m[:, j], mask),
                # did a static UNDER-cover move to nominal (OK/over) under Mondrian?
                "moved_to_nominal": bool(
                    cov_s["picp"] is not None
                    and cov_m["picp"] is not None
                    and not cov_s["underpowered"]
                    and cov_s["direction"] == "under"
                    and cov_m["direction"] != "under"
                ),
            }
        out[key] = {
            "observed_tertile_edges_lo_hi": [float(e) for e in obs_edges],
            "predicted_observed_assignment_agreement": agree,
            "tertiles": tertiles,
        }
    return out


def marginal_pooled(hits, output_keys):
    """Pooled + per-output marginal coverage with the gate's directional CI, for
    the given hit matrix (checks Mondrian does not break the marginal PASS)."""
    per_output = {}
    for j, key in enumerate(output_keys):
        k = int(hits[:, j].sum())
        n = hits.shape[0]
        lo, hi = binom_ci(k, n)
        per_output[key] = {
            "picp": k / n,
            "k_covered": k,
            "n_test": n,
            "ci95": [lo, hi],
            "nominal_in_ci": bool(lo <= NOMINAL <= hi),
        }
    k_pool, n_pool = int(hits.sum()), int(hits.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    return {
        "per_output": per_output,
        "pooled": {
            "picp": k_pool / n_pool,
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
    }


# ============================================================================
# console printing
# ============================================================================


def _fmt(g):
    if g["picp"] is None:
        return "   n=0"
    flag = {"under": "UNDER", "over": "over", "ok": "ok"}[g["direction"]]
    up = "*" if g["underpowered"] else " "
    return f"{g['picp']:.3f}[{g['ci95'][0]:.2f},{g['ci95'][1]:.2f}]{flag:<5}{up}"


def print_cell(slug, split_name, analysis, marg_s, marg_m, output_keys):
    print(f"\n-- {slug} / {split_name} --  (* = UNDERPOWERED, n<{MIN_GROUP_N})")
    for key in output_keys:
        a = analysis[key]
        print(
            f"  output {key}  (assign-agreement predicted/observed = "
            f"{a['predicted_observed_assignment_agreement']:.2f})"
        )
        for lab in MAGNITUDE_LABELS:
            t = a["tertiles"][lab]
            ws = "n/a" if t["mpiw_static"] is None else f"{t['mpiw_static']:.4f}"
            wm = "n/a" if t["mpiw_mondrian"] is None else f"{t['mpiw_mondrian']:.4f}"
            moved = "  -> MOVED to nominal" if t["moved_to_nominal"] else ""
            print(
                f"    {lab:<5} n={t['n']:<3} static {_fmt(t['static'])}  "
                f"mondrian {_fmt(t['mondrian'])}  MPIW {ws}->{wm}{moved}"
            )
    ms, mm = marg_s["pooled"], marg_m["pooled"]
    print(
        f"  MARGINAL pooled: static {ms['picp']:.3f} {'PASS' if ms['nominal_in_ci'] else 'FAIL'}"
        f" -> mondrian {mm['picp']:.3f} {'PASS' if mm['nominal_in_ci'] else 'FAIL'}"
    )


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
    gp_restarts = int(baseline["meta"]["gp_restarts"])

    np.random.seed(SEED)  # mirror the runner (fit RNG is local; harmless)
    out_path = args.out or (
        RESULTS_DIR / f"m1_empa_mondrian{('.' + args.campaign) if args.campaign else ''}.json"
    )

    banner("MONDRIAN (group-conditional) conformal on REAL Empa HiPIMS data")
    print(
        f"baseline : {BASELINE_JSON.name}  (gp_restarts={gp_restarts}, seed={SEED}, alpha={ALPHA})"
    )
    print("grouping : PREDICTED-mean magnitude tertile (edges frozen from the CAL slice)")
    print("eval     : OBSERVED-outcome magnitude tertile (same masks as the conditional study)")
    print(f"fallback : groups with < {MIN_GROUP_N} cal points borrow the POOLED quantile")
    print("question : do the 8 failing high-observed-tertile cells move to nominal, at what width?")

    t_start = time.perf_counter()
    campaigns_out: dict[str, dict] = {}
    fidelity_cells = 0
    moved = []  # (slug/split/output) high-tertile static-under cells that moved to nominal
    still_under = []  # high-tertile static-under cells that did NOT move
    broke_marginal = []  # per-output cells where Mondrian broke a marginal static PASS
    broke_marginal_pooled = []  # POOLED cells where Mondrian broke the marginal PASS (sub-q a)

    for campaign_index, campaign in enumerate(CAMPAIGNS):
        if args.campaign not in (None, campaign.slug):
            continue
        if campaign.slug not in baseline["campaigns"]:
            continue
        X, Y, input_keys, output_keys, units, n, degenerate = campaign_arrays(campaign)
        rec_campaign = baseline["campaigns"][campaign.slug]
        n_train, n_cal = runner.split_indices(n)
        recon_sizes = {"train": n_train, "cal": n_cal, "test": n - n_train - n_cal}
        if recon_sizes != rec_campaign["split_sizes"]:
            raise RuntimeError(
                f"{campaign.slug}: reconstructed split sizes {recon_sizes} != recorded "
                f"{rec_campaign['split_sizes']}"
            )
        splits = reconstruct_splits(n, campaign_index)

        banner(f"CAMPAIGN {campaign.slug}  ({rec_campaign['process_id']})  n={n}")
        campaign_out = {
            "process_id": rec_campaign["process_id"],
            "material": rec_campaign["material"],
            "parameterization": rec_campaign["parameterization"],
            "split_sizes": recon_sizes,
            "splits": {},
        }
        for split_name, (fit_idx, cal_idx, test_idx) in splits.items():
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
            Xc, Yc = X[cal_idx], Y[cal_idx]
            Xt, Yt = X[test_idx], Y[test_idx]

            hits_s, w_s = static_band(model, Xc, Yc, Xt, Yt)
            fid = static_fidelity(
                campaign.slug, split_name, hits_s, output_keys, rec_campaign["splits"][split_name]
            )
            fidelity_cells += 1

            hits_m, w_m, pred_groups, edges, cal_group_sizes = mondrian_band(
                model, Xc, Yc, Xt, Yt, output_keys
            )
            analysis = magnitude_analysis(hits_s, w_s, hits_m, w_m, pred_groups, Yt, output_keys)
            marg_s = marginal_pooled(hits_s, output_keys)
            marg_m = marginal_pooled(hits_m, output_keys)

            # headline bookkeeping over the HIGH observed tertile (the 8 failing cells)
            for key in output_keys:
                hi = analysis[key]["tertiles"]["high"]
                s, mo = hi["static"], hi["mondrian"]
                tag = f"{campaign.slug}/{split_name}/{key}/high"
                if s["picp"] is not None and not s["underpowered"] and s["direction"] == "under":
                    if mo["direction"] != "under":
                        moved.append(tag)
                    else:
                        still_under.append(tag)
                # did Mondrian break a marginal per-output PASS? note the DIRECTION
                # (over-coverage = the safe/conservative break; under = unsafe).
                ps, pm = marg_s["per_output"][key], marg_m["per_output"][key]
                if ps["nominal_in_ci"] and not pm["nominal_in_ci"]:
                    direction = "over" if pm["picp"] > NOMINAL else "under"
                    broke_marginal.append(f"{campaign.slug}/{split_name}/{key} ({direction})")

            # did Mondrian break the marginal POOLED PASS (sub-question a)?
            ps_pool, pm_pool = marg_s["pooled"], marg_m["pooled"]
            if ps_pool["nominal_in_ci"] and not pm_pool["nominal_in_ci"]:
                direction = "over" if pm_pool["picp"] > NOMINAL else "under"
                broke_marginal_pooled.append(
                    f"{campaign.slug}/{split_name} "
                    f"({ps_pool['picp']:.3f}->{pm_pool['picp']:.3f}, {direction})"
                )

            campaign_out["splits"][split_name] = {
                "fidelity_static": fid,
                "mondrian_group_edges_predicted": {
                    key: [float(e) for e in edges[j]] for j, key in enumerate(output_keys)
                },
                "mondrian_cal_group_sizes": cal_group_sizes,
                "marginal": {"static": marg_s, "mondrian": marg_m},
                "magnitude": analysis,
            }
            print_cell(campaign.slug, split_name, analysis, marg_s, marg_m, output_keys)
        campaigns_out[campaign.slug] = campaign_out

    headline = {
        "high_tertile_static_undercover_cells_total": len(moved) + len(still_under),
        "moved_to_nominal_count": len(moved),
        "moved_to_nominal_cells": moved,
        "still_under_after_mondrian_count": len(still_under),
        "still_under_cells": still_under,
        "mondrian_broke_marginal_pooled_pass_count": len(broke_marginal_pooled),
        "mondrian_broke_marginal_pooled_cells": broke_marginal_pooled,
        "mondrian_broke_marginal_per_output_pass_count": len(broke_marginal),
        "mondrian_broke_marginal_per_output_cells": broke_marginal,
    }
    _print_headline(headline)

    payload = {
        "meta": {
            "seed": SEED,
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL,
            "ci": f"exact (Clopper-Pearson) binomial, level {CI_LEVEL}",
            "gp_restarts": gp_restarts,
            "baseline": BASELINE_JSON.name,
            "min_group_n_fallback_to_pooled": MIN_GROUP_N,
            "grouping": "PREDICTED-mean magnitude tertile; edges frozen from the calibration slice (leakage-free)",
            "evaluation": "OBSERVED-outcome magnitude tertile (the same masks as m1_empa_conditional.json)",
            "directional_flag": "nominal 0.90 inside the exact binomial 95% CI (same rule as the recorded gate)",
            "fidelity_gate": {
                "cells_checked": fidelity_cells,
                "status": "PASS -- reproduced STATIC per-output AND pooled k_covered byte-equal to results/m1_empa.json",
            },
            "caveats": [
                "READ-ONLY re-analysis: results/m1_empa.json and m1_empa_conditional.json are never modified",
                "REAL Empa tool data (real_tool) but NOT the signed M1 program gate (M0 venue is a user/PI decision)",
                "group assigned on the PREDICTED mean; tertile evaluated on the OBSERVED outcome -- leakage-free but imperfect alignment (assignment-agreement rate reported per cell)",
                "Mondrian gives coverage conditional on the PREDICTED-magnitude group, NOT oracle per-true-magnitude nor per-point coverage",
                "per-tertile n ~ n_test/3 (~27-43); groups < 20 flagged UNDERPOWERED, point estimate reported only with the flag",
                "BO-driven (BayBE) sampling clusters rows near optima -- the high tertile is the sparse, drifted edge by construction",
            ],
        },
        "campaigns": campaigns_out,
        "headline": headline,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nfidelity gate (static): PASS on all {fidelity_cells} campaign/split cells")
    print(f"results written -> {out_path}")
    print(f"total wall time : {time.perf_counter() - t_start:.1f} s")
    banner("DONE (Mondrian group-conditional coverage on real Empa HiPIMS -- artifacts untouched)")
    return 0


def _print_headline(h):
    banner("HEADLINE -- did Mondrian move the high-tail failures to nominal?")
    print(
        f"high-observed-tertile static UNDER-cover cells: {h['high_tertile_static_undercover_cells_total']}"
    )
    print(f"  moved to nominal under Mondrian : {h['moved_to_nominal_count']}")
    for c in h["moved_to_nominal_cells"]:
        print("     +", c)
    print(f"  still under after Mondrian      : {h['still_under_after_mondrian_count']}")
    for c in h["still_under_cells"]:
        print("     -", c)
    print(
        "Mondrian broke the marginal POOLED PASS in: "
        f"{h['mondrian_broke_marginal_pooled_pass_count']} cell(s)  (over = safe/conservative)"
    )
    for c in h["mondrian_broke_marginal_pooled_cells"]:
        print("     !", c)
    print(
        "Mondrian broke a marginal per-output PASS in: "
        f"{h['mondrian_broke_marginal_per_output_pass_count']} cell(s)"
    )
    for c in h["mondrian_broke_marginal_per_output_cells"]:
        print("     !", c)


if __name__ == "__main__":
    raise SystemExit(main())
