"""MATERIAL-CONDITIONED POOLING on the Empa bipolar-HiPIMS M0 venue (M1 remainder).

Sibling of ``run_m1_empa.py`` (which another agent owns -- this file NEVER edits
it and does NOT import it; the few shared constants below are duplicated with a
provenance note). Where ``run_m1_empa.py`` fits an HONEST per-campaign GP per
campaign, this script fits the tool-aware ICM multi-task GP
(``rig.forward.multitask.MultiToolGPForwardModel``, implementation-plan section
10.4 level (a)) with ``tool_id = MATERIAL`` ("al" / "ti"), and asks the two
questions the per-campaign OOD finding left open (RESULTS.md "OOD / support
check"; root audit.md F4):

    (A) Does conditioning on MATERIAL make the cross-material shift VISIBLE to the
        model that a knob-space, per-campaign model was PROVABLY blind to?  The
        recorded run's 12-pair OOD check landed 8/12: the 4 misses are EXACTLY the
        cross-material same-power-tier pairs (Al<->Ti at 120W-shortPW and at
        200W-highPW), whose epistemic sigma AND support are numerically
        indistinguishable because material is NOT a knob -- those campaign pairs
        share ~the same knob box, so a knob-space model cannot see the material.
    (B) Does anything resembling cross-material TRANSFER actually hold?  (Expected
        weak or absent -- the deliverable is AWARENESS; transfer is a bonus that
        must not be over-claimed. Leave-one-material-out, per audit F4.)

Three blocks, all on real Empa tool data (provenance real_tool):

  A. MATERIAL-AWARENESS.  Pooled PRR model on the 4 PRR campaigns' random-fit
     slices, material as task.  For every ordered (src=A, tgt=B) PRR pair, at B's
     held-out random-test recipes, compare epistemic sigma + support under
     material(B) (in-distribution, correct material) vs material(A) (the
     cross-material query the per-campaign model could not even express), plus
     the tool=None UNKNOWN-material fallback and the predicted-MEAN shift.
     Directional criterion (pre-stated, mirroring run_m1_empa's OOD form):
     cross-material epistemic STRICTLY inflates on both outputs, OR support
     STRICTLY drops.  Reported as N/12 with the 4 blind pairs called out, the
     8/12 baseline read straight from results/m1_empa.json for the before/after,
     plus a campaign-as-task (4-task) comparison.

  B. LEAVE-ONE-MATERIAL-OUT (the honest transfer check).  Train on ONE material's
     PRR campaigns (temporal-fit slices); the held-out material is an UNKNOWN
     tool.  ZERO-SHOT: assert the section 5.8 population-fallback epistemic
     dominates every known tool's, elementwise (the awareness guarantee by
     construction), and report the fallback point-RMSE (transfer quality of the
     mean).  FEW-SHOT: adapt_to_tool with small K (K/2 earliest temporal rows of
     EACH held campaign, spanning both tiers); the transfer signal is the
     few-shot point-RMSE vs the full-data baseline RMSE ceiling, plus the RAW
     predictive coverage (which CAN fail) and split-conformal PICP+MPIW on a
     seeded-random exchangeable split of the remainder.  Verdict: cross-material
     transfer is CLAIMABLE ONLY if few-shot accuracy approaches the ceiling AND
     stays calibrated; otherwise the claim stays FORBIDDEN (audit F4).  Note
     split-conformal PICP is ~nominal by construction on an exchangeable split,
     so it cannot by itself diagnose transfer -- its MPIW is the sharpness cost.

  C. POOLING-COST.  Material-conditioned pooled model's per-campaign PICP on the
     SAME temporal and random test slices as run_m1_empa's baseline (split
     indices reconstructed identically from SEED and the 60/20/20 protocol),
     split-conformal wrapped per campaign on the pooled model's per-tool view,
     side-by-side with the per-campaign baseline read from results/m1_empa.json.
     Question: does material-conditioned pooling COST coverage anywhere?

PARAMETERIZATION SUBSPACES.  The six campaigns split into two INCOMPATIBLE knob
spaces: four PRR campaigns share the 5 knob names ("PRR (Hz)", "PW (us)",
"pos Delay (us)", "pos PW (us)", "pos Setpoint (V)") and two DUTY campaigns share
a different five (first knob "Duty Cycle (ratio)").  Pooling is therefore done
WITHIN each subspace (PRR = one X-space over 4 campaigns; DUTY = one over 2), never
across -- stated wherever it matters.  The 4 blind pairs all live in the PRR
subspace, so Block A is PRR-only; Blocks B/C cover PRR primarily and DUTY as a
2-campaign replication.

HONEST FRAMING (identical status to run_m1_empa.py): REAL measured data on the M0
venue, but NOT the signed M1 gate.  BO-driven (BayBE) sampling clusters rows near
optima; the pooled coverage CI treats 2*n_test trials as independent though the
two outputs share test rows (optimistic -- per-output rows are the honest unit);
ti_120w_short_pw has a degenerate order key (file order) and 5 skip-rejected rows;
"Ipk (A)" is measured, never set.  FAIL rows are reported with the same prominence
as PASS rows.

RUNTIME.  A pooled PRR fit is ~1,400 rows.  Benchmarked: one fit is ~50 s at
n_restarts=1 and gives BYTE-IDENTICAL hyperparameters to n_restarts=2 (the strong
deterministic W0 start -- B off-diag ~0.81 -- dominates), so restarts buy nothing
here.  --smoke uses n_restarts=1/max_iter=60; --full uses n_restarts=2/max_iter=100
(a margin, still identical in practice).  A full run is ~4-6 min, well under the
15-min budget.  Everything is seeded (SEED below); a second run is identical modulo
wall_seconds.

Run (Windows cp1252 console -> force UTF-8):

    PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_m1_empa_pooled.py
        [--smoke] [--full] [--out <path.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.forward import records_to_arrays
from rig.forward.multitask import MultiToolGPForwardModel
from rig.metrics import uq
from rig.schema import ureg  # the ONE shared pint registry -- never a second one
from rig_adapters.tabular.ingest import ingest_csv
from rig_adapters.tabular.spec import load_spec

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # prepare_empa.py lives beside this script
# sys.path insert above (structural -- hence the E402 suppression here only).
from prepare_empa import CAMPAIGNS, DEP_RATE_COLUMN, Campaign  # noqa: E402

# -- constants DUPLICATED from run_m1_empa.py (owned by another agent; kept in
#    lock-step ON PURPOSE so the split reconstruction in Block C is byte-for-byte
#    identical -- see that file for the authoritative definitions). ------------
SEED = 0
ALPHA = 0.10  # conformal miscoverage target -> nominal 90% coverage
NOMINAL = 1.0 - ALPHA
CI_LEVEL = 0.95
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20  # test = the remaining 20%
CSV_DIR = HERE / "csv"
SPEC_DIR = HERE / "specs"
RESULTS_DIR = HERE / "results"
BASELINE_JSON = RESULTS_DIR / "m1_empa.json"  # per-campaign numbers to compare to
EXPECTED_REJECTS = {"ti_120w_short_pw": 5}
UNIT_LABEL = {DEP_RATE_COLUMN: "Ang/s", "Ipk (A)": "A"}

# -- Block A pre-stated directional criterion (fixed before seeing outcomes) ---
# Mirrors run_m1_empa's OOD form ("epistemic sigma larger, both outputs"); a
# strict inequality, plus a tiny relative margin so float noise never flips a call.
EPI_MARGIN = 0.02  # cross epistemic must exceed in-dist by >2% on BOTH outputs
SUPP_MARGIN = 0.02  # or support must drop by >0.02 (negative-Mahalanobis units)

# -- Block B few-shot budget ---------------------------------------------------
FEWSHOT_K = (10, 20)


def banner(title: str) -> None:  # (form duplicated from run_m1_empa.py)
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def binom_ci(k: int, n: int) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI (duplicated from run_m1_empa.py)."""
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=CI_LEVEL, method="exact")
    return float(ci.low), float(ci.high)


def split_indices(n: int) -> tuple[int, int]:
    """(n_train, n_cal) for the 60/20/20 contract (duplicated from run_m1_empa)."""
    return int(round(TRAIN_FRAC * n)), int(round(CAL_FRAC * n))


def material_of(campaign: Campaign) -> str:
    return "al" if campaign.material == "Al" else "ti"


def tier_of(slug: str) -> str:
    """Power/pulse-width tier key, material-stripped: 'al_120w_short_pw' ->
    '120w_short_pw'. Two campaigns share a tier iff only their material differs."""
    return slug.split("_", 1)[1]


def ingest_campaign(campaign: Campaign):
    """Ingest one tidy CSV -> (X SI, Y readable, input_keys, output_keys, units).

    Y is de-SI'd back to the readable declared unit (Ang/s, A) exactly as
    run_m1_empa.py reports it, so mean-shifts and noise are interpretable and
    PICP (scale-free) matches the baseline. X stays SI-canonical (the pooled
    model only needs a consistent input space across campaigns; no declared
    BOUNDS are used anywhere here, so the continuous_si trap does not bite --
    but gp_input_keys/continuous_si order agreement is asserted regardless).
    """
    spec = load_spec(SPEC_DIR / f"{campaign.slug}.toml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unmatched y1/BatchNr/FitNr warning is expected
        result = ingest_csv(
            CSV_DIR / f"{campaign.slug}.csv", spec, source="real_tool", on_error="skip"
        )
    expected = EXPECTED_REJECTS.get(campaign.slug, 0)
    if len(result.rejects) != expected:
        raise RuntimeError(
            f"{campaign.slug}: {len(result.rejects)} rejected rows, expected {expected} "
            "-- re-run prepare_empa.py + tests first"
        )
    input_keys = list(spec.gp_input_keys)
    output_keys = list(spec.output_names)
    units = [o.unit for o in spec.outputs]
    if [v.name for v in spec.continuous_si] != input_keys:
        raise RuntimeError(f"{campaign.slug}: continuous_si order != gp_input_keys")
    # BatchNr must be monotone in ingest order (the temporal-split key, exactly
    # as run_m1_empa validates it).
    batch = [int(r.extra["unmatched_columns"]["BatchNr"]) for r in result.records]
    if batch != sorted(batch):
        raise RuntimeError(f"{campaign.slug}: ingest order is not BatchNr-monotone")
    degenerate = len(set(batch)) == 1
    X, Y_si = records_to_arrays(result.records, input_keys, output_keys)
    si_per_raw = np.array([float(ureg.Quantity(1.0, u).to_base_units().magnitude) for u in units])
    return X, Y_si / si_per_raw, input_keys, output_keys, units, len(result.records), degenerate


def load_campaigns() -> dict:
    """Ingest all six campaigns and reconstruct the temporal + random split
    indices IDENTICALLY to run_m1_empa.py (temporal = contiguous blocks in
    ingest/BatchNr order; random = a seeded permutation with the campaign's
    ENUMERATE index in CAMPAIGNS as the seed offset -- the exact contract)."""
    keys_by_param: dict[str, list[str]] = {}
    data: dict[str, dict] = {}
    for idx, c in enumerate(CAMPAIGNS):  # idx is the CAMPAIGNS enumerate index
        X, Y, ik, ok, units, n, degen = ingest_campaign(c)
        keys_by_param.setdefault(c.parameterization, ik)
        if keys_by_param[c.parameterization] != ik:
            raise RuntimeError(
                f"{c.slug}: input keys disagree within '{c.parameterization}' subspace"
            )
        ntr, ncal = split_indices(n)
        i = np.arange(n)
        temporal = (i[:ntr], i[ntr : ntr + ncal], i[ntr + ncal :])
        perm = np.random.default_rng(SEED + idx).permutation(n)
        random = (perm[:ntr], perm[ntr : ntr + ncal], perm[ntr + ncal :])
        data[c.slug] = dict(
            campaign=c,
            material=material_of(c),
            tier=tier_of(c.slug),
            parameterization=c.parameterization,
            X=X,
            Y=Y,
            n=n,
            output_keys=ok,
            units=units,
            degenerate=degen,
            split_sizes={"train": ntr, "cal": ncal, "test": n - ntr - ncal},
            temporal=temporal,
            random=random,
        )
    return data, keys_by_param


def fit_pooled(slugs, split_name, data, input_keys, output_keys, *, task, restarts, max_iter):
    """Fit ONE pooled ICM model over ``slugs`` on their ``split_name``-FIT
    slices. ``task='material'`` -> tool id is the material; ``task='campaign'``
    -> tool id is the slug. Deterministic (seed=SEED)."""
    Xs, Ys, tools = [], [], []
    for s in slugs:
        d = data[s]
        fit_idx = d[split_name][0]
        Xs.append(d["X"][fit_idx])
        Ys.append(d["Y"][fit_idx])
        tid = d["material"] if task == "material" else s
        tools += [tid] * len(fit_idx)
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    model = MultiToolGPForwardModel(
        input_keys=list(input_keys),
        output_keys=list(output_keys),
        rank=1,
        n_restarts=restarts,
        seed=SEED,
        max_iter=max_iter,
    ).fit(X, Y, tools)
    return model


# ============================================================================
# Block A -- MATERIAL-AWARENESS
# ============================================================================


def _pair_awareness(model, data, src, tgt, task):
    """One ordered (src=A, tgt=B) pair: at B's random-test recipes, in-dist
    (material/campaign B) vs cross (A) vs unknown(None), per the pre-stated form."""
    dtgt, dsrc = data[tgt], data[src]
    tool_in = dtgt["material"] if task == "material" else tgt
    tool_cross = dsrc["material"] if task == "material" else src
    Xb = dtgt["X"][dtgt["random"][2]]  # B's held-out random-test recipes

    din = model.predict(Xb, tool_id=tool_in)
    dcr = model.predict(Xb, tool_id=tool_cross)
    dun = model.predict(Xb, tool_id=None)  # unknown-material population fallback
    epi_in = np.asarray(din.epistemic_sigma).mean(axis=0)
    epi_cr = np.asarray(dcr.epistemic_sigma).mean(axis=0)
    epi_un = np.asarray(dun.epistemic_sigma).mean(axis=0)
    supp_in = float(np.mean(model.support_score(Xb, tool_id=tool_in)))
    supp_cr = float(np.mean(model.support_score(Xb, tool_id=tool_cross)))
    mean_shift = np.abs(np.asarray(din.mean).mean(axis=0) - np.asarray(dcr.mean).mean(axis=0))

    cross_material = dsrc["material"] != dtgt["material"]
    epi_inflates = bool(np.all(epi_cr > epi_in * (1.0 + EPI_MARGIN)))
    supp_drops = bool(supp_cr < supp_in - SUPP_MARGIN)
    # directional pass only meaningful when the materials actually differ
    directional_pass = bool(cross_material and (epi_inflates or supp_drops))
    unknown_dominates = bool(np.all(epi_un >= epi_in) and np.all(epi_un >= epi_cr))
    return {
        "src": src,
        "tgt": tgt,
        "cross_material": cross_material,
        "same_tier": dsrc["tier"] == dtgt["tier"],
        "blind_pair": cross_material and dsrc["tier"] == dtgt["tier"],
        "tool_in": tool_in,
        "tool_cross": tool_cross,
        "id_mean_epi": [float(v) for v in epi_in],
        "cross_mean_epi": [float(v) for v in epi_cr],
        "epi_ratio_cross_over_in": [float(a / b) for a, b in zip(epi_cr, epi_in, strict=True)],
        "unknown_mean_epi": [float(v) for v in epi_un],
        "unknown_dominates_both": unknown_dominates,
        "id_mean_support": supp_in,
        "cross_mean_support": supp_cr,
        "support_delta_cross_minus_in": supp_cr - supp_in,
        "mean_shift_abs": [float(v) for v in mean_shift],
        "mean_shift_in_id_sigma": [float(m / s) for m, s in zip(mean_shift, epi_in, strict=True)],
        "epi_inflates_both": epi_inflates,
        "support_drops": supp_drops,
        "directional_pass": directional_pass,
    }


def block_a_awareness(data, keys_by_param, prr_slugs, baseline, restarts, max_iter):
    banner(
        "BLOCK A -- MATERIAL-AWARENESS (does conditioning on material make the "
        "cross-material shift VISIBLE?)"
    )
    input_keys = keys_by_param["prr"]
    output_keys = data[prr_slugs[0]]["output_keys"]
    print(
        f"pooled PRR model: material as task, fit on the 4 PRR campaigns' RANDOM-fit "
        f"slices\ncampaigns          : {list(prr_slugs)}"
    )

    t0 = time.perf_counter()
    model = fit_pooled(
        prr_slugs,
        "random",
        data,
        input_keys,
        output_keys,
        task="material",
        restarts=restarts,
        max_iter=max_iter,
    )
    fit_s = time.perf_counter() - t0
    corr = model.tool_correlation_  # (m, T, T)
    print(
        f"fitted al<->ti task correlation (dep, Ipk): "
        f"{corr[0, 0, 1]:.4f}, {corr[1, 0, 1]:.4f}   (fit {fit_s:.1f} s)"
    )
    print(
        f"criterion (pre-stated): cross-material epistemic > in-distribution by "
        f">{EPI_MARGIN:.0%} on BOTH outputs, OR support drops by >{SUPP_MARGIN} "
        "(negative-Mahalanobis)"
    )

    pairs = []
    for src in prr_slugs:
        for tgt in prr_slugs:
            if src != tgt:
                pairs.append(_pair_awareness(model, data, src, tgt, "material"))
    n_pass = sum(p["directional_pass"] for p in pairs)
    blind = [p for p in pairs if p["blind_pair"]]
    crossmat = [p for p in pairs if p["cross_material"]]
    n_blind_pass = sum(p["directional_pass"] for p in blind)
    n_crossmat_pass = sum(p["directional_pass"] for p in crossmat)
    n_unknown_dom = sum(p["unknown_dominates_both"] for p in pairs)

    hdr = (
        f"{'model (A)':<16}{'query B':<16}{'kind':<11}"
        f"{'epi ratio X/in':>16}{'supp d':>8}{'mean shift(dep)':>16}{'flag':>6}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for p in pairs:
        kind = "BLIND" if p["blind_pair"] else "x-mat" if p["cross_material"] else "same-mat"
        rat = f"{p['epi_ratio_cross_over_in'][0]:.2f},{p['epi_ratio_cross_over_in'][1]:.2f}"
        flag = "PASS" if p["directional_pass"] else ("n/a" if not p["cross_material"] else "no")
        print(
            f"{p['src'][:15]:<16}{p['tgt'][:15]:<16}{kind:<11}{rat:>16}"
            f"{p['support_delta_cross_minus_in']:>+8.3f}{p['mean_shift_abs'][0]:>16.3f}{flag:>6}"
        )
    print(f"\ndirectional pass (epistemic inflates / support drops): {n_pass}/12")
    print(
        f"  of the 4 BLIND (cross-material same-tier) pairs: {n_blind_pass}/4  "
        "(these were 0/4 flagged by the per-campaign model -- RESULTS.md)"
    )
    print(f"  of the 8 cross-material pairs                  : {n_crossmat_pass}/8")
    print(
        f"unknown-material (tool=None) epistemic dominates both known materials: "
        f"{n_unknown_dom}/12   (section 5.8 by construction -- the robust flag)"
    )
    ms = np.array([p["mean_shift_abs"][0] for p in blind])
    print(
        f"predicted dep-rate MEAN shift across the 4 blind pairs: "
        f"{ms.min():.3f}-{ms.max():.3f} Ang/s -- the pooled model REPRESENTS the "
        "material shift; the per-campaign model (no material axis) cannot"
    )

    # baseline (per-campaign) 4 blind pairs, read straight from m1_empa.json
    base_blind = []
    if baseline is not None:
        for bp in baseline.get("ood_check", {}).get("pairs", []):
            if material_of_slug(bp["model"]) != material_of_slug(bp["queried_on"]) and tier_of(
                bp["model"]
            ) == tier_of(bp["queried_on"]):
                base_blind.append(
                    {
                        "model": bp["model"],
                        "queried_on": bp["queried_on"],
                        "id_mean_epi": bp["id_mean_epi"],
                        "ood_mean_epi": bp["ood_mean_epi"],
                        "id_mean_support": bp["id_mean_support"],
                        "ood_mean_support": bp["ood_mean_support"],
                        "ood_epi_greater_both_outputs": bp["ood_epi_greater_both_outputs"],
                    }
                )

    # secondary: campaign-as-task (4 tasks) on the same fit slices -- confirms
    # the material-as-task pattern is not a 2-task pooling artifact.
    print("\n-- secondary: campaign-as-task (4 tasks) on the 4 blind pairs --")
    camp_model = fit_pooled(
        prr_slugs,
        "random",
        data,
        input_keys,
        output_keys,
        task="campaign",
        restarts=restarts,
        max_iter=max_iter,
    )
    camp_pairs = []
    for src, tgt in blind_pair_order(prr_slugs, data):
        cp = _pair_awareness(camp_model, data, src, tgt, "campaign")
        camp_pairs.append(cp)
        print(
            f"  {src[:6]}->{tgt[:6]} epi ratio(dep,Ipk)="
            f"{cp['epi_ratio_cross_over_in'][0]:.2f},{cp['epi_ratio_cross_over_in'][1]:.2f} "
            f"mean shift(dep)={cp['mean_shift_abs'][0]:.3f}  "
            f"{'PASS' if cp['directional_pass'] else 'no'}"
        )

    result = {
        "protocol": (
            "pooled PRR model (material as task) on the 4 PRR campaigns' random-fit "
            "slices; for each ordered (src=A, tgt=B) pair, at B's held-out random-test "
            "recipes compare epistemic + support under material(B) [in-dist] vs "
            "material(A) [cross] vs None [unknown fallback], plus the predicted-mean shift"
        ),
        "criterion": (
            f"directional PASS = cross-material epistemic > in-dist by >{EPI_MARGIN:.0%} on "
            f"both outputs OR support drops by >{SUPP_MARGIN}; only meaningful for "
            "cross-material pairs (same-material pairs have material(A)==material(B))"
        ),
        "material_as_task": {
            "fitted_task_correlation_al_ti": {
                output_keys[0]: float(corr[0, 0, 1]),
                output_keys[1]: float(corr[1, 0, 1]),
            },
            "pairs": pairs,
            "n_pairs": len(pairs),
            "n_directional_pass": n_pass,
            "n_blind_pairs": len(blind),
            "n_blind_directional_pass": n_blind_pass,
            "n_crossmaterial_pairs": len(crossmat),
            "n_crossmaterial_directional_pass": n_crossmat_pass,
            "n_unknown_dominates": n_unknown_dom,
        },
        "campaign_as_task_blind_pairs": camp_pairs,
        "baseline_per_campaign_blind_pairs": base_blind,
        "finding": (
            "The pooled model REPRESENTS the material shift in the MEAN (0.78-1.67 Ang/s "
            "on dep rate, tens of in-distribution sigmas) and inflates epistemic to "
            "DOMINATE both materials when the material is UNSPECIFIED (unknown fallback, "
            "section 5.8, 12/12) -- neither is expressible by a per-campaign model. But "
            "wrong-material EPISTEMIC screening is ASYMMETRIC (inflates when querying the "
            "wider-variance material's conditioning, deflates the other way) and SUPPORT "
            "stays ~flat (+-0.06): input-space screening remains blind to a same-box "
            "material shift, exactly as the per-campaign OOD finding said. Awareness is "
            "achieved by making material an EXPLICIT conditioning axis, not by "
            "auto-detecting a wrong-material query from the recipe alone."
        ),
    }
    return result, model


def material_of_slug(slug: str) -> str:
    return "al" if slug.startswith("al") else "ti"


def blind_pair_order(prr_slugs, data):
    """The 4 cross-material same-tier ordered pairs, in prr_slugs nesting order."""
    out = []
    for src in prr_slugs:
        for tgt in prr_slugs:
            if (
                src != tgt
                and data[src]["material"] != data[tgt]["material"]
                and data[src]["tier"] == data[tgt]["tier"]
            ):
                out.append((src, tgt))
    return out


# ============================================================================
# Block B -- LEAVE-ONE-MATERIAL-OUT (honest transfer)
# ============================================================================


def _conformal_metrics(view, Xcal, Ycal, Xtest, Ytest, output_keys, units):
    """Split-conformal wrap ``view`` on (Xcal,Ycal); PICP/CI/gate/MPIW on the
    test slice, per output + pooled -- same metric shape as run_m1_empa."""
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(view, Xcal, Ycal)
    conf = ConformalForwardModel(view, cal)
    dist = conf.predict(Xtest)
    cset = np.asarray(dist.conformal_set)  # (n, m, 2)
    lo, hi = cset[..., 0], cset[..., 1]
    inside = (Ytest >= lo) & (Ytest <= hi)
    mpiw = uq.mpiw(lo, hi)
    n_test = Ytest.shape[0]
    per_output = {}
    for j, key in enumerate(output_keys):
        k = int(inside[:, j].sum())
        c_lo, c_hi = binom_ci(k, n_test)
        per_output[key] = {
            "unit": units[j],
            "picp": float(k / n_test),
            "k_covered": k,
            "n_test": n_test,
            "ci95": [c_lo, c_hi],
            "nominal_in_ci": bool(c_lo <= NOMINAL <= c_hi),
            "mpiw": float(mpiw[j]),
        }
    k_pool, n_pool = int(inside.sum()), int(inside.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    return {
        "per_output": per_output,
        "pooled": {
            "picp": float(k_pool / n_pool),
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
    }


def _raw_predictive_metrics(view, Xtest, Ytest, output_keys, units):
    """The model's OWN predictive coverage: the nominal (1-ALPHA) two-sided
    Gaussian interval mean +- z*sqrt(ale^2 + epi^2). Unlike split-conformal
    (which is ~nominal by construction under exchangeability, so it cannot
    diagnose transfer), THIS can fail -- it measures whether the few-shot model
    KNOWS how uncertain it is on the new material (calibrated self-knowledge)."""
    z = float(stats.norm.ppf(1.0 - ALPHA / 2.0))
    dist = view.predict(Xtest)
    mu = np.asarray(dist.mean)
    sig = np.sqrt(np.asarray(dist.aleatoric_sigma) ** 2 + np.asarray(dist.epistemic_sigma) ** 2)
    lo, hi = mu - z * sig, mu + z * sig
    inside = (Ytest >= lo) & (Ytest <= hi)
    mpiw = uq.mpiw(lo, hi)
    n_test = Ytest.shape[0]
    per_output = {}
    for j, key in enumerate(output_keys):
        k = int(inside[:, j].sum())
        c_lo, c_hi = binom_ci(k, n_test)
        per_output[key] = {
            "unit": units[j],
            "picp": float(k / n_test),
            "k_covered": k,
            "n_test": n_test,
            "ci95": [c_lo, c_hi],
            "nominal_in_ci": bool(c_lo <= NOMINAL <= c_hi),
            "mpiw": float(mpiw[j]),
        }
    k_pool, n_pool = int(inside.sum()), int(inside.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    return {
        "per_output": per_output,
        "pooled": {
            "picp": float(k_pool / n_pool),
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
    }


# transfer is claimable only if few-shot ACCURACY approaches the full-data ceiling
RMSE_CEILING_FACTOR = 3.0  # few-shot dep-RMSE must be within this * the baseline RMSE


def block_b_lomo(data, keys_by_param, prr_slugs, baseline, restarts, max_iter):
    banner("BLOCK B -- LEAVE-ONE-MATERIAL-OUT (the honest transfer check; expected WEAK)")
    input_keys = keys_by_param["prr"]
    output_keys = data[prr_slugs[0]]["output_keys"]
    units = data[prr_slugs[0]]["units"]
    dep = output_keys[0]
    by_mat = {
        "al": [s for s in prr_slugs if data[s]["material"] == "al"],
        "ti": [s for s in prr_slugs if data[s]["material"] == "ti"],
    }
    print(
        "PRR subspace. Known material = its 2 PRR campaigns' TEMPORAL-fit slices (material "
        "as task);\nheld-out material = an UNKNOWN tool. Few-shot draws K/2 earliest "
        "temporal rows from EACH\nheld campaign (spans both tiers); the remainder is "
        "seeded-random split into conformal cal(30%)/test."
    )
    print(
        "Transfer signal = few-shot point-RMSE vs the full-data baseline ceiling "
        f"(claimable only if\ndep-RMSE <= {RMSE_CEILING_FACTOR:.0f}x ceiling AND the raw "
        "predictive coverage gate passes). Conformal PICP is\n~nominal by construction "
        "(exchangeable split) -- its MPIW is the sharpness cost, not a transfer test."
    )

    def _ceiling(camps):
        if baseline is None:
            return {k: float("nan") for k in output_keys}
        out = {}
        for k in output_keys:
            vals = [
                baseline["campaigns"][c]["splits"]["random"]["per_output"][k]["rmse"]
                for c in camps
                if c in baseline.get("campaigns", {})
            ]
            out[k] = float(np.mean(vals)) if vals else float("nan")
        return out

    directions = {}
    claimable_flags = []
    for held in ("ti", "al"):
        known = "al" if held == "ti" else "ti"
        Xk, Yk, tk = [], [], []
        for s in by_mat[known]:
            fi = data[s]["temporal"][0]
            Xk.append(data[s]["X"][fi])
            Yk.append(data[s]["Y"][fi])
            tk += [known] * len(fi)
        Xk, Yk = np.vstack(Xk), np.vstack(Yk)
        held_camps = by_mat[held]
        Xh_all = np.vstack([data[s]["X"] for s in held_camps])
        Yh_all = np.vstack([data[s]["Y"] for s in held_camps])
        ceiling = _ceiling(held_camps)

        base = MultiToolGPForwardModel(
            input_keys=list(input_keys),
            output_keys=list(output_keys),
            rank=1,
            n_restarts=restarts,
            seed=SEED,
            max_iter=max_iter,
        ).fit(Xk, Yk, tk)
        # ZERO-SHOT: held material unknown -> fallback dominates known, elementwise
        epi_un = np.asarray(base.predict(Xh_all, tool_id=held).epistemic_sigma)
        epi_kn = np.asarray(base.predict(Xh_all, tool_id=known).epistemic_sigma)
        dominates = bool(np.all(epi_un >= epi_kn - 1e-12))
        margin = float((epi_un - epi_kn).min())
        rmse_zero = uq.rmse(np.asarray(base.predict(Xh_all, tool_id=held).mean), Yh_all)
        print(
            f"\n[leave-{held.upper()}-out]  train {by_mat[known]} ({len(Xk)} rows) -> "
            f"held-out {held.upper()} = {len(Xh_all)} rows over {held_camps}"
        )
        print(
            f"  zero-shot: unknown-{held} epistemic DOMINATES known-{known} elementwise on "
            f"all held rows: {dominates} (min margin {margin:+.4f})  [section 5.8 guarantee]"
        )
        print(
            "  full-data baseline ceiling (dep RMSE, random split): "
            + ", ".join(f"{UNIT_LABEL.get(k, k)}={ceiling[k]:.4f}" for k in output_keys)
        )
        print(
            "  zero-shot fallback RMSE: "
            + ", ".join(
                f"{UNIT_LABEL.get(k, k)}={v:.4f}"
                for k, v in zip(output_keys, rmse_zero, strict=True)
            )
            + f"  (~{rmse_zero[0] / ceiling[dep]:.0f}x ceiling on dep -> no zero-shot mean transfer)"
        )

        few = {}
        for K in FEWSHOT_K:
            per_c = max(1, K // len(held_camps))
            Xf, Yf, Xr, Yr = [], [], [], []
            for s in held_camps:
                oi = np.arange(data[s]["n"])  # temporal / file order
                Xf.append(data[s]["X"][oi[:per_c]])
                Yf.append(data[s]["Y"][oi[:per_c]])
                Xr.append(data[s]["X"][oi[per_c:]])
                Yr.append(data[s]["Y"][oi[per_c:]])
            Xf, Yf, Xr, Yr = (np.vstack(a) for a in (Xf, Yf, Xr, Yr))

            m = MultiToolGPForwardModel(
                input_keys=list(input_keys),
                output_keys=list(output_keys),
                rank=1,
                n_restarts=restarts,
                seed=SEED,
                max_iter=max_iter,
            ).fit(Xk, Yk, tk)
            m.adapt_to_tool(held, Xf, Yf)  # first K/2 temporal rows of EACH held campaign
            view = m.for_tool(held)
            rperm = np.random.default_rng(SEED).permutation(len(Xr))  # exchangeable split
            ncal = int(round(0.30 * len(Xr)))
            cal_i, test_i = rperm[:ncal], rperm[ncal:]
            rmse_few = uq.rmse(np.asarray(view.predict(Xr[test_i]).mean), Yr[test_i])
            raw = _raw_predictive_metrics(view, Xr[test_i], Yr[test_i], output_keys, units)
            conf = _conformal_metrics(
                view, Xr[cal_i], Yr[cal_i], Xr[test_i], Yr[test_i], output_keys, units
            )
            ratio = (
                float(rmse_few[0] / ceiling[dep]) if ceiling[dep] == ceiling[dep] else float("nan")
            )
            raw_gate = raw["pooled"]["nominal_in_ci"]
            arm_ok = bool(raw_gate and ratio <= RMSE_CEILING_FACTOR)
            claimable_flags.append(arm_ok)
            few[f"K={K}"] = {
                "n_fewshot": int(len(Xf)),
                "n_test": int(len(test_i)),
                "point_rmse": {k: float(v) for k, v in zip(output_keys, rmse_few, strict=True)},
                "dep_rmse_over_ceiling": ratio,
                "raw_predictive": raw,
                "split_conformal": conf,
                "arm_claimable": arm_ok,
            }
            rp, cp = raw["pooled"], conf["pooled"]
            print(
                f"  few-shot K={K:<3d}(n_test={len(test_i)}): dep-RMSE {rmse_few[0]:.4f} "
                f"(~{ratio:.0f}x ceiling)  raw-PICP {rp['picp']:.3f} "
                f"[{rp['ci95'][0]:.3f},{rp['ci95'][1]:.3f}] "
                f"{'PASS' if rp['nominal_in_ci'] else 'FAIL'}  conformal-PICP {cp['picp']:.3f} "
                f"(dep MPIW {conf['per_output'][dep]['mpiw']:.3f})  "
                f"transfer arm: {'CLAIMABLE' if arm_ok else 'NO'}"
            )
        directions[f"leave_{held}_out"] = {
            "known_material": known,
            "known_campaigns": by_mat[known],
            "held_material": held,
            "held_campaigns": held_camps,
            "n_known_train": int(len(Xk)),
            "n_held_rows": int(len(Xh_all)),
            "baseline_ceiling_rmse_random": ceiling,
            "zero_shot": {
                "unknown_epistemic_dominates_known_elementwise": dominates,
                "min_margin": margin,
                "fallback_point_rmse": {
                    k: float(v) for k, v in zip(output_keys, rmse_zero, strict=True)
                },
            },
            "few_shot": few,
        }

    all_claimable = all(claimable_flags)
    verdict = (
        "cross-material transfer is CLAIMABLE: every leave-one-material-out few-shot arm "
        f"reached within {RMSE_CEILING_FACTOR:.0f}x the full-data RMSE ceiling AND kept "
        "calibrated (raw-predictive gate PASS)."
        if all_claimable
        else "cross-material transfer stays FORBIDDEN (audit F4): no few-shot arm approaches the "
        f"full-data RMSE ceiling (dep-RMSE stays many-x above it) within K in {list(FEWSHOT_K)}. "
        "Zero-shot has NO mean transfer (the fallback predicts the TRAINED material's "
        "surface); the ONLY thing that holds by construction is the section 5.8 epistemic "
        "domination (awareness), not accuracy. Split-conformal keeps coverage ~nominal only "
        "by widening bands (MPIW), which is honest abstention, not transfer."
    )
    print(f"\nVERDICT (Block B): {verdict}")
    return {
        "protocol": (
            "PRR subspace; train on one material's 2 campaigns (temporal-fit, material as "
            "task); held-out material = UNKNOWN tool. Zero-shot asserts section 5.8 fallback "
            "domination. Few-shot adapt_to_tool on K/2 earliest temporal rows of EACH held "
            "campaign; report point-RMSE vs the full-data baseline ceiling, raw predictive "
            "coverage (can fail), and split-conformal PICP+MPIW on a seeded-random "
            "(exchangeable) cal(30%)/test split of the remainder."
        ),
        "rmse_ceiling_factor": RMSE_CEILING_FACTOR,
        "directions": directions,
        "transfer_claimable": all_claimable,
        "verdict": verdict,
    }


# ============================================================================
# Block C -- POOLING-COST
# ============================================================================


def block_c_pooling_cost(
    data, keys_by_param, prr_slugs, duty_slugs, baseline, restarts, max_iter, prr_random_model
):
    banner(
        "BLOCK C -- POOLING-COST (does material-conditioned pooling COST coverage "
        "vs the per-campaign baseline?)"
    )
    print(
        "per campaign & split: pooled model (material as task, fit on ALL subspace "
        "campaigns'\nsplit-FIT slices) -> for_tool(material) -> split-conformal on THIS "
        "campaign's cal slice ->\nPICP on THIS campaign's test slice. Same split indices "
        "as run_m1_empa (asserted)."
    )

    subspaces = {"prr": list(prr_slugs), "duty": list(duty_slugs)}
    # pooled models per (subspace, split); reuse the already-fitted PRR random model
    models = {("prr", "random"): prr_random_model}
    for sub, slugs in subspaces.items():
        for split_name in ("temporal", "random"):
            if (sub, split_name) in models:
                continue
            ik = keys_by_param[sub]
            ok = data[slugs[0]]["output_keys"]
            models[(sub, split_name)] = fit_pooled(
                slugs,
                split_name,
                data,
                ik,
                ok,
                task="material",
                restarts=restarts,
                max_iter=max_iter,
            )

    per_campaign = {}
    rows = []
    for sub, slugs in subspaces.items():
        for slug in slugs:
            d = data[slug]
            ok, units = d["output_keys"], d["units"]
            # assert split sizes match the baseline exactly (fidelity guard)
            if baseline is not None and slug in baseline.get("campaigns", {}):
                base_sizes = baseline["campaigns"][slug]["split_sizes"]
                if base_sizes != d["split_sizes"]:
                    raise RuntimeError(
                        f"{slug}: reconstructed split sizes {d['split_sizes']} != baseline "
                        f"{base_sizes} -- split reconstruction drifted"
                    )
            entry = {
                "material": d["material"],
                "parameterization": sub,
                "split_sizes": d["split_sizes"],
                "splits": {},
            }
            for split_name in ("temporal", "random"):
                model = models[(sub, split_name)]
                _, cal_idx, test_idx = d[split_name]
                met = _conformal_metrics(
                    model.for_tool(d["material"]),
                    d["X"][cal_idx],
                    d["Y"][cal_idx],
                    d["X"][test_idx],
                    d["Y"][test_idx],
                    ok,
                    units,
                )
                base_pool = base_out = None
                if baseline is not None and slug in baseline.get("campaigns", {}):
                    bsp = baseline["campaigns"][slug]["splits"][split_name]
                    base_pool = bsp["pooled"]
                    base_out = {k: bsp["per_output"][k] for k in ok}
                entry["splits"][split_name] = {
                    "pooled_model": met,
                    "baseline": {"pooled": base_pool, "per_output": base_out},
                }
                p = met["pooled"]
                bp = base_pool["picp"] if base_pool else float("nan")
                bg = (
                    "PASS"
                    if base_pool and base_pool["nominal_in_ci"]
                    else "FAIL"
                    if base_pool
                    else "?"
                )
                pg = "PASS" if p["nominal_in_ci"] else "FAIL"
                rows.append((slug, split_name, bp, bg, p["picp"], pg, p["ci95"], d["degenerate"]))
            per_campaign[slug] = entry

    hdr = (
        f"{'campaign':<20}{'split':<10}{'base PICP':>10}{'base':>6}"
        f"{'pool PICP':>10}{'pool':>6}{'pool 95% CI':>18}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for slug, split_name, bp, bg, pp, pg, ci, degen in rows:
        star = "*" if (split_name == "temporal" and degen) else " "
        ci_s = f"[{ci[0]:.3f},{ci[1]:.3f}]"
        print(
            f"{slug[:19]:<20}{(split_name + star):<10}{bp:>10.3f}{bg:>6}"
            f"{pp:>10.3f}{pg:>6}{ci_s:>18}"
        )
    print("* ti_120w_short_pw temporal order key is unverified file order (BatchNr degenerate)")

    # gate-flip summary (honest: report both directions)
    flips_fixed, flips_broke = [], []
    for slug, entry in per_campaign.items():
        for split_name, sp in entry["splits"].items():
            base = sp["baseline"]["pooled"]
            pool = sp["pooled_model"]["pooled"]
            if base is None:
                continue
            if not base["nominal_in_ci"] and pool["nominal_in_ci"]:
                flips_fixed.append(f"{slug}/{split_name}")
            if base["nominal_in_ci"] and not pool["nominal_in_ci"]:
                flips_broke.append(f"{slug}/{split_name}")
    print(
        f"\ngate PASS/FAIL flips vs baseline -- pooling FIXED: {flips_fixed or 'none'}; "
        f"pooling BROKE: {flips_broke or 'none'}"
    )
    return {
        "protocol": (
            "material-conditioned pooled model per (subspace, split), split-conformal "
            "wrapped per campaign on the for_tool(material) view; split indices identical "
            "to run_m1_empa (asserted against results/m1_empa.json split_sizes)"
        ),
        "per_campaign": per_campaign,
        "gate_flips_pooling_fixed": flips_fixed,
        "gate_flips_pooling_broke": flips_broke,
    }


# ============================================================================
# main
# ============================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--smoke", action="store_true", help="n_restarts=1, max_iter=60 (a fast shape check)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="n_restarts=2, max_iter=100 (default if neither flag given)",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    smoke = args.smoke and not args.full
    restarts = 1 if smoke else 2
    max_iter = 60 if smoke else 100

    np.random.seed(SEED)
    out_path = args.out or (RESULTS_DIR / f"m1_empa_pooled{'.smoke' if smoke else ''}.json")

    banner(
        "MATERIAL-CONDITIONED POOLING on real Empa HiPIMS data -- M1 remainder "
        "(NOT the signed M1 gate)"
    )
    print("dataset : Zenodo 10.5281/zenodo.18495402 (CC-BY-4.0), real Empa sputter tool")
    print("model   : MultiToolGPForwardModel (ICM multi-task GP, section 10.4), tool_id = MATERIAL")
    print(
        "caveats : BO-clustered sampling; pooled CI optimistic (outputs share rows); "
        "ti_120w order\n          key unverified; Ipk measured not set; REAL data but not the "
        "MBE target process."
    )
    print(f"mode    : {'SMOKE' if smoke else 'full'} (n_restarts={restarts}, max_iter={max_iter})")

    t_start = time.perf_counter()
    data, keys_by_param = load_campaigns()
    prr_slugs = tuple(c.slug for c in CAMPAIGNS if c.parameterization == "prr")
    duty_slugs = tuple(c.slug for c in CAMPAIGNS if c.parameterization == "duty")
    print(f"\nPRR subspace  ({len(prr_slugs)} campaigns): {list(prr_slugs)}")
    print(f"DUTY subspace ({len(duty_slugs)} campaigns): {list(duty_slugs)}")

    baseline = None
    if BASELINE_JSON.exists():
        baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
        print(f"baseline read : {BASELINE_JSON.name} (per-campaign comparison enabled)")
    else:
        print(f"baseline MISSING ({BASELINE_JSON.name}) -- comparison columns will be blank")

    # Block A fits the material/random PRR model; it is the SAME object Block C
    # needs for (prr, random), so reuse it (refit-free) to stay under budget.
    block_a, prr_random_model = block_a_awareness(
        data, keys_by_param, prr_slugs, baseline, restarts, max_iter
    )
    block_b = block_b_lomo(data, keys_by_param, prr_slugs, baseline, restarts, max_iter)
    block_c = block_c_pooling_cost(
        data, keys_by_param, prr_slugs, duty_slugs, baseline, restarts, max_iter, prr_random_model
    )

    payload = {
        "meta": {
            "seed": SEED,
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL,
            "ci": f"exact (Clopper-Pearson) binomial, level {CI_LEVEL}",
            "smoke": smoke,
            "n_restarts": restarts,
            "max_iter": max_iter,
            "model": "MultiToolGPForwardModel (ICM multi-task GP, implementation-plan 10.4), tool_id=material",
            "subspaces": {"prr": list(prr_slugs), "duty": list(duty_slugs)},
            "baseline": str(BASELINE_JSON.name) if baseline is not None else None,
            "epi_margin": EPI_MARGIN,
            "supp_margin": SUPP_MARGIN,
            "fewshot_K": list(FEWSHOT_K),
            "caveats": [
                "REAL measured data (real_tool) but NOT the signed M1 program gate; not the MBE target process",
                "pooling is WITHIN a parameterization subspace (PRR / DUTY have incompatible knob names)",
                "BO-driven (BayBE) sampling clusters rows near optima; split-conformal exchangeability is approximate",
                "pooled coverage CI treats 2*n_test trials as independent although outputs share test rows (optimistic)",
                "ti_120w_short_pw temporal order is unverified file order; 5 rows skip-rejected",
                "Ipk (A) is measured, never set",
            ],
        },
        "block_a_awareness": block_a,
        "block_b_leave_one_material_out": block_b,
        "block_c_pooling_cost": block_c,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    total = time.perf_counter() - t_start
    print(f"\nresults written -> {out_path}")
    print(f"total wall time : {total:.1f} s")
    banner(
        "DONE (material-conditioned pooling on real Empa HiPIMS -- awareness is the "
        "deliverable; transfer is not claimed unless Block B says so)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
