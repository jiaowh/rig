"""M1 GATE-FORM validation on REAL Empa bipolar-HiPIMS data (M0 lead candidate).

Runs the RIG forward + conformal stack (GPForwardModel -> SplitConformalCalibrator
at alpha=0.10 -> ConformalForwardModel) per campaign on the six tidy CSVs that
prepare_empa.py builds from Zenodo record 10.5281/zenodo.18495402 (CC-BY-4.0;
Wieczorek et al., Digital Discovery 2026, DOI 10.1039/D6DD00063K), ingested
through the generic WP-H tabular adapter -- the same code path as the Zr
sputtering example (examples/real_data/sputtering/run_m1_sputtering.py), which
this script is modeled on.

Per campaign, BOTH of:

- TEMPORAL split (train = first 60% in BatchNr run order, calibration = next
  20%, test = last 20%) -- the implementation-plan section 15.3 M1 row's
  real-split form; and
- a seeded RANDOM 60/20/20 split as the contrast condition.

The gate check is DIRECTIONAL, exactly as the M1 row words it: the exact
(Clopper-Pearson) binomial 95% CI for the observed conformal coverage given
n_test must CONTAIN the nominal 0.90 -- never a hard +/-2% (at n_test ~ 80-130
the binomial SE alone is ~3%). Reported per output and pooled.

ACI drift path (D4 / implementation-plan section 5.6) -- an ADDITIONAL
evaluation; the static split-conformal blocks above stay in the JSON
unchanged as the baseline. Per campaign and per split, the SAME fitted GP
and the SAME calibration slice are re-evaluated under the ONLINE protocol:
an ACIController (online Adaptive Conformal Inference, Gibbs & Candes 2021;
alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)) streams the test
rows in split order (temporal: BatchNr run order; random: the seeded
shuffled order), and each row's interval is taken at the CURRENT alpha_t
and scored (hit/miss, width) BEFORE observe(x, y) -- an observation never
influences its own interval. What ACI GUARANTEES: asymptotic AVERAGE
coverage -> 1 - alpha under ARBITRARY distribution shift. What it does NOT
guarantee: finite-sample exactness at any fixed horizon -- so the
exact-binomial-CI row computed on the realized online coverage is a
DIRECTIONAL check with the same status as the static gate, not a theorem
test. ACI hyperparameters are the ACIController LIBRARY DEFAULTS
(gamma=0.05, window=50, alpha_clip=(0.001, 0.5), update_scores=True),
identical across ALL campaigns and splits, fixed BEFORE seeing any
per-campaign outcome -- a campaign that still fails under ACI is a
reportable finding, never a knob to turn. The trailing-window
rolling-coverage minimum (full windows only) is recorded as the section
5.6 concrete drift-detector statistic. The RANDOM stream doubles as the
exchangeable CONTROL where ACI should roughly match the static path.
(Method-currency note: section 20.2 makes conformal-PID the online
endpoint with bare ACI a component; this wires and validates the D4 ACI
component only.)

Extras (full 6-campaign run only): a support/OOD directional check across the
four PRR-space campaigns (mean epistemic sigma on a model's own held-out rows
vs on another campaign's recipes), and a pessimistic-inverse demo (section 8)
on Al-shortPW with an explicit INFEASIBLE probe.

HONEST FRAMING (read this): real measured data from a real Empa sputter tool
(provenance real_tool), so this is the M1 gate FORM on a genuine temporal
split -- but it is NOT the signed M1 program gate: the M0 venue decision
(which dataset is THE real dataset) is the user/PI's, and this is not the
project's MBE target process. Data caveats, repeated wherever numbers appear:

- BO-DRIVEN SAMPLING (BayBE): rows cluster near high-rate optima, they are not
  space-filling. Split-conformal exchangeability is therefore an APPROXIMATION,
  and the temporal split doubles as a drift stress test: the BO loop moves
  toward optima over the campaign, so train (early, exploratory) and test
  (late, exploitative) genuinely differ in distribution.
- The pooled coverage CI treats 2 x n_test trials as independent, but both
  outputs share test rows -- the pooled CI is optimistic. Per-output rows are
  the honest unit.
- ti_120w_short_pw is degenerate: BatchNr==1 everywhere (its "temporal" split
  is unverified file order -- flagged, not hidden) and 5/495 rows sit 3e-11 to 4e-11
  outside its full-precision bounds (skip-ingested: 490 kept + 5 rejects).
- "Ipk (A)" is measured, never set -- it is an output here, not a knob.

Run (Windows cp1252 console -> force UTF-8; ~minutes for the full gate):

    PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_m1_empa.py
        [--campaign <slug>] [--smoke] [--out <path.json>]

--campaign runs ONE campaign (results file gets a .<slug> suffix; the OOD
check needs all four PRR campaigns in one run, so it is skipped). --smoke cuts
GP/solver restarts for a fast shape check (results file gets a .smoke suffix).
Deterministic: every stochastic step is seeded (SEED below).
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

# The section 8 solver's diagnostic strings contain Greek (kappa/sigma) and
# section signs; force UTF-8 so a cp1252 Windows console does not crash on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # prepare_empa.py lives beside this script

from prepare_empa import CAMPAIGNS, DEP_RATE_COLUMN, Campaign

from rig.calibration.conformal import (
    ACIController,
    ConformalForwardModel,
    SplitConformalCalibrator,
)
from rig.forward import GPForwardModel, records_to_arrays
from rig.interfaces import Infeasible
from rig.inverse.pessimistic import PessimisticInverseSolver
from rig.metrics import uq
from rig.schema import ureg  # the ONE shared pint registry -- never a second one
from rig_adapters.tabular.ingest import ingest_csv
from rig_adapters.tabular.spec import load_spec

SEED = 0
ALPHA = 0.10  # conformal miscoverage target -> nominal 90% coverage
NOMINAL = 1.0 - ALPHA
CI_LEVEL = 0.95  # binomial CI level for the directional gate check
TRAIN_FRAC, CAL_FRAC = 0.60, 0.20  # test = the remaining 20%
CSV_DIR = HERE / "csv"
SPEC_DIR = HERE / "specs"
RESULTS_DIR = HERE / "results"

# The one campaign that may reject rows (5 rows 3e-11 to 4e-11 outside full-precision
# bounds; see the spec header + tests/test_empa_ingest.py). Any OTHER reject
# count, anywhere, is treated as data corruption and aborts loudly.
EXPECTED_REJECTS = {"ti_120w_short_pw": 5}
DEMO_SLUG = "al_120w_short_pw"  # the pessimistic-inverse demo campaign
PRR_SLUGS = tuple(c.slug for c in CAMPAIGNS if c.parameterization == "prr")
# readable unit labels for console tables (JSON carries the declared units)
UNIT_LABEL = {DEP_RATE_COLUMN: "Ang/s", "Ipk (A)": "A"}


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def binom_ci(k: int, n: int) -> tuple[float, float]:
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion.

    Gate form: "nominal inside this CI" is the exact-test dual of the plan's
    "empirical coverage within the binomial CI of nominal" (implementation-plan
    §15.3 M1, which says "given n_cal"); under the 60/20/20 rounding used here
    n_cal == n_test in all six campaigns, so the two readings coincide.
    """
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=CI_LEVEL, method="exact")
    return float(ci.low), float(ci.high)


def ingest_campaign(campaign: Campaign):
    """Ingest one tidy CSV; enforce the pinned reject expectations."""
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
            f"(first: {result.rejects[0].reason if result.rejects else None!r}) -- "
            "the tidy CSVs or specs changed; re-run prepare_empa.py + tests first"
        )
    # BatchNr (in extra: undeclared by design) doubles as the run-order check:
    # prep stable-sorted the CSVs, so ingest order must be BatchNr-monotone.
    batch = [int(r.extra["unmatched_columns"]["BatchNr"]) for r in result.records]
    if batch != sorted(batch):
        raise RuntimeError(f"{campaign.slug}: ingest order is not BatchNr-monotone")
    degenerate = len(set(batch)) == 1  # ti_120w: BatchNr==1 everywhere
    return spec, result, degenerate


def split_indices(n: int) -> tuple[int, int]:
    """(n_train, n_cal) for the 60/20/20 contract; test = the remainder."""
    n_train = int(round(TRAIN_FRAC * n))
    n_cal = int(round(CAL_FRAC * n))
    return n_train, n_cal


def fit_and_eval(
    Xf: np.ndarray,
    Yf: np.ndarray,
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    input_keys: list[str],
    output_keys: list[str],
    units: list[str],
    gp_restarts: int,
) -> tuple[dict, GPForwardModel, np.ndarray]:
    """Fit GP on the fit slice, conformal-calibrate, evaluate on the test slice.

    Returns (metrics dict, fitted model, per-output mean epistemic sigma on the
    test rows -- the in-distribution reference the OOD check reuses).
    """
    model = GPForwardModel(
        input_keys=input_keys, output_keys=output_keys, n_restarts=gp_restarts, seed=SEED
    ).fit(Xf, Yf)
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    conf = ConformalForwardModel(model, cal)

    dist = conf.predict(Xt)  # batch PredictiveDistribution
    mu = np.asarray(dist.mean)  # (n_test, m)
    ale = np.asarray(dist.aleatoric_sigma)
    epi = np.asarray(dist.epistemic_sigma)
    sig_total = np.sqrt(ale**2 + epi**2)
    cset = np.asarray(dist.conformal_set)  # (n_test, m, 2) conformal band
    lo, hi = cset[..., 0], cset[..., 1]

    n_test = Yt.shape[0]
    inside = (Yt >= lo) & (Yt <= hi)  # (n_test, m)
    rmse = uq.rmse(mu, Yt)
    mae = uq.mae(mu, Yt)
    crps = np.mean(uq.crps_gaussian(mu, sig_total, Yt), axis=0)
    mpiw = uq.mpiw(lo, hi)
    qce = uq.quantile_calibration_error(mu, sig_total, Yt)
    y_rng = Yt.max(axis=0) - Yt.min(axis=0)

    per_output: dict[str, dict] = {}
    for j, key in enumerate(output_keys):
        k = int(inside[:, j].sum())
        ci_lo, ci_hi = binom_ci(k, n_test)
        per_output[key] = {
            "unit": units[j],
            "rmse": float(rmse[j]),
            "nrmse_pct": float(100.0 * rmse[j] / y_rng[j]),
            "mae": float(mae[j]),
            "crps": float(crps[j]),
            "picp": float(k / n_test),
            "k_covered": k,
            "n_test": n_test,
            "ci95": [ci_lo, ci_hi],
            "nominal_in_ci": bool(ci_lo <= NOMINAL <= ci_hi),
            "mpiw": float(mpiw[j]),
            "qce": float(qce[j]),
        }
    k_pool = int(inside.sum())
    n_pool = int(inside.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    metrics = {
        "per_output": per_output,
        # CAVEAT: pooled trials share test rows across outputs -> CI optimistic.
        "pooled": {
            "picp": float(k_pool / n_pool),
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
        "noise_std": {k: float(s) for k, s in zip(output_keys, model.noise_std_, strict=True)},
    }
    return metrics, model, epi.mean(axis=0)


def print_split_table(label: str, metrics: dict, output_keys: list[str]) -> None:
    n_test = metrics["per_output"][output_keys[0]]["n_test"]
    print(f"\n{label}  (n_test={n_test}; nominal coverage {NOMINAL:.2f}; CI = exact binomial 95%)")
    hdr = f"{'output':<20}{'unit':>7}{'RMSE':>9}{'nRMSE%':>8}{'PICP':>7}{'95% CI':>17}{'gate':>6}{'MPIW':>9}"
    print(hdr)
    print("-" * len(hdr))
    for key in output_keys:
        m = metrics["per_output"][key]
        ci = f"[{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]"
        gate = "PASS" if m["nominal_in_ci"] else "FAIL"
        print(
            f"{key:<20}{UNIT_LABEL.get(key, m['unit']):>7}{m['rmse']:>9.4f}"
            f"{m['nrmse_pct']:>8.2f}{m['picp']:>7.3f}{ci:>17}{gate:>6}{m['mpiw']:>9.4f}"
        )
    p = metrics["pooled"]
    ci = f"[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}]"
    gate = "PASS" if p["nominal_in_ci"] else "FAIL"
    print(
        f"{'POOLED (2 outputs)':<20}{'':>7}{'':>9}{'':>8}{p['picp']:>7.3f}{ci:>17}{gate:>6}"
        "   (CI optimistic: outputs share test rows)"
    )


def aci_eval(
    model: GPForwardModel,
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    output_keys: list[str],
    units: list[str],
) -> dict:
    """D4/section 5.6 ONLINE evaluation: ACI on the same GP + calibration slice.

    A FRESH SplitConformalCalibrator is fitted on the SAME calibration slice
    (deterministic -- identical scores to the static path) so ACI's online
    score appends can never touch the static path's state. Test rows are
    streamed in split order; for each row the interval at the CURRENT alpha_t
    is scored FIRST (hit/miss + width), THEN observe(x, y) adapts alpha_t and
    appends the new score. An observation never influences its own interval.

    ONLY alpha_target is passed to ACIController; every other hyperparameter
    is the library default, uniform across all campaigns/splits (see module
    docstring -- no tuning-to-pass). No RNG anywhere in this path.
    """
    t0 = time.perf_counter()
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    controller = ACIController(cal, alpha_target=ALPHA)  # library defaults only

    n_test, m = Yt.shape
    hits = np.zeros((n_test, m), dtype=bool)
    widths = np.empty((n_test, m))
    alphas_used = np.empty((n_test, m))
    rolling = np.empty((n_test, m))
    window = controller._errs.maxlen  # the library-default window (read-only)
    for t in range(n_test):
        x, y = Xt[t], Yt[t]
        alphas_used[t] = np.broadcast_to(np.asarray(controller.alpha_t, dtype=float), (m,))
        itv = controller.interval(x)  # (m, 2) at the CURRENT (pre-update) alpha_t
        lo, hi = itv[..., 0].reshape(-1), itv[..., 1].reshape(-1)
        widths[t] = hi - lo
        miss = ((y < lo) | (y > hi)).astype(float)
        err = controller.observe(x, y)  # scores the SAME pre-update interval
        if not np.array_equal(err, miss):  # bookkeeping guard -- must never fire
            raise RuntimeError(f"ACI hit/miss bookkeeping mismatch at t={t}: {err} vs {miss}")
        hits[t] = err == 0.0
        rolling[t] = controller.rolling_coverage

    per_output: dict[str, dict] = {}
    full = rolling[window - 1 :]  # rolling coverage over FULL windows only
    for j, key in enumerate(output_keys):
        k = int(hits[:, j].sum())
        ci_lo, ci_hi = binom_ci(k, n_test)
        finite_w = widths[np.isfinite(widths[:, j]), j]
        per_output[key] = {
            "unit": units[j],
            "picp": float(k / n_test),
            "k_covered": k,
            "n_test": n_test,
            "ci95": [ci_lo, ci_hi],
            "nominal_in_ci": bool(ci_lo <= NOMINAL <= ci_hi),
            "mean_width": float(finite_w.mean()) if finite_w.size else None,
            "n_infinite_width": int(n_test - finite_w.size),
            "alpha_t": {
                "used_min": float(alphas_used[:, j].min()),
                "used_max": float(alphas_used[:, j].max()),
                "used_mean": float(alphas_used[:, j].mean()),
                "final": float(np.broadcast_to(np.asarray(controller.alpha_t), (m,))[j]),
            },
            "rolling_coverage": {
                "window": window,
                "min_full_window": float(full[:, j].min()) if full.size else None,
                "final": float(rolling[-1, j]),
            },
        }
    k_pool = int(hits.sum())
    n_pool = int(hits.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    return {
        "protocol": (
            "online D4/section 5.6: fresh split calibrator on the SAME calibration slice; "
            "test rows streamed in split order; interval at the current alpha_t scored "
            "BEFORE observe(x, y)"
        ),
        "hyperparameters": {
            "alpha_target": float(controller.alpha_target),
            "gamma": float(controller.gamma),
            "alpha_clip": [float(a) for a in controller.alpha_clip],
            "window": int(window),
            "update_scores": bool(controller.update_scores),
            "provenance": (
                "ACIController library defaults, uniform across all campaigns and splits, "
                "fixed before any per-campaign outcome was seen"
            ),
        },
        "per_output": per_output,
        # CAVEAT (same as static): pooled trials share test rows across outputs.
        "pooled": {
            "picp": float(k_pool / n_pool),
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
        "wall_seconds": round(time.perf_counter() - t0, 1),
    }


def print_aci_table(label: str, aci: dict, output_keys: list[str]) -> None:
    n_test = aci["per_output"][output_keys[0]]["n_test"]
    print(f"\n{label}  (n_test={n_test} streamed; interval BEFORE observe; ACI library defaults)")
    hdr = (
        f"{'output':<20}{'unit':>7}{'PICP':>7}{'95% CI':>17}{'gate':>6}"
        f"{'mean W':>9}{'a_final':>9}{'roll_min':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key in output_keys:
        m = aci["per_output"][key]
        ci = f"[{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]"
        gate = "PASS" if m["nominal_in_ci"] else "FAIL"
        width = "inf" if m["mean_width"] is None else f"{m['mean_width']:.4f}"
        roll = m["rolling_coverage"]["min_full_window"]
        roll_s = "n/a" if roll is None else f"{roll:.3f}"
        print(
            f"{key:<20}{UNIT_LABEL.get(key, m['unit']):>7}{m['picp']:>7.3f}{ci:>17}{gate:>6}"
            f"{width:>9}{m['alpha_t']['final']:>9.4f}{roll_s:>9}"
        )
        if m["n_infinite_width"]:
            print(
                f"  NB {m['n_infinite_width']} step(s) had an INFINITE-width interval "
                "(alpha_t at the low clip vs the score count) -- excluded from mean W"
            )
    p = aci["pooled"]
    ci = f"[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}]"
    gate = "PASS" if p["nominal_in_ci"] else "FAIL"
    print(
        f"{'POOLED (2 outputs)':<20}{'':>7}{p['picp']:>7.3f}{ci:>17}{gate:>6}"
        "   (CI optimistic: outputs share test rows)"
    )


def run_campaign(campaign: Campaign, idx: int, gp_restarts: int) -> tuple[dict, dict]:
    """Ingest + both splits + metrics for one campaign. Returns (summary, artifacts)."""
    spec, result, degenerate = ingest_campaign(campaign)
    records = result.records
    input_keys = list(spec.gp_input_keys)
    output_keys = list(spec.output_names)
    units = [o.unit for o in spec.outputs]  # declared units (raw, readable)
    if [v.name for v in spec.continuous_si] != input_keys:
        raise RuntimeError(f"{campaign.slug}: continuous_si order != gp_input_keys")

    X, Y_si = records_to_arrays(records, input_keys, output_keys)
    # ingest SI-canonicalizes outcomes (angstrom/s -> m/s x1e-10; A -> A x1);
    # report everything back in the readable declared units.
    si_per_raw = np.array(
        [float(ureg.Quantity(1.0, u).to_base_units().magnitude) for u in units]
    )
    Y = Y_si / si_per_raw

    n = len(records)
    n_train, n_cal = split_indices(n)
    n_test = n - n_train - n_cal

    banner(f"CAMPAIGN {campaign.slug}  ({spec.process_id})")
    print(f"rows ingested      : {n}  (+{len(result.rejects)} rejected; expected)")
    print(f"provenance.source  : {records[0].provenance.source}")
    order = "BatchNr (verified run order)" if not degenerate else (
        "FILE ORDER -- BatchNr degenerate (all 1); temporal split UNVERIFIED here"
    )
    print(f"temporal order key : {order}")
    print(f"split sizes        : train={n_train}  calibration={n_cal}  test={n_test}")

    summary: dict = {
        "process_id": spec.process_id,
        "material": campaign.material,
        "parameterization": campaign.parameterization,
        "n_ingested": n,
        "n_rejects": len(result.rejects),
        "order_key": "BatchNr" if not degenerate else "file_order_UNVERIFIED",
        "temporal_split_meaningful": not degenerate,
        "split_sizes": {"train": n_train, "cal": n_cal, "test": n_test},
        "splits": {},
    }
    artifacts: dict = {"spec": spec, "X_all": X, "Y_all": Y, "input_keys": input_keys,
                       "output_keys": output_keys, "units": units}

    # -- temporal split: contiguous blocks in run order (the M1 gate form) ----
    # NB ingest synthesized timestamps (no timestamp column); the split is over
    # the REAL BatchNr run order read from extra, not those synthetic stamps.
    idx_all = np.arange(n)
    splits = {
        "temporal": (idx_all[:n_train], idx_all[n_train : n_train + n_cal],
                     idx_all[n_train + n_cal :]),
    }
    # -- random contrast split: same sizes, seeded permutation -----------------
    rng = np.random.default_rng(SEED + idx)
    perm = rng.permutation(n)
    splits["random"] = (perm[:n_train], perm[n_train : n_train + n_cal],
                        perm[n_train + n_cal :])

    for name, (fit_idx, cal_idx, test_idx) in splits.items():
        t0 = time.perf_counter()
        metrics, model, id_epi = fit_and_eval(
            X[fit_idx], Y[fit_idx], X[cal_idx], Y[cal_idx], X[test_idx], Y[test_idx],
            input_keys, output_keys, units, gp_restarts,
        )
        metrics["wall_seconds"] = round(time.perf_counter() - t0, 1)
        # D4/section 5.6 ACI ONLINE path -- ADDITIONAL evaluation on the same
        # fitted GP + same calibration slice; the static block above is the
        # baseline and stays byte-identical in the JSON.
        metrics["aci"] = aci_eval(
            model, X[cal_idx], Y[cal_idx], X[test_idx], Y[test_idx], output_keys, units
        )
        summary["splits"][name] = metrics
        label = {
            "temporal": "TEMPORAL split (gate form; BO drift makes this the hard one)",
            "random": "RANDOM split (contrast; exchangeability approximately holds)",
        }[name]
        if name == "temporal" and degenerate:
            label = "TEMPORAL-BY-FILE-ORDER split (order key UNVERIFIED -- see above)"
        print_split_table(label, metrics, output_keys)
        aci_label = {
            "temporal": "ACI ONLINE, TEMPORAL stream (D4/section 5.6 drift path; run order)",
            "random": "ACI ONLINE, RANDOM stream (exchangeable control -- expect ~static)",
        }[name]
        if name == "temporal" and degenerate:
            aci_label = "ACI ONLINE, FILE-ORDER stream (order key UNVERIFIED -- see above)"
        print_aci_table(aci_label, metrics["aci"], output_keys)
        artifacts[f"model_{name}"] = model
        artifacts[f"fit_idx_{name}"] = fit_idx
        artifacts[f"test_idx_{name}"] = test_idx
        artifacts[f"id_epi_{name}"] = id_epi
    ns = summary["splits"]["temporal"]["noise_std"]
    print(f"\nfitted aleatoric noise_std (temporal fit): "
          + ", ".join(f"{k}={v:.4f} {UNIT_LABEL.get(k, '')}" for k, v in ns.items()))
    return summary, artifacts


def ood_check(artifacts: dict[str, dict]) -> dict:
    """Support/OOD directional check across the four PRR-space campaigns.

    Uses each campaign's RANDOM-split model (its held-out random-test rows are
    same-distribution, so 'in-distribution epistemic' means what it says; the
    temporal test slice is already drift-shifted by the BO loop and would blur
    the contrast). OOD = ALL rows of another PRR campaign. Directional claim
    only: mean epistemic sigma OOD > in-distribution, per output.

    CAVEAT (also stored in the result): the four campaigns share the same 5-D
    PRR knob space but differ in material (Al/Ti) and power tier -- this is
    cross-campaign shift, not adversarial far-OOD.
    """
    pairs = []
    n_pass = 0
    for src in PRR_SLUGS:
        a = artifacts[src]
        model: GPForwardModel = a["model_random"]
        Xt = a["X_all"][a["test_idx_random"]]
        id_epi = a["id_epi_random"]  # (m,) mean epistemic on own held-out rows
        id_support = float(np.mean(model.support_score(Xt)))
        for tgt in PRR_SLUGS:
            if tgt == src:
                continue
            Xo = artifacts[tgt]["X_all"]
            ood_epi = np.asarray(model.predict(Xo).epistemic_sigma).mean(axis=0)
            ood_support = float(np.mean(model.support_score(Xo)))
            ok = bool(np.all(ood_epi > id_epi))
            n_pass += ok
            pairs.append({
                "model": src,
                "queried_on": tgt,
                "id_mean_epi": [float(v) for v in id_epi],
                "ood_mean_epi": [float(v) for v in ood_epi],
                "id_mean_support": id_support,
                "ood_mean_support": ood_support,
                "ood_epi_greater_both_outputs": ok,
            })
    result = {
        "design": "random-split model; ID = own held-out random-test rows; OOD = all rows of the other campaign",
        "output_order": list(artifacts[PRR_SLUGS[0]]["output_keys"]),
        "pairs": pairs,
        "n_pairs": len(pairs),
        "n_directional_pass": n_pass,
        "caveat": "shared 5-D PRR knob space, but different material (Al/Ti) and power tier -- cross-campaign shift, not adversarial far-OOD",
    }
    banner("SUPPORT / OOD DIRECTIONAL CHECK (4 PRR-space campaigns, 12 ordered pairs)")
    hdr = (f"{'model':<18}{'queried on':<18}{'ID epi (dep,Ipk)':>20}{'OOD epi (dep,Ipk)':>20}"
           f"{'ID supp':>9}{'OOD supp':>10}{'OOD>ID':>8}")
    print(hdr)
    print("-" * len(hdr))
    for p in pairs:
        ide = f"{p['id_mean_epi'][0]:.4f},{p['id_mean_epi'][1]:.3f}"
        oode = f"{p['ood_mean_epi'][0]:.4f},{p['ood_mean_epi'][1]:.3f}"
        print(f"{p['model']:<18}{p['queried_on']:<18}{ide:>20}{oode:>20}"
              f"{p['id_mean_support']:>9.2f}{p['ood_mean_support']:>10.2f}"
              f"{'yes' if p['ood_epi_greater_both_outputs'] else 'NO':>8}")
    print(f"\ndirectional pass: {n_pass}/{len(pairs)} ordered pairs "
          "(epistemic sigma larger on the other campaign's recipes, both outputs)")
    print("caveat: " + result["caveat"])
    return result


def inverse_demo(art: dict, solver_restarts: int | None) -> dict:
    """Section 8 pessimistic-inverse demo on Al-shortPW (random-split model).

    Three queries, three regimes:

    1. NARROW band = the train [0.60, 0.90] dep-rate quantile band. Populated
       territory, but its width (~0.17 Ang/s) is SMALLER than the section 8.4
       credited aleatoric floor 2*kappa*sigma_ale (~0.48 Ang/s at kappa=2), so
       NO recipe can ever satisfy the pessimistic margins -- the correct
       verdict is an explicit INFEASIBLE with a spec-relaxation diagnosis, and
       that abstention-on-an-over-tight-spec is itself the demonstration. The
       expectation is computed from the arithmetic, not hardcoded.
    2. WIDE populated band = the train [0.10, 0.90] quantile band (~0.6+
       Ang/s wide) -- expect FEASIBLE candidates with section 8 margins, each
       verified against the nearest measured run in normalized knob space.
    3. Beyond the observed max -- expect explicit INFEASIBLE, never an
       invented recipe.

    Model choice (documented judgment call): the RANDOM-split train model --
    the temporal-train model has deliberately never seen the late exploitation
    cluster (BO drift), which would conflate this machinery demo with the
    drift story. Target bands are in the readable declared unit (angstrom/s).
    """
    spec = art["spec"]
    model: GPForwardModel = art["model_random"]
    input_keys, output_keys = art["input_keys"], art["output_keys"]
    X_all, Y_all = art["X_all"], art["Y_all"]
    fit_idx = art["fit_idx_random"]
    dep_train = Y_all[fit_idx, 0]
    if output_keys[0] != DEP_RATE_COLUMN:
        raise RuntimeError("dep-rate is expected as output 0")

    si_vars = list(spec.continuous_si)  # SI bounds -- MUST pair with ingested X
    scale = np.array([v.upper - v.lower for v in si_vars])  # NN normalization

    solver = PessimisticInverseSolver(
        model,
        variables=si_vars,
        output_keys=output_keys,
        X_train=X_all[fit_idx],  # derives the section 8.2 support floor (fail-closed)
        n_restarts=solver_restarts,
        seed=SEED,
    )

    banner(f"PESSIMISTIC INVERSE DEMO on {DEMO_SLUG} (implementation-plan section 8)")
    print("NB recipe values are SI (section 3.5): PW / pos Delay / pos PW are in "
          "SECONDS despite the '(us)' in the names (1 us = 1e-6 s); PRR in Hz, "
          "pos Setpoint in V. Targets are in angstrom/s (the declared output unit).")

    demo: dict = {"campaign": DEMO_SLUG, "model": "random-split train model",
                  "target_output": DEP_RATE_COLUMN, "unit": "angstrom/second"}

    # The section 8.4 arithmetic that separates regimes 1 and 2: a band can
    # only ever be pessimistic-feasible if it is wider than the credited
    # aleatoric floor 2*kappa*sigma_ale (necessary, not sufficient).
    sigma_ale = float(model.noise_std_[0])
    min_width = 2.0 * solver.kappa * sigma_ale
    print(f"\ncredited-band floor: 2*kappa*sigma_ale = 2*{solver.kappa:.1f}*{sigma_ale:.4f} "
          f"= {min_width:.4f} Ang/s -- any narrower target band is INFEASIBLE by construction")

    # -- query 1: populated but TOO-TIGHT band (honest abstention demo) -------
    n_lo, n_hi = (float(q) for q in np.quantile(dep_train, [0.60, 0.90]))
    narrow_expected = "INFEASIBLE" if (n_hi - n_lo) < min_width else "FEASIBLE"
    print(f"\nquery 1 (populated, over-tight): dep rate in [{n_lo:.4f}, {n_hi:.4f}] Ang/s "
          f"(train [0.60, 0.90] quantile band; width {n_hi - n_lo:.4f} < floor "
          f"{min_width:.4f}) -- expect {narrow_expected} with a spec-relaxation diagnosis")
    out_n = solver.solve({"targets": {DEP_RATE_COLUMN: (n_lo, n_hi)}, "max_candidates": 3})
    qn: dict = {"targets": [n_lo, n_hi], "quantiles": [0.60, 0.90],
                "band_width": n_hi - n_lo, "credited_floor_2_kappa_sigma_ale": min_width,
                "expected": narrow_expected}
    if isinstance(out_n, Infeasible):
        qn["observed"] = "INFEASIBLE"
        qn["reason"] = out_n.reason
        qn["distance_to_feasible"] = float(out_n.distance_to_feasible)
        print("verdict: INFEASIBLE -- the pessimism refuses a band narrower than what it")
        print("         can credit at kappa=2; this abstention is the correct behavior")
        print("reason :", out_n.reason)
    else:
        qn["observed"] = "FEASIBLE"
        qn["candidates_returned"] = len(out_n)
        print(f"verdict: FEASIBLE ({len(out_n)} candidates)")
    qn["as_expected"] = qn["observed"] == qn["expected"]
    demo["query_populated_overtight"] = qn

    # -- query 2: clearly-populated AND credit-wide band -> expect FEASIBLE ---
    q_lo, q_hi = (float(q) for q in np.quantile(dep_train, [0.10, 0.90]))
    print(f"\nquery 2 (populated, credit-wide): dep rate in [{q_lo:.4f}, {q_hi:.4f}] Ang/s "
          "(train [0.10, 0.90] quantile band) -- expect FEASIBLE")
    out = solver.solve({"targets": {DEP_RATE_COLUMN: (q_lo, q_hi)}, "max_candidates": 3})
    q1: dict = {"targets": [q_lo, q_hi], "quantiles": [0.10, 0.90], "expected": "FEASIBLE"}
    if isinstance(out, Infeasible):
        q1["observed"] = "INFEASIBLE"
        q1["reason"] = out.reason
        q1["distance_to_feasible"] = float(out.distance_to_feasible)
        q1["nearest_achievable"] = {k: float(v) for k, v in out.nearest_achievable.items()}
        print("verdict: INFEASIBLE (NOT as expected)")
        print("reason           :", out.reason)
    else:
        q1["observed"] = "FEASIBLE"
        q1["candidates"] = []
        print(f"verdict: FEASIBLE  ({len(out)} on-support candidate(s))")
        for i, cand in enumerate(out):
            x_cand = np.array([float(cand.recipe[k]) for k in input_keys])
            # nearest MEASURED run in normalized knob space (all campaign rows)
            d = np.linalg.norm((X_all - x_cand) / scale, axis=1)
            nn = int(np.argmin(d))
            nn_dep = float(Y_all[nn, 0])
            nn_in_band = bool(q_lo <= nn_dep <= q_hi)
            iv = {k: (float(a), float(b)) for k, (a, b) in cand.predicted_outcome_interval.items()}
            q1["candidates"].append({
                "recipe_si": {k: float(v) for k, v in cand.recipe.items()},
                "confidence": float(cand.confidence),
                "support_score": float(cand.support_score),
                "credited_interval": iv,
                "nn_distance_normalized": float(d[nn]),
                "nn_measured_dep_rate": nn_dep,
                "nn_in_target_band": nn_in_band,
            })
            rc = {k: f"{v:.4g}" for k, v in cand.recipe.items()}
            print(f"  #{i}: recipe(SI)={rc}")
            print(f"       confidence={cand.confidence:.3f}  support={cand.support_score:.3f}  "
                  f"credited dep interval={iv[DEP_RATE_COLUMN][0]:.3f}..{iv[DEP_RATE_COLUMN][1]:.3f} Ang/s")
            print(f"       nearest measured run: dist={d[nn]:.3f} (normalized), measured "
                  f"dep rate={nn_dep:.4f} Ang/s, in target band: {nn_in_band}")
    q1["as_expected"] = q1["observed"] == q1["expected"]
    demo["query_populated"] = q1

    # -- query 3: strictly above the observed max -> must abstain -------------
    y_max = float(Y_all[:, 0].max())
    t_lo, t_hi = 1.5 * y_max, 2.0 * y_max
    print(f"\nquery 3 (beyond data): dep rate in [{t_lo:.4f}, {t_hi:.4f}] Ang/s "
          f"(observed max {y_max:.4f}) -- expect explicit INFEASIBLE, not an invented recipe")
    out2 = solver.solve({"targets": {DEP_RATE_COLUMN: (t_lo, t_hi)}, "max_candidates": 3})
    q2: dict = {"targets": [t_lo, t_hi], "observed_max": y_max, "expected": "INFEASIBLE"}
    if isinstance(out2, Infeasible):
        q2["observed"] = "INFEASIBLE"
        q2["reason"] = out2.reason
        q2["distance_to_feasible"] = float(out2.distance_to_feasible)
        q2["nearest_achievable"] = {k: float(v) for k, v in out2.nearest_achievable.items()}
        print("verdict: INFEASIBLE (as it must be)")
        print("nearest recipe (SI):", {k: f"{v:.4g}" for k, v in out2.nearest_achievable.items()})
        print("distance-to-feasible:", round(float(out2.distance_to_feasible), 3))
        print("reason              :", out2.reason)
    else:
        q2["observed"] = "FEASIBLE"
        q2["candidates_returned"] = len(out2)
        print(f"verdict: FEASIBLE ({len(out2)} candidates) -- FALSE SUCCESS, investigate!")
    q2["as_expected"] = q2["observed"] == q2["expected"]
    demo["query_beyond_data"] = q2
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", choices=[c.slug for c in CAMPAIGNS], default=None,
                        help="run ONE campaign (default: all six)")
    parser.add_argument("--smoke", action="store_true",
                        help="reduced restarts (GP 2, solver 24) for a fast shape check")
    parser.add_argument("--out", type=Path, default=None,
                        help="results JSON path (default: results/m1_empa[.<slug>][.smoke].json)")
    args = parser.parse_args(argv)

    np.random.seed(SEED)
    gp_restarts = 2 if args.smoke else 5
    solver_restarts = 24 if args.smoke else None  # None -> the solver's 24*dim default
    selected = [c for c in CAMPAIGNS if args.campaign in (None, c.slug)]

    out_path = args.out
    if out_path is None:
        stem = "m1_empa" + (f".{args.campaign}" if args.campaign else "")
        out_path = RESULTS_DIR / f"{stem}{'.smoke' if args.smoke else ''}.json"

    banner("M1 GATE FORM on REAL Empa HiPIMS data -- NOT the signed M1 program gate")
    print("dataset : Zenodo 10.5281/zenodo.18495402 (CC-BY-4.0), real Empa sputter tool")
    print("caveats : BO-clustered sampling (exchangeability approximate); temporal split")
    print("          doubles as a drift stress test; M0 venue choice is the user/PI's.")
    print(f"mode    : {'SMOKE (reduced restarts)' if args.smoke else 'full'};"
          f"  campaigns: {[c.slug for c in selected]}")

    t_start = time.perf_counter()
    summaries: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    for idx, campaign in enumerate(CAMPAIGNS):
        if campaign not in selected:
            continue
        summary, art = run_campaign(campaign, idx, gp_restarts)
        summaries[campaign.slug] = summary
        artifacts[campaign.slug] = art

    ood = None
    if all(s in artifacts for s in PRR_SLUGS):
        ood = ood_check(artifacts)
    else:
        print("\n[OOD check skipped: needs all four PRR-space campaigns in one run]")

    demo = None
    if DEMO_SLUG in artifacts:
        demo = inverse_demo(artifacts[DEMO_SLUG], solver_restarts)

    # -- gate summary + machine-readable results ------------------------------
    banner("GATE SUMMARY (directional: nominal 0.90 inside the exact binomial 95% CI)")
    hdr = f"{'campaign':<20}{'split':<15}{'pooled PICP':>12}{'95% CI':>17}{'gate':>6}"
    print(hdr)
    print("-" * len(hdr))
    for slug, s in summaries.items():
        for split_name in ("temporal", "random"):
            sp = s["splits"][split_name]
            tag0 = split_name
            if split_name == "temporal" and not s["temporal_split_meaningful"]:
                tag0 = "temporal*"
            for tag, p in ((tag0, sp["pooled"]), (tag0 + "+ACI", sp["aci"]["pooled"])):
                ci = f"[{p['ci95'][0]:.3f},{p['ci95'][1]:.3f}]"
                gate = "PASS" if p["nominal_in_ci"] else "FAIL"
                print(f"{slug:<20}{tag:<15}{p['picp']:>12.3f}{ci:>17}{gate:>6}")
    if any(not s["temporal_split_meaningful"] for s in summaries.values()):
        print("* ti_120w_short_pw order key is UNVERIFIED file order (BatchNr degenerate)")
    print("+ACI rows: online (D4/section 5.6) realized coverage, ACI library defaults;")
    print("           asymptotic-average guarantee only -- the CI row is directional")

    payload = {
        "meta": {
            "seed": SEED,
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL,
            "ci": f"exact (Clopper-Pearson) binomial, level {CI_LEVEL}",
            "gate_form": "directional: nominal inside CI (implementation-plan 15.3 M1 row), not a hard +/-2%",
            "smoke": args.smoke,
            "gp_restarts": gp_restarts,
            "solver_restarts": "24*dim default" if solver_restarts is None else solver_restarts,
            "campaigns_run": [c.slug for c in selected],
            "caveats": [
                "REAL measured data (real_tool) but NOT the signed M1 program gate: M0 venue is a user/PI decision and this is not the MBE target process",
                "BO-driven (BayBE) sampling: rows cluster near optima; split-conformal exchangeability is an approximation; the temporal split doubles as a drift stress test",
                "pooled coverage CI treats 2*n_test trials as independent although outputs share test rows (optimistic)",
                "ti_120w_short_pw: temporal order key is unverified file order; 5 rows skip-rejected (3e-11 to 4e-11 outside full-precision bounds)",
                "Ipk (A) is measured, never set",
            ],
            "aci": {
                "status": (
                    "ADDITIONAL online evaluation (D4/section 5.6); the static "
                    "split-conformal blocks are the unchanged baseline"
                ),
                "guarantee": (
                    "asymptotic AVERAGE coverage under arbitrary distribution shift "
                    "(Gibbs & Candes 2021); NOT finite-sample exact -- the binomial-CI row "
                    "on realized online coverage is directional, same status as the static gate"
                ),
                "hyperparameters": (
                    "ACIController library defaults (gamma=0.05, window=50, "
                    "alpha_clip=(0.001, 0.5), update_scores=True), uniform across all "
                    "campaigns and splits, fixed before any per-campaign outcome was seen"
                ),
                "method_currency": (
                    "implementation-plan section 20.2 makes conformal-PID the online "
                    "endpoint with bare ACI a component; this validates the D4 ACI "
                    "component only"
                ),
            },
        },
        "campaigns": summaries,
        "ood_check": ood,
        "inverse_demo": demo,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    total = time.perf_counter() - t_start
    print(f"\nresults written -> {out_path}")
    print(f"total wall time : {total:.1f} s")
    banner("DONE (M1 gate FORM on real Empa HiPIMS data -- venue sign-off is the PI's)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
