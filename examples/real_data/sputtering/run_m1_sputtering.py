"""M1-MACHINERY validation on REAL public magnetron-sputtering data.

Runs the RIG forward + conformal-calibration stack (and a small pessimistic
inverse demo) end-to-end on a genuine measured dataset ingested through the
generic WP-H tabular adapter -- NOT the in-silico machine.

    Dataset : Zr_grid.csv, https://github.com/jarlsanna/gps-for-magnetron-sputtering
              (self-driving-lab magnetron sputtering, single material Zr; a dense
              15x15 power x pressure grid). We keep only the 209 MEASURED rows
              (synthetic flag == 0), dropping the 16 synthetic-augmented ones.

HONEST FRAMING (read this): this is a "the machinery works on genuine measured
data" proof. It is NOT the RIG M1 program gate. The real M1 gate requires the
project's real MBE target-process data, which does not exist yet (that is M0,
still open). Never read the numbers below as "M1 passed".

Run:  PYTHONPATH=src python examples/real_data/sputtering/run_m1_sputtering.py
Deterministic: every stochastic step is seeded (SEED below).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

# The §8 solver's diagnostic strings contain Greek (kappa/sigma) and section
# signs; force UTF-8 so a cp1252 Windows console does not crash on them.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.forward import GPForwardModel, records_to_arrays
from rig.interfaces import Infeasible
from rig.inverse.pessimistic import PessimisticInverseSolver
from rig.metrics import uq
from rig.schema import ureg  # the ONE shared pint registry — never a second one
from rig_adapters.tabular.ingest import ingest_csv
from rig_adapters.tabular.spec import load_spec

SEED = 0
ALPHA = 0.10  # conformal miscoverage target -> nominal 90% coverage
HERE = Path(__file__).resolve().parent
CSV = HERE / "Zr_grid.csv"
SPEC = HERE / "sputtering.toml"

# ng/(cm^2 s) -> SI kg/(m^2 s): report everything back in the readable raw unit.
RATE_SI_PER_RAW = float(ureg.Quantity(1.0, "ng/(cm^2*s)").to_base_units().magnitude)


def banner(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> int:
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    # ---- 1. load spec + ingest the real CSV through the tabular adapter -----
    spec = load_spec(SPEC)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # adapter warns about unmatched columns (expected)
        result = ingest_csv(CSV, spec, source="real_tool", on_error="raise")
    all_records = result.records

    # keep only genuinely MEASURED rows (synthetic flag lives in the unmatched
    # columns the adapter parked in RunRecord.extra).
    def is_real(r) -> bool:
        return str(r.extra.get("unmatched_columns", {}).get("synthetic", "0")) == "0"

    records = [r for r in all_records if is_real(r)]
    banner("1. INGEST (real public data via the generic tabular adapter)")
    print(f"CSV                : {CSV.name}")
    print(
        f"rows ingested      : {len(all_records)}  (unmatched cols kept in extra: "
        f"{list(result.unmatched_columns)})"
    )
    print(
        f"measured rows kept : {len(records)}  (dropped {len(all_records) - len(records)} "
        f"synthetic-augmented rows)"
    )
    print(
        f"provenance.source  : {records[0].provenance.source}  (headline metrics only on real_tool)"
    )

    input_keys = list(spec.gp_input_keys)
    output_keys = list(spec.output_names)
    X, Y_si = records_to_arrays(records, input_keys, output_keys)
    Y = Y_si / RATE_SI_PER_RAW  # work/report in raw ng/(cm^2 s)

    # per-point measurement sigma (aleatoric) from the CSV *_error columns.
    err = np.array(
        [[float(r.extra["unmatched_columns"][f"{k}_error"]) for k in output_keys] for r in records]
    )

    # ---- 2. seeded fit / calibration / test split --------------------------
    n = len(records)
    perm = rng.permutation(n)
    n_test = int(round(0.20 * n))
    n_cal = int(round(0.20 * n))
    test_idx = perm[:n_test]
    cal_idx = perm[n_test : n_test + n_cal]
    fit_idx = perm[n_test + n_cal :]
    Xf, Yf = X[fit_idx], Y[fit_idx]
    Xc, Yc = X[cal_idx], Y[cal_idx]
    Xt, Yt = X[test_idx], Y[test_idx]
    banner("2. SEEDED SPLIT (fit / calibration / test)")
    print(f"fit={len(fit_idx)}  calibration={len(cal_idx)}  test={len(test_idx)}  (seed={SEED})")

    # ---- 3. fit the exact-GP forward model (raw units) ---------------------
    model = GPForwardModel(input_keys=input_keys, output_keys=output_keys, seed=SEED).fit(Xf, Yf)
    banner("3. GP FORWARD MODEL (Matern-5/2 + ARD, per-output; implementation-plan §5.2)")
    print("fitted aleatoric noise_std_ (ng/cm2s):", np.array2string(model.noise_std_, precision=4))
    print("CSV mean measurement error (ng/cm2s) :", np.array2string(err.mean(axis=0), precision=4))
    print("ARD lengthscales per output [pow(W), press(mtorr-SI)]:")
    for k, ls in zip(output_keys, model.lengthscales_, strict=True):
        print(f"  {k}: {np.array2string(ls, precision=3)}")

    # ---- 4. conformal calibration + test-set metrics -----------------------
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    conf = ConformalForwardModel(model, cal)

    dist = conf.predict(Xt)  # batch PredictiveDistribution
    mu = np.asarray(dist.mean)  # (n_test, m)
    ale = np.asarray(dist.aleatoric_sigma)
    epi = np.asarray(dist.epistemic_sigma)
    sig_total = np.sqrt(ale**2 + epi**2)
    cset = np.asarray(dist.conformal_set)  # (n_test, m, 2)  conformal band
    lo, hi = cset[..., 0], cset[..., 1]

    rmse = uq.rmse(mu, Yt)
    mae = uq.mae(mu, Yt)
    crps = np.mean(uq.crps_gaussian(mu, sig_total, Yt), axis=0)
    picp = uq.picp(lo, hi, Yt)  # conformal coverage (target 90%)
    mpiw = uq.mpiw(lo, hi)
    qce = uq.quantile_calibration_error(mu, sig_total, Yt)
    pit = uq.pit_values(mu, sig_total, Yt)  # (n_test, m)
    ks = np.array([stats.kstest(pit[:, j], "uniform").statistic for j in range(pit.shape[1])])
    y_rng = Yt.max(axis=0) - Yt.min(axis=0)

    banner(
        f"4. TEST-SET METRICS  (n_test={len(test_idx)}; conformal target coverage "
        f"{100 * (1 - ALPHA):.0f}%)"
    )
    hdr = f"{'output':<28}{'RMSE':>9}{'nRMSE%':>8}{'MAE':>9}{'CRPS':>9}{'PICP':>8}{'MPIW':>9}{'QCE':>7}{'PIT-KS':>8}"
    print(hdr)
    print("-" * len(hdr))
    for j, k in enumerate(output_keys):
        nrmse = 100.0 * rmse[j] / y_rng[j]
        print(
            f"{k:<28}{rmse[j]:>9.4f}{nrmse:>8.2f}{mae[j]:>9.4f}{crps[j]:>9.4f}"
            f"{picp[j]:>8.2f}{mpiw[j]:>9.3f}{qce[j]:>7.3f}{ks[j]:>8.3f}"
        )
    print(
        "\nunits: RMSE/MAE/CRPS/MPIW in ng/(cm^2 s); PICP is fraction covered "
        f"(nominal {1 - ALPHA:.2f}); nRMSE% = RMSE / test-range."
    )
    print(f"mean conformal PICP over 3 outputs: {picp.mean():.3f}  (nominal {1 - ALPHA:.2f})")
    print(
        "aleatoric ballpark  predicted noise_std / CSV mean error (ratio):",
        np.array2string(model.noise_std_ / err.mean(axis=0), precision=2),
    )

    # ---- 5. small pessimistic inverse demo ---------------------------------
    banner("5. PESSIMISTIC INVERSE DEMO (implementation-plan §8)")
    # Construct a REACHABLE-BY-CONSTRUCTION target: predict qcm_1 at an interior
    # on-grid recipe (guaranteed on-support), then ask the solver to recover a
    # recipe for a generous band around that predicted rate. The solver is NOT
    # told the recipe -- only the target band.
    target_key = output_keys[0]
    # UNITS (audit 2026-07-17 — this example carried a live units defect): ingest
    # SI-canonicalizes every value, so `Xf` pressure is in Pa (0.133..5.73), NOT the
    # declared mTorr. Both the reference point and the solver's bounds must be SI too.
    #   - `x_ref` was `[25.0, 13.0]` commented "25 W, 13 mtorr ... (guaranteed
    #     on-support)". 13 was read as 13 Pa = 98 mTorr — 2.3x beyond the measured
    #     maximum, support_score -5.761 against a floor of -2.044, i.e. the exact
    #     opposite of on-support, and the "reference" rate was a GP extrapolation.
    #   - `variables=list(spec.continuous)` handed the solver declared-unit bounds
    #     1..43, so it searched pressure over 1..43 Pa = 7.5..322 mTorr, a range
    #     whose LOWER bound sits above the data's maximum. Only the §8.2 fail-closed
    #     support floor kept the returned recipes sane; nothing errored.
    # `spec.continuous_si` is the SI-canonical accessor that pairs with ingested data.
    p_si = ureg.Quantity(13.0, "mtorr").to_base_units().magnitude  # 13 mTorr -> ~1.733 Pa
    x_ref = np.array([25.0, p_si])  # interior grid point: 25 W, 13 mTorr
    m1_ref = float(np.atleast_1d(model.predict(x_ref).mean)[0])  # predicted qcm_1 rate
    tol = 6.0
    target_lo_raw, target_hi_raw = m1_ref - tol, m1_ref + tol
    solver = PessimisticInverseSolver(
        model,
        variables=list(spec.continuous_si),  # SI bounds — MUST match the ingested X
        output_keys=output_keys,
        X_train=Xf,  # derives the §8.2 support floor (fail-closed)
        seed=SEED,
    )
    spec_query = {"targets": {target_key: (target_lo_raw, target_hi_raw)}, "max_candidates": 3}
    print(
        f"reference recipe 25 W / 13 mTorr ({p_si:.3f} Pa), support_score="
        f"{model.support_score(x_ref):.3f} -> predicted {target_key} = {m1_ref:.2f} ng/(cm^2 s)"
    )
    print(
        f"target: {target_key} in [{target_lo_raw:.2f}, {target_hi_raw:.2f}] ng/(cm^2 s) (tol +/-{tol})"
    )
    # The variable NAME is the CSV header verbatim, so it carries the dataset's own
    # '[mTorr]' label while the VALUE is SI (Pa) like everything post-ingest. Say so
    # rather than let a reader take 'set_pressure_[mTorr]: 1.82' at face value.
    print(
        "NB recipe values are SI (§3.5): pressure is in Pa despite the column "
        "name's '[mTorr]' label — 1 mTorr = 0.1333 Pa, data spans 0.13-5.73 Pa."
    )
    out = solver.solve(spec_query)
    if isinstance(out, Infeasible):
        print("verdict: INFEASIBLE")
        print("nearest recipe   :", {k: round(v, 3) for k, v in out.nearest_achievable.items()})
        print("distance (sigma) :", round(out.distance_to_feasible, 3))
        print("reason           :", out.reason)
    else:
        print(f"verdict: FEASIBLE  ({len(out)} on-support candidate(s))")
        for i, cand in enumerate(out):
            rc = {k: round(v, 2) for k, v in cand.recipe.items()}
            iv = {
                k: (round(a, 2), round(b, 2))
                for k, (a, b) in cand.predicted_outcome_interval.items()
            }
            print(
                f"  #{i}: recipe={rc}  confidence={cand.confidence:.3f}  "
                f"support={cand.support_score:.3f}"
            )
            print(f"       worst-case credited interval (ng/cm2s): {iv}")

    banner("DONE  (M1-MACHINERY validation on real data -- NOT the M1 program gate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
