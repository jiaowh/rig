# M1-machinery validation on REAL public sputtering data

**What this is:** a proof that RIG's forward-model + conformal-calibration
machinery (plus a small pessimistic inverse demo) runs end-to-end on a *genuine
measured* experimental dataset, ingested through the generic WP-H `tabular`
adapter — the same code path the in-silico MBE work uses, but here fed real data.

**What this is NOT (read carefully):** this is **not** the RIG **M1 program
gate**. The M1 gate (implementation-plan §15.3) requires a calibrated forward model on the
project's *real MBE target-process* data — which does not exist yet (that is
**M0**, still open, the #1 program risk). Nothing here should be read as "M1
passed". This is strictly a *machinery-works-on-real-data* validation on an
unrelated public sputtering process.

## Dataset provenance

- **Source:** `Zr_grid.csv` from
  <https://github.com/jarlsanna/gps-for-magnetron-sputtering> (a self-driving-lab
  study of magnetron sputtering; single material **Zr**). Downloaded from the
  `main` branch raw URL into `examples/real_data/sputtering/Zr_grid.csv`.
- **License:** none stated in the repo. Treated as an **on-prem internal proof
  only** — the file is kept locally, nothing is pushed/redistributed, and no
  recipe values are logged to any cloud service (repo egress guard, implementation-plan §17).
- **Shape:** 225 rows = a **complete 15×15 grid** of source power × chamber
  pressure, both on `{1, 4, 7, …, 43}`. A `synthetic` flag marks **16**
  GP-augmented rows; we drop those and keep the **209 measured rows** only.

### Real column names (verified against the actual file header)

| Role | CSV column | Declared unit → SI |
|---|---|---|
| Input (power) | `source_1_set_power_[W]` | `W` (watt) |
| Input (pressure) | `set_pressure_[mTorr]` | `mtorr` → Pa (×0.1333) |
| Output 1 | `qcm_1_mass_rate_[ng/cm2s]` | `ng/(cm^2*s)` → kg/(m²·s) (×1e-8) |
| Output 2 | `qcm_2_mass_rate_[ng/cm2s]` | `ng/(cm^2*s)` |
| Output 3 | `qcm_3_mass_rate_[ng/cm2s]` | `ng/(cm^2*s)` |
| Per-point aleatoric σ | `qcm_{1,2,3}_mass_rate_[ng/cm2s]_error` | (same rate unit) |
| Other (unused) | `source_1_voltage_[V]_mean/_std`, `n`, `n_error`, `A`, `A_error`, `synthetic` | — |

The three QCM (quartz-crystal-microbalance) columns are measured deposition
rates at three positions; each has a matching `_error` column (measurement σ).
Rates are slightly **negative** at the lowest powers (QCM noise near zero) — the
outputs carry no lower bound, so those rows ingest fine. Note `mTorr` is not a
pint unit; the pint spelling `mtorr` is declared in the spec (same magnitude).

## What was done (pipeline)

1. **Spec** — `sputtering.toml`: two continuous inputs (bounds = observed grid
   extents 1–43), three `scalar_vector` rate outputs. The `_error`/voltage/`n`/`A`/
   `synthetic` columns are *not* declared, so the adapter parks them in
   `RunRecord.extra["unmatched_columns"]` (with a warning). The run script reads
   `synthetic` from there to drop the 16 augmented rows, and reads the `_error`
   columns for the aleatoric ballpark check.
2. **Ingest** — `ingest_csv(...)` → 225 validated, SI-canonical `RunRecord`s
   (provenance `real_tool`); filtered to 209 measured rows.
3. **Arrays** — `records_to_arrays(records, spec.gp_input_keys, spec.output_names)`.
   Metrics are reported back in the readable raw unit ng/(cm²·s).
4. **Fit** — one exact GP per output (Matérn-5/2 + ARD, implementation-plan §5.2) on the fit split.
5. **Calibrate** — `SplitConformalCalibrator(alpha=0.10)` on a held-out
   calibration split; wrapped in `ConformalForwardModel`.
6. **Evaluate** — on a **seeded** (seed=0) random fit/calibration/test split
   (125 / 42 / 42).
7. **Inverse demo** — `PessimisticInverseSolver` (implementation-plan §8) on a
   reachable-by-construction target band.

Everything is seeded; **output is bit-identical across repeated runs** (verified).

## Results (observed; seed=0, `run_m1_sputtering.py`)

### Forward accuracy + calibration — test set (n=42), conformal nominal 90%

| Output | RMSE | nRMSE% | MAE | CRPS | **PICP** | MPIW | QCE | PIT-KS |
|---|---|---|---|---|---|---|---|---|
| `qcm_1_mass_rate` | 2.769 | 6.8% | 1.048 | 0.875 | **0.95** | 6.657 | 0.069 | 0.207 |
| `qcm_2_mass_rate` | 0.453 | 11.0% | 0.275 | 0.211 | **0.88** | 1.126 | 0.032 | 0.133 |
| `qcm_3_mass_rate` | 0.816 | 5.5% | 0.433 | 0.334 | **0.95** | 1.995 | 0.032 | 0.094 |

Units: RMSE / MAE / CRPS / MPIW in ng/(cm²·s). nRMSE% = RMSE ÷ test-set range.
PICP = fraction of test points inside the conformal band (target 0.90).

- **Point accuracy** is good for a 125-point real fit: nRMSE ≈ 5–11% of range.
- **Conformal coverage** lands at **0.95 / 0.88 / 0.95, mean 0.929** vs the 0.90
  nominal — i.e. the split-conformal bands are well-calibrated (slightly
  conservative, as expected for split conformal at this n). QCE and PIT-KS are
  small, consistent with an approximately-calibrated Gaussian predictive.

### Aleatoric-σ ballpark

The GP's fitted constant noise floor `noise_std_` = **[0.879, 0.202, 0.301]**
ng/(cm²·s), versus the CSV's mean per-point measurement error **[0.084, 0.051,
0.058]**. The fitted floor is **~4–10× larger** than the QCM counting error.
That is the honest and expected outcome: the v0 aleatoric floor (implementation-plan §10.3)
is one constant per output that absorbs *everything not explained by the mean* —
run-to-run variation, position/grid-interpolation misfit, and genuine
heteroscedasticity — not just the QCM's reported measurement σ. It is the right
order of magnitude (sub-unit ng/(cm²·s)), just not identical to the pure
metrology error; separating the two needs replicates (heteroscedastic aleatoric
is deferred).

### Inverse demo (implementation-plan §8 pessimistic solver)

- Target constructed to be reachable-on-mean: predicted `qcm_1` at an interior
  grid recipe (25 W / 13 mtorr) = **8.56** ng/(cm²·s); target band = **[2.56,
  14.56]** (±6).
- **Verdict: FEASIBLE** — 3 diverse **on-support** candidates returned (support
  scores −1.0…−2.0, above the §8.2 fail-closed floor), each with its worst-case
  credited outcome interval ⊆ the target box and confidence ≈ 1.0:

  | # | power [W] | pressure [mtorr] | credited `qcm_1` interval |
  |---|---|---|---|
  | 0 | 42.98 | 4.98 | (5.47, 13.27) |
  | 1 | 10.98 | 1.01 | (3.17, 13.83) |
  | 2 | 32.09 | 4.05 | (3.45, 12.74) |

  The solver was told only the target band (not the recipe) and recovered a
  *diverse* pre-image set — a real demonstration of the §8.7 non-injectivity
  handling (several distinct power/pressure recipes land the same rate band).

  (For reference: a tighter earlier target — `qcm_1 ∈ [12, 20]` — returned a
  well-formed **INFEASIBLE** verdict with a §8.8 cause diagnosis and per-output
  relaxation, which is also correct behavior; the machinery reports both
  outcomes honestly.)

## Honest caveats

- **Not the M1 gate.** Different process (magnetron sputtering, not the project's
  MBE target), and the point is machinery validation, not a program milestone.
- **Small & single-material.** 209 measured points, one material (Zr), a clean
  dense grid — an easy, low-noise regression problem; real MBE campaign data will
  be sparser, noisier, drift-prone, and higher-dimensional.
- **No license.** Public repo with no stated license → kept strictly on-prem,
  never redistributed; treat as a demonstration input only.
- **Aleatoric ≠ metrology σ.** The v0 constant noise floor over-estimates the QCM
  counting error (see above) — by design at this modeling tier.
- **Synthetic rows dropped.** Only `synthetic == 0` rows are used, so headline
  numbers are on genuinely measured data.
- **Metrics are one seeded split.** A single seed=0 fit/cal/test split (not a
  cross-validated sweep); reproducible, but not a powered multi-seed study.

## Files

- `sputtering.toml` — the WP-H process spec (real column names, units, bounds).
- `Zr_grid.csv` — the downloaded dataset (kept local, not redistributed).
- `run_m1_sputtering.py` — the deterministic end-to-end script (ingest → GP →
  conformal → metrics → inverse demo). Run:
  `PYTHONPATH=src python examples/real_data/sputtering/run_m1_sputtering.py`.
- `RESULTS.md` — this file.
