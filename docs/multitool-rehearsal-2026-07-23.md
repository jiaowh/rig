# In-silico multi-tool M4 dress rehearsal — 2026-07-23

> **⚠ REHEARSAL — provenance = `physics_sim` — NOT headline evidence.**
> Real multi-tool data does not exist yet. Every number in this document and in
> `docs/multitool-rehearsal.json` is produced by the WP-B in-silico `InSilicoMachine`
> (the fast Arrhenius sim path). This is a **machinery proof** that the whole M4
> story runs end-to-end and reproduces exactly; it is **not** a scientific result and
> is **not** headline-eligible. The M4/M5 claim stays gated on M0 real data. Every
> confirmation run below carries `provenance_source="physics_sim"` and therefore
> `headline_eligible=False` by construction — a §11.4 pre-filter rung, not tool
> qualification.

## What this exercises, and why now

The M4 story is: a *fleet* of tools that differ in ways the recipe vector cannot
see; a *pooled multi-tool surrogate* (ICM, §10.4) that borrows strength across the
fleet and abstains honestly on an unseen tool (§5.8); *EPIG-driven onboarding* of a
new chamber (§10.4); and a *solve → independent confirmation* qualification flow
(§8 + §11.4). None of that has been run together. This rehearsal wires it end-to-end
on the sim so integration bugs surface here instead of on real wafers, and it closes
the **"runner-level qualification auto-invocation"** crumb (BUILD_STATE, open after
2026-07-22): `examples/run_multitool_rehearsal.py` demonstrates
`ConfirmationCampaign.run(solver.solve(spec))` automatically.

Runner: `examples/run_multitool_rehearsal.py` · artifact: `docs/multitool-rehearsal.json`
· tests: `tests/test_multitool_rehearsal.py`. Full run **14.7 s** wall (well under the
~20-min rehearsal budget).

## Fleet design + perturbation justification

Three synthetic "tools" = **one** MBE process, perturbed to emulate as-built
chamber-to-chamber variation the learning stack must **detect** (never being told).
Two independent, sim-supported perturbation axes:

- **Hidden per-tool physics** via `PathologyConfig(tool_perturbation=True)` — the
  sim's own E3 pathology, keyed on `tool_id`: a fixed ±3 % multiplicative offset on
  (substrate emissivity, source `cosine_n`, effective flux). These stand for the
  run-to-run-**invisible** §10.2 chamber differences the RunRecords never carry:
  flux-cell calibration matched only to a few %, wall-coating emissivity history,
  beam-profile spread. ±3 % is the sim's default scale and a plausible chamber-match
  delta — **not tuned**.
- **Build geometry** via `machine_config` (the split-plot HARD_TO_CHANGE whole-plot
  factors): ±5–8 % mechanical build tolerances, symmetric round offsets chosen
  **before any outcome was measured**, all inside the sim's own
  `MACHINE_CONFIG_BOUNDS`.

| tool | geometry offsets | hidden (emissivity, cosine_n, flux) factors — *ground truth, never modelled* |
|------|------------------|------------------------------------------------------------------------------|
| toolA | nominal ("golden" reference chamber) | (0.9719, 0.9804, 0.9741) |
| toolB | gap +8 %, source_height −5 %, heater_radius +5 % | (0.9754, 1.0084, 1.0260) |
| **toolC** | gap −8 %, source_height +5 %, heater_radius −5 % · **held-out "new chamber"** | (0.9991, 0.9805, 0.9824) |

Two modelled outputs make **both** axes visible: `thickness_grown`
(= `film_thickness × flux_eff`, the flux-sensitive channel — the canonical WP-B
handoff KPI) and `T_center` (radiative balance — sensitive to emissivity **and**
geometry). At a fixed reference recipe the tool-to-tool signal sits well above the
metrology-noise floor, so the perturbations are learnable but not a different process:

| output | toolB–toolC gap | metrology noise σ | signal / noise |
|--------|-----------------|-------------------|----------------|
| thickness_grown | 1.09 × 10⁻⁷ m | 5.59 × 10⁻⁹ m | **19.5×** |
| T_center | 12.27 K | 1.95 K | **6.3×** |

## Per-phase results (full run)

### Phase 2 — pooled ICM fit + leave-one-tool-out

- **§5.8 zero-shot domination — all 3 folds PASS.** Hold out each tool in turn, fit
  the ICM on the other two, predict the held-out tool as **unknown**: its epistemic
  σ dominates every fitted tool's, elementwise on both outputs (the §5.8 guarantee by
  construction; the honest "I have never seen this chamber" signal). 3/3 folds.
- **Few-shot pooled vs from-scratch, equal n — pooling HELPS on this fleet** (thickness KPI,
  RMSE vs toolC machine ground truth):

  | K (fixed few-shot rows of C) | pooled warm thk-RMSE | scratch single-tool thk-RMSE | Δ (scratch−pooled) |
  |---|---|---|---|
  | 0 (zero-shot fallback) | 5.45 × 10⁻⁸ | — | — |
  | 10 | 3.78 × 10⁻⁹ | 4.55 × 10⁻⁹ | **+7.7 × 10⁻¹⁰ (pooled wins)** |
  | 20 | 2.39 × 10⁻⁹ | 3.28 × 10⁻⁹ | **+8.9 × 10⁻¹⁰ (pooled wins)** |

  **Honest verdict: HELPS** (mean scratch−pooled RMSE +8.3 × 10⁻¹⁰ m). At equal
  few-shot n the fleet prior sharpens the new tool. *(Note the smoke config, at n=8
  per tool / 1 restart, also lands HELPS at K=4; the result is not a full-config
  artifact.)*

### Phase 3 — new-chamber onboarding (§10.4, EPIG-driven)

Warm arm: warm-start from the pooled {A,B} model, C **unknown**; each batch selects q=4
runs by EPIG-heavy `cost_cooled_acquisition`, runs them on C, refits (`adapt_to_tool`).
Cold arm: a from-scratch single-tool GP seeded by a q=4 Sobol DoE, then EPIG/BALD-selected.
Same C-machine budget (24 runs); held-out thickness RMSE vs ground truth after each batch.

- **EPIG-collapse regression guard: PASS.** On the unknown-tool path EPIG = **2.97 nats**
  (> 0). This is the exact quantity the `posterior_cov`/`predict` law mismatch once
  collapsed to ~0 (audit 2026-07-17); it is now a standing green guard.
- **Runs-to-threshold (loose 1.5× full-data-ceiling target 3.93 × 10⁻⁹ m): TIE at 8 runs.**
  Trajectories (n_C_runs : thk-RMSE):
  - warm: 0:5.4e-8, 4:4.3e-8, **8:2.3e-9**, 12:2.0e-9, 16:6.0e-10, 20:1.0e-9, 24:**7.7e-10**
  - cold: 4:5.1e-9, **8:3.7e-9**, 12:4.4e-9, 16:4.3e-9, 20:3.0e-9, 24:**3.5e-9**
- **Honest verdict: runs-to-a-loose-threshold TIE, but warm start converges to a
  substantially sharper final model.** After the full budget the warm-started model
  reaches **7.7 × 10⁻¹⁰ m — ~4.5× sharper than cold (3.5 × 10⁻⁹ m) and below the
  single-tool full-data ceiling (2.6 × 10⁻⁹ m)**, because it also borrows the fleet's
  data. On this easy 2-D smooth problem a well-placed Sobol seed already clears the
  loose threshold, so *runs-to-threshold* is not discriminating; the *final model
  quality* is where warm start clearly pays. A higher-dim / costlier target (few runs
  not enough) is the regime where warm start's runs-to-threshold advantage would also
  show — the machinery to measure it is now in place.

### Phase 4 — solve + auto-qualification (the crumb)

Bind the onboarded tool (`model.for_tool("toolC")`); §8 solve under the **binding**
policy κ = z_epi = 2.0, delta_frac = 0.02.

- **(i) Direct auto-qualification on a reachable spec.** Spec anchored on an on-support
  reference recipe evaluated at C ground truth (a witness provably exists), tol = 8 % of
  range. Solve → **2 FEASIBLE candidates**; `ConfirmationCampaign.run(solve output)`
  auto-fires a 29-run confirmation batch per candidate on tool C's noisy machine:
  **2/2 certified, 58 confirmation runs**, each 29/29 in spec → Clopper-Pearson lower
  bound 0.9019 ≥ 0.90. `provenance_source=physics_sim`, `headline_eligible=False`.
- **(ii) Unreachable spec → NothingToQualify, ZERO machine calls.** Spec set above the
  achievable thickness → solve returns `Infeasible` ("genuinely unreachable: the
  predicted mean itself is outside the spec box … margin −401σ") → `campaign.run(Infeasible)`
  returns `NothingToQualify` having fired **0** verifier calls (asserted with a counting
  verifier).
- **(iii) `ActiveLearningLoop(qualification=…)` in-loop hook.** A short loop on tool C
  with the qualification hook wired: **hit=True**, stop_reason *"target met (proposal
  in-spec on machine, qualified)"* — i.e. a **solve-driven** proposal was confirmed —
  and the 10 confirmation runs were **charged to the loop budget** (`n_queries`=22).

**Why the direct flow is primary (justification).** `ConfirmationCampaign.run(solver.solve(spec))`
is the faithful runner-level "solve → qualify" flow the crumb asks for, **and** the only
one that produces `NothingToQualify` with zero calls on an `Infeasible` solve (the loop
hook only ever qualifies in-loop machine hits). The `ActiveLearningLoop` hook is exercised
**as well**, to demonstrate in-loop auto-invocation and budget charging.

## Integration observations (no src bugs found)

All four phases ran clean end-to-end; **no `src/` integration bug surfaced.** Two
observations worth recording (neither is a defect):

1. **Onboarded-tool candidates are `calibration_status="model-feasible"`, not
   `"conformal-checked"`.** The multi-tool onboarded view (`ToolBoundForwardModel`) is
   not conformal-wrapped, so `predict().conformal_set is None` and the solver's
   default-on §13.2 `C(x)⊆Z*` gate is (correctly) inert — candidates rest on the raw-σ
   κ margins alone. This is expected/correct behaviour, not a bug. The load-bearing
   independent check is the `ConfirmationCampaign` (D7 non-circularity), which does not
   consult the surrogate at all. A natural M4 enhancement is to wrap the onboarded tool
   view in a `ConformalForwardModel` calibrated on held-out C data **before** solving,
   which would upgrade the status to `"conformal-checked"`; it is out of scope here
   (needs a per-tool calibration split).
2. **The multi-tool model, its `for_tool` view, EPIG/BALD acquisition, the §8 solver,
   and the two qualification flows compose without any adapter glue.** The one historically
   fragile seam — `MultiToolGPForwardModel.posterior_cov` on the unknown-tool branch
   feeding `epig()` — is exercised on the live onboarding path and returns 2.97 nats
   (Phase 3 guard), confirming the 2026-07-17 fix holds under this integration.

## Tests + determinism evidence

`tests/test_multitool_rehearsal.py` (6 tests, 25 s):

- `test_smoke_end_to_end_all_phases` (sim-gated) — every phase produces its verdict.
- `test_smoke_is_deterministic` (sim-gated) — two smoke runs byte-identical modulo
  wall-clock `timings` (`strip_volatile`).
- `test_main_writes_json` (sim-gated) — CLI writes a well-formed artifact.
- `test_auto_qualification_fires_and_charges_budget` (sim-gated) — the loop hook fires
  and charges confirmation runs to budget; the direct flow fires `n_candidates × n_runs`.
- `test_epig_positive_on_unknown_tool_path` (**pure numpy, ungated**) — the EPIG > 0
  regression guard on a tiny 2-tool fleet.
- `test_epig_equals_bald_at_query_point_unknown_tool` (**pure numpy, ungated**) —
  `EPIG(x;{x}) == BALD(x)` on the unknown-tool branch (the self-consistent-law check).

Nothing on this path imports torch (numpy/scipy GP tier), so there is deliberately no
torch gate. Determinism verified out of band: both **smoke** and **full** runs are
byte-identical across two invocations (minus `timings`).

## Reproduce

```bash
# full run -> docs/multitool-rehearsal.json  (~15 s)
PYTHONIOENCODING=utf-8 python examples/run_multitool_rehearsal.py --full

# smoke shape check -> OS temp dir  (~12 s)
PYTHONIOENCODING=utf-8 python examples/run_multitool_rehearsal.py --smoke

# tests
python -m pytest tests/test_multitool_rehearsal.py -q
```

## Phase 4b — conformal wrap of the onboarded tool (added 2026-07-23, later same day)

**What this closes.** The "Integration observations" section above flagged, as an
explicit non-bug: onboarded-tool candidates in Phase 4 carry
`calibration_status="model-feasible"`, not `"conformal-checked"`, because the
onboarded tool's model view (`ToolBoundForwardModel`) is not conformal-wrapped —
`predict().conformal_set is None`, so the solver's default-on §13.2 `C(x)⊆Z*` gate
is (correctly) inert. Phase 4b exercises the enhancement that closes it: carve a
held-out calibration split from the onboarded tool's own runs, wrap its tool view
in the real `ConformalForwardModel`, and re-solve the **same** reachable spec — so
the rehearsal now exercises the full certified §13.2 path, not just the raw-σ κ
margins. Phases 1–4's recorded numbers are **byte-identical** to the block above
(verified programmatically — see "Determinism, extended" below); Phase 4b is
strictly additive and uses its own seed namespace.

### Split rule and the honest small-n_cal finding

A conformal calibration split must be **held out from the model's own fit data**
(the `SplitConformalCalibrator` leakage guard) — so Phase 4b does **not** reuse
Phase 4's fully-onboarded 24-run model. It fits a fresh tool view on a
**fit split** of the onboarded tool's runs and reserves the trailing third as a
**calibration split**: trailing 33% of the 24 accumulated tool-C runs →
`n_fit=16`, `n_cal=8` (chronological — the most-recently-acquired onboarding
batches are held out). A finite split-conformal quantile at `alpha=0.1` needs
`n_cal ≥ 9` (the order-statistic index `k = ceil(0.9·(n+1))` first satisfies
`k ≤ n` at `n=9`) — **so the natural split, at n_cal=8, is one run short.**

**Honest branch (n_cal=8, as naturally collected):** the calibrated quantile is
`kappa=[inf, inf]` — an honestly **infinite** band, no coverage claim possible.
Re-solving the reachable spec against the conformal-wrapped view then returns
`Infeasible` ("conformal-infeasible (§13.2)... the surrogate's raw σ is more
optimistic than its own conformal coverage") — **the gate rejects every
candidate**, including the one the raw κ·σ margins alone had admitted
(1/1 raw-admitted candidates rejected by the honest-branch gate — the interesting
event this phase was built to surface). `ConfirmationCampaign.run(Infeasible)`
correctly returns `NothingToQualify`, 0 machine calls.

**Extra-collection variant:** the onboarding loop collects the `9 − 8 = 1`
additional tool-C run needed for the minimal viable `n_cal=9`, via a plain
seeded Sobol/QMC draw (not EPIG-selected — calibration data should stay
exchangeable with the operating distribution, not actively chosen), charged as
1 extra machine call. With `n_cal=9`, `kappa=[1.462, 2.000]` — **finite** — and
the re-solve returns 1 FEASIBLE candidate now labelled
**`calibration_status="conformal-checked"`** — the upgrade this phase set out to
demonstrate. This time the gate does **not** reject the raw-admitted candidate
(0/1 rejected) — with just enough calibration data, the conformal check and the
raw κ margins agree. `ConfirmationCampaign` then certifies it in the same
29-run confirmation batch as Phase 4 (29/29 in spec, Clopper–Pearson lower bound
0.9019 ≥ 0.90).

### Band widths, raw vs. calibrated

At the probe recipe (the raw-admitted candidate's recipe):

| quantity | thickness_grown | T_center |
|---|---|---|
| raw pessimistic worst-case interval width (Phase 4b's reduced-data model) | 2.24 × 10⁻⁷ m | 20.83 K |
| conformal width, honest n_cal=8 | **∞** | **∞** |
| conformal width, extended n_cal=9 | 1.98 × 10⁻⁸ m | 7.92 K |

With enough calibration data the conformal band is **narrower** than the
pessimistic worst-case-credited interval on both outputs (kappa=1.46 < the
solver's z_epi=2.0 displacement plus the §8.5 δ term it stands in for) — i.e. the
calibrated check was not simply a stricter version of the same margin; it is an
empirically different (here tighter) statement about the same model.

### Budget accounting

24 onboarding runs already charged in Phase 3 (unchanged) + **1** extra
calibration-collection run charged here in Phase 4b. Total additional machine
calls for the whole conformal-wrap exercise: **1**.

### Determinism, extended

- Phases 1–4 JSON blocks (incl. `meta` minus `timings`) are **byte-identical**
  to the previously recorded full-run artifact — verified with a programmatic
  `json.dumps(..., sort_keys=True)` comparison against a copy of the artifact
  saved before this change, not by eyeballing a diff.
- Full run (with 4b): **37.8 s** wall (was 14.7 s before 4b), still well inside
  the ~20-minute rehearsal budget. Two independent full runs (before/after a
  `ruff format` pass) are byte-identical modulo `timings`.
- Two independent smoke runs (`--smoke`) are byte-identical modulo `timings`,
  including the new `phase4b_conformal_wrap` block.
- Smoke run hits the same qualitative story at smaller scale: `n_total=12`,
  `n_fit=8`, `n_cal_natural=4` (well short of `n_cal_min=9`), extra-collection
  needs 5 runs to reach 9, and the same honest-infeasible → conformal-checked
  upgrade occurs.

### Tests (extended)

`tests/test_multitool_rehearsal.py` grew from 6 to **10** tests (all green):

- `test_min_n_cal_matches_the_conformal_quantile_boundary` — pins the
  alpha=0.1 finite-quantile boundary at n=9 (pure numpy, ungated).
- `test_conformal_quantile_infinite_at_tiny_n_cal_synthetic` — unit-scale,
  synthetic, ungated: a tiny (n=4) calibration split at alpha=0.1 gives an
  honestly infinite band on a trivial stub model, and n=9 gives a finite one —
  isolates the honest-infinite-band mechanism from the sim entirely.
- `test_phase4b_conformal_wrap_smoke` (sim-gated) — the natural split is too
  small and INFEASIBLE/NothingToQualify; the extra-collection variant is
  FEASIBLE with `calibration_status="conformal-checked"`; the raw (unwrapped)
  baseline is explicitly checked to be `"model-feasible"` (not
  `"conformal-checked"`) so the upgrade assertion cannot be vacuously true.
- `test_phase4b_wired_into_run_rehearsal` (sim-gated) — `phase4b_conformal_wrap`
  called directly reproduces the same block `run_rehearsal` produces, given the
  same inputs.

**Red-proofed:** the `calibration_status="conformal-checked"` upgrade assertion
was verified to actually fail when it should. The wrap was temporarily disabled
in the source (`wrapped_extended = fit_view` instead of
`ConformalForwardModel(fit_view, calibrator_extended)`), the affected test was
re-run, and it failed loudly — not with an assertion mismatch but with a
`TypeError` inside `_in_box` (`'>=' not supported between instances of
'NoneType' and 'float'`), because the unwrapped model's `conformal_set` is
`None`. The sabotage was then reverted and the full suite re-verified green.

## Bottom line

The full M4 machinery — fleet, ICM pooling + §5.8 abstention, EPIG onboarding, and
solve → independent confirmation — runs end-to-end, reproduces exactly, and closes the
runner-level qualification auto-invocation crumb. On this fleet pooling **HELPS** at
equal few-shot n and warm-start onboarding converges to a **sharper** final model;
both are honest in-silico measurements, **not** headline evidence. Real multi-tool data
(M4/M5) remains owed.
