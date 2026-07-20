# M1 gate (directional form) on the chosen M0 venue — Empa bipolar HiPIMS

**Status (static split-conformal): 5 of 6 campaigns PASS the §15.3 directional
coverage gate on both splits; `ti_200w_high_pw` FAILS both — reported as a finding,
not smoothed over. Status (online ACI, D4/§5.6): all 6 campaigns PASS on both
splits, including `ti_200w_high_pw` — the drift-robust calibrator repairs the one
static failure with library-default hyperparameters and no per-campaign tuning.**

Run: 2026-07-19, `python run_m1_empa.py` (full: gp_restarts=5, solver_restarts=120,
seed fixed), 168.6 s. Results: [results/m1_empa.json](results/m1_empa.json).
Determinism: a second full run (`results/m1_empa.rerun.json`) is **identical modulo
`wall_seconds`**. All 24 per-output Clopper–Pearson CIs were independently recomputed
(orchestrator, scipy) and match exactly. The build was adversarially reviewed by two
independent lenses before this run (units/calibration: CLEAN; leakage/splits/stats:
APPROVE, no blockers) — see BUILD_LOG 2026-07-19.

## What this is — and is not

This is the **M1 gate FORM applied to real tool data on the M0 venue chosen by the
PI (2026-07-19): the Empa bipolar HiPIMS deposition-rate dataset.** Per
implementation-plan §15.3, the real-split check is **directional** — empirical
coverage within the exact binomial 95% CI of nominal (0.90), NOT a hard ±2% — because
~80–130 test points give a ~3–6% binomial SE. The *powered* M1 criterion remains the
in-silico version. This is also not yet a temporal+leave-one-tool-out M1 in the full
plan sense: the dataset has one tool, and one of six campaigns has no verified run
order (below).

## Dataset provenance

- **Source:** Zenodo [10.5281/zenodo.18495402](https://zenodo.org/records/18495402)
  (concept DOI …18495401), **CC-BY-4.0** (verified on the record). Real magnetron
  sputter tool at Empa; autonomous BayBE Bayesian-optimization campaigns.
  Paper: Wieczorek et al., *Digital Discovery* 2026, DOI
  [10.1039/D6DD00063K](https://doi.org/10.1039/D6DD00063K).
- **Shape:** 6 campaigns (Al/Ti × power tier), **n = 3,150 total**; per campaign
  5 continuous pulse knobs (bounds transcribed verbatim from each campaign's own
  `Campaign.json`, test-enforced) → 2 measured outputs: calibrated deposition rate
  (**Å/s**; `y1 ×` per-material factor from `calibration.txt`: Al 1.1684, Ti 0.722838)
  and peak current `Ipk (A)`. Per the paper, **Ipk was measured, not controlled** — it
  is an output here, never a knob.
- **Order key:** `BatchNr` (1..n, monotone) in five campaigns → genuine temporal
  splits. **`ti_120w_short_pw` is degenerate (all BatchNr==1, FitNr null)** — its
  "temporal" split uses unverified file order and is starred everywhere.
- **Known ingest quirk:** `ti_120w_short_pw` bounds are full-precision data extents
  while df values are rounded; exactly 5/495 rows sit 3e-11 to 4e-11 outside bounds
  and are skip-rejected (pinned by test).

## Gate results (pooled per campaign; per-output detail in the JSON)

| Campaign | Temporal PICP [95% CI] | Gate | Random PICP [95% CI] | Gate |
|---|---|---|---|---|
| al_120w_short_pw | 0.875 [0.826, 0.914] | PASS | 0.917 [0.874, 0.948] | PASS |
| al_200w_high_pw | 0.915 [0.875, 0.946] | PASS | 0.873 [0.826, 0.911] | PASS |
| al_250w_low_duty | 0.875 [0.814, 0.922] | PASS | 0.875 [0.814, 0.922] | PASS |
| ti_120w_short_pw * | 0.939 [0.895, 0.968] | PASS | 0.913 [0.865, 0.949] | PASS |
| **ti_200w_high_pw** | **0.817 [0.762, 0.864]** | **FAIL** | **0.950 [0.914, 0.974]** | **FAIL** |
| ti_250w_low_duty | 0.887 [0.828, 0.932] | PASS | 0.900 [0.843, 0.942] | PASS |

\* unverified order key (see above).

**The static failure, honestly and by direction.** Under static split-conformal,
`ti_200w_high_pw` fails temporal by **under-coverage on both outputs** (dep-rate
0.808 [0.726, 0.874], Ipk 0.825 [0.745, 0.888]) — the direction that matters,
consistent with BO-sampling drift a static calibrator cannot track. Its random split
fails by **over-coverage on Ipk** (0.958 [0.905, 0.986]; pooled 0.950) —
conservative intervals, a different and more benign failure mode. Both directions are
reported because they are different diseases. The same dual failure appeared
independently in the reviewer's smoke-restart pre-run, so it is stable, not restart
noise. **This is exactly the case the online path below was built to handle.**

## The online ACI drift path (D4 / §5.6) — the static failure, repaired

The §5.6/D4 remedy for drift is **online Adaptive Conformal Inference** (Gibbs &
Candès 2021): stream the test rows in split order, score each row's interval at the
current adaptive miscoverage level α_t *before* observing it, then update
α_{t+1} = α_t + γ·(α_target − err_t). It targets asymptotic *average* coverage under
arbitrary shift. This is now wired into the same runner as an **additional** path
(the static blocks above are untouched and remain the baseline). **All ACI
hyperparameters are the library defaults (γ=0.05, window=50, clip (0.001, 0.5)),
identical across every campaign and split, fixed before any outcome was seen — no
tuning-to-pass.** The random split doubles as the exchangeable control, where ACI
should merely match static.

| Campaign | Temporal static → +ACI | Random static → +ACI |
|---|---|---|
| al_120w_short_pw | 0.875 PASS → 0.892 PASS | 0.917 PASS → 0.900 PASS |
| al_200w_high_pw | 0.915 PASS → 0.896 PASS | 0.873 PASS → 0.900 PASS |
| al_250w_low_duty | 0.875 PASS → 0.894 PASS | 0.875 PASS → 0.894 PASS |
| ti_120w_short_pw * | 0.939 PASS → 0.913 PASS | 0.913 PASS → 0.893 PASS |
| **ti_200w_high_pw** | **0.817 FAIL → 0.887 PASS** | **0.950 FAIL → 0.908 PASS** |
| ti_250w_low_duty | 0.887 PASS → 0.894 PASS | 0.900 PASS → 0.906 PASS |

ACI **repairs `ti_200w_high_pw` on both splits**: it drives α_t *down* on the
under-covering temporal stream (final α_t 0.066/0.031 for dep/Ipk → wider bands) and
*up* on the over-covering random stream (final α_t 0.20/0.10 → tighter bands) — both
corrections in the direction D4 predicts. The five already-passing campaigns stay
passing; the control moves are small. The **§5.6 rolling-coverage detector still fires
on the drift episode** (trailing-window minimum 0.840/0.860 < 0.90) even though the
online *average* coverage recovers — detector and repair are independently visible,
as they should be.

**Adversarial checks I ran on this claim** (the automated reviewer hit a usage limit;
these were run by hand and reproduce): (1) baseline integrity — the static blocks in
the ACI run are byte-identical to the pre-ACI verified baseline; (2) determinism — two
full runs identical modulo timing; (3) no tuning-to-pass — all 12 campaign/split
hyperparameter blocks equal the library defaults exactly; (4) **the infinite-width
trap** — ACI can trivially "cover" by emitting an unbounded interval when α_t drops
below 1/(n+1). On the temporal `ti_200w_high_pw` stream, 1 (dep) / 5 (Ipk) of 120
steps did exactly that. **Excluding those steps entirely** (numerator and
denominator), Ipk coverage is 0.878 [0.804, 0.932] — 0.90 still inside the CI, so the
PASS is *not* an artifact of trivial intervals. This is disclosed as `n_infinite_width`
in the JSON and must travel with the result.

**What ACI does NOT buy.** Asymptotic average coverage is not finite-sample
exactness, so the binomial-CI row on the realized online rate is still a *directional*
check, same status as the static gate — not a theorem test (its per-step trials are
not i.i.d. once α_t adapts). And `update_scores=True` (the default) means ACI gets
both α-adaptation *and* a growing calibration set during the stream; disentangling the
two would need a non-default arm, deliberately excluded here to keep "library defaults"
literal. §20.2 also flags conformal-PID as the eventual online endpoint with bare ACI
as a component — this validates the D4 ACI component only.

## OOD / support check — the sharpest finding

12 ordered pairs over the 4 PRR-space campaigns: **8/12 pass** (OOD epistemic σ >
in-distribution on both outputs). The 4 failures are **exactly the cross-material,
same-power-tier pairs** (Al-120W↔Ti-120W shortPW; Al-200W↔Ti-200W highPW), where OOD
epistemic and support are numerically indistinguishable from ID (e.g. 0.0131 vs
0.0133). Every cross-*tier* pair passes, because tier shifts move recipes in knob
space. Read plainly: **support/epistemic screening detects knob-space shift and is
provably blind to a shift in a variable that is not an input.** A model trained on Al
will confidently predict Al-like rates for Ti recipes, with high support. Pooling
campaigns therefore requires material as an explicit conditioning variable (§8.3);
per-campaign models are the honest configuration, and are what this gate uses.

## Pessimistic inverse demo (al_120w_short_pw, real data)

- **Over-tight band** [q60, q90] = [1.419, 1.586] Å/s (width 0.167 < the credited
  floor 2κσ_ale = 0.454): **INFEASIBLE** with a §8.8 diagnosis (partly epistemic;
  quantified relaxation 0.205 Å/s). The abstention is the correct §8 behavior — a
  band narrower than what κ=2 can credit is refused by construction.
- **Credit-wide band** [q10, q90] = [0.890, 1.586] Å/s: **FEASIBLE**, 3 on-support
  candidates (confidence ≈0.98, support −1.9 to −2.3); each verified against the
  nearest measured run (normalized dist 0.092/0.201/0.237) — **all three neighbors'
  measured dep rates land inside the target band.**
- **Beyond data** [2.764, 3.685] Å/s (>1.5× observed max): **INFEASIBLE**,
  "genuinely unreachable — mean outside the box, margin −10.8σ; more data will not
  move it in." An explicit refusal with a diagnosis, not an invented recipe.

## Caveats that must travel with these numbers

1. **BO-sampled data** (BayBE, exploitation-clustered) strains split-conformal
   exchangeability; the temporal split makes that strain visible (it is the point),
   and one campaign's coverage did break under it.
2. **§8 feasibility reads raw GP σ, not the conformal band** (known repo-wide
   caveat) — the inverse demo inherits any GP miscalibration.
3. Pooled PICP rows share test rows across the two outputs; per-output rows are the
   honest unit (both are in the JSON; the gate table above uses pooled for
   compactness with per-output FAIL detail spelled out).
4. Single tool, single lab; leave-one-tool-out remains impossible on this dataset.
5. n_cal == n_test (120/120 etc.) makes the §15.3 "given n_cal" reading and the
   implemented n_test CI numerically identical here (documented in `binom_ci`).

## Verdict

The M1 machinery — generic tabular ingest → GP forward → split conformal →
directional gate → §8 pessimistic inverse with NN-verification — ran end-to-end on
real, licensed, recipe→outcome tool data at n=3,150 for the first time. Under the
static calibrator it **passed the directional coverage gate on 5 of 6 campaigns and
failed honestly on the sixth in the direction drift theory predicts**; under the
online **ACI drift path (D4/§5.6) it passes all 6 on both splits with library-default
hyperparameters** — the failure was a calibrator-choice artifact, not a modelling
dead-end, and the designated remedy fixes it without tuning. Alongside coverage, the
inverse **refused two impossible requests with quantified diagnoses and produced three
real-data-verified feasible recipes**. The remaining M1 work is now narrow:
conformal-PID as the §20.2 online endpoint (bare ACI is its component), and
material-conditioned pooling if cross-material transfer is ever claimed (the OOD check
proved support/epistemic screening is blind to a non-knob shift).
