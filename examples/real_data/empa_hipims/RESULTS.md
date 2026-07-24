# M1 gate (directional form) on the chosen M0 venue — Empa bipolar HiPIMS

**Status (static split-conformal): 5 of 6 campaigns PASS the §15.3 directional
coverage gate on both splits; `ti_200w_high_pw` FAILS both — reported as a finding,
not smoothed over. Status (online ACI, D4/§5.6): all 6 campaigns PASS on both
splits, including `ti_200w_high_pw` — the drift-robust calibrator repairs the one
static failure with library-default hyperparameters and no per-campaign tuning.
Status (online conformal-PID, §20.2 — the designated endpoint): all 6 campaigns
PASS on both splits with `n_infinite_width = 0` on every campaign/split/output —
the repair without ACI's unbounded-interval caveat.**

Run: 2026-07-22, `python run_m1_empa.py` (full: gp_restarts=5, solver_restarts=120,
seed fixed), 220.9 s — a strict superset of the 2026-07-19 run: the static and ACI
blocks are **byte-identical** to that verified baseline (checked programmatically at
promotion), plus the new §20.2 PID blocks. Results:
[results/m1_empa.json](results/m1_empa.json). Determinism: a second full run
(`results/m1_empa.rerun.json`) is **identical modulo `wall_seconds`**. All 24
per-output Clopper–Pearson CIs of the 2026-07-19 baseline were independently
recomputed (orchestrator, scipy) and match exactly. The build was adversarially
reviewed by two independent lenses before that run (units/calibration: CLEAN;
leakage/splits/stats: APPROVE, no blockers) — see BUILD_LOG 2026-07-19 and
2026-07-22.

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

## The conformal-PID online endpoint (§20.2) — supersedes bare ACI

§20.2 designates **conformal-PID / decaying-step online CP** as the online
*endpoint*, with bare ACI demoted to a *component*. It is now wired into the same
runner as a **third** path (static and ACI blocks unchanged, byte-verified). A
`ConformalPIDController` streams the test rows in split order — interval at the
current state scored *before* `observe` — but instead of adapting the miscoverage
level α_t (ACI) it tracks the score **threshold q_t directly** (Angelopoulos, Candès
& Tibshirani 2023, NeurIPS): a proportional quantile-tracker
`q_{t+1} = q_t + η·(err_t − α)` plus the paper's log-time `tan` integrator, banding
as `mean ± q_t·σ_total(x)` on the standardized-residual score — the same band shape
as the split calibrator, so the three paths differ only in the multiplier. **All
hyperparameters are library defaults (η=0.1, K_I=2.0, C_sat=7.0, window=50),
identical across every campaign and split, fixed before any outcome — no
tuning-to-pass** (recorded per-block in the JSON as `hyperparameters.provenance`).

**Why the endpoint matters here: it is finite by construction.** ACI "covers" for
free via an unbounded interval whenever α_t drops below 1/(n+1) — the trap that cost
`ti_200w_high_pw`'s ACI temporal PASS 1 (dep) / 5 (Ipk) trivial intervals, disclosed
above. Because q_t is a real number and never a quantile *index*, conformal-PID
**cannot** emit an unbounded interval: on this full run **`n_infinite_width` = 0 on
every campaign, split, and output** — the PASS is real width everywhere, with nothing
to exclude.

| Campaign | Temporal static → +ACI → +PID | Random static → +ACI → +PID |
|---|---|---|
| al_120w_short_pw | 0.875 PASS → 0.896 PASS → 0.887 PASS | 0.917 PASS → 0.904 PASS → 0.908 PASS |
| al_200w_high_pw | 0.915 PASS → 0.896 PASS → 0.896 PASS | 0.873 PASS → 0.900 PASS → 0.896 PASS |
| al_250w_low_duty | 0.875 PASS → 0.894 PASS → 0.887 PASS | 0.875 PASS → 0.894 PASS → 0.881 PASS |
| ti_120w_short_pw * | 0.939 PASS → 0.913 PASS → 0.923 PASS | 0.913 PASS → 0.893 PASS → 0.908 PASS |
| **ti_200w_high_pw** | **0.817 FAIL → 0.887 PASS → 0.875 PASS** | **0.950 FAIL → 0.908 PASS → 0.929 PASS** |
| ti_250w_low_duty | 0.887 PASS → 0.894 PASS → 0.887 PASS | 0.900 PASS → 0.906 PASS → 0.906 PASS |

On the failing campaign the corrections move in the D4-predicted directions, like
ACI but on the threshold scale: temporal (under-covering) q_t **rises** (dep
1.59→2.13, Ipk 1.39→1.76 over the stream; effective α driven down to 0.025/0.058)
= wider bands; random (over-covering) q_t **falls** on dep (final 1.37; effective α
up to 0.20) = tighter bands. The §5.6 rolling detector **still fires on the temporal
drift episode** (trailing-window minimum 0.820/0.840 < 0.90) even as long-run
coverage recovers — detector and repair remain independently visible. The five
already-passing campaigns stay passing with small control moves on the random split.

**What conformal-PID does and does not buy.** Guarantee: long-run coverage under
*arbitrary* distribution shift by direct threshold tracking, finite at every step —
and, unlike this ACI implementation, `observe` appends **no** scores to any
calibration buffer (pure threshold dynamics; calibration scores only warm-start q_0),
so there is no α-adaptation/score-refresh conflation to disentangle. Non-guarantee:
no finite-sample exactness — the binomial-CI row is *directional*, the same status as
the static/ACI gates. The `step="decaying"` variant (Angelopoulos, Barber & Bates
2024) is implemented and OFF by default — and a labeled side study
(`run_pid_step_study.py`, [results/m1_empa_pid_step.json](results/m1_empa_pid_step.json))
measured it on all six campaigns: late-stream threshold volatility drops to ~22% of
fixed-step's uniformly, but `ti_200w_high_pw` flips PASS→FAIL on BOTH splits at these
~100-row stream lengths (drift under-correction on temporal, over-correction on random —
one short-horizon mechanism), so **decaying stays opt-in and is discouraged on drifting
tools**; fixed-step remains the path of record. **This closes the §20.2 online-endpoint M1
item; bare ACI remains beneath it as the validated D4 component.**

## Conditional / per-region coverage — does the pooled PASS hide a regional gap?

`run_conditional_coverage.py` re-analyses the RECORDED static/ACI/PID paths
(results/m1_empa.json, never modified) for group-conditional coverage on four
PRE-STATED groups (fixed before any number was computed; no post-hoc slicing):
(1) knob-space density (near/mid/far — k=5-NN distance to TRAINING recipes,
train-standardized), (2) per-output outcome-magnitude (low/mid/high), (3)
temporal stream-position (early/mid/late — drift phase), (4) Mondrian per-output.
Exact binomial 95% CIs; nominal-in-CI is the directional flag (the recorded
gate's rule); groups < 20 points are flagged UNDERPOWERED (none were: smallest
tertile n=26). A FIDELITY GATE reproduces every static/ACI/PID pooled AND
per-output k_covered byte-equal to the recorded JSON before any conditional
number is trusted — PASS on all 12 campaign/split cells; double-run
byte-identical. Results:
[results/m1_empa_conditional.json](results/m1_empa_conditional.json).

**Finding: pooled PASS DOES hide a regional gap — at the high-outcome tail.**
Across the 180 powered tertile-cells per path, static conformal under-covers 14
(ACI 9, PID 9). The systematic hidden mode is magnitude: the HIGH tertile of the
true outcome under-covers in 8 of 24 campaign/split/output cells (5 of 6
campaigns, both outputs, both splits; e.g. al_250w/random/dep 0.630,
ti_120w/random/Ipk 0.727), while the LOW tertile over-covers (1.000) in 3 — the
marginal 0.90 sits between an over-covered low end and an under-covered high
tail. Six of the eight high-tail failures hide behind a marginal that PASSES.

**The online endpoints repair DRIFT-conditional, not magnitude-conditional,
regional failure.** ACI/PID move all of ti_200w's temporal regional failures
(far-density, mid-stream, low+high magnitude) back to nominal, cutting
under-covering cells 14→9. But 7 of 8 high-tail failures are NOT repaired,
because neither ACI nor PID conditions on outcome magnitude — closing the
high-tail gap would need a group-conditional (Mondrian-by-magnitude) calibrator,
now the named owed remedy. Far-from-data (1 cell) and late-drift-phase (0 cells)
are NOT the dominant hidden modes here; the temporal drift under-coverage on
ti_200w actually concentrates in the mid/early stream, not late. Per-tertile CIs
are wide (n≈27–43); the finding is the consistent DIRECTION repeated across
independent cells, not any one cell.

## Mondrian group-conditional coverage — does grouping by predicted magnitude fix the high tail?

`run_mondrian_coverage.py` reuses the runner's seeded GP fits and the conditional-coverage
study's tertile/coverage machinery (by import) to evaluate a `MondrianConformalCalibrator`
grouped by PREDICTED-mean magnitude tertile (edges frozen from the calibration slice —
leakage-free by construction, since no true outcome exists at predict time) against the
recorded static baseline. Static fidelity is byte-equal to
[results/m1_empa.json](results/m1_empa.json) on all 12 cells. Of the 8 high-observed-magnitude
cells that static conformal under-covered, **6 move to nominal** under Mondrian; the 2 that
don't are exactly the cells where the model's predicted magnitude barely tracks the observed
outcome (assignment agreement 0.66 and 0.48) — an honest mechanistic limit: grouping by
predicted magnitude cannot isolate a tail the model cannot predict. The high tertile pays
1.0–6.0× MPIW; the low tertile (which over-covered) narrows — the intended redistribution.
Mondrian is conservative: on `ti_120w_short_pw` it over-covered enough to break the marginal
pooled gate *from above* (the safe direction — a width cost, not a safety cost). It also
closes the selected-point hole mechanistically: in a synthetic reconstruction of the
false-success study's d=8 miss, the pooled gate admits the marginal candidate and the Mondrian
gate refuses it. Artifact:
[results/m1_empa_mondrian.json](results/m1_empa_mondrian.json). What it does and does not buy:
coverage conditional on the *declared predicted-magnitude group* — not per-point, and not per
true-magnitude group.

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

## Material-conditioned pooling (M1 remainder) — awareness gained, transfer not claimed

`run_m1_empa_pooled.py` fits the tool-aware ICM multi-task GP (§10.4) with
**material as the task**, answering the two questions the per-campaign OOD check
left open (audit F4). Pooling is within each parameterization subspace (PRR — 4
campaigns; DUTY — 2; knob names differ between them). Full run 2026-07-22
(`n_restarts=2`, seed 0), deterministic (no wall-clock keys in the JSON). Results:
[results/m1_empa_pooled.json](results/m1_empa_pooled.json).

**A. The material shift is now representable — the F4 control.** Fitted al↔ti task
correlation 0.9985. For the 4 previously-blind cross-material same-tier pairs the
per-campaign model saw nothing (epistemic ratio ~0.97–1.0, support Δ~0.02, 0/4
flagged). The pooled model now (i) shifts the predicted **mean** by 0.78–1.67 Å/s
(tens of σ_epi) on all 4 — a distinction a per-campaign model structurally cannot
make; (ii) flags 3/4 by epistemic-inflation/support-drop at a pre-stated margin;
(iii) inflates epistemic to **dominate both materials 12/12** when material is left
unspecified (§5.8). Honest catch: the epistemic flag is **asymmetric**
(Al-conditioning carries intrinsically higher epistemic than Ti after pooled
standardization), and **support stays ~flat (±0.06)** — input-space screening is
still blind to a same-box material shift. Campaign-as-task corroborates (1/4).
Awareness comes from making material an explicit conditioning axis (mean +
unknown-material fallback), not from auto-detecting a wrong-material query.

**B. Cross-material transfer does NOT hold (leave-one-material-out).** Zero-shot,
the §5.8 fallback epistemic dominates every known tool (both directions) but the
fallback mean is the trained material's surface (dep-RMSE 9–24× the full-data
ceiling). Few-shot (10–20 runs of the new material): dep-RMSE stays 2–14× the
ceiling and the model's own predictive intervals are mis-calibrated (raw PICP
0.79–0.94, only 1 of 4 arms near nominal); split-conformal holds coverage only by
widening bands to 0.8–1.8 Å/s — near-vacuous on this output range. **No arm is
claimable → cross-material transfer stays FORBIDDEN (audit F4).**

**C. Pooling costs no coverage.** Per-campaign split-conformal PICP under the
material-conditioned pooled model, on the same temporal/random slices as the
baseline: **0 PASS→FAIL flips**; pooling FIXES `ti_200w_high_pw`/random (0.950
FAIL over-coverage → 0.896 PASS) and nudges its temporal 0.817→0.833 (still FAIL —
the BO-drift under-coverage belongs to the online calibrators above). All other 10
cells stay PASS within ±0.04.

**Verdict:** material-conditioned pooling is SAFE (no coverage cost) and delivers
the awareness control audit F4 required; cross-material transfer is NOT demonstrated
and must not be claimed. Per-campaign models remain the honest configuration for
headline accuracy — now with a material-aware pooled screen on top.

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
2. **The inverse demo's §8 feasibility reads raw GP σ, not the conformal band** —
   it runs on the unwrapped GP, so its candidates inherit any GP miscalibration and
   are labeled `model-feasible` accordingly. (Since 2026-07-22 the solver applies the
   §13.2 conformal containment C(x)⊆Z* **by default** whenever its model IS
   conformal-wrapped; this demo deliberately reproduces the recorded unwrapped
   configuration.)
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
real-data-verified feasible recipes**. Of the two M1 items that remained after that:
**conformal-PID (§20.2) is now DONE** — the designated online endpoint passes all 6
campaigns on both splits with library defaults and `n_infinite_width = 0` everywhere
(2026-07-22, above). **Material-conditioned pooling is now DONE too** (2026-07-22,
pooling section above): the material shift the OOD check proved invisible is now
representable and flagged when material is unspecified, and pooling costs no
coverage — while cross-material **transfer** was measured and does NOT hold
(RMSE 2–14× ceiling, mis-calibrated few-shot intervals), so that claim stays
forbidden. Both remaining M1 items are closed; per-campaign models plus the online
calibrators remain the honest headline configuration.
