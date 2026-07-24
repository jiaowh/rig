# BUILD_STATE — single source of truth for the RIG build

> **✅ 2026-07-24 — `docs/rig-explained.html` brought CURRENT with the full build and
> DE-TIMESTAMPED (user directive: comprehensive standing explainer, no dated update notes).**
> Now covers all third/fourth-wave results woven in as standing content: Mondrian conditional
> coverage (§04 + §10, incl. the 6/8-fixed / 2-unfixable honesty), full M2 comparator slate +
> binding-policy survival (the stale "current work item" line fixed), false-success study +
> analytic-gradient parity, M3 v2 acceptance, fleet dress rehearsal + fail-closed conformal
> wrap, E1 ingest validation, D7 ROM cross-check, PGD/typicality hardenings, loop
> qualification hook. Republished to the SAME artifact URL. Keep it in sync content-wise on
> future result changes, but never add dated status lines to the page itself. See BUILD_LOG
> 2026-07-24.
>
> **✅ 2026-07-22 (evening) — SECOND WAVE: five more owed items closed** (5 parallel subagents +
> orchestrator; suite **629 passed / 0 failed / 0 skipped STRICT**, repo ruff check+format exit
> 0, import-linter 1 kept / 0 broken — final bar over the whole tree). (1) **M2 binding-policy
> re-run**: the cost win SURVIVES 2.0/2.0/0.02, attenuated ~35% (M2 row; F3 fully closed).
> (2) **D7 different-physics ROM verifier** for MBE: geometric Knudsen flux ROM, flux-channel
> scope honest, D7 identity check passes with a REAL verifier (WP-B row; Phase-0 owed item
> closed). (3) **WP-E §8 hardening**: PGD δ-box + flow typicality, opt-in, benchmarked +
> red-proofed (WP-E row item 2 done). (4) **Honest M3 v2** on the InSilicoMachine: PASS 6/6 vs
> cold_light 3/6 at 0.19× budget with the restart-budget caveat honestly recorded (M3 row).
> (5) **Loop qualification hook**: opt-in ConfirmationCampaign gating of both target-met stop
> points, budget-honest (F2 loop hookup done). Also: `docs/rig-explained.html` rewritten to
> explain the CURRENT program only (audit/build/test history stripped per user) and
> republished. Still open after this session: ~~runner-level qualification auto-invocation~~
> (CLOSED 2026-07-23 in its honest in-silico form — `run_multitool_rehearsal.py` demonstrates
> solve→ConfirmationCampaign automatically, direct + loop-hook, incl. NothingToQualify/0-calls
> on Infeasible; the REAL-tool invocation remains part of M4/M5), optional ablation@50 M2
> refresh (DONE 2026-07-23), SCBO/TuRBO comparator slate (DONE 2026-07-23 — claim holds vs
> the full slate, see M2 row), prospective real-tool qualification campaign (M4/M5),
> analytic-gradient default (**evidence gathered 2026-07-23, decision NOT made** — parity
> study `docs/gradient-parity-2026-07-23.md`: 97.3% verdict agreement, all 4 disagreements
> are FD false-abstentions analytic resolves correctly, speedup to 11.4× at d=20;
> recommendation flip at d≳8 with the recorded migration checklist; boundary-target follow-up
> advised before finalizing the threshold). **E1 frame validation: DONE
> 2026-07-23** — `rig_adapters.tabular.validation` (blocking missing-columns/dtype/nan-inf +
> advisory bounds/order-key/duplicates, declared-unit bounds per the SI-trap rule, Empa quirks
> pinned row-for-row, 34 tests, red-proofed, non-breaking `IngestResult.frame_report` + opt-in
> `strict=`); see BUILD_LOG 2026-07-23.
>
> **✅ 2026-07-22 — AUDIT.MD REMEDIATED + BOTH REMAINING M1 ITEMS DONE** (4 parallel subagents
> with disjoint file ownership + orchestrator verification). **Suite 581 passed / 0 failed /
> 0 skipped in STRICT mode (`RIG_REQUIRE_MBE_SIM=1` + `RIG_REQUIRE_TORCH=1`), whole-repo
> `ruff check` AND `ruff format --check` exit 0 (the Windows App-Control block has LIFTED),
> import-linter 1 kept / 0 broken (corrected exit-code form).** Nothing committed.
> (1) **audit.md F1–F7: every finding re-verified first-hand against source (all CONFIRMED;
> F3/F7 sharpened), then fixed.** F1 — the §13.2 `C(x)⊆Z*` conformal containment is now
> DEFAULT-ON in `PessimisticInverseSolver.solve()` whenever the model is conformal-wrapped
> (with the anti-false-abstention pool sweep; unwrapped models byte-identical), and
> `RecipeCandidate.calibration_status ∈ {model-feasible, conformal-checked, revalidated}`
> makes the raw-σ-only case explicit; mechanism-twin regression test proven to go red with
> the gate disabled. F2 — `rig.active.campaign.ConfirmationCampaign` wires
> `ConfirmationBatchGate` into a provenance-logged confirmation-campaign path (Infeasible ⇒
> zero machine calls; promotion blocked on rejection; Bonferroni multiplicity knob; 19 tests,
> partition red-proofed; **loop hookup DONE later the same day** — `ActiveLearningLoop` takes
> opt-in `qualification:` gating BOTH target-met stop points, budget-honest, rejection
> continues the loop, default-None byte-identical proven; runner-level auto-invocation remains
> open). F3 — `ActiveLearningLoop` defaults now the binding §8 2.0/2.0/0.02 (solver
> already was) + a defaults-match pin test; `run_m2_sweep.py` keeps its published 1.0/1.0/0.01
> but labels it `FEASIBILITY_POLICY` = ablation in JSON/stdout — **and the binding-policy M2
> re-run is now DONE (same day, evening): the cost win SURVIVES 2.0/2.0/0.02, attenuated ~35%
> (ΔRMST −16,480, p=8.7e-70, holds at every tol_k; see the M2 row) — F3 fully CLOSED**. F5 — explicit approximation ledger in `pessimistic.py` docstrings; **PGD δ-box +
> flow typicality IMPLEMENTED opt-in later the same day** (benchmarked + red-proofed,
> default-off byte-identical — see the WP-E hardening BUILD_LOG entry; remaining GP-tier
> approximations: joint-MC spec-hit, worst-of-K epistemic). F6 — repo-wide ruff green; the audit's "`lint-imports` unavailable"
> was the Store-Python PATH quirk (use the documented `python -c … sys.exit` form; re-verified
> 1 kept / 0 broken, exit 0). F7 — stale revalidation claims annotated in place in
> `docs/dimensionality-2026-07-17.md`.
> (2) **M1 remainder CLOSED.** **Conformal-PID (§20.2 endpoint) BUILT + WIRED + FULL-RUN
> VERIFIED**: `rig.calibration.pid.ConformalPIDController` (P+I quantile tracking directly on
> the score threshold — FINITE by construction; library defaults η=0.1/K_I=2.0/C_sat=7.0
> fixed before outcomes; 10 tests, sign-flip red-proof 7/10) as the THIRD Empa runner path;
> full 6-campaign determinism pair verified (static/ACI/OOD/inverse blocks byte-identical to
> the recorded baseline) and PROMOTED to `results/m1_empa.json`: **12/12 campaign/split PID
> gates PASS with `n_infinite_width = 0` on every campaign/split/output** (ti_200w_high_pw
> temporal 0.817-static-FAIL → 0.875 PID PASS; random 0.950-static-FAIL → 0.929 PID PASS) —
> the ACI repair without ACI's unbounded-interval caveat. **Material-conditioned pooling
> BUILT + FULL RUN** (`run_m1_empa_pooled.py`, ICM §10.4 with material-as-task, no src edits):
> awareness GAINED — the previously-invisible material shift is now representable (mean shift
> 0.78–1.67 Å/s on all 4 blind pairs; unknown-material epistemic dominates 12/12; 0 PASS→FAIL
> coverage flips, even FIXES ti_200w/random static over-coverage) — but **cross-material
> TRANSFER was measured and is NOT claimable** (LOMO zero-shot no mean transfer; few-shot
> dep-RMSE 2–14× ceiling with mis-calibrated intervals) ⇒ per-campaign models stay the honest
> headline config, now with a material-aware pooled screen. `RESULTS.md` carries the §20.2 and
> pooling sections + updated verdict. See BUILD_LOG 2026-07-22 (5 entries).
>
> **✅ 2026-07-19 (late night) — ACI DRIFT PATH (D4/§5.6) WIRED + FULL RUN + VERIFIED: repairs the
> one static M1 failure.** `run_m1_empa.py` gains an online-ACI evaluation alongside the static
> path. Full 6-campaign run (promoted to `results/m1_empa.json`, a superset — static blocks
> byte-identical): **`ti_200w_high_pw` FAIL→PASS on BOTH splits** (temporal 0.817→0.887 via α_t
> down; random 0.950→0.908 via α_t up) with **library-default hyperparameters, no per-campaign
> tuning**; the other 5 stay PASS; the §5.6 rolling detector still fires (window-min 0.84 < 0.90).
> The Fable-5 reviewer agent died on a usage limit → orchestrator (Opus) ran the adversarial
> checks by hand (`scratchpad/verify_aci.py`): baseline byte-identical, deterministic, all 12
> hyperparameter blocks = library defaults, and — the key one — every ACI PASS **survives
> excluding the infinite-width steps** (ti_200w Ipk 0.878 [0.804,0.932] w/o the 5 unbounded
> intervals, 0.90 inside). NET M1 PICTURE: 6/6 pass under the drift-robust calibrator the plan
> designates as primary under drift. Owed: conformal-PID (§20.2 endpoint); material-conditioned
> pooling. See BUILD_LOG 2026-07-19 (~late night, cont.).
>
> **✅ 2026-07-19 (night) — M0 VENUE DECIDED (USER): Empa bipolar HiPIMS. FULL 6-campaign M1
> directional gate RUN on real data: 5/6 PASS both splits; `ti_200w_high_pw` FAILS both**
> (temporal under-coverage dep 0.808/Ipk 0.825 — drift vs static split-conformal, ACI not wired;
> random over-coverage Ipk 0.958). OOD 8/12 (4 fails = exactly the cross-material same-tier
> pairs — material blindness, per-campaign models are the honest config). Inverse demo on real
> data: INFEASIBLE (over-tight, §8.8 diagnosis) / FEASIBLE ×3 all NN-verified in-band /
> INFEASIBLE (beyond data, −10.8σ). Determinism double-run identical modulo wall_seconds; all
> 24 CIs independently recomputed exact. `examples/real_data/empa_hipims/RESULTS.md` is the
> citable summary. NEXT (M1 remainder): D4/§5.6 ACI drift path on the failing campaign;
> material-conditioned pooling if cross-material transfer is ever claimed. See BUILD_LOG.
>
> **🔍 2026-07-19 (evening) — Empa runner ADVERSARIALLY REVIEWED: no blocker; first full
> 6-campaign (smoke-restart) run shows `ti_200w_high_pw` FAILS the gate on BOTH splits**
> (temporal pooled 0.817 under-covers, random pooled 0.950 over-covers) while the other 5
> campaigns pass, and the OOD check lands 8/12 with the 4 failures exactly the cross-material
> same-tier pairs (support/epistemic indistinguishable — material is not a knob; the model is
> blind to that shift). Leakage attacks, CI arithmetic (independently recomputed), and
> determinism (double-run byte-identical) all came up clean. The full-restart 6-campaign run
> is still owed and must be reported WITH its FAIL rows. See BUILD_LOG 2026-07-19 (evening).
>
> **✅ 2026-07-19 (late afternoon) — Empa HiPIMS M1-GATE-FORM RUNNER BUILT (smoke-verified;
> full 6-campaign run NOT yet executed — that is the orchestrator's).**
> `examples/real_data/empa_hipims/run_m1_empa.py` + `README.md`: per campaign, tabular-adapter
> ingest → TEMPORAL (BatchNr-order 60/20/20, the §15.3 M1 real-split form) + seeded RANDOM
> contrast splits → GP + split-conformal (α=0.1) → per-output + pooled PICP with the exact
> binomial 95% CI (gate = directional "0.90 inside CI", NOT ±2%); full runs add the 4-campaign
> PRR-space OOD-epistemic check + a 3-regime §8 inverse demo on Al-shortPW. JSON →
> `results/m1_empa*.json`. Smoke (Al-shortPW, reduced restarts): temporal pooled PICP 0.875
> [0.826,0.914] PASS, random 0.917 [0.874,0.948] PASS; inverse: over-tight [q60,q90] band
> correctly INFEASIBLE (width 0.167 < 2κσ_ale 0.454 credited floor — the abstention IS the
> §8 behavior), [q10,q90] band FEASIBLE (NN-verified in-band), beyond-max INFEASIBLE
> (margin −10.8σ diagnosis). Full run ≈4–6 min; `--campaign`/`--smoke` flags for slicing.
> Nothing committed.
>
> **✅ 2026-07-19 (afternoon) — Empa HiPIMS E1 data-prep slice BUILT (M0 lead candidate).**
> `examples/real_data/empa_hipims/`: deterministic converter (`prepare_empa.py`) → 6 tidy
> per-campaign CSVs (3,150 rows, calibrated `dep_rate_A_per_s` = y1×factor from each
> `calibration.txt`; Al 1.1684 / Ti 0.722838 Å/s) + 6 per-campaign TOML specs with bounds
> VERBATIM from each campaign's own `Campaign.json` + `tests/test_empa_ingest.py` (9 green;
> ingest via the WP-H tabular adapter, source=real_tool, SI spot-checks, `continuous_si`
> trap respected). Data gotchas pinned by tests: `pos. *` columns renamed dot-free (adapter
> reserves '.'), and **Ti-120W-shortPW is degenerate** — all BatchNr==1/FitNr null, and 5/495
> rows sit ~3e-11 outside its full-precision bounds (skip-ingest → 490+5 rejects, documented,
> data not clamped). M0 venue choice remains the USER/PI decision. Nothing committed.
>
> **✅ 2026-07-19 (~13:30) — explainer HTML now carries the data inventory.** New section 12
> `#data` in `docs/rig-explained.html` (republished, same artifact URL): the two current data
> sources (in-silico `mbe_sim`/`InSilicoMachine`; unlicensed `Zr_grid.csv` machinery proof),
> the 4 downloaded M0 candidates with links/licenses/catches (from `data/m0-candidates/
> MANIFEST.md`), the nrel.gov→nlr.gov rot warning, and the "why not all four → lead venue +
> replications" argument (venue = still the user/PI decision). Hero refreshed: **suite 533
> passed** (full re-run this session), 21 defects fixed, "Stress-tested to 20 knobs" (was the
> over-strong "verified"). Nothing committed.
>
> **✅ 2026-07-19 (~03:12) — MFL BAKE-OFF RE-FROZEN + RE-RUN (CITABLE); the caveat two blocks
> below is RESOLVED.** The steelman-REQUIRED fixes are applied (BUILD_LOG 2026-07-19 ~03:12):
> FIX 1 (blocker) — target labels now use the SAME one-sided-slip spec that is stored/scored
> (`_effective_bounds` single source; feasible/hard targets anchored on the slip-feasible
> reachable sub-cloud; every stored witness asserted in-spec); FIX 2 — Loop-A loss restored to
> the paper form `(1/n')Σ‖z'-y'‖²` (immaterial, 6/11 either way). Predictions NOT touched.
> targets.json regenerated + re-frozen: **new hash `02603fc1…`** (was `52e66d3b…`); labels
> re-verified by an independent 20000-pt dense search (feasible/hard ≥385 in-spec pts each,
> infeasible 0, all 15 witnesses valid); hash-freeze proven to refuse a stale hash + a tampered
> target. **CITABLE full run** (`results/full_20260719T031201Z/`): RIG miss **0.0000** /
> false_abst **0.3333** / margin **+10.52** / q=60 (presents+hits all 10 clearly-feasible;
> abstains on all 5 feasible-but-hard AND all 5 infeasible); MFL miss **0.5000** / false_abst 0 /
> margin **−1.45** / q_charit 4060 / q_deploy 12060 (misses exactly the 10 boundary+infeasible
> targets). rig-reval == rig on miss/abstention. The corrected false_abst is 0.333 (not the
> mislabelled-run's 0.533 — the 5 infeasible abstentions are now correctly OUTSIDE the
> denominator). Predictions to be scored by the orchestrator, not here. Nothing committed.
>
> **⚡ 2026-07-19 STATUS: suite 527 passed / 0 failed.** Fresh 6-lens audit DONE (11 confirmed
> findings — 2 major: continuous_si offset-unit mis-conversion, revalidation_model never wired —
> ALL FIXED same-session with red-then-green guards; see BUILD_LOG 2026-07-19). Research docs
> landed: `m0-dataset-candidates-2026-07-18.md` (best: Empa HiPIMS, Zenodo, n>3000, 6-D, CC-BY),
> `mfl-follow-on-research-2026-07-18.md` (MFL: ZERO citations). ⚠ ruff BLOCKED by Windows
> Application Control mid-session (WinError 4551) — user action needed. In flight: MFL bake-off
> build (wf_fc1bfc4a-b02), M0 dataset downloads (data/m0-candidates/).
>
> **✅ 2026-07-19 — MFL BAKE-OFF BUILT + FULL RUN DONE** (was "in flight" above; now landed;
> BUILD_LOG 2026-07-19). Faithful MFL (Gu et al. 2025, Alg. 1) in `src/rig/baselines/mfl.py`
> (torch, lazy; base stays torch-free), + `examples/mfl_bakeoff/` (frozen 20-target set, 4 arms
> rig/rig-reval/mfl-charitable/mfl-deployable, prereg §0 metrics VERBATIM, charitable-vs-
> deployable ledger, --smoke/--full, README with every deviation). 6 new tests pass; 533
> collected unbroken; import-linter 1 kept/0 broken. **Full run (in-silico ONLY): P2 (RIG miss
> 0.0 vs MFL 0.70, 70pp) AND P3 (normalized-margin gap ~20σ; MFL median margin NEGATIVE) both
> HOLD ⇒ the pre-registered "RIG's formulation is better-posed" claim is SUPPORTED on this
> simulator.** Predicted LOSS P5 holds (RIG false-abstention 0.53 — pessimism's price). **P1
> REFUTED in the OPPOSITE direction from the author's prediction**: RIG 60 machine-queries vs
> MFL charitable 4060 (RIG's GP inverse is machine-free at solve time; MFL's on-machine Loop B
> is not) — a legitimate pre-registration finding. P4 tautological, not counted. rig-reval ==
> rig (miss already 0) ⇒ the owed conformal-re-validation experiment's answer is "no change on
> this target set". Two build findings (both fixed + documented as deviations): FD on a NOISY
> machine needs `fd_step` above the noise floor (1e-3→0.05, steelman-adopted); a trivial
> one-sided bound is pathological for the MFL point rule (fixed to a reachable-relative slip
> bound). ruff still blocked (WinError 4551) — code hand-conformed. Nothing committed.
>
> **⚡ IN-FLIGHT TASKING (2026-07-18): session limit cut a 5-part tasking mid-launch.**
> Read `docs/RESUME-STATE-2026-07-18.md` FIRST — it holds the execution plan (audit relaunch,
> research/M0-hunt relaunch, MFL bake-off build per `docs/mfl-bakeoff-build-spec-2026-07-18.md`,
> then the HTML update LAST with the results). Both interrupted workflow scripts are on disk;
> resume commands are in that file. Repo is clean at 511 passed; nothing from that tasking
> has touched the tree yet.

> Update discipline: every agent/session updates its rows here and appends to
> BUILD_LOG.md. Status ∈ {TODO, IN_PROGRESS, DONE, BLOCKED, DEFERRED}.
>
> Last updated: 2026-07-17 (**Session 8 (evening) — 4 new modules built + adversarially
> verified; gradient red-tests resolved**; see BUILD_LOG 2026-07-17 evening for full detail).
> **Suite 511 passed / 0 failed / 0 skipped locally**, ruff clean, import-linter 1 kept / 0
> broken (CORRECTED command). Nothing committed.
>
> **Session 8 summary:** four modules built in parallel — `qualification.py`
> (ConfirmationBatchGate, exact Clopper-Pearson), `forward/distill.py` (§5.7 ensemble→student),
> `active/acquisition.py` (`qlognehvi_phase2`, real botorch qLogNEHVI), and linear-constraint
> support + an opt-in **analytic objective gradient** in `inverse/pessimistic.py` — then all
> audited by isolated-sandbox adversarial agents (mutation-tested, ground-truth-scored).
> **VERDICTS: acquisition SOUND, qualification SOUND, distill PARTIAL (core sound),
> linear-constraints SOUND.** In every module the shipped SOURCE was correct; the only real
> findings were 3 test-coverage gaps (a guard tested only through the thing it guards) — ALL
> fixed and each proven to catch its defect by source mutation: (i) qualification's multi-output
> `np.all` conjunction, (ii) distill's `_box_excess` lower box face (§8.2 fail-closed hole in
> the unsafe direction), (iii) linear-constraints' barrier-off safety test relying on the shared
> checker. The analytic-gradient builder left 2 RED tests in `test_inverse.py`; both diagnosed
> BENIGN and the gradient proven correct three ways (isolated softmax-Jacobian 5e-11;
> well-conditioned e2e 2.4e-7; independent 60-digit mpmath at the exact failing point 5.38e-8) —
> tests rewritten (the `len(fd)==len(an)` premise was false; the FD path UNDER-EXPLORES the
> pre-image relative to the exact gradient — a real finding). Un-audited surface is now zero.
> Still owed: MFL bake-off (only pre-registered, `docs/prereg-mfl-bakeoff-2026-07-17.md`);
> the owed experiments; `revalidation_model` wiring. M0 unchanged (the whole scientific claim).
>
> **P0: the suite had not been running at all.** The `seminn`→`rig` rename listed below
> as PENDING had actually HAPPENED, but the editable install was never re-run, so it
> still pointed at the vanished `...\github\seminn`; every `import rig_adapters` failed
> and pytest died in conftest collection. Fixed by `python -m pip install -e ".[dev]"`.
>
> **10 defects found + fixed** (each reproduced first, each guarded by a test verified to
> fail against the defect): (1) **the documented import-linter command always exited 0
> even on a BROKEN contract** — every past "import-linter clean, exit=0" claim was
> evidence-free (the contract does genuinely hold; the command was the problem);
> (2) **P1 `multitask.posterior_cov` disagreed with `predict` on the unknown-tool
> branch** → EPIG collapsed to ~0 nats (19× under-report) exactly on the §10.4
> chamber-onboarding path and exactly as λ anneals to let EPIG dominate; (3) **P1 a
> UNITS defect in the real-data sputtering example** — declared-unit bounds paired with
> SI-canonicalized data made the solver search pressure over 1–43 **Pa** (lower bound
> ABOVE the data's max) and made the "guaranteed on-support" reference point a 2.3×
> extrapolation (support −5.761 vs floor −2.044; now −0.802); (4–6) three amortized-
> generator defects: `log_prob` returned the u-space density (integrated to 3.36, and
> RE-ORDERED the posterior), `_member_counts` shipped a skewed/truncated sub-mixture at
> small n (the gate certified one law, D2 shipped another), and `sample()` never advanced
> its RNG (repeated calls returned bit-identical rows); (7) `ActiveLearningLoop.u_bound`
> silently not forwarded to the solver (also made M2 reach-asymmetric vs the BO arm);
> (8) `revalidate` could return a FALSE INFEASIBLE after testing only the q diverse picks;
> (9) **CI blind spot A1 recurring one tier up** — 56/382 tests (the ENTIRE WP-E/M3
> surface) skipped silently on every green CI run; `RIG_REQUIRE_TORCH` added;
> (10) `run_m2_sweep.py` crashed on a cp1252 console AFTER all compute and BEFORE writing
> its JSON — **so `docs/m2-result.json` is STALE (pre-IF-1 schema, no `verdict` key) and a
> powered re-run is owed.**
>
> **▶ VERDICT ON THE GOAL (binding framing — do not quote the milestones without it):**
> **the machinery is sound; the goal is NOT demonstrated.** The forward+conformal+inverse
> stack runs end-to-end, reproduces exactly, abstains honestly rather than lying, and beats
> a matched-budget BO on a **fair, verified-non-circular** in-silico cost race. But every
> quantitative claim is either in-silico on a simulator, or on a 2-knob public sputtering
> grid that is a dense 15×15 lookup table (invertible by nearest-neighbour). **M0 is not a
> schedule item — it is the entire scientific claim.**
>
> **▶ NEW un-caveated gaps found by the audit (fix the framing, not the code):**
> **(F9) — CLOSED 2026-07-17, see `docs/dimensionality-2026-07-17.md`.** It WAS true that
> every result sat at 2 input dims (MBE=2, sputtering=2, M3 toy=2) and that "dimension-
> agnostic" was an untested claim. Now measured against GROUND TRUTH (solve, then evaluate
> the TRUE function at the returned recipe — never the model's own opinion):
> **d=2,4,6,8,10,15 → FEASIBLE with 3/3 genuine ground-truth hits**; d=20 @ 240 runs →
> INFEASIBLE with the §8.8 **epistemic-limited** diagnosis, and **doubling to 480 runs
> flipped it to FEASIBLE with a real hit** — while the CONTROL (4× the search budget, 192
> restarts, same 240 runs) stayed INFEASIBLE, proving the abstention was genuinely
> data-limited and not a starved search. **⚠ d=20 @ 800 produced a REAL FALSE SUCCESS
> (2/3 — a certified recipe that misses on the true function), confirmed DETERMINISTIC by
> re-invoking the original code path (`FEASIBLE, 2/3, worst_err 1.046` vs a ±0.8 box — a
> marginal miss, ~31% outside).** It is ONE (seed, config) cell, so NOT a rate — and it is
> **fragile**: dropping the GP's hyperparameter-fit restarts 3→2 makes it vanish (0/9). That
> fragility is itself the finding: **a conservatism guarantee that flips with a model-fit
> detail is not a guarantee.** (An earlier draft here said "did not replicate" — that was a
> mis-correction from a replication that changed the GP fit and so tested a different model.) **The STRUCTURAL point stands regardless and is the durable
> finding:** the §8 margin consumes the model's **RAW** sigmas — `conformal_set` is NOT part
> of the feasibility decision — so the pessimism inherits any miscalibration of the surrogate
> beneath it, and the §13.2 conformal re-validation gate that would catch such a case
> (`revalidation_model`) **defaults to None** and `active/loop.py` never sets it. Worth fixing
> on its own merits. **[UPDATE 2026-07-22/23: FIXED, then MEASURED. The §13.2 containment is
> now DEFAULT-ON when the model is conformal-wrapped (audit F1), and the owed
> false-success-rate-vs-d study is DONE (`docs/false-success-study-2026-07-23.md`): the exact
> d=20/800 crime-scene miss REPRODUCES on the raw arm (excursion 1.046 vs ±0.8) and the
> default-on gate KILLS it (1 conformal-checked genuine hit, 0 FS). Grid caveats to carry:
> at 12·d density false successes are too rare to separate arms (raw FSR ≤ 3.0% 95% CP over
> 122 certified candidates; epistemic abstention dominates); the gate is MARGINAL not
> conditional coverage — the wrapped arm produced its own d=8 selected-point miss (1/84), so
> wrapping is not strictly safer at fixed budget; and the d=20 miss is fragile to the
> optimizer path too (analytic-gradient search → 0 FS at the same cell). Conditional/Mondrian
> conformal is now motivated from TWO independent directions (this + the Empa
> high-outcome-tail study) and is the named owed remedy.]** Also measured: conformal coverage at solver-SELECTED recipes was 1/9 miss
> = 11% vs 10% nominal, i.e. **no evidence** of the selection effect an earlier draft claimed
> (that claim came from a 600-point model and was withdrawn). See
> `docs/dimensionality-2026-07-17.md`. Two real weaknesses found: `n_restarts` was a FIXED 48 for every
> dimension (a starved multi-start → **FALSE INFEASIBLE**; now `max(48, 24·dim)`, which is
> exactly 48 at dim=2 so no 2-D result moves), and solve cost is ~O(d²) because `minimize`
> runs WITHOUT `jac` (SciPy finite-differences: d+1 evals/gradient; 2.6 s@d=2 → 150 s@d=20)
> — **owed work, documented not silently accepted**. Limits: synthetic/smooth, 12·d runs is
> a choice not a law, single seed. **This does NOT touch M0.** **(F7)** the headline PICP 0.929 is bought with fitted
> aleatoric noise **10.5×/3.9×/5.2×** the CSV's measured error (the GP absorbs misfit into
> "noise" until bands cover) — quoted below with no mention. **(F6)** the real-data
> inverse is genuinely correct at the demo's ±6 tol (19/19) but ±6 ≈ the conformal MPIW
> and 56% of all runs already sit in that band; at ±2 it is **0/25 FEASIBLE**. **(F3)**
> the M3 gate is near-tautological (all confidences saturate at 0.9997–1.0) and its real
> signal is **n=1** of 5 targets. **M2 runs RIG at κ=1.0/z_epi=1.0** vs the binding §8
> defaults of 2.0/2.0 — undisclosed in the M2 doc.
>
> — prior: (**Session 6**: **M3 amortized NPE generator DONE + verified**
> — the blocking flow-library decision resolved to **zuko** (user call). Built
> `src/rig/inverse/amortized.py :: AmortizedInverseGenerator` — K conditional neural
> spline flows (zuko `NSF`) deep-ensembled, trained in the UNCONSTRAINED `u`-space of
> `RecipeTransform` (constraint-by-construction: every sample feasible), region-augmented
> box conditioning (D2), mixture sampling + mixture `log_prob`; and the §14.6 **SBC/TARP
> BLOCKING gate** `validate()` reusing WP-G `rig.eval.calibration_gates`. Adversarially
> verified (workflow, 9 agents → 6 CONFIRMED findings, ALL fixed + guarded): **1 HIGH**
> — the gate drew all posterior samples from ONE member per trial (a single component),
> not the shipped even MIXTURE → certified a narrower object than ships; fixed to split
> the draw across all members (`_draw_u_std_mixture`). **2 MED** — default SBC prior was
> a moment-matched Gaussian (invalid for non-Gaussian `u` marginals) → now bootstraps the
> empirical training-`u` rows; and `_spec_context` served unconstrained/one-sided outputs
> a 6σ (width-12) box far off the trained region-augmentation manifold → clamped to the
> max trained box width (`2·region_hw[1]`). **3 LOW** — inverted spec box for extreme
> one-sided bounds (structurally fixed by the same rewrite), `sample_array(spec, 0)`
> opaque numpy crash → clean empty, and a redundant `sample()` u↔recipe round-trip →
> shared draw path. Suite **358 green** (+16 M3 tests incl. 7 fix-guards), ruff +
> import-linter (1 kept, 0 broken) clean, torch-free base import preserved. Nothing
> committed. **► NEXT = no hard blocker in code; the standing USER items are M0 real
> data (#1 program risk) and the repo-folder rename `seminn`→`rig` (session-lock).**
> — prior: **Session 5**: spec `nnplan.md`→`implementation-plan.md`
> (DONE); repo folder `seminn`→`rig` (PENDING — blocked by the live-session lock; run
> `Rename-Item seminn rig` from the parent dir with the session closed, then migrate the
> `...github-seminn` memory dir). Two more WP-E slices landed + adversarially
> verified (4 LOW findings, all fixed+guarded): (a) **inner-loop surrogate** —
> `DeepEnsembleForwardModel.posterior_cov` (makes the ensemble EPIG-capable → the §9 AL
> loop now runs on backend B) + `sngp_member_view`/`inner_loop_surrogate` (fast
> single-member view, §5.7 option B, ~K× cheaper) + guarded AL-loop wiring + opt-in
> `revalidation_model` on the §8 solver (full-ensemble + §13.2 conformal re-validation;
> default None = M2 path unchanged); (b) **`BoTorchBO`** — production `SingleTaskGP`
> Matérn-5/2 + qLogEI/qLCB via `optimize_acqf`, the fair matched-budget comparator that
> closes M2 BF-1b (continuous BoTorch baseline). Suite **342 green**, ruff+import-linter
> clean, torch-free base import preserved. **► NEXT = BLOCKING USER DECISION: the M3 NPE
> generator flow library** (sbi vs zuko-only vs hand-roll — none installed). Nothing
> committed. — prior: Session 4: **WP-E UNBLOCKED — torch stack installed**.
> torch 2.11.0+cu128 + gpytorch 1.15.2 + botorch 0.18.1 live; RTX 5050 sm_120
> (Blackwell) GPU compute + autograd verified. First WP-E slice landed: the
> **deep-ensemble β-NLL + spectral-normalized SNGP forward tier** (D3 backend B),
> `src/rig/forward/ensemble.py` — same canonical `PredictiveDistribution` as the GP,
> OOD epistemic inflates ~14× (§5.9 inv. 1), conformal-wrapped PICP ≈ nominal, CUDA
> path green. 13 new tests, full suite **321 green** (torch-free base import
> preserved via lazy `__getattr__`), ruff + import-linter clean. Nothing committed.
> — prior: Session 3: **M2 empirical result** produced & honest —
> `docs/M2-result-2026-07-16.md`; v1 config adversarially refuted then rebuilt;
> real BO scalarization scale-bug fixed in `warm_bo.py`; suite 308 green,
> ruff/import-linter clean. Also: M0 dataset shortlist `docs/m0-dataset-candidates.md`;
> RTX 5050 laptop confirmed adequate for WP-E torch (needs cu128 / torch≥2.7).
> Session 3 (cont.): **IF-1 fully resolved** — feasibility-attribution readout in
> `_inverse_readout`; ablation + binding study prove the M2 cost win = *inverse loop
> beats BO* and §8 delivers calibrated feasibility where it binds (≥15σ). See M2 row
> + BUILD_LOG 2026-07-16 (cont.).)
>
> **▶ WHAT'S NEXT (suite 382 green, nothing committed):** (1) **M0 [USER]** — secure a
> real recipe→outcome dataset (shortlist in `docs/m0-dataset-candidates.md`); **#1 program
> risk and, per the Session-7 audit, the entire scientific claim** — no code blocked on it.
> (2) **DIMENSIONALITY — DONE 2026-07-17** (`docs/dimensionality-2026-07-17.md`,
> `examples/run_dimensionality_study.py`): the inverse works to **d=20** with ground-truth
> hits; the restart budget now scales with dim. **Follow-on owed:** an ANALYTIC objective
> gradient (`∂σ_epi/∂x`, `∂J/∂x` — a second derivative of the GP mean) to kill the O(d²)
> finite-difference cost; until then d ≳ 20 needs torch autograd or a cut budget. Also
> worth doing: a high-dim run on a NON-synthetic process once M0 lands.
> (3) **Repo-folder rename `seminn`→`rig` — the rename itself is DONE** (the repo is live
> at `...\github\rig`; Session 7 found the editable install still pointed at the old path
> and fixed it). **Still owed:** the orphaned `...github-seminn` memory dir was never
> migrated (both dirs exist). (4) **Re-run `run_m2_sweep.py` (~40-60 min)** — the shipped
> `docs/m2-result.json` is STALE (pre-IF-1 schema, no `verdict` key); its crash-before-write
> bug is now fixed, so the documented reproduce command finally works. (5) **M3 D2
> integration + end-to-end acceptance run — ALREADY DONE** (an unlogged session; see
> `src/rig/inverse/d2.py`, `docs/M3-acceptance-2026-07-17.md`) — but the audit rates the
> gate near-tautological with n=1 of real signal on a toy tanh; a re-run on the
> InSilicoMachine with a non-saturating pass rule is the honest version.
> (6) WP-E remainder: ~~qLogNEHVI~~/SCBO/TuRBO slate; ~~ensemble distillation~~;
> flow-typicality + PGD δ-box for §8; DKL deploy path. (qLogNEHVI Phase-II + distillation
> DONE 2026-07-17 eve, slice 5.) (7) Owed from the audit — **RESOLVED 2026-07-17 eve:**
> `ConstraintSet` is now WIRED into the §8 solver (hard reject independent of a soft barrier,
> verified SOUND — worst A@x−b over certified recipes = 0.0); **`ruff format` ADOPTED** (whole
> tree formatted, `ruff format --check` gate added to CI, formatter of record in CLAUDE.md);
> **Pandera DROPPED** (was declared but imported nowhere — removed from pyproject; frame
> validation stays an E1 item for when M0 lands).
> Session 2: full-codebase audit + all-33-findings remediation (272→306) + WP-D/F/G +
> M2 baseline. Session 1: audit + WP-A/B/C/H/I. Next hard blocker: WP-E torch stack
> needs a USER install decision (hardware is not the blocker; the install is).

## ✅ Audit findings (2026-07-16, docs/audit-2026-07-16.md) — ALL 33 FIXED

Remediation landed 2026-07-16 (see BUILD_LOG). Suite **306 passed** (34 new
guarding tests), ruff clean, import-linter KEPT. CI-path (no sim) = 265 passed,
41 skipped WITH a loud banner. Nothing committed.

- **P0 A1 (CI blind spot) — FIXED:** `tests/conftest.py` adds `RIG_REQUIRE_MBE_SIM=1`
  strict switch (missing sim → hard error) + a prominent skip banner; `ci.yml` runs
  `-rs`. CI no longer silently green-lights an un-run sim layer.
- **P1 defects — FIXED:** B1 simplex sum-to-1 (schema.py); B2 E2 split on the
  RecipeRecord path (mbe/machine.py); B3 JSONL completeness/catchable error
  (tabular/ingest.py); B4 `gp_input_keys` order-safe accessor (tabular/spec.py);
  B5 `posterior_cov` on the ICM model + for_tool (forward/multitask.py).
- **P1 test-gaps (C1–C11) — FIXED:** value-pinning tests added; each verified to
  fail against its described regression before landing.
- **P2 (D1–D14) — FIXED:** D1 OOB-inverse raise, D3 max_candidates guard, D4 arity
  annotation, D8 sobol guards, D10 dimensionless unit gate + the rest (docs/tests).

Two findings were correctly REFUTED and intentionally left as-is (recipe
completeness guarded by tested encoders; independent-variance RMST is by-design
per §12.4). No critical/wrong-answer defect existed; this was hardening, and it is
done — not a blocker for M0/WP-E.

## Program-level gates (implementation-plan §15.3)

| Gate | Status | Notes |
|---|---|---|
| M0 — secure real dataset (D1) | **GO — VENUE DECIDED BY USER 2026-07-19: Empa bipolar HiPIMS** (real tool, n=3,150, CC-BY-4.0, local + ingested; real temporal split exists for 5/6 campaigns via BatchNr — §15.3 M0 "GO if ≥ one process with a real split is secured" is met, with caveats: single tool → no leave-one-tool-out; BO-sampled; ti_120w order key unverified; QCM dep-rate + Ipk outcomes only. A fab/vendor agreement or own campaign remains the only path to a FULL-bar dataset). Previous status: | 2026-07-18 hunt: 25 candidates rated, 4 STRONG, **none meets the FULL bar** (`docs/m0-dataset-candidates-2026-07-18.md`). All 4 STRONG **downloaded + ground-truthed** into `data/m0-candidates/` (382 MB, `MANIFEST.md`): lead = **Empa bipolar HiPIMS** (real sputter tool, n=3,150, two 5-D pulse subspaces + categoricals, CC-BY-4.0, MD5-verified) — compromises: QCM-rate-only outcome, timestamps unverified. Runners-up: Ada SDL 2022_09 (n=177/180, license file 404s) + 2021_01 (n=253, no wall-clock timestamps); NREL HTEM degraded (nrel.gov dead → nlr.gov, API partly broken). **2026-07-19 (afternoon): the Empa E1 data-prep slice is BUILT** (`examples/real_data/empa_hipims/` — converter, 6 tidy CSVs n=3,150, 6 per-campaign TOML specs with verbatim BayBE bounds, 9 ingest tests green incl. the Ti-120W 5-row bounds-rounding edge and its all-BatchNr==1 degeneracy; see BUILD_LOG). NEXT: user picks the M0 venue; then temporal/LOTO split design (NB Ti-120W has NO run-order key; use oscilloscope serials) and the M1 gate re-run on REAL data. A fab/vendor agreement or own campaign remains the only path to a full-bar dataset. |
| Phase 0 — MBE bootstrap (§15.2) | IN_PROGRESS | WP-B below is its first slice (E2/E3). |
| M1 — calibrated forward on real split | **REAL-SPLIT DIRECTIONAL CHECK RUN ON THE M0 VENUE 2026-07-19 (full restarts, deterministic, orchestrator-verified). STATIC split-conformal: 5/6 PASS both splits; `ti_200w_high_pw` FAILS both (temporal under-cover 0.817 / random over-cover 0.950). ONLINE ACI (D4/§5.6) now WIRED: 6/6 PASS both splits — ti_200w FAIL→PASS with library defaults, no tuning; verified NOT to be an infinite-width artifact (finite-only 0.878 [0.804,0.932]).** Citable summary: `examples/real_data/empa_hipims/RESULTS.md`. The POWERED in-silico M1 criterion remains the primary gate per §15.3. **2026-07-22: BOTH remaining M1 items DONE — conformal-PID (§20.2 endpoint): 12/12 campaign/split gates PASS with `n_infinite_width=0` everywhere (the finite-by-construction repair); material-conditioned pooling: awareness gained (mean shift representable, unknown-material dominates, 0 coverage flips), cross-material transfer measured and NOT claimable (`results/m1_empa_pooled.json`).** **Conditional/per-region coverage study DONE 2026-07-23** (`run_conditional_coverage.py`, fidelity gate byte-equal to m1_empa.json 12/12, 4 pre-stated groups): pooled PASS hides a HIGH-OUTCOME-TAIL under-coverage (high-magnitude tertile under-covers 8/24 cells, 6 behind passing marginals; static 14 → ACI/PID 9 under-covering tertile-cells of 180); the online endpoints repair drift-conditional but NOT magnitude-conditional failure → **named owed remedy: Mondrian-by-magnitude group-conditional calibrator — BUILT AND EVALUATED same day 2026-07-23** (`rig.calibration.mondrian`, zero-solver-edit interface parity with the §13.2 gate; Empa: **6/8 hidden high-tail cells → nominal**, the 2 failures are the lowest predicted/observed agreement cells 0.66/0.48 — grouping-by-predicted can't isolate a tail the model can't predict; high-tertile MPIW 1.0-6.0×; over-conservative on ti_120w, broke its marginal from ABOVE; the selected-point d=8-style miss is mechanistically rejected). Far-from-data/late-drift are NOT the dominant hidden modes. **PID decaying-step side study DONE 2026-07-23** (`run_pid_step_study.py`, fidelity 12/12 exact): volatility cut to ~22% of fixed's uniformly BUT ti_200w flips PASS→FAIL on both splits at n≈100 stream lengths (short-horizon step-decay mechanism) → `step="decaying"` stays opt-in, discouraged on drifting tools; fixed remains the path of record. Build trail: machinery proven on real PUBLIC data 2026-07-16 | **2026-07-19: `examples/real_data/empa_hipims/run_m1_empa.py`** runs the §15.3 M1 directional check (nominal 0.90 inside the exact binomial 95% CI, per-output + pooled) on a REAL temporal split (BatchNr run order) of each Empa campaign, with a random-split contrast, a 4-campaign PRR-space OOD-epistemic check, and a 3-regime §8 inverse demo — smoke numbers in the banner above; full-run numbers owed. Still not the SIGNED M1 gate: M0 venue = user/PI decision, and this is not the MBE target process. The M1 GATE as originally scoped still needs the real MBE target-process split (M0). But the forward+conformal+inverse machinery is now demonstrated on GENUINE measured data (not in-silico): magnetron-sputtering SDL (`examples/real_data/sputtering/`, 209 measured runs, power×pressure → 3 QCM rates) ingested via the WP-H tabular adapter → GP + conformal → mean PICP **0.929** vs 0.90 nominal; §8 inverse returns diverse on-support recipes + correct INFEASIBLE. WP-C is the in-silico groundwork. **AUDIT CAVEATS 2026-07-17 (quote these WITH the 0.929, never without):** (a) **the coverage is bought with inflated noise** — fitted aleatoric is **10.5×/3.9×/5.2×** the CSV's own measured error, i.e. the GP absorbs model misfit into "noise" and widens bands (MPIW 6.66 on a ~40-wide output) until they cover; (b) **the inverse only fires at near-vacuous tolerances** — at the demo's ±6 it is genuinely right (19/19 nearest measured runs in-tol) but ±6 ≈ the conformal MPIW and 56% of all 209 runs already sit inside it; at ±2/±1/±0.5 it is **0/25 FEASIBLE** (honest abstention, but the certified real-data inverse only fires where the answer is nearly free); (c) this dataset is a dense **15×15 lookup grid over 2 knobs** — invertible by nearest-neighbour, so it is not a meaningful test of a *learned* inverse; (d) a real **UNITS defect** here (declared-unit bounds vs SI data) was found+fixed 2026-07-17 — see the `continuous_si` decision below. |
| M2 — inverse beats warm-started BO in-silico | PASS in-silico vs GP-EI BO (fixed-pool AND continuous-acquisition); cost verdict SURVIVED v2 re-validation + the BF-1a crux test (RIG ~2× cheaper, ΔRMST −17,850, p=1.8e-31); constrained-BO/BoTorch slate = WP-E | Empirical RESULT produced 2026-07-16 (`docs/M2-result-2026-07-16.md`, `docs/m2-result.json`). Powered 40-seed × 4-target sweep on the InSilicoMachine with **metrology noise ON**, coupled `T_center×bow` target, **metrology-anchored** tol + sensitivity curve, scale-fixed BO: RIG ΔRMST=−2.55e4 (p≈1e-156, win 93%, hit 1.00 vs BO 0.42); both-hit ΔRMST=−1.4e4; **RIG wins at every tol_k∈{2,3,4,6,8}** (not a knob artifact). Win attributed 100% to the inverse exploit. NOTE: a v1 config was adversarially REFUTED (9 attacks, TS1 fatal) → rebuilt honestly; found+fixed a real BO scalarization scale-bug (`warm_bo._distance_to_box` now tol-normalized). v2 6-lens re-validation DONE → 2 CONFIRMED claim-narrowings, BOTH now resolved: (BF-1a) added continuous-acquisition BO — doubles BO hit 0.45→0.90 but RIG still ~2× cheaper (ΔRMST −17,850); (IF-1) **fully resolved 2026-07-16** — feasibility-attribution readout wired into `_inverse_readout` (`verdict ∈ {FEASIBLE_CERTIFIED, INFEASIBLE_FALLBACK}`), ablation shows pessimism adds only a marginal non-sig edge on M2 cost (ΔRMST −1,083, p=0.053) so the cost win = *inverse loop beats BO*, and a binding study shows §8 *does* bind (0% FEASIBLE at 6σ → 100% at ≥15σ, spec-tol-driven not data-driven) and delivers **calibrated feasibility** there (0% vs 7.8% false-accept, ~95% credited-interval coverage) — a payoff on an axis orthogonal to M2 cost. Runs: `scratchpad/if1_{diagnose,ablation,binding}.py`. REMAINING: **BF-1b PARTIALLY CLOSED 2026-07-17** — `rig.baselines.BoTorchBO` adds the production continuous BoTorch comparator (SingleTaskGP Matérn-5/2 + Hvarfner √D prior + qLogEI/qLCB via `optimize_acqf`, fair matched-budget, bit-identical warm-start, plain posterior); ~~still owed: constrained-BO/SCBO + qLogNEHVI/TuRBO slate~~ **BoTorch slate DONE 2026-07-23** (`rig.baselines.trust_region_bo`, faithful TuRBO-1 + SCBO, steelmanned by known-answer sanity; 50 seeds × 2 targets: the ~2× claim HOLDS vs the full slate — 1.65× vs BoTorchBO the strongest arm, 2.46× TuRBO, 2.89× SCBO, all significant, no comparator wins; `docs/m2-botorch-slate-2026-07-23.md`; optional follow-on: binding-policy slate re-run). Remaining: full-pathology-surface rerun (A1-1). **Binding-policy re-run DONE 2026-07-22** (`--policy` flag; `docs/m2-result-binding.json`, 50 seeds, 138 min): the M2 cost-win **SURVIVES** the binding §8 2.0/2.0/0.02 policy, attenuated ~35% — pooled ΔRMST −16,480 (was −25,530 ablation), CI [−18.2k, −14.7k], p=8.7e-70, win 82%, rig hit 0.99, rig RMST +61%; both-hit median saving 15k→5k; holds at every tol_k 2–8; BO arm provably untouched and invariant (40→50 seeds) → the attenuation is the policy, not the seed count. Binding abstains ~9× harder (mean distance_to_feasible 1.60σ→14.84σ); at 6σ both policies abstain 100% (consistent with IF-1). Determinism: 3-seed subset byte-identical (24/24). **Audit F3 binding-policy item CLOSED.** **Ablation@50 refresh DONE 2026-07-23**: docs/m2-result.json refreshed (post-IF-1 schema, 50 seeds); published numbers reproduce to <1% (pooled ΔRMST ≈ −25.75k, CI [−27.4k, −24.1k]; both-hit −15k/median 15k); equal-seed comparison confirms the binding attenuation ≈36% (−25.75k → −16.48k @ 50 seeds each). Stale-JSON caveat closed. |
| M3 — amortized posterior + AL loop | **AMORTIZED GENERATOR + SBC/TARP GATE DONE (2026-07-17, Session 6); end-to-end M3 acceptance run owed** | WP-F AL loop DONE (numpy tier); **2026-07-17: the AL loop runs on backend B (the deep ensemble) end-to-end** — `DeepEnsembleForwardModel.posterior_cov` makes it EPIG-capable, inner loop on the SNGP-member fast view (§5.7). **The amortized NPE posterior (§14.3) is now built + gated**: `rig.inverse.AmortizedInverseGenerator` — deep-ensembled zuko conditional neural spline flows (`NSF`) trained in `RecipeTransform` `u`-space (constraint-by-construction), region-augmented box conditioning (D2), mixture `sample`/`log_prob`; §14.6 **SBC/TARP blocking gate** `validate()` (WP-G `calibration_gates`) that PASSES a calibrated flow and BITES an undertrained one. Adversarially verified (6 CONFIRMED findings, all fixed+guarded incl. the HIGH single-member-vs-mixture gate bug + the empirical-prior/spec-context-width MED bugs). 17 tests. **CORRECTED 2026-07-17 (Session-7 audit): D2 integration and the end-to-end M3 acceptance run are DONE, not owed** — `src/rig/inverse/d2.py::AmortizedRefiner`, `examples/run_m3_acceptance.py`, `docs/M3-acceptance-2026-07-17.md` (verdict PASS, 5/5, 0.281× cold-heavy budget; reproduces byte-identically). That work landed in an **unlogged session** and this row said "owed" for a day. **But read the PASS narrowly:** the audit rates the gate near-tautological — the rule is `d2_light_conf ≥ cold_heavy_conf − 0.02` while every confidence saturates at 0.9997–1.0, `cold_light` already succeeds in 4/5 cells, so the "amortization fills the gap" claim rests on **n=1 of 5 targets**; it is scored against the GP fit on the same 220 samples that trained the generator (ground truth never called during eval — 10/10 returned recipes were separately verified to hit it, so harmless here); and it runs on a **toy tanh, not the InSilicoMachine**. Session-7 also fixed 3 real generator defects (u-space `log_prob`, skewed small-n sub-mixture, frozen sampling RNG). **Honest M3 re-run DONE 2026-07-22** (`examples/run_m3_acceptance_v2.py` + `docs/M3-acceptance-v2-2026-07-22.md`, verdict PASS, deterministic): InSilicoMachine + metrology noise, non-saturating GROUND-TRUTH pass rule, pre-registered cold_light-probe targets (3 genuine INFEASIBLE + 3 controls), §14.6 gate PASS as a blocking precondition, binding 2.0/2.0 policy. d2_light 6/6 = cold_heavy 6/6 > cold_light 3/6 at 0.19× budget. **Read narrowly:** the agent's own adversarial control shows 9 RANDOM restarts also rescue 3/3 — the win is restart-budget, not amortized-vs-random; that stronger claim is NOT demonstrated on this smooth near-invertible 2→2 map and is not made. Scope was 2→2 because a near-deterministic identity output (`thickness_grown`) breaks SBC calibration — recorded gotcha. v1's PASS stays on record as the toy-tanh result it was. |
| M4/M5 — real campaign / certification | BLOCKED on M0 | |

## Work packages (session-sized, dependency-ordered)

| WP | Scope (implementation-plan refs) | Status | Depends on |
|---|---|---|---|
| WP-A: Foundation — pyproject (uv-compatible), src layout, `rig.interfaces` (ProcessAdapter, ForwardModel, InverseSolver, QualificationGate, PredictiveDistribution), `rig.schema` (RunRecord/RecipeRecord/OutcomeRecord/Provenance, Pint SI canonicalization), `rig.registry` (entry-point discovery), `rig.constraints` + `rig.transforms` (box-sigmoid, softmax/ILR simplex + round-trip), import-linter contract, pytest+Hypothesis suite, CI stub (§3, §13) | **DONE** (2026-07-15, 42 tests passing; ruff + import-linter clean) | — |
| WP-B: MBE adapter + in-silico pathology machine — `rig_adapters.mbe`: recipe-vs-config split (E2), snapshot()→OutcomeRecord translation, cost model + DoE hooks; `InSilicoMachine` wrapper with injectable seasoning / first-wafer offset / heteroscedastic metrology noise / second perturbed "chamber" / tool_id + seeded determinism (E3, §10, §15.2); Sobol seed-design data generation → RunRecords | **DONE** (2026-07-15, 36 tests; entry point `mbe` live; fixture `tests/fixtures/mbe_silico_smoke.jsonl`). Next: finite-difference sensitivities for Sobolev distillation (§6.1, Phase 0 task ii). **D7 different-physics ROM verifier DONE 2026-07-22** — `verifier.py::GeometricDepositionVerifier` (geometric Knudsen line-of-sight flux ROM, no shared lineage/code with the fast path, mechanically test-enforced), bounds the flux-scale `thickness_grown` channel; combined nonuniformity is ~98% thermal → honestly out of scope; wired into `independent_verifier`, D7 identity check passes with a real verifier | WP-A |
| WP-C: Forward surrogate v0 (little-data regime, D3) — exact GP (Matérn-5/2 + ARD, numpy/scipy), heteroscedastic constant-floor noise v0, `PredictiveDistribution` provider, `support_score` (Mahalanobis fallback), conformal layer (split-CQR-lite + jackknife+/CV+ + online ACI), metrics (CRPS closed-form, PICP/MPIW, PIT) (§5, D4) | **DONE** (2026-07-15, `rig.forward` + `rig.calibration` + `rig.metrics`; 27 new tests, full suite 105 green, ruff + import-linter clean). Next: wire to WP-B's Sobol RunRecords for an end-to-end in-silico fit; heteroscedastic aleatoric + time-decay kernel deferred (needs replicates / drift data). | WP-A |
| WP-I: Tool-aware forward surrogate (§10.4 level (a), chamber matching) — `rig.forward.multitask.MultiToolGPForwardModel`: ICM multi-task GP (Bonilla 2008), kernel k_Matern52·B[s,t] with B = WWᵀ+diag(v); few-shot `adapt_to_tool`; unknown-tool population fallback with inflated epistemic; per-tool support_score; `records_to_arrays_with_tools`; `for_tool()` view for tool-blind wrappers. Motivated by 2026-07-15 user signal (multi-tool switching) | **DONE** (2026-07-15, 23 new tests, full suite 188 green; in-silico proof: 40 A-runs + 4 B-runs beats pooled-blind AND scratch-on-B; §5.8 LOTO epistemic check passes; MBE integration multi 4.0e-12 m vs pooled 3.0e-8 m RMSE on `thickness_grown`). Next: RGPE negative-transfer fallback (future WP); torch-era per-tool FiLM/ANP (WP-E); Mondrian per-tool conformal once a tool has a few dozen runs. **In-silico M4 dress rehearsal DONE 2026-07-23** (`run_multitool_rehearsal.py`, physics_sim REHEARSAL): 3-tool fleet, §5.8 LOTO 3/3, few-shot pooling HELPS at equal n, EPIG onboarding 2.97 nats (collapse guard green on the historically fragile posterior_cov→epig seam), solve→auto-qualification end-to-end (2/2 certified reachable; NothingToQualify/0-calls unreachable). **Phase 4b (onboarded-tool conformal wrap) DONE 2026-07-23**: per-tool fit/cal split (trailing-1/3 chronological), candidates upgrade to conformal-checked with ZERO src edits; the natural n_cal=8 case honestly yields an infinite band → gate rejects the raw-admitted candidate (fail-closed demonstrated); +1 exchangeability-preserving run → finite band + certified campaign. Phases 1-4 byte-identical to the recorded artifact. Pattern note for real M4: calibrate on a held-out slice of the onboarded tool's runs, never on the fully-fit model. | WP-B, WP-C |
| WP-D: Per-query pessimistic inverse (§8) — `rig.inverse.PessimisticInverseSolver` (InverseSolver protocol) + `SpecBox`/`parse_targets`. Reparameterized (RecipeTransform) Sobol multi-start + L-BFGS-B; robust worst-case credited interval ⊆ spec box; epistemic worst-member (z_epi·σ_epi) + δ box (‖J⊙Δ‖₁ §8.5 Taylor) + κ·σ_ale credited band + support fail-closed reject (§8.2); farthest-point diverse pre-image; INFEASIBLE with nearest point + relaxation + 4-way §8.8 cause diagnosis (via nominal-feasibility probe). Recipes GIVEN a tool via `for_tool` (never searched) | **DONE** (2026-07-15, 25 new tests [23 synthetic + 2 sim-gated MBE], full suite 213 green; ruff + import-linter clean; 2 adversarial review rounds → 3 code defects + 4 test gaps fixed & guarded). Next: Loop-B online hook (ACI + realized-vs-predicted gap → κ,τ) is WP-F. **2026-07-17 evening: linear constraints DONE** (`ConstraintSet` box+coupling via hard reject `_admissible` INDEPENDENT of the soft log-sigmoid barrier — verified SOUND: worst `A@x−b` over adversarial certified recipes = 0.0, simplex+constraint raises `NotImplementedError` fail-closed at construction) **and an opt-in analytic objective gradient DONE** (`analytic_grad=True`, `_GPTermProvider`: closed-form ∂σ_epi/∂x + Hessian ∂J/∂x replacing SciPy finite-differencing, the F9-owed item — proven correct to 5.38e-8 vs a 60-digit mpmath reference). qLogNEHVI multi-obj Pareto + amortized generator are WP-E (both now DONE). **2026-07-22 (audit F1/F5): the §13.2 conformal containment `C(x)⊆Z*` is DEFAULT-ON in `solve()` whenever the model is conformal-wrapped (no `revalidation_model` needed); candidates carry `calibration_status ∈ {model-feasible, conformal-checked, revalidated}` — "model-feasible" is explicitly NOT a calibrated guarantee; docstrings carry the §8 approximation ledger.** | WP-B, WP-C |
| WP-E: Torch stack — deep ensemble + β-NLL + SNGP trunk (D3 >300-run regime), distillation (§5.7); ALSO the deferred torch pieces of WP-D/F/G (normalizing-flow typicality, PGD, amortized generator/NPE flow for SBC/TARP, qLogNEHVI Phase-II) | **IN_PROGRESS — UNBLOCKED 2026-07-17** (torch 2.11.0+cu128 / gpytorch 1.15.2 / botorch 0.18.1 installed; RTX 5050 sm_120 GPU verified). **Backend-B forward tier DONE:** `DeepEnsembleForwardModel` (`src/rig/forward/ensemble.py`) — K-member heteroscedastic β-NLL(β=0.5) ResMLP + spectral-normalized (bi-Lipschitz) trunk + RFF-GP SNGP-Laplace last layer; same canonical `PredictiveDistribution`; epistemic = Var[μ_m] + E[SNGP-Laplace var]; support_score = Mahalanobis in the spectral latent (§8.2/§11); autograd jacobian; AdamW/cosine/early-stop (§5.7). 13 tests (§5.9 inv-1 OOD ~14×, β-NLL no-collapse, conformal PICP≈nominal, determinism, CUDA path). | torch install DONE. WP-A |
| — WP-E slice 2 DONE (2026-07-17): **SNGP-single-member inner-loop path** (§5.7 option B) — `DeepEnsembleForwardModel.{posterior_cov, sngp_member_view, inner_loop_surrogate}` + guarded AL-loop wiring (`loop.py`: fast view for the §8 solver + EPIG/BALD; GP path unchanged) + opt-in `revalidation_model` on the §8 solver (full-ensemble + §13.2 conformal C(x')⊆Z* re-validation). `posterior_cov` = ensemble-spread + SNGP-Laplace joint cov (diagonal == epistemic_sigma², so EPIG is consistent) → the ensemble is now a full `_JointModel` and the §9 AL loop runs on backend B. Adversarially verified (4 LOW findings, all fixed+guarded). REMAINING here: ensemble DISTILLATION → single distributional net (§5.7 option A, the ≥20× / `/invert` serving path). | **DONE** | ensemble tier DONE |
| — WP-E slice 3 DONE (2026-07-17): **`rig.baselines.BoTorchBO`** — production continuous BoTorch comparator (`SingleTaskGP` Matérn-5/2 + §20.5 Hvarfner √D prior + qLogEI/qLCB via `optimize_acqf`); fair matched-budget (bit-identical warm-start, plain posterior, machine-query budget, same hit rule/Trajectory), continuous-only, lazy-imported (base torch-free). Closes M2 **BF-1b** (continuous BoTorch baseline). 10 tests incl. m2_sweep drop-in. REMAINING: qLogNEHVI (multi-obj) + SCBO/constrained-EI + TuRBO for the full §12.3 slate; a powered M2 re-run with BoTorchBO. | **DONE** | ensemble tier DONE |
| — WP-E slice 4 DONE (2026-07-17, Session 6): **amortized NPE flow generator (§14.3) + SBC/TARP gate (§14.6) = the M3 gate** — flow-library decision resolved to **zuko** (user). `rig.inverse.AmortizedInverseGenerator` — K deep-ensembled zuko `NSF` conditional neural spline flows trained in `RecipeTransform` `u`-space (constraint-by-construction), region-augmented box conditioning (D2), mixture `sample`/`sample_array`/`log_prob`; `validate()` = the §14.6 SBC/TARP blocking gate on WP-G `calibration_gates`. Lazy-imported (base torch-free). Adversarially verified (workflow, 6 CONFIRMED, all fixed+guarded): HIGH single-member→**mixture** gate draw; MED default-prior→**empirical bootstrap**; MED `_spec_context` unconstrained/one-sided→**max trained box width** (was 6σ/width-12 OOD); LOW inverted-box, `n==0` crash, redundant `sample()` round-trip. 16 tests. | **DONE** | ensemble tier DONE |
| — WP-E slice 5 DONE (2026-07-17 evening): **ensemble distillation (§5.7 option A)** — `src/rig/forward/distill.py` (`distill_ensemble`, `DistilledForwardModel`): a single distributional student trained on the deep-ensemble teacher, preserving the aleatoric/epistemic SPLIT (verified: tracks teacher aleatoric corr 0.99999, epistemic corr 0.9999), canonical PredictiveDistribution, out-of-transfer-box guard `_box_excess`. **AND Phase-II `qlognehvi_phase2`** (`src/rig/active/acquisition.py`) — real botorch qLogNEHVI multi-objective acquisition (was `NotImplementedError`). Both adversarially verified SOUND (distill PARTIAL: core sound, one OOD-guard test-gap fixed). 26+25 tests. | **DONE** | slices 2-4 DONE |
| — WP-E remaining (dependency-ordered): (1) ~~SCBO/constrained-EI + TuRBO BoTorch slate + powered M2 re-run~~ **DONE 2026-07-23** (see the M2 row: claim holds vs the full slate, 1.65-2.89×); (2) ~~normalizing-flow typicality + PGD δ-box for §8~~ **DONE 2026-07-22** (opt-in `delta_mode="pgd"` + `typicality=FlowTypicalityScore`; benchmarked + red-proofed; 15 tests in `test_inverse_hardening.py`; see BUILD_LOG); (3) DKL / SNGP-single deploy path. | (1),(3) TODO | slices 2-5 DONE |
| WP-F: Active learning loop (§9) — `rig.active`: cost-cooled `[λ·EPIG+(1−λ)·BALD]/cost^β` (BALD closed-form; EPIG via GP `posterior_cov`), anneal λ 0.2→0.9 / β 1→0 (CArBO), greedy-submodular diverse batch (§9.5), closed loop (Sobol DoE → refit → re-solve inverse → 1 exploit+(q−1) explore → stop target/budget/stall). qLogNEHVI Phase-II = WP-E (NotImplementedError) | **DONE** (2026-07-15, numpy tier; ~22 tests incl. sim-gated MBE cost-to-target; 3-lens adversarial review → HIGH hit-detection bug [cost-to-target counted only the exploit, biasing M2 vs the BO baseline] + 2 test gaps, all fixed & guarded). Next: qLogNEHVI Phase-II hand-off, multi-fidelity KG, drift-kernel/ADWIN, offline re-distillation — all WP-E/later | WP-C, WP-D |
| M2-baseline: Warm-started GP-EI BO (§9.8/§12.3) — `rig.baselines.WarmStartedBO`, the fair matched-budget comparator; + `tests/test_m2_comparison.py` (RIG loop vs BO → difference-in-RMST) | **DONE** (2026-07-15). Next: BoTorch qLogEI/qLogNEHVI/TuRBO + MFL-round-trip + cINN/NPE baselines = WP-E | WP-F, WP-G |
| WP-G: Evaluation harness (§12) — `rig.eval`: KM/RMST/diff-in-RMST cost-to-target survival (infeasible-excluded), inverse metrics (hit-rate/false-success/false-abstention/constraint-sat/robust-hit), §12.1 exploitation stress test, Vendi diversity, SBC+TARP gates | **DONE** (2026-07-15, 36 new tests incl. sim-gated non-circular exploitation; 5-lens adversarial review → 6 confirmed fixed incl. HIGH TARP-discreteness bug). Next: log-rank caveated-secondary, CD-diagrams/BH-FDR reporting layer, wire SBC/TARP to the WP-E amortized flow | WP-B..D |
| WP-H: Generic tabular adapter (E5 seed + E1 slice) — `rig_adapters.tabular`: declarative process-spec config (variables/bounds/units/categoricals/outputs/cost) + CSV/JSONL → RunRecord ingestion, adapter conformance checks, onboarding doc. Motivated by user requirement 2026-07-15: "I might feed data that's not MBE" | **DONE** (2026-07-15, 60 new tests, full suite 165 green; entry point `tabular` live; TOML spec format; `examples/*.toml` + `docs/new-process-onboarding.md` shipped; end-to-end CSV→GP+conformal proof in `tests/test_tabular_ingest.py`). Next: full E5 conformance harness (constraints-vs-data, transform round-trips); curve_1d/field_2d modalities; censoring-at-ingest. **E1 frame validation DONE 2026-07-23** — `rig_adapters.tabular.validation`, wired non-breakingly into `ingest_csv` (`IngestResult.frame_report`, opt-in `strict=`) + `prepare_empa.py` summary; Empa quirks pinned row-for-row; 34 tests, red-proofed. Open question recorded: whether `strict=True` becomes the default for newly onboarded processes once E5 lands. | WP-A |
| E1 full ETL (MES/metrology join) / E4 budget / E5 full runbook | TODO | user input needed for E1 source + E4 numbers |

## 2026-07-21 inverse-capability audit outcome

`audit.md` records that RIG is **not deployment-ready** for output-to-input prediction: default
inverse feasibility does not require conformal revalidation, and independent real-tool
qualification is not wired into inverse/active-loop execution. The audit also flags the active
loop/M2 1.0/1.0 conservatism settings against the binding 2.0/2.0 policy, incomplete robust
objective features, stale revalidation documentation, limited prospective data evidence, and
current ruff failures. See `audit.md` for remediation order.

**REMEDIATION STATUS 2026-07-22 (see the top banner + BUILD_LOG 2026-07-22):** F1 fixed
(conformal containment default-on when wrapped + `calibration_status` labeling + red-proofed
regression test); F2 fixed at the orchestration layer (`ConfirmationCampaign`) AND wired into
`ActiveLearningLoop` as an opt-in hook gating both target-met stop points (budget-honest,
rejection continues the loop, default-None byte-identical; runner-level auto-invocation still
open); F3 fully closed (loop 2.0/2.0/0.02; M2 labeled ablation AND the binding-policy re-run
ran same day — the cost win survives, attenuated ~35%, see the M2 row); F4 addressed to the extent currently possible — material-conditioned pooling
built, awareness demonstrated, **cross-material transfer measured and explicitly NOT claimed**
(prospective multi-tool/replicate data remains the real F4 closure and is M4/M5); F5 made
explicit in APIs/docstrings (PGD + flow typicality remain WP-E TODO); F6 fixed (repo ruff
green; import-linter verified with the corrected form); F7 fixed (docs annotated). The audit's
bottom-line product statement ("research prototype that can propose and abstain; not
demonstrated reliable output-to-input prediction for deployment") REMAINS accurate until
prospective qualification on real tools exists — the machinery for that qualification now
exists and is tested.

## Session-1 audit outcome

implementation-plan.md audited and corrected 2026-07-15 — see `docs/audit-2026-07-15.md`.
Verdict: sound and buildable; 12 editorial/citation fixes applied; 0 hallucinated
citations; E2/E3 repo claims verified true against `..\MBE sim`.

## Standing decisions made during the build (add here as they happen)

- 2026-07-15: package name `rig`, src layout (`src/rig`, `src/rig_adapters`); ASCII
  field names `aleatoric_sigma`/`epistemic_sigma` for the canonical
  PredictiveDistribution; adapters entry-point group `rig.adapters`.
- 2026-07-15: session-1 scope deliberately torch-free (numpy/scipy GP = the plan's
  primary small-n backbone anyway, D3); torch/BoTorch stack is WP-E with its own
  install step.
- 2026-07-15: "the machine" for Phase 0 = the MBE sim's fast Arrhenius/regime path
  wrapped in the pathology layer; the kMC `ZoneEnsemble` path = high-fidelity check.
  NB (D7/E2): the regime path shares physics lineage with kMC ⇒ neither can be the
  independent verifier; the different-physics ROM verifier is still owed in Phase 0.
  — **RESOLVED 2026-07-22:** `rig_adapters.mbe.verifier.GeometricDepositionVerifier`
  supplies it (purely-geometric flux ROM, no shared lineage/code with the fast path);
  bounds the flux-scale `thickness_grown` channel only — thermal-dominated nonuniformity
  and thermo-mechanical outputs explicitly out of scope. See BUILD_LOG 2026-07-22.
- 2026-07-15 (WP-A): core interfaces are structural `typing.Protocol`s, not ABCs —
  adapters conform without importing rig base classes; `registry.get_adapter()`
  always runs `interfaces.validate_adapter()` (includes the D7 physics≠verifier
  identity check). Entry points in group `rig.adapters` must resolve to a
  zero-arg/kwargs-only factory returning the adapter instance.
- 2026-07-15 (WP-A): units — single shared pint registry `rig.schema.ureg` (defines
  `sccm`, `slm`); all `Quantity{magnitude, unit}` values SI-canonicalized inside the
  Pydantic validator. Never create a second `UnitRegistry`.
- 2026-07-15 (WP-A): compositional recipe values are flattened as
  `"<variable>.<component>"` keys (e.g. `alloy.ga`) in `RecipeRecord.values`,
  `RecipeTransform` dicts, and `ConstraintSet` — binding for WP-B/WP-D.
- 2026-07-15 (WP-A): simplex reparameterization = ALR / fixed-gauge softmax
  (`x = softmax([u, 0])`, inverse `u_i = log(x_i/x_K)`), chosen over ILR
  (rationale in `src/rig/transforms.py` docstring).
- 2026-07-15 (WP-A): Windows Store Python puts no console scripts on PATH — run
  tools as `python -m ruff ...`, `python -m pytest ...`, and import-linter via
  `python -c "import sys; from importlinter.cli import lint_imports; sys.exit(lint_imports())"`
  (CI uses `lint-imports` which works on GitHub runners).
  **CORRECTED 2026-07-17 (audit):** the form used through Session 6 —
  `python -c "from importlinter.cli import lint_imports; lint_imports()"` —
  DISCARDS the return value. `lint_imports` is a plain function returning an int
  (1 = broken), NOT a click command that calls `sys.exit`; only the
  `lint-imports` console script wrapper propagates it. The old one-liner
  therefore exits 0 **even when a contract is BROKEN** (reproduced 2026-07-17 by
  injecting `rig.registry -> rig_adapters.tabular.spec`: report said "0 kept, 1
  broken", exit was 0). Every "import-linter clean, exit=0" claim before
  2026-07-17 is evidence-free on the exit-code half — the printed report was
  still correct, so a human reading stdout was fine, and the contract does
  genuinely hold today (verified with the corrected form: 1 kept, 0 broken,
  exit 0). NB a second trap: redirecting stdout (`> /dev/null`) makes the `§` in
  the contract name raise UnicodeEncodeError on the Windows console encoding →
  exit 1, a FALSE FAILURE. Read the printed report, or pipe through
  `PYTHONIOENCODING=utf-8`.
- 2026-07-15 (WP-C): ACI update rule (binding, §5.6/D4): per-output
  `alpha_{t+1} = clip(alpha_t + gamma*(alpha_target - err_t), 0.001, 0.5)`,
  gamma=0.05 default; err_t is scored against the PRE-update interval; each
  observed score is appended online to the split calibrator's buffer (this,
  not alpha alone, is what re-widens bands after drift). Rolling coverage
  over a trailing window (default 50) is the drift detector.
- 2026-07-15 (WP-C): jackknife+ vs CV+ switch point: LOO for n <= 40,
  K-fold CV+ (K=10) above. Conformal quantiles return +inf when
  ceil((1-alpha)(n+1)) > n (never a clamped finite band).
- 2026-07-15 (WP-C): forward-model shape contract (binding for WP-D):
  `predict((d,))` -> fields shaped (m,), `predict((n,d))` -> (n,m);
  `conformal_set` = (m,2) / (n,m,2) interval array (None when unwrapped);
  `jacobian` takes a single (d,) point -> (m,d) in raw units;
  `support_score((d,))` -> float (higher = more in-distribution; it is a
  NEGATIVE Mahalanobis distance, max 0 at the training mean), batch -> (n,).
- 2026-07-15 (WP-C): aleatoric floor v0 is a constant fitted GP noise std
  per output (§10.3 identifiability-honest at small n); heteroscedastic
  aleatoric waits for replicate data.
- 2026-07-15 (WP-B): the E2 recipe-vs-config split (binding for WP-D/WP-F):
  RECIPE (EASY, the inverse's search space) = `T_heater` [K, 1150-1500] +
  `film_thickness` [m, 2e-7..5e-6]; MACHINE-CONFIG (HARD_TO_CHANGE,
  split-plot whole-plot, conditioning-not-free) = `heater_radius, gap,
  source_offset, source_height, aim_offset` (DEFAULT_KNOBS bounds, mirrored
  + drift-tested against the sim repo). `expert_ranges`/`seed_design` span
  RECIPE vars only; the run's machine config is recorded in
  `RunRecord.extra["machine_config"]`.
- 2026-07-15 (WP-B): declared MBE outputs (all scalar_vector) =
  nonuniformity_pct [percent -> serialized as dimensionless FRACTION],
  T_center [K], slip_max_ratio [-, upper_spec=1.0], bow_cooldown_um [um ->
  serialized m], thickness_grown [m]. thickness_grown is the flux-scale
  channel: normalized uniformity outputs are blind to a pure flux change,
  so seasoning/tool flux pathologies act via achieved thickness.
- 2026-07-15 (WP-B): all sim access via `rig_adapters.mbe.simlink`
  (`MBE_SIM_PATH` env var; lazy import; `sim_available()` gates test skips).
  Fast Arrhenius path only (24 nodes / 24 phi default); kMC ZoneEnsemble is
  never called by adapter, machine, tests, or datagen.
- 2026-07-15 (WP-B): `mbe` entry point is live in pyproject
  ([project.entry-points."rig.adapters"]) — re-run
  `python -m pip install -e ".[dev]"` after edits to entry points, or
  `registry.get_adapter("mbe")` serves stale metadata.
- 2026-07-15 (WP-B): InSilicoMachine determinism contract: same
  (PathologyConfig, seed, machine_config, run sequence) => bit-identical
  serialized RunRecords (uuid5 run ids, base+1h*run_index timestamps,
  sha256(tool_id)-keyed perturbations, per-run-index noise streams). Hidden
  state never enters RunRecords; `state_snapshot()` is the ground-truth
  hook for §12.1 figures. Censoring flags live in
  `RunRecord.extra["censored"]` (WP-A follow-up: consider a first-class
  OutcomeRecord field at E1 ingest time).
- 2026-07-15 (WP-H): process-spec format for the generic tabular adapter is
  **TOML primary** (stdlib tomllib; JSON accepted by suffix). Specs reuse
  the WP-A variable/output dataclasses; validation is strict at load
  (`SpecError` names the offending key). The E5 mixture-vs-flows tag is
  enforced at load: a `compositional` block with a non-dimensionless unit
  (e.g. sccm) is rejected, citing implementation-plan §3.1.
- 2026-07-15 (WP-H): parameterized-factory pattern (BINDING template for
  future config-driven adapters): entry-point factory takes its config as an
  optional kwarg (`spec_path=None`) → falls back to a documented env var
  (`RIG_TABULAR_SPEC`) → else raises an actionable LookupError. Never a
  silent default. Call paths: `TabularAdapter.from_spec(path)`,
  `registry.get_adapter("tabular", spec_path=...)`, or env var + bare
  `get_adapter("tabular")`.
- 2026-07-15 (WP-H): ingestion contract: CSV values are read in the
  SPEC-DECLARED units and SI-canonicalized via the shared `rig.schema.ureg`;
  compositional CSV columns are `"<var>.<component>"`; missing required
  columns hard-error, unmatched columns → warning +
  `extra["unmatched_columns"]`; `on_error="raise"|"skip"` (skip returns a
  rejects report in `IngestResult`); sum-to-1 atol 1e-6; absent timestamp
  column ⇒ deterministic synthetic ladder + `extra["synthetic_timestamp"]=
  True` (temporal splits over synthetic order are meaningless, §12.4).
- 2026-07-15 (WP-H): GP fitting over a compositional process must drop one
  component per compositional variable (sum-to-1 ⇒ exact collinearity) —
  documented in docs/new-process-onboarding.md and the integration test.
- 2026-07-15 (user signal): switching between machines/tools is an expected usage
  pattern — prioritize implementation-plan §10.4 level-(a) chamber matching (tool-aware surrogate:
  partial pooling / per-tool adaptation, few-shot new-tool onboarding, RGPE fallback)
  when scheduling the post-WP-D work. v0 GP is tool-blind; interim guidance = per-tool
  fit or pooling, with support_score/ACI-coverage as the "model doesn't transfer" alarm.
  [RESOLVED same day by WP-I for the GP era: `MultiToolGPForwardModel`; RGPE +
  torch FiLM/ANP remain future work.]
- 2026-07-15 (WP-I): ICM tool kernel (binding while the GP backend is primary):
  k((x,s),(x',t)) = k_Matern52_ARD(x,x')·B[s,t], B = WWᵀ + diag(v), rank default 1,
  v log-parameterized (>0), UNIT-variance Matérn factor (sf2 lives in B); one
  constant noise sn2 per output shared across tools (the §10.3 floor v0).
  Standardization pooled across tools (per-tool y-standardization would erase the
  tool offsets B must learn). Fit = exact NLML, analytic gradients, same
  L-BFGS-B multi-start as GPForwardModel via the shared private module
  `rig.forward._gp_common` (extracted from gp.py; GPForwardModel public
  behavior unchanged).
- 2026-07-15 (WP-I): unknown-tool prediction (zero runs, `tool_id=None`, or
  declared-but-dataless) is the B-weighted population fallback — per output:
  w_t ∝ Σ_s max(B[t,s],0); μ_u = Σ w_t μ_t; σ²_epi,u = max_t σ²_epi,t +
  Σ w_t(μ_t−μ_u)² + (1−ρ̄²)·mean_t B[t,t], ρ̄ = mean pairwise tool correlation
  clipped to [0,1] (0 at T=1). Unknown-tool epistemic therefore dominates every
  known tool's elementwise (§5.8 LOTO check holds by construction). Never
  silently treat an unseen tool_id as known.
- 2026-07-15 (WP-I): tool-blind consumers (ConformalForwardModel, WP-D inverse)
  take the tool-bound view `model.for_tool(tool_id)` (ForwardModel-conformant);
  the wrappers themselves stay unmodified. `records_to_arrays_with_tools` is the
  opt-in (X, Y, tools) sibling in rig.forward.data; the tool-blind function's
  signature is untouched.
- 2026-07-15 (E4 compute baseline, user hardware question): workload is
  small-model/many-experiments — GPU VRAM is not the constraint; throughput + CPU are.
  Recommended single-box spec: RTX PRO 5000 Blackwell 48GB (~$4.2k Jul-2026; 2x RTX
  PRO 4500 32GB for parallel sweeps at same cost; PRO 6000 96GB only for paper-2
  scale), Threadripper PRO ~32c, 128-256GB ECC, 2-4TB NVMe + DVC remote, WSL2/Linux
  for the WP-E torch stack. Current numpy/scipy stack needs NO GPU; defer purchase
  until WP-E is scheduled (pro-GPU prices moved >50%/yr).
- 2026-07-15 (WP-D): pessimistic-inverse §8 → GP-tier realization (binding while
  the GP backend is primary): the §8.1 objective is a ROBUST worst-case CREDITED
  INTERVAL ⊆ spec box, per constrained output j with standardized margins
  `u_hi=(U−μ−s)/σ_ale`, `u_lo=(μ−L−s)/σ_ale`, displacement
  `s = z_epi·σ_epi + Σ_i|J_ji|·Δ_i`. FEASIBLE iff `min_j min(u_hi,u_lo) ≥ κ`.
  Defaults: κ=2.0 (credited-band bar §8.4), z_epi=2.0 (epistemic worst-member
  proxy — deep-ensemble worst-of-K is WP-E), delta_frac=0.02 (δ box, §8.5 exact
  ℓ∞ Taylor via the ANALYTIC Jacobian; PGD is WP-E), λ_m=0.3 (soft support
  reward). Epistemic enters ONCE (no κ·U_epi double-count). confidence =
  Π_j(Φ(u_hi)+Φ(u_lo)−1) (per-output independence approx; joint-residual-cov MC
  is WP-E).
- 2026-07-15 (WP-D): §8.2 anti-reward-hacking is FAIL-CLOSED — the ctor REQUIRES
  `support_floor` or `X_train` (default floor = 5th-pct train support_score); a
  survivor must clear it (hard reject) AND get the soft `λ_m·support` reward.
  support_score = negative Mahalanobis (§8.2 cheap fallback; normalizing-flow
  typicality is WP-E). This is the defense against the §8.2 "σ_epi spuriously
  small in a far-OOD hole" failure mode — never disable it.
- 2026-07-15 (WP-D): tool conditioning (§8.3 split-plot) — bind the tool with
  `model.for_tool(tool_id)` BEFORE constructing the solver; the solver's free
  `variables` are the RECIPE vars only, in the model's input-vector order
  (`_flat_keys`). NEVER search over tool_id. spec['tool_id'] is informational
  (enriches the "collect runs" message only).
- 2026-07-15 (WP-D): INFEASIBLE is a first-class outcome (never a clipped point):
  `Infeasible(nearest_achievable, distance_to_feasible, reason)` + per-output
  relaxation (raw units, via the FLOORED σ scale the margins used, not raw
  σ_ale). The §8.8 cause is diagnosed by a NOMINAL-FEASIBILITY PROBE — a second
  epistemic-free multi-start (`ignore_epi=True`, adds `z_epi·σ_epi/σ` back to the
  margins) — because the pessimistic search AVOIDS high-epistemic regions, so the
  mean-feasible point is never in the primary restarts. 4-way verdict: (a)
  off-manifold → expand support (§9); (b) epistemic-limited (mean-robust sans
  epistemic, `margin_no_epi ≥ κ`) → collect runs; (c) partly-epistemic tight box
  (mean in box, epistemic helps but box < ±κσ band) → data cuts the needed
  relaxation + relax spec; (d) mean OUT of box → hard conflict, relax target /
  change process. The probe ALSO promotes an on-support pessimistically-feasible
  point it finds that the primary missed (never return INFEASIBLE when a feasible
  recipe exists). [Both the always-true `epi_dominated` bug and this taxonomy
  came out of 2 adversarial review rounds — see BUILD_LOG 2026-07-15 WP-D.]
- 2026-07-15 (WP-D): §8.7 diversity = farthest-point (max-min) selection in
  normalized recipe space (the k-DPP stand-in; qLogNEHVI Pareto over COMPETING
  KPIs + the amortized generator sampling the pre-image are WP-E). Anchors on the
  highest-confidence recipe; stops early (fewer than q) when the pre-image is
  (near) a single point — never pads with near-duplicates. `parse_targets`
  REJECTS a zero-width box (bare `{'target': t}` with no tol) with an actionable
  error (no κσ margin fits a point; fail-loud, matches WP-H no-silent-default).
- 2026-07-15 (WP-F): §9.4 acquisition (binding, GP tier): Phase-I blend
  `α = [λ·EPIG_S + (1−λ)·BALD] / cost^β` — BALD/EPIG both in NATS (linearly
  blendable), cost enters by DIVISION (CArBO), fixed `c_batch` NOT in the ratio
  (it is the §11 stop rule). BALD = `0.5·log(1+σ_epi²/σ_ale²)` per output summed
  (H[total]−E[H[ale]], never raw variance). EPIG = GP joint-covariance info gain
  about the inverse's target points x* (needs `GPForwardModel.posterior_cov`;
  BALD needs only public `predict`). anneal λ 0.2→0.9, β 1→0. Phase-II qLogNEHVI
  is a SEPARATE (non-nats) acquisition = WP-E; `qlognehvi_phase2` raises. The loop
  = 1 exploit (inverse best, checked in-spec on the real machine for the
  cost-to-target event) + (q−1) greedy-diverse explore per lot; refit fresh GP
  each batch; stop on target-met/budget/stall.
- 2026-07-15 (WP-G): §12 harness conventions (binding): cost-to-target is
  survival data — "event"=spec hit, "censor"=budget-exhausted, SMALLER RMST is
  better. difference-in-RMST (Uno 2014) is the PRIMARY comparator (log-rank
  invalid under crossing → deferred caveated secondary). Infeasible targets are
  EXCLUDED from the survival analysis (never censored); they go to the
  feasibility/abstention metrics. "hit" = the SINGLE top-ranked recipe in tol
  (best-of-q reported separately). SBC/TARP simultaneous band = simulation
  (KS sup-deviation), credibilities/ranks CONTINUITY-CORRECTED `(K+U)/(M+1)`
  before the band (the TARP small-M fix — else calibrated coarse posteriors
  over-reject). exploitation `interval_violation_fraction` is JOINT (≈1−(1−α)^m);
  `per_output_violation` is the ≈α per-KPI check. Vendi is scale-invariant
  (standardized + median bandwidth).
- 2026-07-17 (D2, RETROACTIVE — landed in an UNLOGGED session, recorded by the
  Session-7 audit): **`spec['warm_start_recipes']` is a BINDING key on the §8
  solver.** `PessimisticInverseSolver.solve` honours it (`_warm_start_u`) by
  seeding the multi-start from given recipes instead of only cold Sobol points;
  `rig.inverse.d2.AmortizedRefiner` REQUIRES it (that is how an amortized
  proposal becomes a warm start). It is the D2 contract between §14.3 and §8 —
  do not remove or rename it. It was absent from these standing decisions for a
  day, so a session reading only BUILD_STATE would not have known it exists.
- 2026-07-17 (WP-H / §3.5, BINDING — audit): **`ProcessSpec.continuous_si` is the
  accessor for anything touching ingested data; `.continuous` is NOT.** Ingest
  SI-canonicalizes every VALUE, but `.continuous` must keep the SPEC-DECLARED
  units because ingest needs them to read CSV cells. Pairing declared-unit BOUNDS
  with SI-canonical DATA silently searches the wrong space, and the bug is
  INVISIBLE whenever the declared unit is already SI (W stays W) — it only bites
  the scaled ones. This was live in `examples/real_data/sputtering/`: pressure
  declared `1..43 mtorr`, data at `0.133..5.73 Pa`, solver handed `1..43` → it
  searched `1..43 Pa ≈ 7.5..322 mTorr`, a range whose LOWER bound sits above the
  data's maximum, and only the §8.2 fail-closed support floor kept the answers
  sane. Use `continuous_si` for fitting, the §8 `variables`, support scores and
  plots. NB a variable's NAME may still carry a unit label from its source column
  (`set_pressure_[mTorr]`) while its VALUE is SI — say so in any output.
- 2026-07-17 (WP-I, BINDING — audit): **`posterior_cov`'s diagonal MUST equal
  `predict`'s `epistemic_sigma**2` for the same `tool_id`, on EVERY branch.**
  `epig()` takes `var_f_star` from `predict` and `Cov(f(x*),f(x))` from
  `posterior_cov`; two different laws break Cauchy-Schwarz and silently collapse
  the info gain to ~0 nats (measured 19× under-report on the unknown-tool arm).
  `predict`'s unknown-tool `max_t var_t` is itself binding (it is what makes
  unknown-tool epistemic dominate every known tool, §5.8 LOTO), so
  `posterior_cov` is the side that must be reconciled — via `_unknown_tool_cov`'s
  congruence rescale, which preserves PSD-ness and the mixture's correlation
  structure while matching the diagonal exactly. The sharpest guard is the
  identity `EPIG(x; {x}) ≡ BALD(x)`, which holds for any self-consistent joint
  model. Every forward tier must keep this test.
- 2026-07-17 (§14.3, BINDING — audit): **the gate must certify the law that
  ships.** Two separate bugs have now come from this one principle (Session 6's
  HIGH single-member draw; Session 7's skewed small-n `_member_counts`). Any
  change to how `sample`/`sample_array` draws MUST be mirrored in what
  `validate()` draws, and the even-mixture assumption in `log_prob`
  (`logsumexp − log K`) is part of that contract. Also: `log_prob` returns a
  density over RECIPE space — a u-space density is NOT a monotone-reparam
  substitute, because a monotone reparam preserves the ordering of the VARIABLE,
  not of the DENSITY (`RecipeTransform.log_abs_det_du_dx` is the required term).
- 2026-07-17 (tooling, BINDING — audit): **verify with
  `python -c "import sys; from importlinter.cli import lint_imports; sys.exit(lint_imports())"`.**
  The old documented form discarded the return value and exited 0 even on a
  BROKEN contract. Generally: before trusting any command's exit code as
  evidence, prove it BITES by injecting the failure it is supposed to catch —
  every "clean, exit=0" claim through Session 6 was reading a constant.
- 2026-07-17 (WP-D / F9, BINDING): **the §8 multi-start budget scales with the SEARCH
  dimension.** `n_restarts=None` (default) => `max(_MIN_RESTARTS=48, _RESTARTS_PER_DIM=24
  * RecipeTransform.dim)`. Three things are load-bearing and must not be "simplified":
  (a) it keys on `_rt.dim` — the FREE u-coordinates the optimizer searches (K-1 per
  simplex), NOT the recipe key count; (b) `24*dim` == **exactly 48 at dim=2**, which is
  why M2 / the AL loop / every 2-D result stayed bit-for-bit valid across this change —
  change the constant and you silently move every published 2-D number; (c) an explicit
  int still wins. WHY it matters: a FIXED budget is dense in 2-D and vanishing in 20-D,
  and a starved multi-start does not fail loudly — it returns a **FALSE INFEASIBLE** (we
  failed to FIND a recipe, and reported that none EXISTS), the exact confusion the §8.8
  taxonomy exists to prevent. Measured evidence + the ground-truth protocol:
  `docs/dimensionality-2026-07-17.md`, `examples/run_dimensionality_study.py`.
- 2026-07-17 (WP-D, OWED — do not mistake for done): the §8 solver calls `minimize(...,
  method="L-BFGS-B")` **without `jac`**, so SciPy finite-differences the objective: `d+1`
  evaluations per gradient step, each running BOTH `predict` and `jacobian`. With the
  (correctly) dimension-scaled restart budget this is ~O(d^2): 2.6 s at d=2 -> ~150 s at
  d=20. The fix is an ANALYTIC objective gradient, which needs `d(sigma_epi)/dx` and
  `dJ/dx` (a SECOND derivative of the GP mean). Until it lands, d >~ 20 needs the torch
  tier's autograd or a deliberately cut budget. Documented in the `pessimistic.py` module
  docstring so nobody re-derives it from a stopwatch.
- 2026-07-17 (evaluation, BINDING): **an inverse is scored against GROUND TRUTH, never
  against the surrogate that proposed the recipe.** Solve, then evaluate the TRUE function
  at the returned recipe. Scoring `model.predict(x)` for a recipe the model itself chose
  measures self-consistency, not correctness — the circularity the audit flagged in the M3
  gate (F2). `examples/run_dimensionality_study.py` and
  `test_inverse_returns_ground_truth_hits_above_two_dimensions` are the reference pattern.
