# Analytic-gradient vs finite-difference PARITY study (2026-07-23)

**Question.** `PessimisticInverseSolver(analytic_grad=True)` replaces SciPy's
finite-difference (FD) objective gradient with a closed-form one — `d+1` model
evaluations per L-BFGS-B step collapse to 1 (`src/rig/inverse/pessimistic.py`). It is
opt-in; the default stays FD because it changes the optimizer's SEARCH PATH, not the
objective, so results are equal to FD tolerance, not bitwise. Flipping the default is
owed (`docs/BUILD_STATE.md`: "analytic-gradient default") but risky — it would silently
move every recorded FD-path number. This study is the evidence for/against flipping. It
does **not** flip the default; no `src/` edits were made.

Reproduce: `python examples/run_gradient_parity_study.py --full` (~43 min measured).
Smoke + determinism: `python examples/run_gradient_parity_study.py --smoke`.
Raw data: `docs/gradient-parity.json`. Scorer + tests: `tests/test_gradient_parity.py`.

## Design and power achieved

- `d ∈ {2, 4, 8, 15, 20}`, `n_train = 12·d` (the house scaling), **15 seeds** per
  `(d, target_class)` — 150 FD/analytic pairs total.
- Two target classes per (cell, seed), on the **same** fitted GP (so target difficulty
  is isolated from data/truth variation): `reachable` (the false-success study's own
  `±0.8` box around truth at a held-out point) and `hard` (same center, `±0.3` — a
  boundary/marginal box, imported logic reused, only the tolerance shrunk).
- Both arms use the **same** `n_restarts=16, max_iter=40` (a fixed, reduced,
  shared-across-arms budget — see the "compute note" below), the same binding solver
  policy (`kappa=z_epi=2.0, delta_frac=0.02`), the same seed.
- Truth family, training design, and the `reachable` target are reused **by import**
  from `run_false_success_study.build_cell` / `run_dimensionality_study.make_truth`
  — not redefined.
- Deterministic: `--smoke` runs a tiny grid twice and diffs the timing-stripped view
  byte-for-byte (`assert v1 == v2` — verified PASS, see Tests below).
- **Power achieved:** 150 pairs is enough to see the true verdict-agreement rate to
  within the Clopper-Pearson 95% CI reported per cell (typically ±10–20 points at
  n=15, tighter at n=150 overall: **[93.3%, 99.3%]**). It is not enough to bound a
  *false-success* rate precisely — none were observed in either arm at this n (see
  Results) — so, per the house honesty convention, that is reported as "0 observed at
  n=150," not a claim of exactly zero.

**Compute note.** Production `n_restarts=None` scales `24·dim` (480 at d=20), measured
at ~150 s FD/solve (`docs/dimensionality-2026-07-17.md`). This study fixes
`n_restarts=16` for **both** arms so the ~150-solve-pair grid fits the compute budget
(measured: 42.7 min wall). The restart count is irrelevant to the parity *question*
(agreement between arms) as long as it is shared, which it is at every cell — but this
study is not a re-measurement of the production speedup ratio at production budget;
that number lives in `docs/dimensionality-2026-07-17.md` / the false-success study.

## Results

### Verdict agreement (exact binomial CI)

| d | target | n | verdict agree | rate | 95% CI |
|---|---|---:|---:|---:|---|
| 2 | reachable | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 2 | hard | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 4 | reachable | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 4 | hard | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 8 | reachable | 15 | 13/15 | 0.867 | [0.60, 0.98] |
| 8 | hard | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 15 | reachable | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 15 | hard | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| 20 | reachable | 15 | 13/15 | 0.867 | [0.60, 0.98] |
| 20 | hard | 15 | 15/15 | 1.000 | [0.78, 1.00] |
| **overall** | — | **150** | **146/150** | **0.973** | **[0.933, 0.993]** |

All 4 disagreements are on the `reachable` class, concentrated at d=8 (2 cases) and
d=20 (2 cases) — never at d=2, d=4, d=15, and never on the tighter `hard` class (where
both arms mostly abstain together — see the ground-truth section below).

### Ground truth at the disagreements — the endpoint that matters

Every one of the 4 disagreements has the **same shape**: FD returned `INFEASIBLE`,
analytic returned `FEASIBLE`, and analytic's certified top candidate **genuinely hits
ground truth** (`hit_an=True`, `excursion_an=0.0` in every case). Scored via
`verdict_favors` (`examples/run_gradient_parity_study.py::score_pair`):

| d | seed | target | fd | an | an's recipe hits truth? | `verdict_favors` |
|---|---|---|---|---|---|---|
| 8 | 6 | reachable | INFEASIBLE | FEASIBLE | yes | `fd_false_abstention` |
| 8 | 13 | reachable | INFEASIBLE | FEASIBLE | yes | `fd_false_abstention` |
| 20 | 7 | reachable | INFEASIBLE | FEASIBLE | yes | `fd_false_abstention` |
| 20 | 11 | reachable | INFEASIBLE | FEASIBLE | yes | `fd_false_abstention` |

**Zero of the 4 disagreements involve a false success from either arm.** All 4 are FD
**false abstentions** — the exact under-exploration failure mode
`tests/test_inverse.py::test_analytic_grad_and_fd_paths_are_both_ground_truth_valid`
already documented at d=6 ("the FD path finds 1 recipe, analytic finds 3, and ALL are
in-box"): with a fixed, budget-starved restart count, FD's finite-difference gradient
estimate is noisier and converges to fewer/worse local optima, so it sometimes fails to
*find* a recipe that genuinely exists — the false-`INFEASIBLE` confusion §8.8 exists to
name. `an_feasible_fd_infeasible` / `fd_false_abstention` = **evidence FOR flipping**
(analytic finds a real hit FD's search missed); `an_false_abstention` /
`an_false_success` (evidence AGAINST) were **never observed** at n=150.

`n_disagreements_favoring_flip = 4`, `n_disagreements_favoring_keep_fd = 0`.

### Among AGREEING feasible pairs — recipe distance, margin, and ground truth

Where both arms agree FEASIBLE, they agree on ground truth **100% of the time**
(`gt_agreement_rate_among_agree_feasible = 1.0` in every cell that had agreeing-feasible
pairs — `gt_split = 0` everywhere, no cell where one arm's top pick hits and the other's
misses). Recipe distance and margin *do* differ between arms — expected, since an exact
vs. finite-difference gradient drives L-BFGS-B down different paths to possibly
different optima — but per the pre-registered scoring rule, **that is fine as long as
both genuinely hit**, which they always did here:

| d (reachable) | n agree-feasible | mean recipe dist (normalized) | mean \|margin diff\| | max \|margin diff\| |
|---|---:|---:|---:|---:|
| 2 | 14 | 0.039 | 0.72 | 4.06 |
| 4 | 13 | 0.235 | 1.47 | 5.05 |
| 8 | 9 | 0.161 | 0.65 | 2.40 |
| 15 | 3 | 0.304 | 0.30 | 0.44 |
| 20 | 1 | 0.017 | 0.009 | 0.009 |

(`hard`-class agree-feasible counts are 0 or near-0 at every d≥4 — the tight box mostly
drives both arms to abstain together, which is exactly why the `hard` class's verdict
agreement is a clean 1.000 everywhere: there's little live disagreement surface, and
what there is shows up on `reachable` instead.)

### Wall-time (speedup), at the study's shared reduced budget

| d | mean t_fd (s) | mean t_an (s) | speedup (fd/an) |
|---|---:|---:|---:|
| 2 | 1.48 | 0.78 | 1.90× |
| 4 | 4.66 | 1.46 | 3.20× |
| 8 | 11.01 | 1.86 | 5.91× |
| 15 | 20.53 | 2.42 | 8.47× |
| 20 | 33.44 | 2.92 | **11.43×** |

Monotonically increasing with `d`, consistent with FD's `O(d)` extra evaluations per
gradient compounding with the search's own `O(d)` restart-budget scaling — the `O(d²)`
story `docs/dimensionality-2026-07-17.md` already measured at production budget. This
study's absolute numbers are smaller (reduced shared restart budget), but the *trend* —
and the fact that analytic is never slower — replicates.

## Recommendation

**Flip the default to `analytic_grad=True`, starting at `d ≳ 8`. Below that, either
default is empirically safe but the win is small enough that keeping FD as the
universal default (with `analytic_grad=True` recommended above the threshold) is the
lower-risk rollout.**

Justification, from the measurements above:

1. **Verdict agreement is high (97.3%, CI [93.3%, 99.3%]) and every disagreement
   observed favors analytic.** All 4 disagreements (d=8 ×2, d=20 ×2) are FD **false
   abstentions** that analytic resolves correctly (genuine ground-truth hits). Zero
   disagreements went the other way, and zero false successes were introduced by
   either arm at this n — flipping the default at d≥8 would have found MORE genuine
   recipes, never fewer, in this study.
2. **Where both arms agree, they agree on ground truth 100% of the time** — different
   recipes and different margins are common (an exact vs. approximate gradient takes
   different paths), but "both genuinely hit" held in every single agreeing-feasible
   case observed. Flipping does not risk trading a hit for a different, non-hitting
   recipe in this data.
3. **The speedup is real and grows with d** (1.9× at d=2 → 11.4× at d=20, at a reduced
   shared budget; production-budget numbers in `docs/dimensionality-2026-07-17.md`
   show ~150s FD vs measured ~16s analytic at d=20, a similar-shaped curve). At d=2
   the win is marginal (1.9×) and 2-D is exactly where every currently-published
   number (M2, the AL loop) lives — the highest-cost place to be wrong. Above d≈8 the
   win is large (≥6×) and the evidence for correctness is, if anything, better than
   FD's (fewer false abstentions).
4. **This study's own compute is deliberately reduced** (`n_restarts=16` vs
   production's `24·dim`) — a threshold recommendation from it should be read as
   directional, not as a precise cutoff. `d≳8` is where this study's own agreement
   rate first dips below 100% and where analytic's advantage first becomes decisive;
   it is a reasonable, conservative reading of the data in hand, not a tuned line.

**Explicitly not recommended by this data:** an unconditional flip. Two honesty
caveats keep this from being unconditional:
- n=150 pairs cannot rule out a rare false success in either arm (the CI on "0
  observed" is itself wide at this n) — this mirrors the exact lesson of
  `docs/dimensionality-2026-07-17.md`'s d=20/n=800 false success, which needed 800
  points and a specific GP-fit-restart detail to surface. A threshold-flip decision
  should be re-checked against a LARGER seed count before being treated as final,
  especially past d=20 (untested here).
- The `hard` (boundary) target class showed almost no live disagreement surface at
  d≥4 (both arms abstain together) — so this study's disagreement evidence is
  concentrated on the generous `reachable` class. It does not directly test whether
  a marginal, boundary-sitting FEASIBLE verdict (the scenario closest to the
  d=20/n=800 false success) is where the two paths would diverge; that scenario
  needs the crime-scene-style larger-n cell, not this study's 12·d cells.

### Migration checklist, if/when the default flips

Every one of these was confirmed (by reading the calling code) to construct
`PessimisticInverseSolver` **without** passing `analytic_grad`, i.e. it rides the FD
default today and its recorded numbers would need re-verification against the analytic
path before being trusted post-flip:

- **`docs/dimensionality-2026-07-17.md`** (`examples/run_dimensionality_study.py`,
  `probe()`) — the entire d=2..20 ground-truth table, incl. the d=20/n=800 false
  success and its n_restarts=48/192 control arms. This is the highest-value re-run:
  it is the one place a false success was ever observed, and it predates
  `analytic_grad` existing at all.
- **M2** (`docs/M2-result-2026-07-16.md`, `docs/m2-result-binding.json`,
  `examples/run_m2_sweep.py`) — via `rig.active.loop.ActiveLearningLoop`, which
  constructs `PessimisticInverseSolver` with no `analytic_grad` kwarg
  (`src/rig/active/loop.py` ~line 460). The binding-policy M2 re-run (BUILD_STATE,
  2026-07-22) is FD; its ΔRMST headline would need re-verification.
- **M3 acceptance** (`docs/m3-acceptance-v2.json`, `docs/M3-acceptance-v2-2026-07-22.md`,
  `examples/run_m3_acceptance.py`, `examples/run_m3_acceptance_v2.py`) — same
  `ActiveLearningLoop` path, FD.
- **`tests/test_active_loop.py`** — assertions pinned against the FD-path loop
  behavior; would need re-pinning (not necessarily re-writing, but re-verifying)
  against the analytic path.
- **`tests/test_inverse.py::test_analytic_grad_is_off_by_default_and_leaves_two_dim_results_untouched`**
  (the `_FD_PIN` fixture) — this test's entire premise is "the default is FD"; it
  would need to become an explicit `analytic_grad=False` pin rather than a default
  pin, and a new test would be needed to pin the (now-default) analytic numbers.
- **mfl_bakeoff** (`examples/mfl_bakeoff/run_bakeoff.py`, its `README.md`) — FD,
  no `analytic_grad` passed.
- **Real-data inverse demos**: `examples/real_data/empa_hipims/run_m1_empa.py`
  (feeds `results/m1_empa.json`'s inverse block) and
  `examples/real_data/sputtering/run_m1_sputtering.py` (feeds `RESULTS.md`) — both FD.
- **`examples/run_multitool_rehearsal.py`** — FD (via the loop path).
- **NOT affected** (already immune): `docs/false-success-study.json`'s main grid
  already used `analytic_grad=True` by CLI default; only its `crime_scene_reproduction`
  arm explicitly hardcodes `analytic_grad=False` regardless of the library default
  (`run_false_success_study.py::crime_scene_reproduction`), so a default flip changes
  nothing there either way.

None of the above were edited by this study (file-ownership boundary respected).

## Tests and red-proof

`tests/test_gradient_parity.py` — **13 passed**:
- `public_margin` checked against 2 hand-computed values (with/without an epistemic
  displacement term), using a fake `ForwardModel` (public API only — `predict` /
  `jacobian` / `support_score` — never solver internals).
- `score_pair`: both-INFEASIBLE agreement; agreeing-FEASIBLE with recipe distance +
  ground truth recorded; an agreeing-FEASIBLE case that **splits** on ground truth
  (`gt_split`), proving verdict agreement does not imply ground-truth agreement;
  all 4 disagreement shapes (`fd_false_success`, `an_false_abstention`,
  `fd_false_abstention`, `an_false_success`), each scored by ground truth, not by
  which arm certified.
- `aggregate()` tallies disagreement direction correctly on a 2-record fixture.
- **Red-proof**
  (`test_red_proof_ground_truth_polarity_matters` +
  manual break/restore of `score_pair`'s disagreement branch): the scorer's
  `an_false_abstention` label was hand-replaced with a constant (ground-truth-blind:
  "the certifying arm is always right"), the suite was re-run, and **3 tests went
  red** (`test_score_pair_disagreement_fd_false_success_favors_flip`,
  `test_aggregate_tallies_disagreement_direction`,
  `test_red_proof_ground_truth_polarity_matters`) — confirming the scorer has teeth
  and genuinely depends on ground truth, not just on which arm certified. The code
  was then restored and the suite re-confirmed green (13/13).
- `test_analytic_grad_and_pgd_delta_mode_raise_at_construction` pins the documented
  `analytic_grad=True` + `delta_mode="pgd"` construction-time boundary this study
  never exercises (both arms use the default `delta_mode="taylor"`).
- Smoke end-to-end (`test_smoke_end_to_end_and_determinism`) + the standalone
  `--smoke` CLI run: **double-run byte-identical** (`deterministic_view(g1) ==
  deterministic_view(g2)`), confirmed both via pytest and via a direct
  `python examples/run_gradient_parity_study.py --smoke` invocation.

`ruff check` and `ruff format --check` are clean on both new files
(`examples/run_gradient_parity_study.py`, `tests/test_gradient_parity.py`).

## Files

- `examples/run_gradient_parity_study.py` — the study (new)
- `tests/test_gradient_parity.py` — unit + smoke + red-proof (new)
- `docs/gradient-parity.json` — raw grid (design, aggregate, per-seed records)
- `docs/gradient-parity-2026-07-23.md` — this file
