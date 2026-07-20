# MFL bake-off — BUILD SPEC (2026-07-18)

The pre-registration `docs/prereg-mfl-bakeoff-2026-07-17.md` is BINDING (metric definitions §0,
protocol §4, predictions §2 frozen). This file is the implementation spec for the build that the
session-limit outage prevented. Build → steelman-verify → smoke → full run → analyze vs the
frozen predictions. Do NOT commit. ruff format everything. PYTHONIOENCODING=utf-8.

## Files to create

1. `src/rig/baselines/mfl.py` — faithful Model Feedback Learning (Gu et al., arXiv:2505.16060,
   Algorithm 1), torch, lazy-imported (base stays torch-free; follow the existing lazy pattern).
2. `tests/test_mfl_baseline.py` — tests below, written FIRST.
3. `examples/mfl_bakeoff/pre_register_targets.py` — targets.json BEFORE any arm runs.
4. `examples/mfl_bakeoff/run_bakeoff.py` — arms + ledger + metrics + JSON out.
5. `examples/mfl_bakeoff/README.md` — repro commands + every deviation from the paper, honestly.

## MFL Algorithm 1 (transcribed from the paper — implement EXACTLY; cite lines in comments)

Inputs: machine M; reverse model R_θ (θ⁰); emulator data {(x_i, z_i)}_{i=1..n}; targets
{z'_j}_{j=1..n'}; learning rates α1 > α2; periods T, T0, τ, τ0.

- Step 1: train emulator E on {(x_i, z_i)} supervised, WITH domain randomization (zero-mean
  Gaussian noise added to inputs during E training).
- Loop A (t = 0..T−1) on E: x'_{t,j} = R_θ(z'_j); y' = E(x'); L = (1/n')Σ‖z'_j − y'_j‖²;
  if t ≥ T0 and mean_j s_E(x'_{t,j}) ≥ δ → lr = α2 else α1; gradient flows THROUGH E
  (their Eq. 4: [∂R/∂θ]ᵀ[∂E/∂x]ᵀ(y′−z′)) — NOT a detached target.
- Loop B (h = 0..τ−1) on the MACHINE M: same, with ∂M/∂x, sensitivity s_M, onset τ0.
- Sensitivity s_f(x) = induced L2 norm of ∂f/∂x.
- Table 10 defaults (expose as kwargs): α1=0.01, α2=0.99·0.01, hidden=64 MLP, emulator epochs=700,
  T=1200, T0=1150, τ=200, τ0=150, δ=0.9. ONE R conditions on z' and serves the whole target set
  (their framing = target distribution). Input bounds: clip (the literal reading; document).

## ∂M/∂x on a non-differentiable machine (prereg §3 resolution — document in README)

InSilicoMachine has no autograd. BOTH arms compute ∂M/∂x by finite differences (forward
differences, d+1 evals, state so). Arms differ ONLY in the query LEDGER:
- **mfl-charitable**: FD probe evals NOT counted (their setting: M is a differentiable deployed model).
- **mfl-deployable**: EVERY machine eval counts, FD probes included.
Identical compute path, different accounting — cleaner than faking a differentiable machine.

## Targets (`pre_register_targets.py`)

≥20 on InSilicoMachine outputs: ≥10 clearly-feasible, ≥5 feasible-but-hard (near reachable-set
boundary), ≥5 ground-truth-INFEASIBLE. Ground truth = the machine's NOISE-FREE eval (find the
deterministic path; else average many seeded replicates, documented). Feasibility by DENSE SEARCH
(Sobol ≥4096 + local refine) on the noise-free function — never by a method under test.
targets.json: per-target spec box, feasible_truth, ground-truth witness x if feasible, seed.
FROZEN once written — runner refuses to run if the targets.json hash changes between arms.

## Arms (`run_bakeoff.py`)

Shared: N seed runs (Sobol over recipe box, seeded, default N=60, flag-exposed), identical for all
arms; machine noise ON for queries; ground-truth scoring uses the noise-free path.
- **rig**: GPForwardModel on seed runs → PessimisticInverseSolver.solve per target (defaults).
- **rig-reval**: + revalidation_model = ConformalForwardModel(GP, SplitConformalCalibrator fit on
  a held-out slice of seed runs) — doubles as the owed "does conformal re-validation change the
  miss rate" experiment.
- **mfl-charitable / mfl-deployable**: same trained R, different ledger. MFL presents R(z'_j) per
  target (cannot abstain). Point-target rule for boxes (prereg §1): box CENTER for two-sided;
  (bound + 1·σ_seed) for one-sided, σ_seed = output std over seed runs. Document.

## Metrics (prereg §0 names, EXACTLY)

- `certified_miss_rate`: among presented recipes (rig: FEASIBLE returns only; mfl: all), fraction
  out-of-spec on noise-free ground truth.
- `yield_under_noise`: per presented recipe, fraction of 200 noisy replicates in-spec.
- `normalized_margin`: min over outputs of distance-to-nearest-spec-edge / noise σ at that x
  (σ from the same 200 replicates).
- `machine_queries`: full ledger — seed_runs, loopB_evals, fd_probe_evals, revalidation_evals,
  totals under both accountings.
- `false_abstention_rate`: among feasible_truth targets, fraction refused (mfl: structurally 0,
  emit with a 'cannot abstain' note — prereg P4 tautology warning).
Output: `examples/mfl_bakeoff/results/<timestamp>/bakeoff_results.json` + text summary table.
Seeded everywhere. `--smoke` (4 targets: 2 easy/1 hard/1 infeasible, N=30, reduced T/τ, labeled
non-citable) and `--full`.

## Tests (`tests/test_mfl_baseline.py`) — FIRST, all must pass

1. MFL recovers a known linear inverse (M(x)=Ax+b analytic): ‖M(R(z'))−z'‖ small on held-out z'.
2. Loop B improves on Loop A alone when E is deliberately biased from M (their Fig 9 claim).
3. Conservative-LR gate actually fires: high-sensitivity construction → α2 selected ≥once after
   T0 (expose a counter).
4. FD Jacobian of a quadratic matches analytic to 1e-4.
5. Ledger exactness: one deployable loop-B step on n' targets adds the documented eval count;
   charitable differs by exactly fd_probe count.
6. Scorer mutation-proofed: `certified_miss_rate` flags a PLANTED miss (>0), and counts rig
   abstentions as non-presented, not as misses.

## Steelman pass (prereg §4.6 — REQUIRED before the full run)

An independent adversary must try to prove the MFL arm UNFAITHFUL or UNDER-TUNED:
faithfulness vs Alg 1 (two loops, grad-through-E, domain randomization, sensitivity gate,
Table 10 constants, one-R-for-all-targets, bounds); under-tuning (more budget / 128 hidden —
if a cheap change materially strengthens MFL, ADOPT it before running); unfair scoring
(point-target rule applied at its most favorable reasonable reading; ledger per README).
Then run `--smoke` end-to-end and verify every §0 metric appears for all 4 arms.

## Full run + analysis

Full run as a BACKGROUND Bash from the main session (not inside an agent — 10-min cap).
Analysis scores the frozen predictions P1–P5; verdict rule: "better-posed" claim is supported
ONLY if P2 AND P3 hold; P4 never counts (tautological); expected LOSSES on P1/P5 are reported
as losses, prominently. Both venues in-silico — say so in every write-up.
Optional exploratory (non-prereg, label as such): a d=11 synthetic ground-truth venue matching
the paper's dimensionality (the dimensionality-study fixtures generalize).
