# RIG vs MFL bake-off

A pre-registered comparison of RIG's §8 pessimistic inverse against **Model Feedback
Learning** (MFL; Gu et al., *Few-Shot Test-Time Optimization Without Retraining for
Semiconductor Recipe Generation and Beyond*, arXiv:2505.16060v1, 2025), on RIG's MBE
`InSilicoMachine`.

- **Pre-registration (BINDING):** [`docs/prereg-mfl-bakeoff-2026-07-17.md`](../../docs/prereg-mfl-bakeoff-2026-07-17.md).
  Its metric definitions (§0), protocol (§4), and predictions P1–P5 (§2) are frozen. Do
  **not** edit the predictions; a refuted prediction is a legitimate result.
- **Build spec:** [`docs/mfl-bakeoff-build-spec-2026-07-18.md`](../../docs/mfl-bakeoff-build-spec-2026-07-18.md).

> **Both venues are simulators (prereg §5).** Nothing here is about real hardware. Two
> in-silico methods disagreeing tells you which is better *on this simulator*, and nothing
> about MFL's performance on *their* plasma-etch task, nor about the M0 gap that remains the
> whole scientific claim for both projects. Say so in every write-up.

## Reproduce

```bash
# 1. Freeze the target set (dense noise-free search; run ONCE, before any arm).
PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/pre_register_targets.py

# 2a. Smoke: 4 targets, reduced budget, ~30 s — NOT citable (labeled so in the output).
PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/run_bakeoff.py --smoke

# 2b. Full: all 20 frozen targets, Table-10 defaults, 200-replicate yield.
PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/run_bakeoff.py --full
```

Results land in `examples/mfl_bakeoff/results/<label>_<timestamp>/bakeoff_results.json`
plus a printed summary table. `targets.json` is **hash-frozen**: the runner recomputes the
sha256 over `{meta, targets}` and refuses to run on a mismatch (prereg §4.1).

## What runs

Four arms, all scored against **ground truth** (the machine's noise-free path), never
against any method's own surrogate (prereg §4.3):

| arm | what |
|---|---|
| `rig` | `GPForwardModel` on the seed runs → `PessimisticInverseSolver.solve` per target (§8 defaults κ=2, z_epi=2). Presents a recipe iff **FEASIBLE**. |
| `rig-reval` | + a `ConformalForwardModel` re-validation gate (§13.2), fit on a held-out slice of the **same** seed runs — the owed "does conformal re-validation change the miss rate" experiment. |
| `mfl-charitable` | MFL (Alg. 1). FD Loop-B probes **not** counted (their setting: a differentiable deployed `M`). |
| `mfl-deployable` | The **same** trained MFL; **every** machine touch counted (a real tool). |

The two MFL arms share **one** trained reverse model `R` and differ **only** in the
machine-query ledger — the prereg §3 charitable-vs-deployable resolution. MFL always
presents a recipe (no abstention branch, prereg P4).

### Metrics (prereg §0 names, verbatim)

`certified_miss_rate` (headline), `yield_under_noise` (200 noisy replicates),
`normalized_margin` (min-over-outputs distance-to-edge / aleatoric σ), `machine_queries`
(the full ledger, both accountings), `false_abstention_rate`.

### Targets (`targets.json`, frozen)

20 targets over three controlled outputs — `nonuniformity_pct` and `T_center` (two-sided)
and `slip_max_ratio` (one-sided upper). **10 clearly-feasible, 5 feasible-but-hard**
(tiny boxes on the reachable-set boundary), **5 ground-truth-infeasible** (boxes off the
reachable manifold). Every `feasible_truth` label is set by **dense search** (Sobol 4096 +
local refine) on the noise-free path — never by a method under test — and the generator
asserts each class is what it claims.

### Ground truth = the machine's noise-free path

The deterministic mechanism used is
`rig_adapters.mbe.adapter.evaluate_physics(recipe, machine_config)` with its **nominal**
keyword params (`emissivity=NOMINAL_EMISSIVITY`, `cosine_n=NOMINAL_COSINE_N`,
`flux_eff=1.0`): the fast-Arrhenius path with **no** metrology noise, **no** first-wafer
offset, **no** seasoning, **no** tool perturbation — i.e. exactly an
`InSilicoMachine(PathologyConfig())` (all pathologies OFF) run, which the machine's
determinism contract makes bit-identical. Taken through `metrics_to_outcomes` so its SI
canonicalization matches the noisy path byte-for-byte. **A genuine deterministic path
exists, so the seeded-replicate-average fallback is not used.** Noisy queries use
`InSilicoMachine(PathologyConfig(metrology_noise=True))`.

## ∂M/∂x on a non-differentiable, noisy machine

`InSilicoMachine` has no autograd, so **both** MFL arms compute `∂M/∂x` by **forward finite
differences** (`d` probes/point). The arms run the identical compute path and differ only
in the **ledger**: charitable does not count FD probes; deployable counts every machine
touch (`d+1` per point per Loop-B iteration — prereg §3b). This is the sharpest technical
criticism of the paper made concrete: on a real tool MFL's Loop-B query cost grows with the
input dimension while RIG never needs the machine's Jacobian at all (a *design* difference,
not a result).

## Deviations — from the paper AND from the build spec (honest ledger)

**From the paper (Gu et al. 2025):**

1. **All NN I/O is standardized** (train-set mean/std). Their data is Gaussian-sampled
   ~unit-scale; MBE recipes/outputs span ~20 orders of magnitude (K vs metres), so the net
   cannot train without it. Consequently the sensitivities `s_E/s_M`, the `δ=0.9` gate, the
   FD step, and the clip bounds are all in **standardized** units — the only scale-free
   reading of `δ`.
2. **`∂M/∂x` by forward finite differences** (their `M` is a differentiable MLP; ours is
   not). This is the prereg §3 crux, not an incidental choice.
3. **`fd_step=0.05`** (standardized), not a paper value — the paper's `M` is noiseless so it
   gives no guidance. On our **noisy** machine the FD step must exceed the standardized
   metrology-noise floor or Loop B injects pure-noise gradients that destroy `R` (measured:
   `fd_step=1e-3` drives target-recovery error 0.03 → 2.4; `0.05` restores it). This is a
   **steelman** adjustment adopted per prereg §4.6 — a real noisy-tool tension (small step →
   noise-dominated; large step → biased Jacobian) that also *sharpens* the deployable-arm
   critique.
4. **MLP depth = 2 hidden layers of width 64** (the paper fixes width 64, not depth).
5. **Optimizer = plain SGD** with the scheduled lr, so `α1/α2` act as literal learning
   rates (Adam would re-scale them).
6. **`domain_randomization=0.05`** input-noise std (the paper says "domain randomization"
   without a magnitude).

**Faithful to the paper (verified by tests):** two loops with the gradient flowing
**through E** in Loop A (Eq. 4, not a detached target); the Loop-A objective
`L = (1/n')Σ_j ‖z'_j − y'_j‖²` (sum over outputs, mean over targets — consistent with
Loop B's `2/n'` gradient; see steelman fix 2 below); domain randomization in Step 1; the
conservative-LR sensitivity gate (α2 = 0.99·α1, δ=0.9); one `R` conditioning on `z'` serving
the whole target set; the Table-10 constants (`α1=0.01`, hidden 64, epochs 700, T=1200,
T0=1150, τ=200, τ0=150); input-bound clip.

**From the build spec (`docs/mfl-bakeoff-build-spec-2026-07-18.md`):**

- **One-sided spec bound set relative to the reachable range** (`slip_max_ratio ≤` the 85th
  percentile of reachable slip), not the nominal `≤1`. Slip only reaches ~0.3 here, so `≤1`
  is trivially satisfied **and** makes the spec's one-sided MFL point rule (`bound − σ ≈ 0.95`)
  chase an unreachable value and wreck the other two outputs (measured: it forced
  `certified_miss_rate → 1.0`). A reachable-relative bound is a genuine, satisfiable-with-
  effort one-sided constraint and keeps the point rule sensible. The build spec's point rule
  itself is applied verbatim; only the *bound* is made reachable.
- **Controlled-output set fixed to three outputs** (`nonuniformity_pct`, `T_center`,
  `slip_max_ratio`) shared by every target, so one `R` conditions on a fixed-dim `z'`. The
  build spec left the output choice open.
- **Targets selected from a labelled pool** to guarantee the class quotas (≥10/≥5/≥5) after
  the joint (3-output) dense-search feasibility labelling, rather than assuming a generated
  box has its intended feasibility.

**Steelman-mandated fixes (2026-07-19, prereg §4.6 audit trail).** The independent adversary
rated the MFL arm faithful and not under-tuned but found two required corrections; both are
recorded here so the audit trail lives with the code:

- **Fix 1 (blocker) — target-label / scored-spec mismatch.** `pre_register_targets.py`
  originally labelled feasibility against the **two-sided** slip box while the stored/scored
  spec overrode slip to **one-sided** `[None, slip_upper]`. A 16384-point dense search proved
  the consequences: `hard_01/03/04` had **zero** feasible recipes under the stored spec yet
  were labelled `feasible_truth=True`, and **6 stored witnesses violated the stored slip
  bound**. Fixed by a single `_effective_bounds()` that applies the one-sided-slip override
  used for **both** labelling and storage, anchoring the feasible/feasible-but-hard targets on
  the **slip-feasible** reachable sub-cloud (the reachable set *under the stored spec*), and
  asserting in the generator that every stored witness satisfies the stored spec. The target
  pool was **regenerated** and `targets.json` **re-frozen** (its hash changed, as required);
  labels were re-verified by an independent 20000-point dense search (feasible/hard: ≥385
  in-spec points each; infeasible: 0; all witnesses valid).
- **Fix 2 (recommended) — Loop-A loss scaling.** `_loop_a` used `((ys−zt)²).mean()` (mean over
  all elements). Restored to the paper's `((ys−zt)²).sum(dim=1).mean()` = `L = (1/n')Σ‖z'−y'‖²`,
  consistent with Loop B's `2/n'` gradient. The two forms differ only by the constant `1/z_dim`;
  the steelman verified this is **immaterial to the outcome** (6/11 either way) — adopted as the
  paper-faithful form, not to change the result.

## Scoring the frozen predictions

The analysis scores P1–P5 against `bakeoff_results.json`. Verdict rule (prereg §2): the
"RIG's formulation is better-posed" claim is supported **only if P2 AND P3 both hold**; P4
never counts (it is the tautology that MFL has no abstention branch); the predicted
**losses** on P1 (query count, charitable) and P5 (false abstention) are to be reported as
losses, prominently. Both venues in-silico — say so in every write-up.
