# Pre-registration: RIG vs MFL bake-off

**Written 2026-07-17, BEFORE any bake-off code exists or any run is executed.**
Commit this before running. If any number below is edited after a run, the edit is a
finding about me, not about the methods — record it in BUILD_LOG as such.

Comparator: Gu et al., *Few-Shot Test-Time Optimization Without Retraining for
Semiconductor Recipe Generation and Beyond*, arXiv:2505.16060v1 (21 May 2025).
UC Berkeley / Virginia Tech / **Lam Research** / UCL. Method = **Model Feedback
Learning (MFL)**.

## Why pre-register

This session has already produced three overclaims in a row on the d=20 study: reported
ahead of the refuting run, wrongly retracted on a sloppy replication, then reinstated.
See [[claims-die-in-the-follow-up-run]]. The failure mode is *deciding what counts as a
win after seeing the numbers.* This document exists to make that impossible. I am the
author of RIG's design and I am predicting RIG wins on one axis and loses on another —
if I am unwilling to write the losing prediction down now, the comparison is worthless.

---

## 0. Terminology trap — fix BEFORE measuring

`rig.eval.inverse_metrics.false_success_rate` means **"a hit on a ground-truth-INFEASIBLE
target."** That is **NOT** what this session has been calling "the d=20 false success"
(= solver certified FEASIBLE, and the top recipe MISSES on ground truth, on a target that
was itself feasible). Two distinct things share the name "false success". Conflating them
is a ready-made way to claim victory on whichever definition happens to look good.

For this bake-off the metrics are **renamed and defined here, in advance**:

- **`certified_miss_rate` (the headline).** Among recipes the method *presents as meeting
  the spec*, the fraction that are out of spec on ground truth. Defined for both methods:
  - RIG presents a recipe iff it returns FEASIBLE.
  - MFL always presents a recipe (it has no abstention) — so every MFL recipe counts.
  This is the fab-relevant question: *of the recipes you told me were good, how many were
  actually good?*
- **`yield_under_noise`.** For a presented recipe, P(in spec) over the machine's noise at
  fixed x. Estimated by N=200 repeat evaluations at the returned x.
- **`normalized_margin`.** For each output j, distance from the achieved value to the
  nearest spec edge, divided by the machine's aleatoric σ_j at that x. One-sided
  constraints use the single finite edge. Report `min_j` per recipe.
- **`machine_queries`.** Count of ground-truth evaluations, **including every evaluation
  spent estimating a Jacobian** (see §3 — this is the crux).
- `false_abstention_rate` keeps its existing meaning (feasible target wrongly refused).

**Open question, NOT a finding (unreproduced):** `false_success_rate` as currently written
may be structurally ≈0 for every method — if a target is truly infeasible, no x lands in
tolerance, so `hit` is unreachable and the metric may be vacuous. The docstring blames
`clip()` for manufacturing these, which I do not yet follow. **Check this by reading and
executing the metric before using it.** Do not report it either way until reproduced.

---

## 1. What each method actually is

|                     | RIG (§8 pessimistic inverse)                  | MFL (Alg. 1)                              |
|---------------------|-----------------------------------------------|-------------------------------------------|
| Target form         | spec **region** Z* (handles one-sided natively) | **point** z′ + MSE                        |
| Uncertainty         | aleatoric/epistemic split, conformal available | none                                      |
| Decision rule       | `min_j min(u_hi,u_lo) ≥ κ`, κ=2               | argmin MSE                                |
| Abstention          | explicit INFEASIBLE + §8.8 diagnosis          | none — always emits an x                  |
| Machine queries     | **values only** (RunRecords)                  | **values + ∂M/∂x** (Eq. 4, Loop B)        |
| Amortized           | no (per-query, 24·dim restarts)               | yes (7 kB reverse model, forward pass)    |
| Sensitivity ‖∂/∂x‖  | shrinks the margin: `Σ_i |J_ji|·Δ_i`          | schedules the learning rate (§4.2)        |

Both are **in-silico only**. MFL §5: *"the experiments are conducted in simulation."* Its
`M` is an MLP on Gaussian-sampled data. Neither method has touched real hardware. Any
framing of this bake-off as "which is better for a fab" is therefore **out of scope** — it
compares two simulators. Say so in every write-up.

---

## 2. Predictions (falsifiable, numeric, written blind)

Scored on RIG's `InSilicoMachine` (MBE), pre-registered target set, ground-truth scored.

- **P1 — MFL wins on query count. (confidence: high)**
  Ignoring Jacobian cost, MFL reaches spec in **≥3× fewer** machine queries than RIG's
  per-query inverse. *Refuted if* MFL's median `machine_queries` ≥ RIG's / 3.
  **I expect to lose this one and it should be reported as a loss, prominently.**

- **P2 — RIG wins on certified_miss_rate. (confidence: MEDIUM — this is the real claim)**
  MFL's `certified_miss_rate` exceeds RIG's by **≥15 percentage points** under process
  noise. *Refuted if* the gap is <15pp, or if RIG's is higher.
  Medium, not high, because RIG's own gate reads **raw** sigmas, not the conformal band —
  it inherits GP miscalibration and already produced a real certified miss at d=20
  ([[rig-pessimism-inherits-model-miscalibration]]). **RIG could plausibly lose this.**

- **P3 — MFL's recipes hug spec edges. (confidence: high)**
  Median `normalized_margin` for MFL < RIG's, by ≥1.0σ. This is the direct consequence of
  MSE-to-a-point having no margin term. Their own Table 1 shows it: they targeted etch
  depth **2260** (10 nm above a hard floor of 2250), landed at **2255.55** — so their
  residual (4.45) is **80% of the remaining margin** (5.55). *Refuted if* the median gap
  is <1.0σ.

- **P4 — MFL cannot abstain, so on infeasible targets it emits confident garbage.
  (confidence: high, but see §0 — the metric may be vacuous)**
  On ground-truth-infeasible targets MFL presents a recipe 100% of the time; RIG declares
  INFEASIBLE ≥80%. *Refuted if* RIG abstains <80%. **Note this is near-tautological** (MFL
  has no abstention branch) and must NOT be reported as an empirical win. It is a
  restatement of the design, and the write-up must say so.

- **P5 — RIG pays for its pessimism. (confidence: medium)**
  RIG's `false_abstention_rate` on feasible-but-hard targets is **≥10%**, i.e. materially
  worse than MFL's 0%. **Another predicted loss.** *Refuted if* RIG abstains <10%.

**Overall pre-registered verdict rule:** the bake-off supports "RIG's formulation is
better-posed" **only if P2 AND P3 both hold.** P4 alone does not count (tautological).
If P2 fails, the honest headline is *"RIG's margin machinery did not deliver a lower
miss rate; the formulation argument is unsupported by evidence"* — and that goes in the
explainer and BUILD_STATE, not just here.

---

## 3. The crux: MFL's Loop B needs ∂M/∂x

§4.1: Loop B's gradient is Eq. (4) *"replacing the emulator E with the true machine M."*
Eq. (4) contains `[∂E(x)/∂x]ᵀ` → becomes **`[∂M(x)/∂x]ᵀ`**. Line 20 needs
`s_M(x) = ‖∂M(x)/∂x‖` too. **You cannot backprop through a plasma etcher.** It is free in
their paper only because `M` is an MLP.

This must be measured, not asserted. Run **both arms**:

- **3a. Charitable arm (their setup):** M differentiable, autograd Jacobian, cost 1 query.
  This reproduces their claim on our machine and is the arm P1 is scored on.
- **3b. Deployable arm (a real tool):** Jacobian by finite differences, **d+1 queries per
  iteration** (d=11 in their setup → ≥12). Their "5 iterations" → **~60+ queries**, vs
  Lam's 20 and the human's 84 (their Table, sourced from Kanarik et al. *Nature* 2023).

**Pre-registered claim:** in arm 3b, MFL's query advantage over RIG **inverts or vanishes**.
*Refuted if* MFL still wins arm 3b by ≥3×. This is the sharpest technical criticism of the
paper and it is the one most likely to be wrong in my favour by accident — so score it
with the SAME `machine_queries` counter for both methods, and make the counter count
finite-difference probes. Do not let RIG's surrogate-Jacobian advantage go unremarked:
RIG never needs the machine's Jacobian, and that is a *design* difference, not a result.

---

## 4. Protocol (fix before running)

1. **Targets pre-registered before any run:** ≥20 targets, including ≥5 known-infeasible
   and ≥5 feasible-but-hard. Feasibility established by ground-truth search, not by either
   method under test.
2. **Same machine, same seeds, same noise realizations** for both arms. Seeded end to end.
3. **Score against ground truth only** — never against either method's own surrogate
   (audit F2). MFL's emulator E must never score MFL.
4. **Identical data budget.** MFL gets the same RunRecords RIG's GP was fit on. If MFL's
   emulator is trained on more data, the comparison is void.
5. **Config control:** every knob for both arms recorded in the results file. Any run whose
   config differs from another is a DIFFERENT EXPERIMENT and is not a replication of it
   ([[claims-die-in-the-follow-up-run]]).
6. **Steelman MFL.** I am the author of the competing method — the failure mode is a weak
   MFL implementation. Requirements: implement Alg. 1 faithfully (two loops, domain
   randomization, conservative LR α2 = 0.99·α1, δ=0.9, MLP hidden 64, α1=0.01); use their
   Table 10 hyperparameters where applicable; **have an independent adversarial agent try
   to refute the MFL arm specifically as under-tuned** before any result is reported. If
   MFL loses, the first hypothesis is that I built it badly.
7. **The refuting arm runs BEFORE the write-up**, not after. No result is reported while a
   run that could refute it is executing.

## 5. What this bake-off CANNOT show

- Nothing about real hardware. Both arms are simulators. Two in-silico methods disagreeing
  tells you which is better *on this simulator*.
- Nothing about MFL's real performance on **their** emulator/task. Different machine,
  different dimensionality, different noise. A loss here is not a refutation of their paper.
- Nothing that generalizes past MBE. RIG's machine is an MBE sim; theirs is plasma etch.
- It cannot settle the M0 gap, which remains the entire scientific claim for both projects.

Related: [[rig-pessimism-inherits-model-miscalibration]], [[claims-die-in-the-follow-up-run]],
[[m2-honest-vs-v1-artifact]], `docs/dimensionality-2026-07-17.md`.


---

# OUTCOMES — scored 2026-07-19 (append-only; predictions above UNTOUCHED)

Corrected run `full_20260719T031201Z` (targets re-frozen `02603fc1…` after the steelman-mandated
label fix; labels independently re-verified by a 20,000-pt Sobol search on a different seed;
MFL arm steelmanned FAITHFUL, grad-through-E exact to 8.3e-17, tuning plateau confirmed).

| | Prediction | Outcome |
|---|---|---|
| P1 | MFL wins queries ≥3× (high conf) | **REFUTED — OPPOSITE.** RIG 60 vs MFL 4,060 (charitable) / 12,060 (deployable). The author's own prediction was wrong: RIG's surrogate inverse is machine-free at solve time — the GP *is* the amortization. |
| P2 | RIG wins certified_miss_rate ≥15pp (medium) | **HOLDS.** 0.00 vs 0.50 — 50pp. |
| P3 | MFL hugs edges, margin gap ≥1.0σ (high) | **HOLDS.** +10.5σ vs −1.45σ (≈12σ gap); MFL's MEDIAN presented recipe is out-of-spec. |
| P4 | MFL cannot abstain (tautological) | Not counted, per the frozen rule. (MFL presented 5/5 infeasible; RIG abstained 5/5.) |
| P5 | RIG false-abstains ≥10% (predicted LOSS) | **HOLDS AS THE LOSS.** 33.3% (5/15): RIG abstained on ALL 5 feasible-but-hard targets. Pessimism's full price, honestly paid. |

**Frozen verdict rule (P2 ∧ P3): "RIG's formulation is better-posed" — SUPPORTED, on this
simulator only.** Both venues in-silico; nothing here is about real hardware or MFL's own
plasma-etch task (§5 of this document stands in full).

Structural read: perfect class separation. Both methods hit all 10 clearly-feasible targets.
The entire difference is the boundary and beyond — RIG refuses (5 correct refusals of
infeasible targets, 5 over-cautious refusals of hard ones), MFL presents all 10 and misses all
10. Of recipes RIG certified, 100% were genuinely in spec; of MFL's, 50%. rig-reval == rig
(miss already 0), so conformal re-validation changed nothing on this target set — the owed
d=20 revalidation experiment remains open on its own venue.
