# M2 BoTorch comparator slate — RIG vs BoTorchBO / SCBO / TuRBO (2026-07-23)

**Scope:** in-silico machinery proof, NOT real-tool evidence. The real-data M2/M3
headline stays gated on M0 (Empa HiPIMS). This slate answers one reviewer
question: *does the M2 "RIG reaches spec ~2x cheaper than BO" claim survive the
two BoTorch families a reviewer would demand next — constrained BO (SCBO) and
trust-region BO (TuRBO) — implemented faithfully and steelmanned?*

Artifacts: [`docs/m2-botorch-slate.json`](m2-botorch-slate.json) (full result +
raw per-campaign rows); driver `examples/run_m2_botorch_slate.py`; new comparators
`src/rig/baselines/trust_region_bo.py`; tests `tests/test_botorch_slate.py`.

## Verdict

**The M2 claim HOLDS against the full BoTorch slate.** RIG reaches spec strictly
cheaper than all three BoTorch comparators (ΔRMST < 0, every pairing significant at
p ≪ 0.05), with an equal-or-higher hit rate on every one. No comparator beats RIG
on cost or hit rate, pooled or per-target. `claim_holds_vs_slate = true`,
`beaten_by = []`.

The "~2x cheaper" figure is honest but arm-dependent: against the **strongest**
comparator (`BoTorchBO`, the production SingleTaskGP + Hvarfner + qLogEI arm) RIG's
RMST is **1.65×** lower; against TuRBO **2.46×**; against SCBO **2.89×**. So "~2x"
is a fair characterization of the slate, and a *floor* of ~1.65× against the best
BoTorch arm.

## Configuration (fairness contract, identical across all four arms)

| Knob | Value |
|---|---|
| Machine | `InSilicoMachine(MBE)`, `metrology_noise=True` (stochastic) |
| Targets | 2 joint `T_center × bow_cooldown_um` (coupled, non-identity) |
| Seeds | 50 (independent noisy machine realizations, CRN-paired across arms) |
| Spec | metrology-anchored, `tol = 6σ`; σ = {T_center 2.08, bow_cooldown_um 3.95e-7} |
| Budget / q / n_seed | 40 / 4 / 8 machine queries; identical warm-start Sobol DoE |
| Cost model | Kanarik (`cost_recipe=1000`, `c_batch=1000`); horizon = 49001 |
| RIG feasibility policy | `ablation` (κ=z_epi=1.0, δ=0.01) — the published-M2 config |
| Bootstrap | 5000 paired resamples, seed 0 |

Every arm gets the bit-identical warm start, same budget/cost/hit-rule, the same
box-sigmoid interior search domain, and the same GP tier (SingleTaskGP + input
Normalize + outcome Standardize + §20.5 Hvarfner √D dim-scaled Matérn-5/2). The §8
feasibility knobs are applied to RIG only (they are RIG's conservatism, not BO's;
injecting them into the comparators would handicap BO). A **hit** = a single
in-spec observation on the noisy machine, for ALL arms — see the false-accept note
below.

## Slate table (RIG reference; ΔRMST = RMST_rig − RMST_cmp, negative ⇒ RIG cheaper)

| Comparator | RMST_cmp | RMST_rig | ΔRMST | 95% CI | p | P(RIG better) | win-rate | hit_cmp | hit_rig | RMST ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BoTorchBO | 27050 | 16400 | −10650 | [−12350, −9000] | 5.7e-30 | 1.00 | 0.81 | 0.97 | 1.00 | 1.65× |
| TuRBO | 40350 | 16400 | −23950 | [−25702, −22000] | 7.2e-118 | 1.00 | 0.97 | 0.68 | 1.00 | 2.46× |
| SCBO | 47401 | 16400 | −31001 | [−32102, −29751] | ~0 | 1.00 | 0.99 | 0.13 | 1.00 | 2.89× |

Per-target (both joint targets agree — no target flips the verdict):

| Comparator | target | ΔRMST | p | hit_cmp | hit_rig | win-rate |
|---|---|---:|---:|---:|---:|---:|
| BoTorchBO | joint_0.30 | −11100 | 6.0e-28 | 1.00 | 1.00 | 0.90 |
| BoTorchBO | joint_0.70 | −10200 | 1.9e-12 | 0.94 | 1.00 | 0.72 |
| TuRBO | joint_0.30 | −25000 | 9.1e-73 | 0.78 | 1.00 | 0.98 |
| TuRBO | joint_0.70 | −22900 | 1.1e-56 | 0.58 | 1.00 | 0.96 |
| SCBO | joint_0.30 | −32901 | 3.7e-208 | 0.12 | 1.00 | 0.98 |
| SCBO | joint_0.70 | −29101 | ~0 | 0.14 | 1.00 | 1.00 |

## Honest reading of the comparators

- **BoTorchBO is the strongest comparator** (hit 0.97, RMST 1.65× RIG) — as
  expected, since it is a fully-tuned production GP-EI arm. RIG's win over it is
  the load-bearing result: 1.65× cheaper, win-rate 0.81, p = 5.7e-30.
- **TuRBO (hit 0.68)** trails. The trust-region shrink/expand machine spends
  budget contracting around a local incumbent; under a 40-query budget in a 2-D
  coupled space with a 6σ box on a tiny-scale KPI, it converges more slowly than
  global EI/qLogEI — a known TuRBO trade-off (it is built for high-dimensional
  global optimization, not tight low-D target-hitting).
- **SCBO is the weakest arm here (hit 0.13)** — and this is a genuine,
  non-rigged finding, not a broken comparator. SCBO models the `bow_cooldown_um`
  box as separate outcome constraints at a native scale of ~4e-7 (box half-width
  ~2.4e-6); its constrained-Thompson feasibility search rarely drives a candidate
  into that pinhole box within 40 queries. The **steelman sanity tests prove the
  SCBO implementation itself is correct** (see below): on a well-scaled bowl the
  seed DoE misses, SCBO hits 8/8 seeds where pure random search hits 0/8. So the
  low MBE hit rate is a property of SCBO-vs-this-problem, reported plainly, not a
  silent implementation failure faking a RIG win.

## Steelman / known-answer evidence (why a broken comparator can't fake this)

A comparator that silently failed to optimize would look like random search and
manufacture a RIG "win". The discriminating check (in `tests/test_botorch_slate.py`,
`test_known_answer_optimizes_where_random_cannot`): a 2-D bowl whose feasible box
is a tight disk (radius ~0.024) the 8-point seed DoE misses. At the same 60-query
budget:

- pure scrambled-Sobol random search: **0 / 8** seeds hit;
- TuRBO: **8 / 8** hit — and the hit comes from the optimization loop
  (`stop_reason` excludes "seed DoE"), i.e. Thompson sampling inside the trust
  region converged to the known optimum region;
- SCBO: **8 / 8** hit via constrained-Thompson.

Both arms demonstrably optimize where random cannot — the sanity check that would
have caught a faked win.

## Implementation fidelity

- **TuRBOBaseline** — TuRBO-1 (Eriksson et al., NeurIPS 2019, arXiv:1910.01739),
  the canonical BoTorch tutorial `TurboState` machine verbatim: length 0.8 →
  [0.5⁷, 1.6], `failure_tolerance = ⌈max(4/q, d/q)⌉`, success_tolerance 10,
  relative-1e-3 improvement rule, lengthscale-shaped Sobol candidate set with the
  perturbation mask (prob min(20/d,1)), Thompson selection via
  `MaxPosteriorSampling`, restart on collapse below `length_min`.
  *Declared simplifications:* TuRBO-**1** (single region), not TuRBO-m — the
  standard/strongest choice at a ~40-eval, d≈2 budget; and restart retains
  observation history (can only help the comparator).
- **SCBOBaseline** — SCBO (Eriksson & Poloczek, AISTATS 2021, arXiv:2002.08526):
  the spec box as 2m outcome constraints `c(x) ≤ 0`, one GP per constraint
  (`ModelListGP`), constrained Thompson sampling via
  `ConstrainedMaxPosteriorSampling` (feasible-first, else minimum-violation),
  inside the TuRBO trust region driven by *feasible* improvement (the canonical
  constrained `update_state`). *Declared simplification:* the objective is the
  shared box-distance scalarization `f=-g`, redundant with the constraints by
  construction (the spec *is* the box), so SCBO here is a constrained-feasibility
  searcher; the objective is kept non-degenerate rather than constant.

Both share BoTorchBO's fairness contract exactly (warm start, budget, cost, hit
rule, interior domain, GP tier); no RIG-only trick (the μ−κσ pessimistic
objective) leaks into either. Continuous recipe spaces only; a compositional
variable raises, matching BoTorchBO.

## Caveats / false-accept

- In this cost-to-target harness a **hit is a single in-spec observation on the
  noisy machine for ALL arms** — RIG here runs *without* the F2 confirmation hook
  (the published-M2 posture), alongside the three BO arms. So false-accept
  exposure is identical across arms and is **not** a differentiator in this
  comparison; a confirmation-gated variant is the F2 `ConfirmationCampaign`, out of
  slate scope. We report this rather than manufacture an arm-specific
  certified-miss metric.
- **In-silico only.** These are machinery numbers on the calibrated MBE
  `InSilicoMachine` with `metrology_noise` on. Nothing here is real-tool evidence;
  the real-data headline remains gated on M0.
- RIG runs under the **ablation** feasibility policy (the more permissive
  published-M2 config). The binding §8 2.0/2.0/0.02 re-run is a separate owed item
  (`docs/m2-result-binding.json`), not re-done here.

## Reproduce

```
python examples/run_m2_botorch_slate.py --seeds 50 --targets 2 --tol-k 6 \
    --bootstrap 5000 --out docs/m2-botorch-slate.json
```

Determinism: every campaign is seeded off the campaign seed (torch Sobol/MC
sampling included); the paired bootstrap uses seed 0. Per-arm run-to-run
byte-identity is pinned by `test_deterministic_same_seed` for both comparators.
Full run wall-time ≈ 52 min (400 campaigns) under concurrent load.
