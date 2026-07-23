# F9 — the inverse above 2 input dimensions (2026-07-17)

**The gap (audit finding F9):** every result in this repo sat at **2 input dimensions** —
MBE recipe = `T_heater` + `film_thickness`, sputtering = power × pressure, the M3 toy =
flux × temperature. The core is dimension-agnostic *by construction* (the GP, the
`RecipeTransform`, the §8 margin all take vectors), but "it generalizes" was an
**untested claim about the exact axis where inverse problems get hard**. Nothing
anywhere said so.

Closed by measurement, not assertion. Reproduce: `python examples/run_dimensionality_study.py`.

## Method — scored against GROUND TRUTH, not the model

The trap in evaluating an inverse is scoring the returned recipe with the same surrogate
that proposed it: the model agrees with itself and you learn nothing (this is exactly the
circularity the audit flagged in the M3 gate, F2). So here the loop is:

1. Define a smooth `d`-dim, 2-output process with **every input dimension active**
   (decaying random weights, `tanh`/`sin` couplings — not a 2-D problem in a `d`-dim coat).
2. Train a GP on `12·d` scrambled-Sobol runs + noise (σ=0.05).
3. Pick a reachable target: evaluate the **true** function at a held-out on-support point,
   ask for that outcome ±0.8 on both outputs.
4. Solve. Then evaluate the **TRUE function** at each returned recipe and check whether it
   really lands in the spec box. `truth()` is never called during the solve.

## Result — it works

| d | n_train | verdict | ground-truth hits | worst err | t_fit | t_solve |
|---|---|---|---|---|---|---|
| 2 | 24 | FEASIBLE | **1/1** | 0.531 | 0.05 s | 2.6 s |
| 4 | 48 | FEASIBLE | **3/3** | 0.609 | 0.17 s | 7.6 s |
| 6 | 72 | FEASIBLE | **3/3** | 0.479 | 0.31 s | 13.5 s |
| 8 | 96 | FEASIBLE | **3/3** | 0.274 | 0.44 s | 30.9 s |
| 10 | 120 | FEASIBLE | **3/3** | 0.352 | 0.95 s | 34.2 s |
| 15 | 180 | FEASIBLE | **3/3** | 0.181 | 3.48 s | 70.9 s |
| 20 | 240 | INFEASIBLE | — | — | 3.72 s | 150.8 s |
| **20** | **480** | **FEASIBLE** | **1/1** | — | — | 97.1 s |
| **20** | **800** | **FEASIBLE** | **2/3 ⚠ FALSE SUCCESS (real, deterministic)** | 1.046 vs tol 0.8 | — | 335.0 s |

Through **d=15 every returned recipe genuinely hit the spec box on the true function** —
not the model's opinion of the recipe, the actual function. **At d=20 with 800 runs, 1 of 3
certified recipes MISSED — a REAL, deterministically reproducible false success**, though a
marginal one (1.046 against a ±0.8 box) that vanishes under a small change to the GP's
hyperparameter fit. Read the false-success section before quoting it: one cell is not a rate.

### The d=20 abstention was HONEST — confirmed by the control

At `d=20, n=240` the solver returned `INFEASIBLE` with the §8.8 **epistemic-limited**
diagnosis — *"the target is reachable; you have not earned it yet; collect runs."*

Two arms distinguish an honest abstention from a false one (we failed to FIND a recipe and
reported that none EXISTS):

| arm | n_train | restarts | verdict |
|---|---|---|---|
| add DATA | 480 | 48 | **FEASIBLE**, genuine ground-truth hit |
| add SEARCH (the control) | 240 | **192** (4×) | **still INFEASIBLE**, same epistemic diagnosis |

**Quadrupling the search budget did NOT flip it; doubling the data did.** So the
abstention was genuinely data-limited. The solver correctly diagnosed that 240 points in a
20-dim box is too sparse to certify anything, said which of the four §8.8 causes applied,
and named the fix — and the fix worked. That is the §8.8 taxonomy earning its keep on a
problem 10× larger than anything it had seen.

### ⚠ FALSE SUCCESS at d=20 — REAL, deterministic, marginal, and config-fragile

**Status: CONFIRMED as a real observation in its exact configuration. It is one (seed, config)
cell — not a rate.**

`d=20, n=800` returns FEASIBLE with **2 of 3** candidates genuinely in spec: the third is
certified by the solver and **misses on the true function**. That is a false success — worse
than a false abstention, because it is the pessimistic guarantee failing to be pessimistic, the
one thing §8 exists to prevent.

**Reproduced exactly.** Re-invoking the original code path with its own arguments and seed —
`dim_probe.probe(20, 800, seed=0, n_restarts=48)` — returns `FEASIBLE, 2/3, worst_err = 1.046`
deterministically. Against a tolerance of ±0.8 that is a **marginal miss: 1.046 vs 0.8, ~31%
beyond the box** — not a wild extrapolation, a recipe that crept just outside.

**But it is fragile to the model fit, which is itself informative.** Two nearby configurations
show ZERO false successes:

| config | difference from the original | candidates | false successes |
|---|---|---|---|
| original | `GPForwardModel(n_restarts=3)`, all 800 rows train | 3 (seed 0) | **1** (reproduced deterministically) |
| replication A | **`n_restarts=2`** GP fit, independent 256-pt calibration draw | 9 (seeds 0,1,5) | 0 |
| replication B | 200 of 800 held out ⇒ a **600-point model** | 6 (seeds 0–1) | 0 |

Dropping the GP's hyperparameter-fit restarts from 3 to 2 makes it vanish. So the false success
is real but sits on a knife edge: it depends on which local optimum the marginal-likelihood fit
lands in. **A conservatism guarantee that flips with a hyperparameter-fit detail is not a
guarantee** — that is the actual finding, and it is worse news than a robust-but-rare failure
would be, because it means the margin has no safety buffer at d=20 for some fits.

*(Process note: this section previously said "did not replicate" on the strength of replication
A. That was wrong — A changed the GP fit and so tested a different model. The determinism check
is the arbiter and it confirms the original. Recorded because the mis-correction is itself an
instance of the lesson: replicate the EXACT config before believing either a claim or its
retraction.)*

**The structural mechanism — and this stands regardless of how rare the anomaly is:** the §8
feasibility test consumes the model's **RAW** sigmas. The margin math reads `mean`,
`aleatoric_sigma`, `epistemic_sigma` off `predict`; **`conformal_set` is not part of the
feasibility decision at all.** So the κ margin is only ever as trustworthy as the surrogate's
own uncertainty estimate — the pessimism **inherits any miscalibration beneath it**, and at
d=20 with 20 ARD lengthscales that estimate is evidently optimistic (one candidate's true error
reached **3.12× its own claimed σ** — roughly a 1-in-500 Gaussian event, in 9 draws). The
defence that would catch exactly this — the §13.2 `C(x') ⊆ Z*` re-validation gate
(`revalidation_model`) — **defaults to `None`**, and `active/loop.py` never sets it. The default
path a user gets has no conformal check on feasibility whatsoever.

> **UPDATE 2026-07-22 (F1/F7, audit 2026-07-21) — the two claims in the paragraph above are
> now stale; corrected here, original left in place as the record (never rewrite history).**
> (a) `active/loop.py` DOES set `revalidation_model`: since the WP-E slice-2 work it
> auto-sets it to the full surrogate whenever a fast/full ensemble split exists
> (`inner is not surrogate`), so "`active/loop.py` never sets it" is no longer true.
> (b) BUT that auto-revalidation was conformally INERT on every default path anyway,
> because nothing in the loop conformal-**wraps** the surrogate — its §13.2 component
> re-checked only margins/support (`_conformal_in_box` returns `True` when
> `conformal_set is None`). So the substance of the finding — the default path had no
> conformal check on feasibility — held, for a subtler reason than the sentence gives.
> (c) As of 2026-07-22 `PessimisticInverseSolver.solve` applies the §13.2 `C(x) ⊆ Z*`
> containment **by default** whenever its own `model` IS conformal-wrapped (no
> `revalidation_model` needed), using the same anti-false-abstention pool sweep the
> reval path uses; and every emitted candidate now carries an explicit
> `calibration_status` ∈ {`model-feasible`, `conformal-checked`, `revalidated`}, so a
> raw-σ recommendation is no longer indistinguishable from a conformally-accepted one.
> `model-feasible` (no conformal model present — e.g. a **bare GP**, which is what every
> d=20 row above used) is still ONLY the raw-σ κ margin and remains explicitly NOT a
> calibrated guarantee — so the false-success verdicts above are unchanged. Owed item 2
> below (does a conformal-wrapped model reject the bad candidate?) is now mechanized as
> the default path and covered by `tests/test_conformal_feasibility.py`.

**Owed, in priority order:**
1. Quantify the false-success RATE vs d across many seeds AND across GP-fit restarts
   (`rig.eval` already has a `false_success` metric). One cell is not a rate.
2. Test whether a conformal-wrapped `revalidation_model` rejects the bad candidate. If it does,
   that is a strong argument for making conformal re-validation the DEFAULT at high d.
3. Ask whether `z_epi=2.0` is simply too little paranoia once d is large, and whether the
   epistemic term is systematically optimistic with many ARD lengthscales on few points.

### The selection effect — a real structural concern, but NOT observed here

**Recorded as an open question, not a finding. The measurement does not support it.**

The reasoning is sound and worth writing down. Conformal prediction guarantees **marginal**
coverage over *exchangeable* draws. The inverse solver does not hand it exchangeable draws — it
**searches** for recipes where the model looks most favourable (the entire point of the §8.6
multi-start) and reports those. Selection is a distribution shift, so coverage at chosen points
is not what split conformal promises, and nothing in the construction makes it conditional.
This matters structurally because the §13.2 re-validation gate (`C(x') ⊆ Z*`) — the designated
defence against a false success — evaluates conformal bands *precisely at the points the solver
chose*.

**But it did not show up.** Measured at d=20, model fit on all 800 runs, conformal calibrated on
an independent 256-point draw, 9 certified candidates across 6 seeds:

| quantity | observed | nominal |
|---|---|---|
| conformal band misses truth at SELECTED recipes | **1 / 9 (11%)** | 10% (α = 0.1) |
| max true error ÷ model's own claimed σ | 3.12× (one candidate) | ~1× typical |

**11% against a nominal 10% is coverage behaving as advertised.** An earlier run showed 2/6
(33%) and was briefly written up here as evidence for the selection effect — that was wrong to
report: it came from a **600-point model** (200 of the 800 runs were held out for calibration,
so it was not the model under test), and 6 points is not a rate. The larger, cleaner
measurement supersedes it.

The one number that survives as notable: a single candidate whose true error was **3.12× the
model's own claimed σ**. Under a Gaussian that is roughly a 1-in-500 event, and it turned up in
9 draws — suggestive that the tail is heavier than the model believes at d=20. One observation;
and the conformal band is exactly what would absorb it.

Owed (now clearly worth doing rather than assumed): quantify coverage at solver-SELECTED points
vs RANDOM points, same model, across seeds and d. That contrast is the actual test of the
selection effect and neither run performed it. If it degrades, the fix direction is
conditional/Mondrian conformal or a selection-inflation term — not a bigger κ.

**Owed work, in priority order:**
1. Reproduce with more seeds — this is n=1 and could be a marginal miss; quantify the
   false-success RATE vs d (the §12 `false_success` metric already exists in `rig.eval`).
2. Check whether a conformal-wrapped `revalidation_model` catches the bad candidate. If it
   does, that is a strong argument for making conformal re-validation the DEFAULT at high d
   rather than an opt-in.
3. Investigate whether the GP's epistemic term is systematically optimistic at high d
   (ARD with 20 lengthscales on 800 points), and whether `z_epi=2.0` is simply too small a
   paranoia setting once d is large.

Do not read the d≤15 rows as a clean bill of health for d=20.

## Two real dimensional weaknesses found (both now handled)

**1. The restart budget was FIXED at 48 regardless of `d` — FIXED.**
48 scrambled-Sobol starts densely covers a 2-D box and is vanishing in 20-D. This is not
a mere slowdown: a starved multi-start degrades into a **false `INFEASIBLE`** — we fail to
*find* a recipe and report that none *exists*, the precise confusion §8.8 is built to
prevent, and the same false-abstention class as the re-validation bug fixed the same day.
`n_restarts=None` (new default) now scales as `24·dim`, floored at 48.

> `24·dim` evaluates to **exactly 48 at dim=2**, so M2, the AL loop, and every existing
> 2-D result are bit-for-bit unchanged. The budget keys on `RecipeTransform.dim` — the
> *free u-coordinates the optimizer actually searches* (K−1 per simplex), not the recipe
> key count. Passing an explicit int still wins.

**2. Cost grows ~O(d²), and it is the GRADIENT, not the model — DOCUMENTED, NOT FIXED.**
`minimize(..., method="L-BFGS-B")` is called **without `jac`**, so SciPy finite-differences
the objective: `d+1` evaluations per gradient step, each running `predict` *and*
`jacobian`. Multiply by the (correctly) growing restart budget and solve time goes 2.6 s
at d=2 → ~150 s at d=20.

An analytic objective gradient needs `∂σ_epi/∂x` and `∂J/∂x` — the latter is a *second*
derivative of the GP mean. That is real work and it is **owed**, not silently accepted.
Until it lands, treat `d ≳ 20` as needing the torch tier's autograd or a cut budget.

## Honest limits of this study

- **Synthetic and smooth.** A random-weight `tanh`/`sin` process is friendly to a
  Matérn-5/2 GP. It shows the machinery composes at `d`; it does not show real
  high-dim process physics is learnable from `12·d` runs.
- **`12·d` scaling is a choice, not a law.** Real campaigns will not grant 240 runs for a
  20-knob process. The honest reading of the d=20 row is *"the method needs data
  proportional to the space, and it tells you so instead of guessing."*
- **Still not M0.** This closes "does the inverse work above 2-D" (yes). It does not touch
  the program's actual claim, which needs a real recipe→outcome dataset on the target
  process. See BUILD_STATE.
- Single seed per cell. Direction and margins are unambiguous; the exact numbers are not
  a powered estimate.
