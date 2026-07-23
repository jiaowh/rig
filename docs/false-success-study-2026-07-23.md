# False-success rate vs input dimension — the gate OFF vs ON (2026-07-23)

**Safety-central.** RIG's central promise is "no false successes": a CERTIFIED recipe
must really land in spec on the TRUE function, not merely on the surrogate's opinion of
itself. The known crack (`docs/dimensionality-2026-07-17.md`): at d=20 / n=800 one
certified recipe missed ground truth — a real, deterministic FALSE SUCCESS whose
mechanism is that the §8 pessimistic κ·σ margins read the GP's RAW σ, which at high d was
optimistic. Since 2026-07-22 the §13.2 conformal containment gate `C(x) ⊆ Z*` runs BY
DEFAULT when the solver's model is conformal-wrapped (`_conformal_screen`), and every
candidate carries a `calibration_status`. This study measures what that fix buys and what
it costs, gate OFF (raw) vs ON (wrapped), across dimension and seeds.

Reproduce: `python examples/run_false_success_study.py --full` (JSON:
`docs/false-success-study.json`). Smoke + determinism: `... --smoke`.

## Pre-registered question (fixed before results)

Does conformal-wrapping the surrogate (arm B: §13.2 gate default-on) reduce the rate of
CERTIFIED false successes vs the raw-σ solver (arm A: gate inert), and at what cost in
abstention and genuine-hit rate? Directional hypotheses, either direction a finding:
**H1** wrapped FSR ≤ raw FSR at every d; **H2** wrapped abstention ≥ raw (the gate can
only remove candidates); **H3** wrapped genuine-hit-rate/candidate ≥ raw. Honesty guards:
0 observed ⇒ report the 95% Clopper–Pearson UPPER BOUND, never a claim of zero; the
wrapped surrogate is fit on FEWER runs (calibration carved from the SAME budget — the real
cost); an infinite conformal quantile (cal too small) is flagged, not silently scored.

## Design and achieved power

- **Grid:** d ∈ {2, 8, 15, 20} at n = 12·d, PLUS the d=20 / n=800 CRIME SCENE — 5 cells ×
  2 arms × **20 seeds**. The truth family varies per seed (`make_truth(d, seed)`), so each
  seed is an independent draw from the smooth 2-output process family AND its N(0, 0.05)
  label noise — a population estimate, not one fixed function. Reachable target =
  true function at a held-out on-support point, ±0.8 on both outputs (the house pattern).
- **Arms, same data:** RAW = unwrapped GP fit on all n runs (gate structurally inert →
  `model-feasible`). WRAPPED = the SAME GP wrapped by the REAL `SplitConformalCalibrator`
  + `ConformalForwardModel`, calibrated on a held-out block of `n_cal = round(n/3)` carved
  FROM the budget (effective train n reported per cell) → gate default-on →
  `conformal-checked`.
- **Solver policy:** binding κ/z_epi/δ = 2.0 / 2.0 / 0.02; GP hyperparameter-fit restarts
  = 3; multi-start budget = 48 (the house/crime-scene budget — see the config note below).
- **Search path (grid):** `analytic_grad=True`. Timed empirically first (below): under
  concurrent load a d=20 FD solve is 222 s vs 16 s analytic AT THE SAME VERDICT — analytic
  is the established high-d speedup and is what makes 20 seeds feasible in ~93 min. The
  crime scene is run SEPARATELY in the exact original FD path to reproduce the miss byte-
  for-byte.
- **Achieved power:** with the solver's high abstention at 12·d density, certified
  candidates are scarce (7–40 per cell), so the powered grid can only bound rare rates: a
  0-count cell gives a 95% CP upper bound of ~8–41%/candidate (tighter where more
  candidates survive). Pooled over the grid the raw arm is **0 / 122 candidates → FSR ≤
  3.0 % (95%)**; the wrapped arm **1 / 84 → 1.2 % [0.03, 6.4] %**. False successes are too
  RARE at this NSEEDS to separate the arms in the grid — the honest headline is these
  bounds plus the crime-scene reproduction below, NOT a point rate.

## The false-success table

| cell | arm | eff train n | n_cal | feas/seeds | abstain | cand | FS | hit-rate/cand | FSR/cand [95% CI] | FSR/seed [95% CI] | worst miss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d2 n24 | raw | 24 | 0 | 19/20 | 5% | 33 | 0 | 100.0% | 0.0% [0.0, 10.6]% | 0.0% [0.0, 16.8]% | 0.000 |
| d2 n24 | wrapped | 16 | 8 | 0/20 | 100% | 0 | 0 | — | — (no candidates) | 0.0% [0.0, 16.8]% | 0.000 |
| d8 n96 | raw | 96 | 0 | 15/20 | 25% | 40 | 0 | 100.0% | 0.0% [0.0, 8.8]% | 0.0% [0.0, 16.8]% | 0.000 |
| d8 n96 | wrapped | 64 | 32 | 11/20 | 45% | 33 | **1** | 97.0% | **3.0% [0.1, 15.8]%** | 5.0% [0.1, 24.9]% | 0.237 |
| d15 n180 | raw | 180 | 0 | 7/20 | 65% | 20 | 0 | 100.0% | 0.0% [0.0, 16.8]% | 0.0% [0.0, 16.8]% | 0.000 |
| d15 n180 | wrapped | 120 | 60 | 8/20 | 60% | 24 | 0 | 100.0% | 0.0% [0.0, 14.2]% | 0.0% [0.0, 16.8]% | 0.000 |
| d20 n240 | raw | 240 | 0 | 3/20 | 85% | 7 | 0 | 100.0% | 0.0% [0.0, 41.0]% | 0.0% [0.0, 16.8]% | 0.000 |
| d20 n240 | wrapped | 160 | 80 | 4/20 | 80% | 9 | 0 | 100.0% | 0.0% [0.0, 33.6]% | 0.0% [0.0, 16.8]% | 0.000 |
| d20 n800 | raw | 800 | 0 | 9/20 | 55% | 22 | 0 | 100.0% | 0.0% [0.0, 15.4]% | 0.0% [0.0, 16.8]% | 0.000 |
| d20 n800 | wrapped | 533 | 267 | 8/20 | 60% | 18 | 0 | 100.0% | 0.0% [0.0, 18.5]% | 0.0% [0.0, 16.8]% | 0.000 |

CI = exact Clopper–Pearson 95%. "worst miss" = raw-units excursion of the worst certified
miss beyond the spec box (0 = every certified candidate genuinely in spec).

## The gate's measured effect — reduction AND abstention cost

**1. In the powered grid, the gate's dominant MEASURED effect is abstention cost, not a
measurable FSR reduction** — because at 12·d density the raw solver is ALREADY extremely
abstemious (5 %→85 % of solves return INFEASIBLE with the §8.8 *epistemic-limited*
diagnosis), so certified candidates are scarce and genuine false successes rarer still
(raw 0/122). There is simply little for the gate to remove. Where the gate DOES add
abstention beyond the raw arm's own pessimism (Δ = wrapped − raw):

| d | raw abstain | wrapped abstain | Δ | wrapped abstention cause |
|---|---|---|---|---|
| 2 | 5% | 100% | **+95pp** | conformal (all 20 solves have an INFINITE band) |
| 8 | 25% | 45% | +20pp | still epistemic (gate rarely the binding cause) |
| 15 | 65% | 60% | −5pp | epistemic |
| 20 (n240) | 85% | 80% | −5pp | epistemic |
| 20 (n800) | 55% | 60% | +5pp | epistemic (1 unreachable) |

The wrapped arm's INFEASIBLE reasons are almost entirely `epistemic`, not `conformal`,
for d ≥ 8 — i.e. the pessimistic κ margins + high epistemic σ abstain BEFORE the §13.2 gate
ever binds. So beyond the degenerate d=2 cell the gate's *extra* abstention cost at this
data density is small (~0–20 pp), and it costs **no** genuine hits at d ≥ 15 (hit-rate
1.00 both arms). The real abstention comes from the surrogate being weaker on fewer runs
(eff train n = ⅔·n), which is the honest price of calibration, not the gate rejecting good
recipes.

**2. d=2 wrapped is the small-budget DEGENERATE case (report, don't hide).** At n=24 the
honest ⅓ split leaves n_cal=8, and `ceil((1−α)(n_cal+1)) = 9 > 8`, so the α=0.1 conformal
quantile is **+∞** — an infinite band, which the containment gate `C(x) ⊆ Z*` rejects for
every candidate. Result: **100 % abstention** (`n_band_infinite = 20/20`). The gate cannot
help where the budget cannot fund a finite calibration quantile; this is a cost, not a
benefit, and d=2 is excluded from any gate-effect claim.

**3. The gate is NOT a certificate — the wrapped arm produced its OWN false success
(d=8).** One wrapped d=8 solve certified a `conformal-checked` candidate that misses truth
by 0.237 (FSR/candidate 3.0 % [0.1, 15.8] %), while the raw d=8 arm had 0/40. This is not
the gate failing to remove a raw-arm miss — the arms fit DIFFERENT surrogates (wrapped:
64 runs; raw: 96) — but it is the honest lesson: split-conformal gives MARGINAL coverage
over exchangeable draws, and the solver hands it SELECTED points (it searches for where the
model looks best), so a marginal miss can slip through at a solver-chosen recipe. Carving
calibration data out of a fixed budget weakens the surrogate, and the gate's marginal
coverage does not catch every selected-point miss. **H1 is therefore NOT supported at fixed
budget: wrapping is not strictly safer at every d** — at d=8 the weaker wrapped surrogate +
non-conditional coverage produced a miss the stronger raw surrogate did not. The fix
direction the dimensionality doc named — conditional/Mondrian conformal, or a
selection-inflation term — is what would close this, not a bigger κ.

## The d=20 / n=800 crime scene — does the original miss reproduce, does arm B kill it?

Run in the EXACT 2026-07-17 config (FD search, 48 restarts, GP restarts=3, seed 0):

| arm | verdict | candidates | genuine hits | false successes | worst miss | calibration_status |
|---|---|---|---|---|---|---|
| RAW (gate inert) | FEASIBLE | 3 | 2 | **1** | **0.2461** | model-feasible |
| WRAPPED (gate on) | FEASIBLE | 1 | 1 | **0** | 0.000 | conformal-checked |

**The original miss REPRODUCES exactly.** The raw arm's worst certified miss excursion is
**0.2461** — i.e. the true outcome landed **1.046 against the ±0.8 box**, the exact
`worst_err 1.046` recorded on 2026-07-17. One of three certified recipes is a real,
deterministic false success on the FD default path.

**Arm B kills it.** The default-on §13.2 gate rejects the two candidates whose calibrated
band spills the box (including the false-success one), returns the single survivor whose
band fits, and that survivor GENUINELY hits truth — `calibration_status =
"conformal-checked"`, 0 false successes. This is the mechanism the gate was built for (the
raw σ was optimistic; the calibrated band caught it), demonstrated on the exact crime
scene, and it is why the crime scene lives in the data-rich (n=800) regime where the solver
certifies enough to make the gate matter.

**Fragility revisited — the miss is search-path fragile too.** The SAME (d=20, n=800,
seed 0) cell under the grid's `analytic_grad` search path returns 3/3 genuine hits, **0
false successes** — the marginal miss (1.046 vs 0.8) flips to a hit when L-BFGS-B takes
exact-gradient steps instead of finite-difference ones. This extends the 2026-07-17 finding
("vanishes under a small change to the GP hyperparameter fit") to the search path: **the
d=20 raw-σ margin has no safety buffer — a false success sits on a knife-edge of GP fit AND
optimizer path.** A conservatism guarantee that flips with such details is exactly why the
calibrated gate, not the raw κ margin, must be the acceptance test at high d.

## Determinism

- `examples/run_false_success_study.py --smoke` double-runs a d2+d8 grid and asserts the
  timing-free view is **byte-identical (True)**; `tests/test_false_success_study.py::
  test_smoke_end_to_end_and_determinism` pins the same property in CI.
- The full grid is deterministic given `(base_seed=0, nseeds=20, n_restarts=48,
  analytic_grad=True)`: every sub-seed derives from `base_seed + seed_index` (data draw,
  GP fit, target, solver all keyed on it). Seeds 0–19 are recorded in the JSON `per_seed`
  (via the checkpoint) so any cell is re-runnable.

## Honest limits

- **Synthetic and smooth.** A random-weight tanh/sin process is friendly to a Matérn-5/2
  GP; this shows the safety machinery composes and where it cracks, not that real high-dim
  process physics is learnable from 12·d runs.
- **Underpowered for rare events.** 20 seeds × ~1–2 certified candidates/solve at high d is
  too few to estimate a sub-percent false-success rate; the grid delivers CP UPPER BOUNDS
  (raw ≤ 3.0 % pooled) and the crime-scene existence proof, not a powered rate curve.
- **Grid vs default path.** The grid measures the `analytic_grad` search path (verdict-
  equivalent to the FD default per the solver's own verification, and the only way to
  afford 20 seeds); the crime scene anchors the exact FD default. The one grid false
  success (d=8) and the crime-scene FD miss are both marginal (~0.24 beyond a 0.8 box),
  consistent across paths.
- **The gate is marginal, not conditional.** It caught the crime-scene miss but let a
  d=8 selected-point miss through — do not read "gate on" as a certificate.
