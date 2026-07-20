# RIG — Recipe Inverse Generator

**Calibrated, uncertainty-aware inverse recipe generation for process manufacturing.**
Given the *outcome* you need, RIG returns the *recipes* that get you there — each with a
certified safety margin — or it says **INFEASIBLE** with a diagnosis instead of inventing
one.

> **Honest status (read this first).** RIG's machinery is built, tested (542 tests), and
> audited. What it does *not* yet have is a certified result on the project's real target
> process. Every headline number is either in-silico (against a physics simulator) or on
> public third-party datasets — the largest of which, a real magnetron-sputter tool
> dataset, was ingested and gated for the first time on 2026-07-19 (see
> [Real-data results](#real-data-results)). This is a **research prototype**, not a
> production qualification tool. Claims below are scoped to exactly what was measured.

---

## The problem it solves

Every model in ML runs *forwards*: settings in, result out. Manufacturing needs the
opposite. A process engineer starts from a **specification** — "deposition rate in
[1.0, 1.4] Å/s, film uniformity under 1.5%" — and asks the backwards question: *what
recipe do I run, how confident should I be, and when is it impossible?*

That inverse is hard in four specific ways, and each is a design driver:

1. **Non-injective** — many recipes hit the same outcome, so the answer is a *set*, not a point.
2. **Reward-hackable** — a naive optimizer walks into the model's blind spots, where it is
   confidently wrong, and hands you a wafer-destroying recipe that looks perfect on paper.
3. **Noisy** — a recipe sitting exactly on the spec limit is a coin flip, not an answer.
4. **Drifting knobs** — the tool delivers 1,341.8 K when you dialed 1,340 K.

RIG answers each with a **pessimistic inverse**: it only certifies a recipe that stays in
spec *even if the model is as wrong as it admits it could be, the knobs drift the worst way,
and there is still margin to spare.*

## How it works (four stages)

| Stage | What it is |
|---|---|
| **Forward model** | A Gaussian process (Matérn-5/2 + ARD) fits the process from very little data and reports *two* uncertainties — aleatoric (irreducible noise) and epistemic (model ignorance) — kept strictly separate. |
| **Conformal calibration** | Turns the model's error bars into ones with a *guaranteed* coverage rate, distribution-free. Includes an online **ACI** path (adaptive conformal inference) that keeps coverage honest under drift. |
| **Pessimistic inverse (§8)** | Searches for recipes whose worst-case margin `s = z_epi·σ_epi + Σ|J|·Δ` still clears the spec box by `κ` noise-widths. Returns a diverse, ranked candidate set — or an explicit INFEASIBLE with a cause diagnosis. |
| **Qualification gate** | The final anti-reward-hacking check before a recipe is trusted. *(Interface defined; concrete gate + independent verifier still owed — see roadmap.)* |

A longer, illustrated walkthrough (with live interactive figures) is in
[`docs/rig-explained.html`](docs/rig-explained.html).

## Architecture

RIG is a **process-agnostic core** plus **per-process adapters**, with a hard,
CI-enforced boundary between them:

```
src/rig/            process-agnostic core — MUST NEVER import from rig_adapters/
  forward/          GP forward model + predictive distribution
  calibration/      split-conformal, jackknife+, ACI
  inverse/          pessimistic per-query solver + amortized generator
  active/           active-learning loop (BALD / EPIG)
  qualification.py  the certification interface
  schema.py         RunRecord (Pydantic v2 + Pint, SI-canonicalized at ingest)
  registry.py       adapter discovery via importlib.metadata entry points

src/rig_adapters/   per-process adapters (self-register; core never names them)
  tabular/          generic CSV/JSON → RunRecord ingest (used for real datasets)
  mbe/              MBE physics-sim adapter + in-silico "machine" with pathologies
```

The `rig → rig_adapters` import ban is enforced by
[import-linter](https://pypi.org/project/import-linter/) in CI. Adapters register
themselves through the `rig.adapters` entry-point group.

### Canonical interfaces

```python
ForwardModel.predict(x) -> PredictiveDistribution(
    mean, aleatoric_sigma, epistemic_sigma, conformal_set)
ForwardModel.support_score(x)      # "does this recipe look like anything we've run?"
ForwardModel.jacobian(x)           # sensitivity, used to price knob-drift risk
InverseSolver.solve(spec) -> list[RecipeCandidate]   # with feasibility flags / INFEASIBLE
QualificationGate.certify(recipe)
```

## Install

Requires **Python ≥ 3.12**. The core is deliberately **torch-free** (pydantic v2, pint,
numpy, scipy only), so `import rig` works with just:

```bash
python -m pip install -e ".[dev]"
```

The optional deep-learning tier (ensembles, SBI flows, BoTorch BO/AL) needs torch. On
CUDA hardware install torch from the appropriate index *first*, then:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128  # example: CUDA 12.8
python -m pip install -e ".[torch]"
```

Run the tests:

```bash
python -m pytest tests/ -q      # 542 tests
```

## Quickstart

Ingest a tabular dataset through the generic adapter, fit the calibrated forward model,
and run the pessimistic inverse:

```python
from rig_adapters.tabular.spec import load_spec
from rig_adapters.tabular.ingest import ingest_csv
from rig.forward import GPForwardModel, records_to_arrays
from rig.calibration.conformal import SplitConformalCalibrator, ConformalForwardModel
from rig.inverse.pessimistic import PessimisticInverseSolver

spec = load_spec("examples/real_data/empa_hipims/specs/al_120w_short_pw.toml")
records = list(ingest_csv("examples/real_data/empa_hipims/csv/al_120w_short_pw.csv", spec))

# ... split, fit GPForwardModel, calibrate, wrap in ConformalForwardModel,
# then PessimisticInverseSolver(...).solve(spec_box) -> [RecipeCandidate | INFEASIBLE]
```

Complete, runnable examples:

- [`examples/real_data/empa_hipims/`](examples/real_data/empa_hipims/) — the real
  sputter-tool M1 gate run end-to-end (see its `README.md` and `RESULTS.md`).
- [`examples/real_data/sputtering/`](examples/real_data/sputtering/) — a second real
  dataset (Zr magnetron sputtering) as a machinery proof.
- [`examples/mfl_bakeoff/`](examples/mfl_bakeoff/) — the pre-registered head-to-head
  against a published baseline.

## Real-data results

On **2026-07-19** the full pipeline ran end-to-end on a genuine measured dataset — the
Empa bipolar-HiPIMS deposition-rate campaigns (a real magnetron-sputter tool, n = 3,150,
CC-BY-4.0), ingested through the generic tabular adapter:

- **Coverage gate (directional, per implementation-plan §15.3):** under a static
  conformal calibrator, **5 of 6 campaigns pass** on both a temporal (run-order) and a
  random split; one campaign fails in the under-coverage direction that drift theory
  predicts for the Bayesian-optimization sampling. Wiring in the **online ACI drift
  calibrator** (library-default settings, no per-campaign tuning) **repairs that
  campaign — all 6 pass** — and the repair was independently verified *not* to be an
  artifact of infinite-width intervals.
- **Pessimistic inverse:** refused an over-tight spec band and an out-of-reach one, each
  with a quantified diagnosis, and returned **three feasible recipes whose nearest
  actually-measured runs all landed inside the target band.**
- **Support/OOD screen:** 8/12 cross-campaign checks pass; the 4 failures are exactly the
  cross-*material* pairs — a documented limitation (the model is blind to a shift in a
  variable that is not a knob).

Full write-up with caveats: [`examples/real_data/empa_hipims/RESULTS.md`](examples/real_data/empa_hipims/RESULTS.md).

**This is the M1 gate *form* on real data, not the signed M1 program gate** — that
requires the project's own target-process data (still the #1 open risk). The powered M1
criterion remains the in-silico version.

### Comparison to prior work

A pre-registered, steelmanned bake-off against **Model Feedback Learning** (Gu et al.,
*Few-Shot Test-Time Optimization…*, arXiv:2505.16060) supports RIG's formulation being
**better-posed on the simulator tested**: RIG's certified recipes had a 0.00 certified-miss
rate vs 0.50, and positive vs negative safety margins. RIG's honest cost is on the record
too — it *abstains* on hard-but-feasible boundary targets where the baseline confidently
answers (and misses). **Both systems are evaluated in-silico**; neither has a real-tool
result yet. Details: [`docs/prereg-mfl-bakeoff-2026-07-17.md`](docs/prereg-mfl-bakeoff-2026-07-17.md).

## Status & roadmap

RIG follows a gated build order (implementation-plan §15.3). Current state:

| Gate | Status |
|---|---|
| **M0** — secure a real recipe→outcome dataset | **Venue chosen (Empa HiPIMS), ingested.** A full-bar dataset (real target process, temporal + leave-one-tool-out split, clean license) still requires a fab/vendor agreement or an own campaign. |
| **M1** — calibrated forward on a real split | Directional real-split check run: 6/6 campaigns pass under the ACI drift path. Powered in-silico criterion + conformal-PID endpoint owed. |
| **M2** — inverse beats warm-started Bayesian optimization (in-silico) | Passing in-silico (~2× cheaper cost-to-target); BoTorch/SCBO comparator slate owed. |
| **M3** — amortized posterior + active-learning loop | Generator + SBC/TARP calibration gate built. |
| **M4 / M5** — prospective real campaign / certification | Blocked on M0 full-bar data. |

Known gaps kept honest (not hidden): the `QualificationGate` has no concrete
implementation yet; the D7 independent verifier is unbuilt; a documented false-success at
20 knobs traces to the feasibility test reading raw model σ rather than the conformal band.
See [`docs/BUILD_STATE.md`](docs/BUILD_STATE.md) for the live, authoritative status and
[`docs/BUILD_LOG.md`](docs/BUILD_LOG.md) for the session-by-session journal.

## Repository layout

```
src/rig/              process-agnostic core
src/rig_adapters/     per-process adapters (tabular, mbe)
tests/                542 tests (pytest + Hypothesis; seeded/deterministic)
examples/             runnable end-to-end examples
docs/                 the authoritative spec (implementation-plan.md), audits,
                      results, and the illustrated explainer
implementation-plan.md   the binding design spec (decisions D1–D9, invariants)
```

## Development

- `python -m pytest tests/ -q` — the full suite (deterministic; seeded everything).
- `python -m ruff check .` and `python -m ruff format --check .` — lint + format of record.
- `lint-imports` — enforces the `rig ⊄ rig_adapters` boundary (import-linter).

All three are green in CI. Contributions should keep the core torch-free and the
adapter-import boundary intact.

## Provenance

The authoritative design is [`implementation-plan.md`](implementation-plan.md) (decisions
D1–D9 and the §2.1 invariants are binding). This is exploratory research software; treat
every quantitative claim as scoped to the specific in-silico or public-dataset experiment
that produced it, with the caveats stated alongside it.
