# A Calibrated, Uncertainty-Aware System for Learned Process Simulation and Inverse Recipe Generation in Semiconductor Manufacturing

> **Working title for the program.** Internally we call the system **RIG** (Recipe Inverse Generator). This document is the build spec and the paper plan. Read it top to bottom once before writing code; the early sections fix vocabulary, invariants, and the contradiction-resolutions that the later sections assume.

---

## 0. North Star

Given *any* semiconductor unit process exposed through a thin per-process adapter, **learn a calibrated, uncertainty-aware, drift-tracking forward simulator purely from `recipe → outcome` data** (the neural network *is* the simulator), then **propose a diverse set of manufacturable recipes that hit a possibly multi-objective, ranged target** — with conformal confidence that stays valid under drift, refusing infeasible targets instead of clipping, never exploiting surrogate error, reaching spec in the **fewest real machine runs (cost-to-target)**, independently qualified before deployment, and proven on **real machine data** across drift / OOD / leave-one-tool-out splits. MFL ("Model Feedback Learning" — Gu et al., arXiv:2505.16060) showed a neural network can invert another neural network in simulation; we show a *system* can invert a *machine* under uncertainty, on one shared core, for processes from MBE through lithography.

**Who reads this.** You are a strong ML engineer starting the build. You have: (i) deep familiarity with semiconductor recipe generation across MBE, CVD/PECVD/LPCVD/MOCVD/ALD, PVD, plasma/RIE/ICP etch and ALE, and lithography; (ii) an existing physics-based **MBE process simulator** (Python) that emits labelled `recipe → wafer-scale outcome` data (thickness, uniformity, bow, stress, composition/wavelength) and finite-difference sensitivities; (iii) a "cost-is-not-an-issue" compute posture — prefer the most rigorous SOTA option, add ensembles, active learning, and extra validation freely. Treat the MBE simulator strictly as a **bootstrap for one process instance**: a data generator, an OOD-safe physics prior, and an in-silico stand-in "machine" for validation *before* real data. **Most target processes ship no simulator; design the data-only case as the default and treat a physics model as an optional plug-in.** Final claims rest on real data.

**Scope is the central requirement.** The system is **not MBE-specific and not etch-specific**. The same core machinery — data schema, forward surrogate, inverse engine, robust optimizer, active-learning loop, evaluation — serves any process for which `recipe → outcome` pairs exist. Heterogeneity in input dimensionality, output **modality** (scalar KPIs / 1-D profiles & curves / 2-D fields & images), noise, cost-per-run, and physics is absorbed by a **pluggable per-process adapter**, never by forking the core. MBE is *one* plug-in instance among many.

---

## 1. How This Improves on MFL

MFL is the closest prior art and the baseline we reimplement and beat. It has two good ideas we keep — **amortized inverse via a round-trip/cycle-consistency loss**, and a **cheap-surrogate-then-real two-loop structure** — and a set of first-order defects we replace.

| MFL component | Our replacement | Why it is better |
|---|---|---|
| Deterministic MLP emulator `E` (point predictor) | Data-regime-conditional **probabilistic ForwardModel**: GP/DKL below ~300 real runs, deep-ensemble + spectral-normalized SNGP above; heteroscedastic β-NLL heads; shift-robust conformal wrapper; online drift update | Emits calibrated epistemic + aleatoric uncertainty; **run-indexed conformal gives marginal, drift-adaptive coverage** (OOD is handled by the trust-region/typicality gate of §8.2 — conformal alone gives *no* conditional-on-`x` OOD guarantee); together these block surrogate reward-hacking |
| Single-point reverse map `R` (regresses the mean of a non-injective pre-image) | **Amortized conditional posterior** (NPE normalizing-spline flow; **FMPE** for high-dim scalability, **Simformer** for arbitrary partial/masked specs) returning a *diverse recipe set* + per-candidate confidence, with **explicit infeasibility declaration** | Handles the many-to-one map and infeasible targets instead of averaging valid recipes into invalid regions |
| Round-trip loss `‖Z' − E(R(Z'))‖²` against a deterministic `E` | Round-trip through the *probabilistic* surrogate + **risk-averse pessimistic objective**: uncertainty penalty + data-manifold term + robust min-max + closed-loop oracle refit | Optimizes the *physical* manifold, not the learned one (the headline failure mode); autofocuses the surrogate where proposals actually land |
| Clipping as constraints | **Constraint-by-construction** (simplex/box reparameterization, physics-parameterized outputs) on the Banad & Sharif coupling ladder; projection/DC3 only where structure cannot be parameterized away | Respects mixture/coupling/hard-to-change structure; no boundary-piling; every proposal feasible by construction |
| Loop B via black-box machine Jacobian `dM/dx` (finite differences, noise-dominated) | **Gradient-free, cost-aware batch Bayesian active learning** (BoTorch qLogEI / qLogNEHVI inside a TuRBO trust region), warm-started, closing each batch back into the surrogate | The real machine is never differentiated; noise-robust; minimizes cost-to-target |
| "Conservative learning" = Jacobian-gated learning rate | **Uncertainty-gated trust region + in-distribution penalty**; the first-order Taylor of the robust min-max *recovers* the Jacobian penalty as a special case | Conservative about where the model is *wrong*, not merely where it is steep |
| In-sim validation (an NN inverting an NN on Gaussian synthetic data); BO run without its priors; no calibration, no OOD/temporal split, thin statistics | **Real-data protocol**: temporal + leave-one-tool-out splits; **cost-to-target** as primary metric (survival analysis); warm-started BO/DoE + offline-MBO + re-implemented-MFL baselines; physics-fidelity benchmark; coverage/CRPS/PIT; ≥5–10 seeds + CIs; pre-registered prospective hardware campaign | Claims survive the emulator-vs-reality gap, drift, and top-venue review |
| No deployment story | **Independent QualificationGate** (Banad & Sharif rung 5 / AEC-Q100-style), physical wafer as final arbiter | The surrogate cannot self-certify; provides the trust artifact a fab requires |
| MBE/etch-specific framing | **`ProcessAdapter` + modality-tagged cores + optional physics plug-in + multi-process transfer** | One core across MBE → CVD → PVD → etch/ALE → litho and scalar / 1-D / 2-D modalities; data-only by default |

---

## 2. Design Principles, Invariants & the Decisions That Resolve Cross-Section Conflicts

### 2.1 Non-negotiable invariants

Any component that violates one of these is rejected regardless of headline accuracy.

1. **Uncertainty-first.** The forward model returns a *predictive distribution*, never a point, decomposed into **epistemic** (reducible model ignorance; must inflate off-manifold) and **aleatoric** (irreducible run-to-run / metrology noise; heteroscedastic).
2. **Robustness to surrogate error (the headline invariant).** The inverse objective must *penalize*, never reward, regions where the surrogate is unconstrained. Enforce defense-in-depth: (a) epistemic-uncertainty penalty on the objective; (b) a trust region in input space; (c) a data-manifold constraint; (d) closed-loop oracle refitting so the surrogate is corrected exactly where proposals land. Without (d) the loop reward-hacks by construction.
3. **Respect the data manifold.** Proposed recipes must lie in the support of realizable inputs.
4. **Calibrated, and calibrated under shift.** Point UQ is insufficient; wrap the model in conformal prediction and hold coverage under drift and per-tool.
5. **One-to-many aware.** The inverse returns a diverse candidate set with per-candidate confidence and *declares infeasibility* for unreachable targets.
6. **Physics-grounded where possible, data-only by default.** A physics model is an optional per-process plug-in — augmenter, gray-box residual backbone, or independent verifier — never a requirement.
7. **Rigorous real-data evaluation.** Final claims rest on real machine data with temporal and leave-one-tool-out splits, and **cost-to-target** as the primary success measure — not surrogate RMSE.
8. **Certified deployment path.** Before any recipe reaches production it passes an *independent* qualification gate external to the training/inversion loop; the physical wafer is the final arbiter.
9. **Reproducible.** Fixed seeds, versioned data/splits, logged configs, deterministic calibration sets, released ablations; every headline number carries seeds and CIs.

### 2.2 Canonical decisions (these resolve the contradictions across the source sections — treat them as binding)

The methodology sections agree on philosophy and disagreed only on *which mechanism instantiates it*. Those are settled here, once, and referenced everywhere. **§20 folds in a dedicated 2024–2026 SOTA research pass; where it refines a decision below (e.g. the backbone-crossover threshold in D3, or DPS→twisted-SMC for calibrated diffusion posteriors), §20 wins on method currency while the design decisions D1–D9 and the invariants stand.**

- **D1 — Real-data asset is Milestone 0, gated before build (see §15).** The single claim that separates us from MFL is "we invert a *machine*." Public real recipe→outcome datasets barely exist and fab recipe data is trade-secret, so securing a real dataset is the #1 program risk and has an owner. Absent it, the paper de-scopes to explicitly-labelled in-silico.
- **D2 — Inverse engine = amortized proposal + single per-query pessimistic refinement.** Primary generator: **NPE with a conditional normalizing-spline flow** (`sbi`); switch to **FMPE** for high-dimensional recipe spaces (its edge is estimator *scalability*, not conditioning) and to **Simformer** for arbitrary *partial/masked* conditioning (dropping a subset of KPIs at test time without retraining); ranged *box* targets are served by the region-augmentation training trick with either. This amortized posterior is the **offline "instant-answer" service**. Every emitted candidate is then polished by **one** per-query risk-averse optimizer — the pessimistic min-max of §8 solved by multi-start — not by three different refiners. **SVGD is a fallback refiner** (it underestimates posterior variance); **cINN/cVAE are within-family generator fallbacks** under data scarcity or flow instability; **direct gradient inversion (MFL's `R`)** is kept only as a baseline. **Calibration attaches to the amortized proposal and to the conformally re-validated selected set — never to the refined output**, which optimizes a deliberately different risk-reweighted objective.
- **D3 — Forward/UQ backbone is data-regime-conditional.** Below **~300 real runs/process** (the little-data regime Kanarik's premise implies), the primary is a **GP / Deep Kernel Learning / Conditional Neural Process**; above ~300 runs, or for 1-D/2-D field outputs, the primary is a **deep ensemble (K=5 dev, K=10 final) of heteroscedastic β-NLL nets with a spectral-normalized SNGP last layer**. Both expose the same canonical `PredictiveDistribution(mean, aleatoric_σ, epistemic_σ, conformal_set)` provider (§3.2) so downstream code is backend-agnostic. **Do not bootstrap ensemble members** — init/seed diversity only (bootstrap usually hurts on small data).
- **D4 — Small-n conformal recipe.** You cannot hold out a 1000-point real-machine calibration set from a ~100-run budget. Therefore: (i) **pre-set** the conformal machinery in the in-silico phase; (ii) make **online Adaptive Conformal Inference (ACI) / conformal-PID keyed on run index the *primary* real-data coverage path**, updating run-by-run rather than needing a static held-out block; (iii) use **cross-conformal / jackknife+** (finite-sample valid, data-efficient) instead of split conformal when n is tiny; (iv) per-tool **Mondrian** conformal only when a tool has ≥ a few dozen runs — otherwise report marginal + ACI and say so.
- **D5 — Loop-B acquisition is a single cost-cooled blend with a three-stage emphasis** (exact schedule in §9.4): **early** batches lean on **BALD + space-filling** (global epistemic reduction — EPIG *cold-starts* because the inverse engine `R` is unreliable until the surrogate stabilizes); **mid** loop, once `R`'s proposals stop moving between batches, **EPIG** becomes the goal-oriented anti-exploitation core (it up-weights `R`'s high-epistemic optima, spending real runs exactly where `R` would cheat); **late** batches shift to **cost-aware qLogNEHVI** toward the spec box. This annealed emphasis *is* Kanarik's Human-First/Computer-Last V-curve and is the one canonical Loop-B rule. (Note: an earlier draft mislabeled EPIG as the *early* driver; it is BALD-early → EPIG-mid → qLogNEHVI-late.)
- **D6 — Online cadence.** During Loop B, use the **per-query pessimistic optimizer** against the freshly-updated surrogate (cheap to re-solve, no generator retraining). **Re-distill the amortized posterior offline** on a schedule (nightly / every *k* batches), not every batch. This is why amortization survives the online loop.
- **D7 — MBE independent-verifier circularity.** The MBE simulator cannot be *both* the gray-box physics prior *and* the independent verifier for the same claim. When only one solver exists, the independent verifier is **the real tool only** (no in-silico fidelity claim for that process), or you build a deliberately **different-physics / reduced-order** second model (e.g. an analytic ROM vs the kMC sim) as the verifier.
- **D8 — Process-agnostic *interface* claim now; empirical *transfer* claim later.** Paper 1 demonstrates the same core running on ≥2 processes via adapters (interface generality). The foundation-trunk claim (multi-process pretraining lowers cost-to-target on a held-out process) is deferred to a later paper with more data. Keep the negative-transfer guards but mark them paper-2.
- **D9 — Data-only globally; gray-box is the default *only when* a physics model exists.** No conflict: the core API never imports a simulator; when an adapter supplies one, the default becomes the physics-anchored gray-box for that process.

### 2.3 Unified notation and canonical choices (use these symbols and defaults everywhere)

| Symbol / knob | Canonical choice | Notes |
|---|---|---|
| `M` | the real machine (black box; never differentiated) | |
| `f_θ`, `E` | learned probabilistic forward surrogate | returns `p(Y \| X)` |
| `R`, `q_φ` | amortized inverse posterior | NPE flow / FMPE / Simformer |
| `Z*` | target spec = box / set of ranges / partial spec | not a point |
| `κ` | **credited-band (aleatoric) multiplier** `q̂(x)=κ·σ` — *not* a separate epistemic penalty (epistemic enters via the worst-of-`K`, §8.1) | default `κ=2.0` static, or annealed 3→1 when adapting; = the conformal band width |
| `λ_m` | data-manifold / in-distribution penalty weight | set by self-adaptive loss-balancing |
| Ensemble size `K` (the machine is `M`) | **5 (dev), 10 (final claims)** | init-diversity only, no bagging |
| Aleatoric loss | **β-NLL (β=0.5)**; **CRPS** for skewed/heavy-tailed KPIs | β-NLL fixes variance-collapse of plain Gaussian NLL |
| Acquisition | **qLogEI** (single-obj), **qLogNEHVI** (noisy multi-obj) | Ament 2023 log-forms supersede qEI/qNEHVI; qNParEGO if >4 objectives |
| Trust region | **TuRBO** anchored on real data | |
| OOD / manifold score | **composite requiring BOTH normalizing-flow *typicality* AND ensemble/GP epistemic disagreement**; Mahalanobis in a spectral-normalized latent as the cheap fallback | raw flow density alone is unsafe (models over-assign likelihood to OOD) |
| Conformal | **CQR** + **online ACI / conformal-PID** (primary real-data); cross-conformal/jackknife+ for tiny n; Mondrian per-tool when data allows | |
| Batch size `q` | **4** (one lot; adapter-overridable) | Kanarik cadence |
| Backbone threshold | **~300 real runs/process** | GP/DKL below, ensemble+SNGP above |

### 2.4 Ownership map (each cross-cutting concept is defined once; everywhere else cross-references)

To keep the document readable and non-repetitive, each recurring concept has exactly one owning section:

- **Constraint-by-construction:** box/simplex/coupling → owned by **§8.3**; physics-guaranteed *monotonicity* → owned by **§6.3**.
- **Physics-fidelity benchmark:** definition → **§14.4**; evaluation/suite → **§12.7**. **Certification/qualification deployment** → **§11.4** + **§14.9**.
- **Drift-aware / shift-robust conformal** → owned by **§5** (forward surrogate calibration).
- **Cost-to-target definition and its statistics** → owned by **§11** (definition/cost model) and **§12** (survival-analysis statistics).
- **MFL teardown table** → owned by **§1**.
- **Uncertainty decomposition + backbone choice** → owned by **§5**.
- **Composite OOD / trust-region score** (typicality AND disagreement) → owned by **§8.2** (cross-referenced by §6.4, §10.5, §13.2, §14.7).

---

## 3. Process-Agnostic Interfaces & Data Contract

Everything downstream talks through four interfaces. The hard boundary is between **process-agnostic machinery** (models, inverse, UQ, active learning, training) and **per-process knowledge** (schemas, constraints, physics plug-ins), connected only through typed interfaces and a registry. The training loop, UQ, inverse solver, and evaluation import only `interfaces` and `registry`; they never `import mbe`. Enforce with an **import-linter** CI contract. **(This holds transitively only if `registry` itself never statically imports adapters — so adapters *self-register* via packaging entry points, discovered at runtime via `importlib.metadata`, not by a static import list in `registry`.)**

### 3.1 `ProcessAdapter` — declares the process, owns all process-specific knowledge

- **Input schema**, typed variables: continuous (with bounds), categorical, **mixture/compositional** (simplex, sum-to-1 — *true compositional fractions*: alloy mole fractions, or a blend defined as fractions; **NB independent MFC gas setpoints in sccm are NOT a simplex** — they are independent box-bounded flows whose total is not fixed, so the adapter declares which factors are genuinely compositional vs. independent flows), and a **change-cost class** per variable (hard-to-change tool/chamber vs easy temperature/dose — split-plot structure). Constraints (box, linear/mixture, monotone, nonlinear coupling) declared here for constraint-by-construction.
- **Output schema** with an explicit **modality tag** (`scalar_vector`, `curve_1d`, `field_2d`) plus per-output tolerance/spec semantics.
- **Cost model**: `$/run`, `$/batch`, batch size, change-over penalties. Distinguish **fixed per-batch cost** `c_batch` from **variable per-recipe cost** `c_recipe(x)` — this split drives acquisition (§8) and the stop/continue rule (§11).
- **DoE / warm-start hooks**: expert-constrained ranges and a space-filling seed design (scrambled **Sobol'** primary; maximin/Morris–Mitchell LHS fallback for d≤8) so active learning strictly *generalizes* RSM/CCD/BBD/Taguchi rather than ignoring them.
- **Optional physics plug-in** (`f_physics(x)`, optional `∂f/∂x`) and an **independent verifier** distinct from the physics prior (D7). Absent by default.
- **Encoders**: modality-appropriate input embedding and output-head selection so downstream cores are modality-agnostic.

### 3.2 `ForwardModel` — learned probabilistic simulator (§5)

`predict(x) → PredictiveDistribution(mean, aleatoric_σ, epistemic_σ, conformal_set)` for any modality **(this exact name, field set, and order are canonical — use verbatim everywhere; do not write `OutcomeDist`, `predict→scalar`, or a `_var` tuple)**; `support_score(x)` (in-distribution density / distance-to-support); `jacobian(x)` (for sensitivity reporting). Backend A: GP/DKL/CNP; Backend B: ensemble+SNGP (D3). Wrapped by the shift-robust conformal calibrator (§5.6). Ingests each Loop-B batch to fine-tune and recalibrate (invariant 2d).

### 3.3 `InverseSolver` — target → set of recipes with calibrated confidence (§8 objective/solver; §14.3 amortized posterior)

`solve(spec) → List[(recipe, confidence, predicted_outcome_interval, feasibility_flag, support_score)]`, where `spec` is a multi-objective / set-of-ranges target plus optional constraints and a cost budget. Returns a *set*, or an explicit **INFEASIBLE** verdict with the nearest achievable Pareto point and its distance-to-feasible.

### 3.4 `QualificationGate` — independent deployment certification (§11, §14)

`certify(recipe) → {pass/fail, evidence}` using a verifier *outside* the training/inversion loop (independent physics solver or a fixed confirmation batch on the real tool). No recipe reaches production without a logged qualification record.

### 3.5 The data record

Every row is a `RunRecord`, validated on read *and* write (Pydantic v2 + Pandera + Pint units). This is where we beat MFL's "Gaussian arrays with no metadata."

```python
class RunRecord(BaseModel):
    run_id: UUID
    process_id: str
    tool_id: str                 # chamber/tool identity → leave-tool-out splits & drift
    timestamp: datetime          # → temporal splits & drift monitoring
    recipe: RecipeRecord         # values: dict[str, pint.Quantity | CategoricalValue | Fraction]
                                 #   numeric→validated vs adapter ranges; categorical→validated vs enumerated levels;
                                 #   Fraction→simplex member (NOT a bare Quantity — see §3.1 categoricals/mixtures)
    outcomes: list[OutcomeRecord]# modality-tagged; profiles/fields are DVC-tracked array refs
    provenance: Provenance       # source ∈ {physics_sim, real_tool}, operator, calib state, data hash, git sha
```

`Provenance.source` is load-bearing: **all headline metrics are computed on `source == real_tool`**; the physics sim is only bootstrap/prior. Units are canonicalized to SI at ingest (kills the sccm-vs-slm / °C-vs-K silent bug). Mixture and hard-to-change tags are first-class so the inverse treats seasoning-like state as *observed-but-not-actionable* and enforces split-plot structure.

---

## 4. Framing: This Is Offline Model-Based Optimization

State this explicitly in the paper. Generating a recipe by optimizing against a learned forward model *is* offline model-based optimization (MBO), and **surrogate exploitation** — the optimizer driving inputs into regions where the surrogate is optimistically wrong — is the canonical, documented failure of that literature (Conservative Objective Models, Trabucco et al. 2021; Design-Bench, Trabucco et al. 2022; autofocused oracles, Fannjiang & Listgarten 2020; CbAS, Brookes et al. 2019). It is identical to Banad & Sharif's (2026) "the inverse optimizes the *learned* manifold, not the *physical* one." MFL's weak defenses (input clipping — as used in its LSRS-LR baseline; its reverse-model constraint mechanism is not fully specified in the paper — plus the crude Jacobian-gated "conservative learning") are exactly what COMs shows is insufficient. Our two headline defenses follow the offline-MBO playbook: **(a) pessimism under uncertainty** (the objective is penalized by epistemic uncertainty — note the piece that specifically **subsumes MFL's Jacobian-gated "conservative learning" is the robust min-max over the input-tolerance box**, whose first-order Taylor *is* a Jacobian-magnitude penalty, §8.5; the epistemic penalty is a *distinct, complementary* axis — input-steepness ≠ model-ignorance) and **(b) a trust-region gate** that refuses candidates outside the data-supported region, closed with **oracle refitting** every batch. Correspondingly the primary metric is **cost-to-target**, not held-out surrogate R².

**Where the gradients come from (this dissolves MFL weakness 4).** We never differentiate the real machine. All inverse gradients flow through the differentiable *surrogate* (and, when available, a differentiable physics simulator in the gradient path). The real machine `M` is queried only for labels that update the surrogate — zeroth-order w.r.t. `M`, first-order w.r.t. the surrogate. This is the standard offline-MBO / BO posture and removes the black-box-Jacobian requirement entirely.

**Warm-start, never pure-from-scratch (Human-First / Computer-Last).** Kanarik et al. (2023) show from-scratch BO beats a human expert in <5% of trajectories, while an expert doing rough tuning *and constraining the search range* halves cost-to-target (a V-shaped optimal hand-off). We initialize the surrogate from physics-simulator augmentation and expert/physics priors, confine the inverse to expert- or physics-constrained ranges by default, and **never claim unconstrained search suffices**.

---

## 5. The Probabilistic Forward Surrogate (the learned simulator)

**Owner of: uncertainty decomposition, backbone choice, calibration, drift-aware conformal.**

### 5.1 Role and the non-negotiable requirement

The surrogate `f_θ: X → p(Y|X)` *is* the simulator; the inverse engine, active learning, and cost accounting all consume its predictive *distribution*. MFL's `E` is a deterministic ~64-unit MLP, so its inverse loop is free to drive `R` into regions where `E` is confidently wrong. A surrogate that cannot say "I don't know here" cannot defend against that. So for every output the surrogate emits **aleatoric** (input-dependent process/metrology noise) and **epistemic** (model uncertainty that *grows off the training manifold*) separately, plus a **trust region and reachable set** the inverse consumes (a distance-aware epistemic gate + a conformal band per output). A target `Z*` is declared *feasible-and-trustworthy* only if some `x` in the supported region has its conformal band inside `Z*`. **A method that cannot inflate epistemic uncertainty OOD is disqualified** — this rules MC-dropout out of the primary role.

### 5.2 Data-regime-conditional backbone (decision D3)

**Below ~300 real runs/process (the default little-data regime):**
- **Primary: GP with Matérn-5/2 + ARD** (single/low output count) → principled epistemic UQ and native acquisition; recency weighting is trivial via a kernel time-decay. **Deep Kernel Learning** (Wilson et al. 2016) when learned features help — but guard against DKL feature-collapse (Ober et al. 2021) with spectral normalization and by monitoring latent-vs-input distance correlation. **Conditional/Attentive Neural Processes** (Garnelo et al. 2018; Kim et al. 2019) for amortized few-shot adaptation across many small related datasets (multi-tool bring-up).
- **Gray-box option** (when a physics model exists, §6): use physics as GP mean function / residual prior.

**Above ~300 runs, or for 1-D/2-D field outputs:**
- **Primary: deep ensemble (K=5 dev, K=10 final) of heteroscedastic ResMLP surrogates with a spectral-normalized SNGP last layer.** Independent inits + input-domain randomization; **no bagging**. Predictive mixture `p(y|x)=(1/K)Σ N(μ_m, σ_m²)`; total variance = `E[σ_m²]` (aleatoric) + `Var[μ_m]` (epistemic).

**Honest qualification.** Plain deep ensembles are *not* automatically well-behaved far OOD — members can agree confidently in never-seen regions. The distance-awareness that *promotes* OOD inflation — approximately, via the bi-Lipschitz argument, **not a hard guarantee** — comes from the **spectral-normalized trunk + SNGP/RFF-GP last layer** (Liu et al. 2020), not from ensembling per se. We combine them: the ensemble buys robustness and mixture-level aleatoric averaging; the spectral/SNGP component buys the (approximate) OOD distance-awareness. SNGP is also the **single-model deploy path** for the inverse inner loop (§5.7). Compute fallbacks for large field surrogates where 10 copies are too heavy: **last-layer Laplace** (Daxberger et al. 2021) or SNGP as single-pass epistemic. **Rejected for primary:** MC-dropout (epistemic is a fixed function of dropout rate, does not inflate OOD) and evidential deep regression (Amini et al. 2020 — its epistemic term is not a proper posterior and is miscalibrated OOD, exactly the regime we care about; Meinert et al. 2023, Bengs et al. 2022). Evidential is acceptable only as a cheap screening surrogate in a tight inner loop, never as the arbiter of feasibility.

### 5.3 Shared trunk (process-agnostic)

Pre-LayerNorm **ResMLP** (residual + pre-LN for stable deep gradients on the inverse backward pass): `d=256`, `L=4`, GELU, dropout 0.1 for regularization only. **Spectral normalization on residual branches** is load-bearing (it forces latent distances to track input distances). Input adapter:
- Continuous knobs → per-feature z-score using **train-fold statistics only** (fit every transform inside the CV fold — leakage here silently inflates apparent calibration).
- Categorical / `tool_id` / `process ∈ {MBE,PECVD,RIE,litho,…}` → **learned embeddings** (dim ≈ min(50, ⌈card/2⌉)). The process-type embedding is the hook for multi-process sharing (D8, paper-2).
- **Mixture/compositional inputs** → **ILR (isometric-log-ratio) transform** so the simplex constraint holds and the net never sees a spurious extra DOF.
- **Hard-to-change factors** tagged with a whole-plot flag consumed by active learning and grouped splitting, not the trunk.
- **FiLM conditioning** (Perez et al. 2018): the process/tool embedding produces per-block scale/shift so one trunk specializes per process without separate weights.
- Positive outputs → `log1p` (log-normal predictive); bounded outputs → logit; angles → `(sin,cos)`; non-standard marginals → `QuantileTransformer` as fallback (conformal then restores distribution-free guarantees).

### 5.4 Modular output heads — one per (process, modality)

| Modality | Head (default) | Aleatoric likelihood | Milestone |
|---|---|---|---|
| **Scalar KPIs** | 2-layer MLP → `(μ_k, s_k=logσ²)` per output | heteroscedastic Gaussian on transformed target | **Phase 1 (build now)** |
| **1-D profiles / curves** | GRU / 1-D-CNN decoder (implicit field `g(z,t)` optional) | per-location heteroscedastic Gaussian; low-rank+diagonal spatial covariance | **Phase 2** |
| **2-D fields / images** | small U-Net / CNN decoder (implicit field optional; FNO/DeepONet when a PDE prior exists) | per-pixel heteroscedastic Gaussian; optional low-rank spatial-GP layer | **Phase 2** |

The schema carries all three modalities from day one, but **simpler GRU/CNN/U-Net decoders are the defaults** and implicit neural fields are the optional variant — building implicit fields for a scalar-KPI first deliverable is over-engineering. A new modality is "a new head on the shared trunk," not a core change.

### 5.5 Aleatoric heads: the loss matters more than the architecture

Naïve Gaussian NLL lets the model explain away hard regions by inflating σ, starving the mean of gradient (variance collapse / mean underfit; Seitzer et al. 2022). **Default: β-NLL, β=0.5** — reweight the NLL gradient by `stopgrad(σ^{2β})`, recovering most of MSE's mean-fit quality while keeping calibrated variance. **For skewed/bounded/heavy-tailed KPIs, prefer CRPS** (Gneiting & Raftery 2007) — a proper scoring rule robust to non-Gaussianity, with a closed form for Gaussian predictives (Gneiting et al. 2005), so it is a drop-in differentiable loss. Choose CRPS when residual QQ-plots show non-Gaussian tails; otherwise β-NLL.

### 5.6 Calibration — the guarantee layer (owns drift-aware conformal; implements D4)

Trained UQ is approximately calibrated at best. Layer post-hoc:

1. **Temperature / variance scaling** (scalar `s` minimizing NLL of `N(μ, s·σ²)`) and optional **isotonic CDF recalibration** (Kuleshov et al. 2018) for non-affine miscalibration — these need a held-out block, so **they are fit in the in-silico Phase-0 (abundant data) and *pre-set*, not re-fit on the ~100 real runs** (D4). **On the real stream the operative layer is the online/data-efficient conformal of item 2** (ACI + cross-conformal/jackknife+), which needs no held-out block.
2. **Split-conformal / CQR** (Romano et al. 2019) as the load-bearing distribution-free finite-sample layer — the honest replacement for MFL's clipping.
   - **Multi-output KPIs:** per-output conformal plus a joint region via **copula-based conformal** (Messoudi et al. 2021) or **calibrated multi-output quantile regression** (Feldman et al. 2023); Bonferroni (α/J per dim) / Mahalanobis only as coarse fallbacks. Fields/curves: **functional conformal bands** (Diquigiovanni et al. 2021).
   - **Mondrian (group-conditional) conformal** by tool/chamber/process-type so coverage holds *per process* — but only when a group has ≥ a few dozen runs (D4).
   - **Exchangeability-safe grouped splitting.** Nested lot/wafer/die structure means residuals within a lot are correlated and are *not* exchangeable under a random row split, even absent drift. Calibrate with **leave-lot-out / grouped** splits.
   - **The small-n + drift reality (D4).** Split/CQR conformal assumes exchangeability, which our own temporal and leave-one-tool-out splits deliberately break, and a ~100-run budget cannot supply a 1000-point held-out block. Therefore make **online ACI (Gibbs & Candès 2021) / conformal-PID (Angelopoulos, Candès & Tibshirani 2023) keyed on run index the primary real-data coverage path**, pre-set in-silico and updated run-by-run; use **cross-conformal / jackknife+** (Barber et al. 2021) for tiny n; pair with **weighted conformal** (Tibshirani et al. 2019) under covariate shift. **Rolling coverage below nominal is our concrete DRIFT detector** (Banad & Sharif failure mode 2), tying to run-indexed time-varying modeling (Cho, Shao & Mesbah 2024).

### 5.7 Training protocol and defaults

- **Splits:** K=5 CV **grouped by lot** for model selection, plus **explicit leave-one-tool-out and leave-latest-runs-out** splits reported separately with calibration *per split*.
- **Optimizer:** AdamW, lr 1e-3 cosine-decay, weight decay 1e-4, batch 128 (scalar) / 16–32 (fields), early stop on validation β-NLL (patience 30).
- **Loss weighting:** multi-output → uncertainty-weighting across heads (Kendall et al. 2018); physics-hybrid → inverse-Dirichlet (Maddu et al. 2022) or self-adaptive weights (McClenny & Braga-Neto 2023).
- **Inverse-loop cost:** the inverse inner loop runs thousands of surrogate passes, so K=10 heteroscedastic forwards/step is a real cost. Resolve it *not* by shrinking UQ but by **ensemble distribution distillation** (Malinin et al. 2020) into a single distributional net, or by using the **SNGP single member** inside the loop, then **re-validating final candidates against the full ensemble + conformal**. (This removes the naïve "cost is not an issue" contradiction: *training* stays full-ensemble; the *inner loop* is distilled.)

### 5.8 Metrics — report all, per split, per output dimension

Point accuracy (RMSE/MAE on physical scale, **as a ratio to the aleatoric noise floor** = the *metrology* floor from Gage R&R **plus** genuine run-to-run *process* variance from nested REML per §10.3 — Gage R&R alone is only the metrology part and understates the floor), probabilistic accuracy (**CRPS** primary, NLL secondary; a single exploding NLL dim is a variance-collapse *diagnostic*, not the ranking metric), **interval score** (Winkler) with **PICP/MPIW** as its diagnostics at 50/80/90/95%, **regression quantile-calibration error + PIT histograms** (not classification ECE), and an **OOD epistemic check** (mean epistemic on leave-one-tool-out must exceed in-distribution). Among calibrated models, prefer smallest MPIW/CRPS.

### 5.9 Pitfalls and detection

Variance collapse (β-NLL fixes it; else mean head under-capacity) · overconfident OOD (leave-one-tool-out epistemic must exceed in-distribution; else verify spectral norm / rely on SNGP) · DKL feature collapse (correlate latent- vs input-distances) · calibration leakage (nominal-vs-empirical coverage on a grouped, temporal held-out split) · drift breaking exchangeability (rolling ACI coverage) · ensemble mode collapse (low `Var[μ_m]` everywhere → add seed/architecture diversity) · inverse-loop cost blowup (distill/SNGP inner loop).

---

## 6. Physics-Informed / Gray-Box Hybrid (optional per-process plug-in)

Default is **data-only** (D9). When an adapter supplies a physics model, the default *for that process* becomes the physics-anchored gray-box below. The goal is not on-support accuracy alone; it is to make the surrogate **hard to exploit off-support** by anchoring it to a physical prior and, where physics guarantees it, to shape constraints that hold by construction. We operate at the two strongest *forward-model* rungs of Banad & Sharif's coupling ladder — *simulator-in-the-loop* and *constraint-by-construction* — plus the fifth rung, **independent certification** (§14).

### 6.1 Primary architecture: distilled differentiable emulator + Kennedy–O'Hagan residual

**Stage 1 — distill the slow simulator into a fast differentiable emulator `P(x,θ)`.** Train an MLP (or FNO/DeepONet for fields) on abundant simulator draws with **Sobolev training** (Czarnecki et al. 2017): match values *and* Jacobians, `L = ‖P−f‖² + λ_J‖∂P/∂x − J_sim‖²`, start `λ_J=0.1`, tune on a held-out gradient-cosine metric.
- **Sensitivities are not free.** A full FD Jacobian costs `O(d)` extra sim evals per point. Prefer, in order: (a) exact forward-mode AD / adjoint at `O(1)–O(d)`; (b) **stochastic directional Sobolev** — supervise `⟨∂P/∂x, v⟩` on 1–4 random probe directions per point (unbiased, near-constant cost); (c) full FD only on a subsample of anchors. FD Jacobians are noisy — down-weight `λ_J` where FD conditioning is poor and supervise gradient *direction/sign* (cosine, monotonicity) on ill-conditioned channels.
- **Payoff (the single biggest concrete win over MFL):** MFL discarded FD sensitivities (used only as a scalar LR gate); we turn them into dense supervision, cutting sample complexity and giving a **physically consistent Jacobian for the inverse loop** so the inverse cannot ride spurious gradients a black-box MLP would invent.

**Stage 2 — wrap `P` in a Kennedy–O'Hagan (2001) discrepancy model on real data:** `Ŝ(x) = ρ·P_θ̂(x) + δ_ψ(x) + ε(x)`, with `ρ>0` (constrained so it cannot flip physical ordering), `θ̂` calibrated physics parameters, `δ_ψ` a *constrained* NN discrepancy trained on residuals, and **heteroscedastic** noise `ε(x)~N(0,σ²(x))`. `δ_ψ` is a 5–10-member deep ensemble so the residual carries epistemic UQ. Residual-of-physics is primary (needs only sim *outputs*, so it is black-box-solver compatible; isolates model-form error into an inspectable object; little real data goes far). PINN-style PDE-residual losses are the fallback, used only when the governing PDE is in the gradient path and the output is a field.

### 6.2 Calibration and principled domain randomization (sim-to-real done right)

Randomize over **physics parameters** `θ` (sticking coefficients, activation energies, effective fluxes) with physically-motivated priors — *not* MFL's naïve input jitter. Make the emulator `P(x,θ)` take `θ` as input, then **calibrate `p(θ|real)` by simulation-based inference** — Neural Posterior Estimation (`sbi`; Greenberg et al. 2019; Ramos et al. 2019). This yields a *parametric (epistemic)* physics-uncertainty band (not aleatoric). **Calibrate `θ` *jointly* with the shape-constrained discrepancy `δ`, not against the bare simulator alone — the Brynjarsdóttir & O'Hagan (2014) trap:** if `θ` is fit to `P_θ` with `δ` absent (as a naïve SBI-then-residual ordering does), `θ` silently absorbs the sim-vs-real gap and its posterior is biased and overconfident. Constrain `δ` (shrink-to-zero prior + the §6.3 monotonicity/Lipschitz caps) so `(θ, δ)` are jointly identifiable, run the SBI over the *discrepancy-augmented* generative model, and **state the θ/δ confounding explicitly** as a limitation. **Mandatory calibration check:** amortized SBI posteriors are frequently overconfident (Hermans et al. 2022); before any `p(θ|data)` is consumed, verify with **simulation-based calibration** (Talts et al. 2018) and **expected-coverage/TARP** (Lemos et al. 2023). If under-covered, widen via post-hoc conformal or ensemble the SBI posterior; **fallback** to MAP/least-squares `θ` + GP-KOH discrepancy if SBI is unstable.

### 6.3 Hard shape constraints — constraint-by-construction (see §8 for the general treatment)

Where physics *guarantees* a qualitative law, encode it so it cannot be violated: **monotonicity** via Lipschitz Monotonic Networks (Nolte et al. 2023) / Constrained Monotonic NNs (Runje & Shankaranarayana 2023) (Deep Lattice Networks, You et al. 2017, as interpretable fallback); **sign/boundedness/simplex** via output links (softplus/sigmoid/softmax); **Lipschitz control** on `δ` via spectral norm (GroupSort/orthogonal layers, Anil et al. 2019, for a tight certified bound). *Cut as over-engineering:* the ICNN convexity constraint — process KPIs are rarely provably convex; re-add only for a KPI with a derivable convexity law.

### 6.4 Uncertainty-gated blend (OOD-safe fallback)

`Ŝ_final(x) = w(x)·Ŝ_data(x) + (1−w(x))·ρP_θ̂(x)`, `w(x)→1` in-support, `→0` OOD, from the composite OOD score (§8). Off-support the surrogate *returns physics* (plausible, honest wide uncertainty) — this is what stops the inverse from reading an optimistic hallucinated optimum out of a data void. Two corrections a reviewer will demand: (a) the `x`-varying blend can **break the §6.3 monotonicity guarantee on the deployed object** — re-establish/verify the constraint on `Ŝ_final`. **A scalar monotone output link (softplus/sigmoid/isotonic) does NOT fix this**: post-composing a monotone `g` gives `g'·∂Ŝ_final/∂x_j`, the *same sign*, so it cannot restore input-axis monotonicity once the blend's `w'(x)·(A−B)` term has flipped it. The **working** remedies are (i) make `w` depend only on an OOD distance **orthogonal to the constrained axes** (so `w'_j=0` along them), or (ii) pass `(x_mono, blend)` through a **Lipschitz Monotonic Network** constrained monotone in `x_mono`; either way the mandated post-hoc constraint check on `Ŝ_final` is the backstop; (b) the handoff must be **C¹-smooth** (sigmoid of a calibrated OOD distance) so the gradient-based inverse cannot stall in or exploit a kink. Augment sparse real data with `P_θ` samples but **tag them, down-weight them, and time-order validation** so synthetic points cannot mask drift; give the discrepancy a run/time index `δ_ψ(x,t)` so the surrogate itself tracks drift.

### 6.5 Gray-box vs pure-data decision, per output channel

Do **not** use hard statistical test switching — at n≈30–100 real points those tests (nested CV, HSIC, prior-sensitivity refits) have almost no power. Instead, obtain each candidate's leave-one-out predictive log-density **the right way *per model*** — PSIS-LOO (Vehtari et al. 2017) requires pointwise log-likelihoods over a Bayesian posterior with importance weights, which only the GP supplies; a deep ensemble has no such posterior and a physics-only point predictor has no density at all. So: **closed-form LOO for the GP** (or PSIS-LOO where a valid Bayesian posterior exists); **explicit grouped (leave-lot-out) CV predictive log-density for the deep-ensemble and for the physics-only model** (the latter needs an added noise model to *have* a predictive density). Then combine those LOO predictive densities by **Bayesian stacking** (Yao et al. 2018); the stacking weights *are* the soft decision and degrade gracefully at small n. (Remove the DeLong test — it is for ROC-AUC, the wrong tool for regression densities.) **Default under statistical uncertainty: keep physics + hard shape constraints** (safe, exploit-blocking, cost no statistical power). Log `R²_phys`, `ρ̂`, `‖δ‖`, and stacking weights per channel as model-card metrics.

### 6.6 MBE independent-verifier resolution (D7)

The MBE simulator cannot be both the gray-box prior and the independent verifier. For MBE, the independent verifier is **the real tool only** (no in-silico fidelity claim), or you build a deliberately **different-physics reduced-order MBE model** as the verifier. Make this a line item in the physics-bootstrap plan (§15, Phase 0).

### 6.7 Quantifying the extrapolation benefit (report as a headline)

**Extrapolation-gap curve** (held-out error vs distance-to-support for {pure-data, gray-box, physics-only}; the gray-box gap widens with distance), **sample-efficiency / cost-to-target curve** (gray-box vs pure-data vs DoE/RSM and BO-with-priors; multi-fidelity/residual methods typically show ~2–10× fewer runs — Meng & Karniadakis 2020; Perdikaris et al. 2017 — treat as expectations to reproduce, not guarantees), and the **inverse robustness / physics-fidelity benchmark** (§14). Also elevate the **gradient-cosine-vs-sim** check into a headline *interpretability* result: the learned surrogate's Jacobian signs/monotonicities should match known physics.

---

## 7. Reserved

*(Section number intentionally retained so that §8=optimization, §9=active learning read in the requested order; content folded into §5–§6 and §8–§9.)*

---

## 8. Robust, Uncertainty-Aware & Constrained Optimization (the inverse objective)

**Owner of: constraint-by-construction; the pessimistic objective; the composite OOD/trust-region score.**

This section specifies the inverse-search objective and its per-query solver — the single canonical refiner of D2. It is risk-averse by construction: the optimizer must never be rewarded for driving inputs where the surrogate is confidently wrong.

### 8.1 One objective

The inverse problem is not "find `x` minimizing `‖f(x)−z*‖`." It is: **find a set of recipes that provably lie inside the spec box under the worst plausible combination of (model error, process variation), stay on the data manifold, satisfy hard input/coupling constraints — and if none exists, say so.** The canonical per-candidate objective (using the unified symbols of §2.3):

```
maximize_u   J(u) = log P̂_lcb(Y ∈ Z* | x)   +   λ_m · log p̂(x)
             P̂_lcb = pessimistic spec-hit probability: inner worst case taken over
                     k ∈ [K] (ensemble members = the EPISTEMIC term) and δ ∈ Δ (input tolerance box);
                     the credited band inside P̂ uses the conformal width q̂(x) ≡ κ·σ (§8.4) = the ALEATORIC term
subject to   x = g(u)   (constraint-by-construction reparameterization: box/mixture hold for every u)
```

- `P̂(Y∈Z*|x)` is a **differentiable "hits the box" probability estimated by Monte Carlo over *joint* predictive/ensemble samples**, not a product of per-output Gaussian CDFs — process outputs are correlated (etch depth and rate are not independent). The independent-CDF product is only a fast approximation valid under conditional independence. **Rare-event caveat (tight multi-KPI specs — the operationally-critical regime):** under the pessimistic worst-member/worst-δ inner problem, `P̂(Y∈Z*|x)≈0` for most `x` when the box is tight across correlated KPIs, so its gradient *vanishes* over most of the multi-start landscape — exactly where the method must win. For the **gradient**, use a *smooth* surrogate — a **temperature-relaxed soft box indicator (log-sum-exp / sigmoid, τ annealed)** or the **multivariate-Gaussian box probability over the calibrated residual covariance** — and reserve the hard joint-MC estimate for *final ranking* only; use **common random numbers** across restarts for variance reduction.
- **Epistemic pessimism enters *once*, via the inner worst-case `min over k∈[K]`** (worst ensemble member; the across-member spread ≈ `Var_k μ_k`). We deliberately do **not** also subtract a standalone `κ·U_epi` term on top — that would double-count epistemic uncertainty (an earlier draft did, and `U_epi` was never defined; it is removed). `min over δ` (input tolerance) is complementary to the band (output uncertainty at fixed input), not redundant.
- `κ` is the **credited-*band* multiplier** (§2.3/§8.4 — *not* a separate epistemic penalty): a match is credited only inside the conformal band `q̂(x) ≡ κ·σ`, so betting where the surrogate is uncertain is penalized automatically. `λ_m` (the *additive* manifold-penalty weight) is set by **self-adaptive loss-balancing** (McClenny 2023 / Maddu 2022 / Wang 2021). **`κ` is NOT set by loss-balancing** — it sits *inside* `P̂` as a band width, not as an additive loss coefficient, so it is set by the §8.4 protocol (the conformal band `q̂`, or the cost-to-target `κ∈{1,2,3}` sweep, annealed 3→1). Starting values `κ=2.0`, `λ_m≈0.3`; both tuned so *realized* coverage on held-out/confirmation-run data is nominal — not by hand.

This subsumes and upgrades every MFL knob: clipping → constraint-by-construction; Jacobian-gated LR → the robust `min over δ` term whose first-order Taylor *is* a Jacobian penalty (§8.5); point emulator → ensemble pessimism.

### 8.2 The composite OOD / trust-region score (the anti-reward-hacking substrate)

Pessimism alone is leaky: `σ_epi` can be spuriously small in far-OOD holes the ensemble happens to agree on. A recipe is **on-manifold only if it is BOTH (i) *typical* under a normalizing flow of the input marginal `p(x)` AND (ii) low-disagreement under the ensemble/GP epistemic score.** Either signal alone is spoofable:
- Raw flow density is unsafe: deep generative models assign *higher* likelihood to some OOD inputs than to in-distribution data (Nalisnick et al. 2019). Use a **typicality test** — reject when `|log p_flow(x) − E_train[log p_flow]|` exceeds a calibrated band (atypically high *or* low).
- **Cheap fallback:** Mahalanobis distance in a **spectral-normalized latent** (valid only because the trunk is distance-preserving), or the ensemble-disagreement penalty alone. Switch to the full flow when the input space is >~15-D or strongly non-Gaussian.
- Confine the whole search to a **TuRBO trust region** (Eriksson et al. 2019) anchored on real data. `penalty_manifold(x)=max(0, τ−score(x))`; **hard-reject** below `τ` (default = 5th-percentile train score). This imports MOReL's pessimistic MDP (Kidambi et al. 2020) and MOPO's uncertainty penalty (Yu et al. 2020) into recipe design.

### 8.3 Constraints — climb the coupling ladder, don't clip

Default to **constraint-by-construction**; use penalties only for what cannot be parameterized away.
- **Box ranges:** `x = ℓ + (h−ℓ)·sigmoid(u)` — feasible for all `u`.
- **Mixture / compositional** (Σ=1 gas/precursor blends): `x = softmax(u)` or additive-log-ratio / stick-breaking — non-negativity and sum-to-1 exact.
- **Hard-to-change / split-plot factors** (tool, chamber): treated as *conditioning* `c`, not free variables — generate recipes *given* the fixed setting.
- **Coupled linear/convex constraints:** differentiable convex projection — **cvxpylayers** (Agrawal et al. 2019); OptNet (Amos & Kolter 2017) is the QP special case.
- **Hard nonlinear equality/inequality:** **DC3** (Donti et al. 2021), which completes and corrects onto the constraint set, differentiably.
- **Chance constraints** `P(g(x)>0)≤α` on outputs/safety: Gaussian reformulation `μ_g+κ_α·σ_g≤0` using the *calibrated* uncertainty, or the distribution-free **scenario approach** (Calafiore & Campi 2006).
- **Top rung is certification (§14).** Constraint-by-construction and pessimism guarantee feasibility on the *learned* manifold, never the *physical* one; every emitted recipe is a *candidate* for independent qualification, never a substitute for it.

### 8.4 The pessimism weight κ — defaults + protocol (do not leave it free)

- **Default:** `κ=2.0` (≈ one-sided 97.5% under Gaussianity).
- **Rigorous setting:** replace `κ·σ` with a **conformalized (CQR) band `q̂(x)`** calibrated on the real-machine stream (via ACI, §5.6/D4) at joint level `1−α` (`α=0.1`), giving a distribution-free, input-adaptive credited band; widen if empirical joint coverage on the OOD/temporal split undershoots.
- **Cost-to-target tuning:** sweep `κ∈{1,2,3}` in in-silico validation and pick the smallest `κ` whose realized on-machine spec-hit rate ≥ target — cost-to-target, not surrogate accuracy, is the objective.
- **Adaptivity:** anneal `κ` downward as the trust region tightens (early `κ=3`, late fine-tuning `κ=1`) — the principled version of Kanarik's V-shaped hand-off.
- **Failure detector:** track the **realized-vs-predicted mismatch gap** on every machine validation query; if realized ≫ predicted (surrogate optimistic), automatically raise `κ` and shrink `τ` *before* the next batch.

### 8.5 Distributional robustness — the principled replacement for "conservative learning"

Replace MFL's Jacobian-gated LR with an explicit **robust min-max** over the tolerance box `Δ` (Gage R&R / tool repeatability, e.g. ±1–3%/range) and over the ensemble: `min_x max_{k, δ∈Δ} L_k(x+δ)`. Solve the inner max with a few **PGD** steps (5–7, step `Δ/4`) plus an explicit worst-member `max` (group-DRO; Sagawa et al. 2020). **The MFL connection to state in the paper:** first-order Taylor of `max_δ L(x+δ)` over an ℓ∞ box gives `L(x)+‖∇_x L(x)‖₁·Δ` — a Jacobian-magnitude penalty; MFL's Jacobian-gated LR is a crude surrogate for exactly this term, and our robust objective *recovers their heuristic as its linearization* and generalizes it to worst-case and to model uncertainty. Fallback: the analytic first-order sensitivity penalty `γ·‖J_f(x)‖_F` when PGD cost dominates.

### 8.6 The per-query solver loop (concrete)

```
Inputs: ForwardModel (ensemble/GP + aleatoric heads), flow p_flow, spec box Z*, Δ, constraints, real-machine ACI state
1. Reparameterize x=g(u) so box+mixture hold by construction.
2. Multi-start: R = 512 restarts, u ~ Sobol/LHS DoE seed (generalizes RSM).
3. Per restart, Adam(lr=1e-2, 300 steps), minimize the §8.1 objective:
     inner max over δ (PGD 5 steps) and over k (explicit worst member),
     + λ_m·manifold penalty (reject if flow-atypical OR high disagreement),
     + chance-constraint terms.
4. Deduplicate + rank by pessimistic P̂(Y∈Z*); non-dominated sort / qLogNEHVI → Pareto set.
5. Emit top-q recipes (q≈4) with mean, joint-conformal interval, pessimistic spec-hit prob, typicality, disagreement.
6. After machine validation: update ACI, log realized-vs-predicted gap → adjust κ,τ; hand novel-but-valid gaps to active learning.
```

**Inner-loop compute budget (quantify it — the one place a "cost-is-no-issue" posture can still bite).** Per proposal batch the solver is `R=512 restarts × 300 Adam steps × (5 PGD inner + worst-of-K) × K forwards`, with EPIG (nested MC) above it — naïvely `~10⁷` ensemble-member forwards per batch at K=10 (512×300×5×K ≈ 7.7M; `~10⁶` on a single distilled model). So the inner loop **must** run on the distilled single-model surrogate or the SNGP member (§5.7), **not** the K=10 ensemble, with final candidates re-validated on the full ensemble + conformal. Set explicit targets, measured and logged to W&B: **≥20× wall-clock speedup from distillation** and a **per-batch inner-loop budget of a few GPU-minutes**; the interactive `/invert` serving path always uses the distilled surrogate. Restart/step counts are the first knobs to cut if the budget is exceeded.

### 8.7 Multi-objective reality — return a Pareto set (fixes MFL weakness #2)

MFL weakness #2 has two causes: **(a) competing objectives** (etch depth vs CD vs uniformity; thickness vs bow vs composition) → **qLogNEHVI** (Daulton et al. 2021; Ament et al. 2023) over the *pessimistic* objectives (feed `μ−κσ`-style estimates), 128 MC samples, reference point = spec-box nadir + 10% (hypervolume is acutely sensitive to it); **augmented-Chebyshev scalarization sweep** for >4 objectives; and **(b) non-injectivity** (many pre-images for one target) → the amortized generator (§14.3, driven per §9) *samples* the pre-image manifold; present a diverse *set* + a **diversity/DPP penalty (default: k-DPP over the §9.5 kernel, weight `w_div=0.1`; ablated as −diversity in §12.4)**. **An empty pessimistic-feasible set is a reportable outcome** — return an explicit infeasibility verdict with the tightest achievable spec relaxation, never a clipped point.

### 8.8 Pitfalls and detection

Ensemble collapse (disagreement must grow off-manifold; else add init/architecture diversity or a repulsive-ensemble term, D'Angelo & Fortuin 2021) · flow declares OOD "typical" (score on a held-out OOD batch; fall back to disagreement-only) · conformal calibrated on the surrogate not the machine (audit the calibration fold contains real `y_machine` — this is MFL's weakness #1 fix) · over-conservatism → empty set (relax `κ` and check whether feasibility appears smoothly; real infeasibility stays empty) · trust region kills novel-but-valid recipes (hand off to active learning to *expand* support rather than lowering `τ` blindly).

---

## 9. Active Learning & Optimal Experimental Design (the closed-loop experiment selector)

This layer decides **which real experiments to run** — the load-bearing answer to MFL's "given enough (x,z) pairs," which MFL never earns (its "Loop B" is 5 ungoverned, myopic, uncertainty-free, unbatched steps). It operates on the abstract `(recipe x, outcome y, cost c, fidelity f)` interface, never assuming scalar KPIs.

### 9.1 Objective and protocol: cost-to-target, not model accuracy

Primary metric = **cost-to-target** (§11 defines it; §12 does its statistics). Report physics-fidelity (§14) as first-class, calibration only as diagnostic. **Two coupled objectives run in the loop:** (i) reach-and-qualify the spec cheaply; (ii) keep the surrogate trustworthy *in the pre-image region of the spec* so the inverse cannot exploit optimistic error. Acquisition serves both, blended per D5.

### 9.2 Warm start / initial DoE (Phase 0 of a campaign)

Never search from scratch (Kanarik HF-CL). Expert-constrained ranges define `X`; **space-filling seed** = scrambled Sobol' primary (`n_seed ≈ max(2d+2, 8)` runs, or one lot), maximin/Morris–Mitchell LHS fallback for d≤8. Use the physics sim three ways: **pretrain** the surrogate on cheap sim samples; **low-fidelity source** in multi-fidelity BO *with an explicit learned discrepancy* (Kennedy & O'Hagan 2001; Le Gratiet 2014 — a biased sim used naively as low fidelity misleads MF acquisition); **GP/BNN prior mean**. This makes the loop a strict generalization of RSM/CCD/BBD/Taguchi.

### 9.3 Surrogate + calibration for acquisition

Backbone per D3. **Aleatoric noise needs replication** — reserve a small budget to anchor the heteroscedastic head and set the aleatoric floor, **replicating not only the incumbent-best but a few *space-filling anchors* too, so `σ²(x)` is identifiable at more than one point** (§10.3 identifiability caveat); propagate nested lot/wafer/die variance (§9.6 budget). Recalibrate via Kuleshov (2018) / CQR *before* trusting acquisition and after every refit — an overconfident surrogate makes BALD/EPIG mine noise.

### 9.4 Acquisition: the single cost-cooled blend (resolves D5)

**The acquisition is a *two-phase schedule*, not one linear blend — because qLogNEHVI is an expected-hypervolume-improvement quantity (outcome-volume units), NOT nats, so it cannot be linearly added to the information terms.** *Phase I (explore)* blends the two information families (both in nats, hence linearly blendable); cost enters by **division/cost-cooling**, not by subtracting dollars from nats:

```
α(x) = [ λ·EPIG_S(x) + (1−λ)·BALD(x) ] / cost(x)^β
```

- **BALD** (Houlsby et al. 2011), decomposed as `H[total] − E[H[aleatoric]]` (never raw variance — that chases aleatoric noise): global epistemic reduction, dominates early.
- **EPIG** (Bickford Smith et al. 2023): prediction-targeted information about outcomes at `p*(x)` = the inverse engine's candidate recipes for targets in `S` (K≈256 draws). This is the real improvement over MFL — accuracy *where the inverse will propose*. Caveats handled: cold-start → anneal `λ` **up** from ~0.2 (early loop on BALD/space-filling; trust EPIG only once R's proposals stop moving); circularity → always mix in a Sobol' fill and rely on the epistemic penalty to prevent mode collapse.
- **`β` (cost-cooling exponent) annealed 1→0 over the run** (CArBO; Lee et al. 2020): **cost-frugal early** (high exponent — build the surrogate with cheap runs) and **cost-agnostic late** (exponent → 0 — pay for the run that actually reaches spec). *This is the CArBO direction; an earlier draft had it inverted (0→1), which would make you most cost-averse exactly when you should spend to hit spec.* (Plain EI-per-cost, Snoek et al. 2012 EIpu, over-explores cheap regions early and can underperform plain EI — which is why the exponent *decays*.) **Fixed `c_batch` does not enter the per-recipe ratio** — it enters the stop/continue decision (§11).
- ***Phase II (exploit) — a SEPARATE acquisition selected by a hand-off, not a term in the Phase-I blend.*** After a **hand-off trigger** (R's proposals stable between batches AND ≥ a budget fraction `φ` consumed — the Kanarik V-curve vertex, swept per §11.2), acquisition **switches** from the nat-space blend to a **separately cost-cooled, feasibility-weighted `qLogNEHVI(x) / cost(x)^β`** toward the spec box (native outcome constraints; §11.3). These are **two acquisitions chosen by a schedule** — the nats-additive justification does *not* extend to qLogNEHVI. (So §9.8's `λ`:0.2→0.9 governs the Phase-I BALD→EPIG slide **only**; the EPIG→qLogNEHVI shift is the *phase switch*, not a further `λ` move.) **Anti-reward-hacking:** R's candidates are never trusted until queried on the real machine, and Phase-I acquisition up-weights high-epistemic R-optima so real runs are spent exactly where R would cheat. **Established fallback:** Stepwise Uncertainty Reduction / level-set estimation for the pre-image set (Bect et al. 2012–2014).
- **Cost-/fidelity-/constraint-awareness:** multi-fidelity via trace-aware Knowledge Gradient (Wu et al. 2019) or MF-MES (Takeno et al. 2020) on the discrepancy-corrected MF model; unknown constraints via **SCBO** (Eriksson & Poloczek 2021), constrained-EI fallback; tool-damaging regions via **safe exploration** (SafeOpt; Sui et al. 2015). **Non-myopic look-ahead is mostly cut** — one-step Knowledge Gradient (Frazier et al. 2008) is the pragmatic choice; full rollout (GLASSES/BINOCULARS) off by default.

### 9.5 Batch active learning with diversity (real runs come in lots)

`q≈4–8` per lot. Naïve top-q picks near-duplicates. **BatchBALD** (Kirsch et al. 2019, greedy submodular) for the information family; combine with a **k-DPP** over a kernel mixing input distance, last-layer **BADGE gradient embeddings** (Ash et al. 2020), and predictive covariance. **TuRBO** trust-region batch Thompson sampling (fallback when d≳15 or highly multimodal); parallel Thompson sampling as a strong cheap baseline. **Batch must respect split-plot structure** — constrain each batch to share whole-plot (tool/chamber) factors, varying only sub-plot factors, by solving batch acquisition under an equality constraint on hard-to-change dims.

### 9.6 Real-run budget allocation

The replication/variance-components budget competes with the optimization budget. **Allocate the ~100 real runs explicitly:** `n_seed` DoE (space-filling) + `k` replicates for the noise floor (default: duplicate incumbent-best once per ~3 batches) + remainder for closed-loop. **Estimate the noise floor largely in-silico and from historical logs to spare real budget** — Gage R&R and nested REML can be pre-computed; only a few real replicates anchor them. State the split in every campaign's pre-registration.

### 9.7 Online cadence, drift, failed runs, stopping

**Cadence (D6):** warm-start-refit the surrogate on all real data (few epochs), recalibrate; the **per-query pessimistic optimizer** (§8) runs against the updated surrogate every batch; the **amortized posterior is re-distilled offline** on a schedule, not every batch. **Drift:** **Run-Indexed Time-Varying BO** (Cho, Shao & Mesbah 2024) — run index as a context feature with a temporal kernel down-weighting stale data; a rolling ACI coverage statistic flags drift or exploitation. **Failed/censored runs** (MFL and generic BO ignore these): model run-failure as a learned classifier feeding the safety constraint; treat unmeasurable/out-of-range metrology as **censored observations** in the likelihood — never silently drop them. *Feedback-loop guard:* recency weighting + AL can chase noise — hold an effective-sample-size floor (`ESS≥~30`) and a minimum inter-recalibration interval. **Stopping (any triggers):** (i) target met + qualified (calibrated spec-satisfaction ≥0.95 *and* confirmed by an independent real qualification run); (ii) budget exhausted; (iii) acquisition stall (max α < ε for 2 batches → report **distance-to-feasible**, not a bogus recipe); (iv) diminishing returns (cost-to-target EWMA improvement < threshold).

### 9.8 Baselines to beat and defaults

Random; one-shot DoE (RSM/CCD, Taguchi); **GP-qLogEI/LCB with expert priors and constrained ranges** (the *fair* BO baseline MFL omitted); TuRBO; Dragonfly; Run-Indexed TV-BO (drift); and MFL's own round-trip method. Ablate: −EPIG (BALD-only), −MF, −drift-kernel, −split-plot. Libraries: **BoTorch/Ax** (Balandat et al. 2020); BAAL/custom for BatchBALD/EPIG; Olympus/Summit for self-driving-lab benchmarking. Defaults: Sobol `n_seed=max(2d+2,8)`; ensemble **K=5 members (dev), 10 (final)** Adam 1e-3; `q=4`; `λ` annealed 0.2→0.9 (BALD→EPIG); **`β` (cost-cooling) annealed 1→0** (cost-frugal early, cost-agnostic late — CArBO); candidate pool 256 R-proposals ∪ 2048 Sobol fill re-drawn each batch; low-fidelity (sim) allowed only in the first half of the budget, qualification always real.

---

## 10. Process Reality: Drift, Hidden State, Noise, Transfer

MFL exercised none of these (its "machine" was a static NN on Gaussian synthetic data). Every mechanism here is testable **in-silico first** by injecting the pathology into the MBE simulator (seasoning, first-wafer offsets, measurement noise, a second "chamber" with perturbed rate constants) before any real-data claim.

### 10.1 Drift / non-stationarity

Reframe MFL's one-shot two-loop handshake as **persistent, batch-aware continual adaptation**: offline pre-train on history + physics augmentation; after every real batch, *drift-detect → conditionally recalibrate → re-solve the inverse*. **Detection:** **ADWIN** (Bifet & Gavaldà 2007) on the surrogate's standardized *residual* stream per-KPI (residuals, not raw `y`, so recipe-driven variation doesn't trigger), in parallel with **Page–Hinkley** (catches slow ramps ADWIN is sluggish on); **Benjamini–Hochberg** across KPIs for multiplicity; freeze `σ̂` from a pooled reference window early (small-n `σ̂` is noisy). Covariate-shift complement: **kernel MMD** (Gretton et al. 2012) or **classifier two-sample test** (Lopez-Paz & Oquab 2017) between reference and trailing windows. Decision logic (2×2, completed): residual drift + no covariate shift ⇒ **conditional shift** (physics moved → recalibrate) **OR a metrology-tool recalibration event** (a label-channel shift hitting *all* KPIs at once, which needs the *opposite* response — re-anchor to metrology, hold the process model); **disambiguate with a metrology-drift channel** — periodic re-measurement of a control/standard wafer — because Gage R&R only pins a *static* floor and tracks nothing over a months-long campaign. Covariate shift alone ⇒ you moved in recipe space (check OOD, don't necessarily recalibrate). **Residual drift AND covariate shift together ⇒ ambiguous — quarantine and require a standard-wafer confirmation run.** Note **MMD/C2ST vs a *frozen* reference fires persistently by construction under continuous AL exploration**, so window the reference and test *residuals*, not raw inputs. Ship classical EWMA/CUSUM SPC alongside as the interpretable deployed-fab incumbent. **Recalibration:** triggered exponential recency weighting (`w_i=γ^Δruns`, `γ∈[0.98,0.995]`) + replay buffer + **anchor regularization** (L2-SP, Li et al. 2018; or EWC-lite) against catastrophic forgetting; on the GP backend, kernel time-decay makes this trivial. Optimization layer adopts Run-Indexed TV-BO (Cho, Shao & Mesbah 2024).

### 10.2 Hidden state / memory / path-dependence

Chamber seasoning, first-wafer effects, wafer-index-within-lot, time-since-clean make `Y` depend on more than current `X`. **Detection:** residual autocorrelation (Ljung–Box / Durbin–Watson on run-ordered residuals) and a **replicate-scatter / variance-components test** (repeated center points → nested mixed-effects; if between-*occasion* variance for an identical recipe exceeds the measurement floor, `Y` carries state). **Modeling — primary: fold state in as engineered context features** (runs-since-clean, cumulative deposited/etched thickness since clean, time-since-PM, wafer-index-in-lot, lag features of previous wafer's recipe+outcome), tagged "state/context" so the inverse treats them as *observed-but-not-actionable*. Fallback only if autocorrelation persists: a **GRU encoder over the recent run sequence** FiLM-conditioning the forward net; last resort, a **Deep State-Space / Deep Markov Model** (Rangapuram et al. 2018; Krishnan et al. 2017). **Cut S4/long-range SSMs as over-engineering** for the ~100-run regime.

### 10.3 Noise and run-to-run variability

Decompose with the heteroscedastic β-NLL ensemble/GP (§5). **Separate measurement from process noise** with a replicate design + nested/mixed-effects REML (lot⊃wafer⊃die) + a Gage R&R study to pin the metrology floor — a distinction MFL cannot make and the inverse *must* have (chasing below the metrology floor burns budget). **Propagate to the inverse** via the chance-constrained risk-averse objective of §8 (`min E[loss] + κ√Var[y|x]` s.t. `P(y∈spec)≥1−α`), refusing recipes that hit the target *mean* with unacceptable *spread*. **Identifiability caveat (critical):** at n≈100 with replication concentrated at the incumbent, only a **constant** aleatoric floor is identified — **`σ²(x)` as a function of `x` is unconstrained off the replicated points**, yet the chance constraint, the credited band `q̂=κσ` (§8.4), and this spread-refusal all consume `σ²` at *proposed, unreplicated* recipes. Mitigations: (i) **distribute the replication budget** across a few **space-filling anchors**, not only the incumbent, so `σ²(x)` is pinned at more than one location; (ii) lean on the **distribution-free conformal band (CQR/ACI)** — valid coverage *regardless* of whether `σ²(x)` is correct — as the backstop for the credited band; (iii) **trust `σ²(x)` only within a calibrated distance of a replicated anchor**, else fall back to the conservative constant floor + a flag, and test that the fitted variance surface is trustworthy before relying on it off-anchor.

### 10.4 Transfer — two levels (interface claim now, empirical claim later; D8)

**Level (a): across tools/chambers of the SAME process (chamber matching).** Primary: **hierarchical / partial-pooling with per-tool adapters** — shared trunk + per-tool FiLM (Perez et al. 2018) or a small adapter/LoRA head fit on a few per-tool runs; for few-shot onboarding of a *new* chamber, condition an **Attentive Neural Process** (Kim et al. 2019) on the handful of runs — the principled generalization of MFL's "5 machine iterations," with UQ and cross-tool priors. Also **Ranking-Weighted GP Ensemble (RGPE)** (Feurer et al. 2018), robust to negative transfer because weights track rank agreement on new data. Fallback if negative transfer detected: per-tool models + Deep CORAL alignment, or a multi-task GP with a tool-covariance kernel (Bonilla et al. 2008).

**Level (b): across DIFFERENT process types.** Shared foundation trunk + per-process adapters + process embedding + FiLM, heterogeneous outputs via per-process decoder heads. **This is the paper-2 empirical claim (D8).** Paper 1 demonstrates only the *interface* claim (same core on ≥2 processes via adapters). Keep the negative-transfer guards but mark them paper-2: held-out per-process transfer gain, **task-affinity grouping** (Fifty et al. 2021), **gradient-conflict monitoring + PCGrad** (Yu et al. 2020), loss balancing via GradNorm/inverse-Dirichlet. **Meta-learning** (MAML/Reptile) is the alternative for an *open-ended* stream of new processes; prefer ANP/multi-task for a fixed known set.

### 10.5 OOD gating and STOP-trusting-the-surrogate

Composite OOD score (§8.2) gating a trust-region inverse; **calibrated intervals under drift via ACI** (§5.6). Testing (non-negotiable, MFL lacks all of it): **temporal (forward-in-time) split**, **leave-one-tool-out**, **leave-one-process-out**, **out-of-range recipe holdout** (the OOD detector must flag them). Detector quality metric: **AUROC of (OOD score vs |surrogate residual|)** — the detector is useless if its score doesn't track actual error.

---

## 11. Cost-to-Target, Human-in-the-Loop & Deployment/Certification

**Owner of: the cost-to-target definition and cost model; deployment/certification operationalization.**

### 11.1 Cost-to-target is the objective function, not accuracy

**Cost model (adapter-supplied):** `C = Σ_batches c_batch + Σ_recipes c_recipe(x) + Σ metrology cost + Σ c_fail(x)`, where **`c_fail(x)` is the failure/rework/scrap cost** (wasted wafer, re-run, or a scrapped lot near a tool-damage boundary) — **omitting it biases the headline survival comparison, because methods differ systematically in near-boundary proposal rate**; Kanarik defaults `c_recipe≈$1k`, `c_batch≈$1k`, ~4 recipes/batch; multi-fidelity costs (in-silico ≈ $0, single-wafer confirm, full qual lot). The fixed/variable split (§3.1) drives acquisition and the stop/continue rule. **Feasibility first (the half of MFL weakness #2 they miss):** before spending on `Z*`, test whether it lies inside the reachable output set: **feasible ⟺ `Z*` is (weakly) *dominated by* the estimated attainable Pareto front** (equivalently, positive posterior-predictive mass on outcomes that reach `Z*`). **A `Z*` that is *non-dominated* relative to the front is better-than-achievable = INFEASIBLE** — do not invert this polarity (a "non-domination = feasible" test would pass exactly the infeasible targets). If infeasible with high confidence, return *"target infeasible; nearest achievable is `Z°` at distance d, driven by inputs …"* instead of burning budget.

**Primary metrics (report distributions, not point values):** **Expected cost-to-target (ECT)** over ≥50 seeds × held-out target specs; **success-rate-at-budget** (fraction hitting spec within a fixed \$ budget); **CTT vs. expert baseline** ratio (Kanarik's halved-cost bar) and vs. from-scratch BO. **Cost-aware acquisition** = CArBO cost-cooling generalized to qLogNEHVI weighted by *variable* recipe cost; the fixed `c_batch` gives the principled "run another \$1k batch vs. stop" rule via multi-fidelity KG.

### 11.2 Human-First / Computer-Last and the V-curve

Three warm-start channels, ranked by leverage: **(1) physics prior as GP mean / gray-box residual** (primary for any process with a simulator — extrapolation defaults to physics, not an unconstrained NN); **(2) expert-constrained search box** (hard box constraints + informative GP lengthscale priors — the source of Kanarik's cost halving); **(3) cross-tool/process transfer** (RGPE primary; FSBO/ABLR with ≥dozens of prior campaigns). **Study the V-curve explicitly:** sweep the hand-off fraction φ and plot CTT vs φ; report the empirical minimum *per process* (don't assume Kanarik's optimum transfers); ship it as a tunable, not a hard-coded 5-iteration loop. **Trust/interpretability:** every proposed recipe ships with calibrated KPI bands, **SHAP attributions** (Lundberg & Lee 2017), a **counterfactual** in *settable* knobs ("+2 °C substrate, +8% group-III cell flux → +8 nm thickness, uniformity −0.3%" — note V/III is a *derived readout* in MBE, not a directly-set variable, so counterfactuals must be phrased in the knobs the tool actually sets), and an explicit **flag on sacrificial/exploratory recipes** with their expected information gain (Kanarik: unexplained sacrificial moves are the #1 trust failure).

### 11.3 Batch experimentation

`q=4` (adapter-overridable); qLogNEHVI with **native outcome-constraint (feasibility-weighted) form**; **reference point from the spec-box nadir**, not a default. qNParEGO when >4 objectives. Split-plot scheduling (hard-to-change factors constant within a lot, varied across lots) and simplex constraints for mixtures, both in the acquisition optimizer's feasible set. First batch = DoE space-filling. Async (Kandasamy et al. 2018) if lots return staggered.

### 11.4 Deployment & certification (always-on gate)

Certified deployment is mandatory regardless of generator. **The independent qualifier shares no parameters with the generator or its surrogate** (else it certifies the same hallucination). Crucially, the independent solver is itself fallible: to prevent Goodhart-on-the-qualifier, it is **query-limited, cost-gated, and never a training signal for the generator**, and the **physical single-wafer / pilot / qual lot is always the final arbiter** — the solver gate is only a cheap pre-filter that decides when to spend the physical query. Staged, cost-ordered gates: `in-silico independent-solver gate → single-wafer confirmation → small pilot lot → qualification lot with Cpk acceptance (default Cpk≥1.33; ≥1.67 automotive)`, — **the recipe/process-step's own certification is Cpk + process-window + SPC** (above). **AEC-Q100 HTOL/temperature-cycling qualifies the finished *packaged device*, not a single process step**, so it is the *downstream* device-level reliability gate the eventual product (not the recipe) must pass — an explicit pluggable *downstream* stage (Banad & Sharif flag automotive qualification as an open gap), not the recipe's direct certification. Post-deployment: EWMA/CUSUM on residuals; drift via Run-Indexed TV-BO; every deployed recipe logged with a **model card + provenance** for audit and rollback.

### 11.5 Safe autonomous / self-driving-lab operation

Runs unattended *only* behind guardrails. **Never propose outside the safe envelope** (constraint-by-construction). **OOD stop:** Mahalanobis in the spectral-normalized latent beyond the 99th percentile → defer to human. **Uncertainty stop:** deep ensemble + ACI conformal; if `predictive_std/tolerance > 0.5` (per-adapter default, ideally the decision-theoretic "defer when expected cost of a bad run > cost of deferring") or disagreement exceeds a calibrated band → defer. The Mahalanobis and conformal gates are **complementary** (conformal gives calibrated intervals when exchangeability holds; Mahalanobis catches the cases where it doesn't). **Trust-region anti-exploitation:** inverse search inside a TuRBO region anchored on real data + epistemic penalty + **required independent-solver agreement before any machine run**. **Governance:** tiered sign-off (low-uncertainty in-box → auto-run; exploratory/near-envelope → human sign-off; OOD/high-uncertainty → human design session); kill-switch + rollback to last certified recipe on any SPC alarm; immutable audit log.

---

## 12. Evaluation, Benchmarking & Statistical Rigor

**Owner of: cost-to-target statistics (survival analysis); the benchmark suite.**

The measurement layer that makes the work publishable and makes MFL's failures impossible to hide. Every metric is computed per-modality with an explicit scalar reduction; every headline number carries a bootstrap CI and a **pre-registered** comparison; every accuracy/hit-rate is interpreted against the **aleatoric noise floor** — the *metrology* floor from Gage R&R **plus** genuine run-to-run *process* variance (nested REML, §10.3); Gage R&R alone is only the metrology part and understates the floor. You cannot beat this floor, and every tolerance τ and "surrogate is wrong" claim is stated relative to it. No metric is a single seed's point value.

### 12.1 Forward-surrogate metrics

Calibration and sharpness dominate point accuracy (see §5.8). Additionally, the **surrogate-exploitation stress test (likely headline figure):** take the recipes the inverse engine *actually proposes* and run them on the **OOD / pathology-injected in-silico machine of §15.2(iii)** — one whose injected structure (hidden state, a perturbed second chamber, a held-out physics regime) the surrogate was **not** trained on, so exploitation is possible **non-circularly**. (Running against the *same* sim family the surrogate learned is partly circular: real-machine structure the sim lacks cannot be exploited by construction, so that venue *understates* the failure.) Plot **surrogate-predicted vs realized outcome on those proposed points specifically** (not the i.i.d. test set); report the **optimism gap** = mean(predicted − realized improvement) and the fraction of proposals whose realized value falls outside the surrogate's 90% predictive interval. Well-behaved ⇒ optimism gap ≈ 0 (within noise floor), interval violations ≈ 10%. **On real hardware this figure is n≈1 and underpowered** (state it — the §12.4 power caveat is scoped to cost-to-target, not this figure); the non-circular OOD-injected in-silico run is its primary venue. This is the cleanest demonstration we fixed MFL weakness #3.

### 12.2 Inverse-recipe-generation metrics

- **PRIMARY: target-hit-rate @ N real queries within tolerance τ**, evaluated ON GROUND TRUTH (machine, or sim in-silico), τ floored at Gage-R&R repeatability, pre-registered. Report at N∈{1,5,10,20,batch-budget} and the full success-vs-budget curve. **Fixed set→deployment policy (definitional — the metric is ill-defined without it):** the returned set of `q≈4` recipes counts as `q` real queries toward cost; a target is **"hit" iff the single top-ranked recipe** (final ranking = pessimistic P̂ × robustness × cost, per §8.6 step 4) lands in tolerance. Best-of-`q` is reported *separately* (it inflates hit-rate and multiplies cost — never conflate it with the primary).
- **Cost-to-target as right-censored survival data (two statistical corrections):** **(i) infeasibility ≠ censoring.** Only *feasible-but-unhit-within-budget* targets are legitimately right-censored; a **truly-infeasible pre-registered target is never hit at any budget** — an "immune/cured" subject that violates KM's eventual-event assumption — so **exclude infeasible targets from the cost-to-target survival analysis entirely** (they belong only to the feasibility/abstention metrics below). **(ii) crossing curves ⇒ not log-rank.** Report the **Kaplan–Meier survival curve** and the **restricted-mean survival time (RMST)** to the budget horizon; because the curves are *expected to cross* (expert fast early, algorithm faster near tight tolerances), the **primary comparator is a difference-in-RMST test** (Uno et al. 2014) — **not** log-rank, whose power collapses under exactly the non-proportional hazards that crossing implies (log-rank is reported only as a caveated secondary). Convert to dollars via the Kanarik cost model. Bar to beat: halved cost-to-target vs HF-CL.
- **Feasibility & abstention calibration (MFL-absent):** the target set is **pre-registered to include known-infeasible targets**; report fraction correctly flagged and **false-success rate on infeasible targets (must be ≈0)** — a "hit" on an infeasible target is a bug. MFL's clip() manufactures fake feasibility; this exposes it.
- **FALSE-ABSTENTION (Type-I) rate (new — structurally induced by our own machinery):** on a pre-registered set of **known-feasible-but-hard** targets, report the fraction the system wrongly declares infeasible / refuses. Pessimism + the trust region push the *estimated* reachable set inward, so wrongly refusing a reachable target is a direct, headline-relevant cost of the "refuses-instead-of-clips" differentiator — and it is invisible unless pre-registered alongside the false-success rate.
- **Constraint-satisfaction rate reported BEFORE any projection/clipping** — honestly penalizes a method that needs heavy clipping; constraint-by-construction scores ≈1.0 by design.
- **Multimodality / posterior recovery (the second MFL-killer, framed as SBI):** **Simulation-Based Calibration** (Talts et al. 2018) with **ECDF-difference simultaneous bands** (Säilynoja et al. 2022); **posterior coverage via TARP** (Lemos et al. 2023); **diversity of *valid* solutions** among in-tolerance proposals **and, separately, on the refined-and-deployed recipe set** (**Vendi Score**, Friedman & Dieng 2023, + mode-count + pairwise-L2 cross-check). Two subtleties: conditioning on in-tolerance is essential (else a method games diversity with diverse garbage); and **SBC/Vendi on the amortized *proposal* is not enough** — the multi-start-pessimism + k-DPP selection that produces the *shipped* recipes can mode-collapse into one basin, so the multimodality/coverage test must also run on what actually deploys.
- **Robustness of chosen recipes:** perturb inputs by known actuation noise; report **robust-hit-rate** (fraction still in tolerance) — turning MFL's untested "domain randomization" into a ranked metric that rewards flat basins.

### 12.3 Baselines (strong, tuned, matched-budget)

Every baseline gets the **same query budget, the same warm-start/prior information, and independent tuning with the same tuning budget** (reported). **MFL reimplemented** (round-trip Loop A + machine Loop B, Jacobian-gated LR). **BO done right:** GP-**LogEI** (Ament et al. 2023) + GP-qLCB (BoTorch/Ax, Matérn-5/2, input warping), **warm-started from expert-constrained ranges**; multi-objective qLogNEHVI + qParEGO; constrained cEI/SCBO; high-dim TuRBO/SAASBO; drift Run-Indexed TV-BO. **Evolutionary:** CMA-ES, NSGA-II. **Learned inverse:** cVAE, cINN (Ardizzone et al. 2019), **NPE via `sbi`** (Greenberg et al. 2019; Tejero-Cantero et al. 2020), and **direct gradient inversion through our own frozen surrogate** (ablates everything we add on top of a differentiable emulator ≈ MFL without R). **Classical/DoE floors:** Sobol/LHS, RSM/CCD+Taguchi, grid. **Primary claim:** beats well-tuned warm-started qLogNEHVI/SCBO, MFL, and cINN/NPE on cost-to-target and posterior recovery, at matched budgets.

### 12.4 Splits, ablations, statistics

**Three splits always, side by side:** (1) random i.i.d. (sanity floor only); (2) **OOD/extrapolation** (leave out a design-space corner *and* a whole level of a hard-to-change factor — a chamber/tool/precursor — *and* target regions beyond the training outcome hull); (3) **temporal/drift** (order by run-index, train on past, test on future; **fit all normalization/conformal stats on the training window only** — leakage guard). Report the **sim-to-real gap** explicitly (Phase-0 in-silico vs Phase-2 hardware on the same targets). **Ablations, one per claimed component:** UQ head (deterministic → ensemble/heteroscedastic → *then* conformal, which *composes* with base UQ, it is not an alternative; we deliberately avoid evidential regression, pre-empting "why not evidential"); physics prior (data-only → +residual loss → simulator-in-the-loop); robust penalty on/off; AL acquisition vs DoE; constraint handling (clip vs project vs by-construction); multimodal mechanism (point R vs flow/NPE); **−diversity** (k-DPP weight `w_div`=0 vs 0.1 — proves the diversity penalty recovers distinct valid recipes). **Statistics MFL had none of:** ≥5 seeds (10 preferred, varying init, data order, *and* target set); report **distributions** (KM curves, median+IQR, 10k-resample bootstrap bands) never bare means; **paired Wilcoxon signed-rank** (primary) + paired bootstrap + **difference-in-RMST** (Uno et al. 2014; log-rank only as a non-proportional-hazards-caveated secondary) for censored survival, infeasible targets excluded (§12.2); multi-method ranking via **pairwise Wilcoxon–Holm** (Benavoli et al. 2016) with a **critical-difference diagram** (Demšar 2006, leading with Wilcoxon–Holm over the low-power Nemenyi post-hoc); **effect sizes not just p-values** (a significant 2% improvement is a non-contribution); **Benjamini–Hochberg FDR** across the metric×method grid; **pre-register the primary metric, split, and test**. **Power & the in-silico/hardware split (the n≈100 reality):** the ≥5–10-seed survival/Wilcoxon/CD-diagram statistics are *powered only in-silico*, where seeds are cheap; the hardware Phase-2 is a **single** pre-registered prospective campaign (~100 total runs buys essentially one trajectory), so its claim is **descriptive** (per-target hit/miss + interval coverage) with honestly-stated limited power. Do a **design-stage power / sample-size analysis** for how many *targets* the hardware campaign must include to detect the pre-declared effect, and never present a hardware number with in-silico-grade error bars.

### 12.5 Real-data validation protocol (prospective, pre-registered)

1. **Phase 0 — in-silico dress rehearsal** (MBE sim + public sims): validate the *entire* closed loop including Loop-B-style queries and **lock every hyperparameter**. Physics sim belongs here, explicitly *not* where the final claim lives.
2. **Phase 1 — retrospective on real logged data** with OOD/temporal splits.
3. **Phase 2 — PROSPECTIVE, pre-registered on hardware:** before touching the machine, register a hash of {trained model + code + fixed target list (including infeasible ones) + primary metric/split/test} as a signed timestamped manifest. For each target emit recipe(s) *and a calibrated predicted-outcome interval up front*; run batch-aware. **Success = pre-declared in-tolerance hit-rate within pre-declared budget**; calibration judged by whether realized outcomes fall in pre-declared intervals at nominal rate. Include a **human-expert comparator under identical constraints where feasible** (§12.6).
4. **Certification gate:** an *independent* qualification pass (metrology / independent solver not used in training), reported as separate pass/fail.

### 12.6 Human-expert comparator — scoped, not asserted

"Beats human experts" is not asserted without a protocol. **Either** run the comparator as a milestone (recruit N process engineers, blinded pre-registered target set, identical expert-constrained box, Kanarik-style) **or** downgrade the headline to "beats warm-started BO / HF-CL trajectory" and state the human comparison as future work. Decide at M2 based on whether a fab/academic partner can supply engineers. **The expert-comparator is human-subjects research: obtain IRB / ethics-board approval and informed consent, and register the protocol, *before* recruiting** — fold this into the M2 decision and the §17 governance checklist.

### 12.7 Multi-process benchmark suite & physics-fidelity benchmark

Ship a **named suite** spanning ≥4 process types and all three modalities, public where possible: **Etch (scalar)** on a public level-set/feature-profile simulator (head-to-head with MFL/BO on their home turf); **Deposition (scalar+1-D)** CVD/PECVD/ALD thickness+uniformity with mixture inputs (tests the physics-hybrid backbone); **Lithography (2-D field)** — **LithoBench** (S. Zheng et al. 2023, NeurIPS D&B) supplies mask/layout → aerial-/resist-image + printability data (use it for the 2-D-field + mask-optimization + physics-fidelity task; it is the public precedent for physics-fidelity-over-FID). **NB LithoBench does *not* contain overlay, sidewall-angle, or dose–focus (FEM) metrology** — a *process-window / dose–focus* task needs a separate litho simulator or dataset; scope that claim to the separate source, not LithoBench; **MBE (scalar, bootstrap)** in-house sim, one instance, the in-silico rehearsal machine and a FD-sensitivity sanity check, not the headline; **(stretch)** an additional public tabular process (CMP/implant/anneal) for the transfer ablation. **Physics-fidelity benchmark (replaces FID/likelihood):** fraction passing an **independent** solver (D7), violation magnitude, distance-to-feasible.

---

## 13. Software Architecture, Reproducibility & MLOps

Hard boundary between process-agnostic core and per-process adapters (§3), enforced by import-linter. MFL shipped as a single script inverting a single MLP on synthetic data; the deltas below make the process-agnostic claim *enforced*, surrogate exploitation *detectable/gated/non-circularly benchmarked*, drift *handled in production*, and every result *regenerable from one command*.

### 13.1 Stack (decisive)

| Concern | Primary | Fallback / switch |
|---|---|---|
| Modeling | **PyTorch 2.x + Lightning 2.x**, `torch.func` (vmap/grad/functional_call) + `torch.compile` for the inverse inner loop | JAX/Equinox **only** as a profile-gated, isolated escape hatch — not a default dependency |
| Config | **Hydra 1.3 + OmegaConf** structured configs | — |
| Tracking | **Weights & Biases** (Artifacts + Sweeps + Registry) | **MLflow** self-hosted if data cannot leave the fab; swap via a `Logger` shim |
| Data/model versioning | **DVC 3.x** + remote (S3/MinIO) | lakeFS only if fab volume outgrows DVC |
| Data validation | **Pydantic v2** + **Pandera** + **Pint** (units) | — |
| Env | **uv** (`uv.lock`) + **Docker** (pinned digest) | conda-lock only for conda-only binaries |
| UQ / numerics | deep ensembles + **laplace-torch** + **conformal** (ACI); **BoTorch/GPyTorch** for BO/AL | **SNGP** as single-model distance-aware alternative |
| Serving | **FastAPI + Pydantic** (single-process batched ensemble via `vmap`) | **Ray Serve** only if members are large FNO/CNN decoders |
| Testing | **pytest + Hypothesis** + `torch.autograd.gradcheck` | — |
| CI | **GitHub Actions** (or GitLab on-prem), matrix over Python × adapter | — |

**JAX is not a default (correction to a common over-engineering instinct):** `torch.func` + `torch.compile` cover vmapped ensemble gradients and fused kernels in-framework; shipping JAX by default incurs two checkpoint formats and two seeding regimes. Enable it only if profiling shows the inner loop dominates wall-clock (>~30%) *and* the physics prior is a differentiable function; gate with a `torch↔jax` numerical-parity test.

### 13.2 Layered defense against surrogate exploitation (enforced in code)

Ensemble-disagreement penalty alone is insufficient (ensembles share correlated errors). Layered, cheap→hard: **(1) constraint-by-construction** in `featurize` (simplex/box); **(2) distributional objective** (epistemic + aleatoric penalty); **(3) trust-region / in-distribution gate** (`density_gate.py`, the composite OOD score of §8.2 — the actual guard against correlated ensemble error); **(4) conformal wrapping** (ACI) — the calibrated interval the operator sees; **(5) mandatory real-tool verification** — the loop closes on `source==real_tool`, never on the surrogate alone. The `ForwardModel.predict` **returns a `PredictiveDistribution`, not a scalar**; `InverseSolver.solve` **returns a set with feasibility flags** (canonical names/tuple per §3.2/§3.3 — used verbatim everywhere: `predict(x)→PredictiveDistribution(mean, aleatoric_σ, epistemic_σ, conformal_set)`, `solve(spec)`); `support_score` makes exploitation a first-class testable signal.

### 13.3 Testing (a large part of the rigor delta)

Unit (adapter unit round-trips, constraint satisfaction, splitters) · **property-based (Hypothesis)** — for every adapter, random valid recipes satisfy invariants; **scoped correctly**: the "every proposal satisfies `ConstraintSet` for all targets" guarantee is asserted only for **by-construction** constraints; for projection-only constraints assert *satisfy-or-explicitly-reject*, never silent violation · **gradient checks** (`gradcheck` on custom heads; FD-vs-autograd parity on `ForwardModel.jacobian` against the MBE sim's FD sensitivities) · **calibration regression tests** (frozen fixture: **regression quantile-calibration error / PIT** — *not* classification ECE, per §5.8 — plus NLL and 90%-interval-coverage in 88–92% must not regress) + a **reward-hacking canary** (on a held-out high-disagreement region, the inverse's chosen candidates must carry inflated predicted uncertainty *and* low `support_score`) · **physics-fidelity benchmark non-circular by construction** (`independent_verifier` must differ from `physics_prior` — D7, enforced by the interface split) · data-contract tests (Pandera in CI and on every `dvc repro`) · integration (`train→calibrate→invert→benchmark` on a tiny fixture, deterministic + a loose cost-to-target bound). **CI matrix** over `{python 3.11,3.12}×{mbe,pecvd,rie_etch,litho}` = ruff + mypy/pyright + import-linter boundary + pytest; calibration/cost-to-target benchmarks nightly on GPU → W&B. **Three subsystem tests the list omits — the very places the plan caught its own sign bugs:** (1) an **AL-schedule directional regression test** on a known-answer synthetic (assert EPIG dominates *mid* not early, the cost-cooling exponent decays 1→0, and the `λ` direction is correct); (2) a **drift-decision branching test** (inject a known drift *type* — conditional-shift vs covariate-shift vs metrology-recal — and assert the §10.1 2×2 fires the correct branch); (3) an **automated `Ŝ_final` monotonicity re-verification** (§6.4 can break §6.3 — assert monotonicity on the *blended, deployed* surrogate, not just the parts).

### 13.4 Determinism & serving

Central `seed_everything(workers=True)`, `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed `worker_init_fn`; a "same-seed-twice → bitwise-equal metrics" CI test; document any op lacking a deterministic impl, never silently drop determinism. Serving reuses the *same* schema classes as training (no train/serve drift): `POST /invert` → ranked `RecipeCandidate` set (each with predicted distribution, ACI interval, feasibility flag, `support_score`, constraint-satisfaction proof); `POST /observe` → append real run, trigger batch AL (qLogNEHVI/qLogEI, Sobol-seeded, split-plot-aware). **Drift:** monitor served-model coverage + input distribution over `timestamp`; on drift, switch AL to Run-Indexed TV-BO; **shadow-eval gate** promotes a candidate model only if calibration *and* held-out cost-to-target don't regress; `GET /model` returns git SHA + data hash + calibration report for audit. **Campaign-pipeline liveness monitoring (this is a multi-week closed loop, not just the served model):** alert on each cadence step — *metrology export arrived → refit succeeded → calibration held → next batch proposed* — with a **hard alert on a silently-missing metrology file or a diverging refit**, and a defined **on-call / SLA for the §11.5 "defer to human" path** over a months-long run.

---

## 14. Theory, Well-Posedness, Guarantees & Honest Limits

**Owner of: physics-fidelity benchmark definition, coupling ladder, certification path.**

The theoretical spine and honesty layer: what "solving the inverse" means, what we can prove, what we provably cannot, and a detector for every way the system can lie. The single organizing objective is **cost-to-target**, not surrogate accuracy.

### 14.1 The inverse is ill-posed; the correct object is a posterior

Recipe generation is **Hadamard-ill-posed**: existence fails (infeasible `Z*` outside `image(M)`), uniqueness fails (`M` is many-to-one; the pre-image is generically a *manifold*), stability fails (near-singular `∂M/∂x`). **This is exactly what MFL's `R:Z→X` gets wrong** — a single-valued `R` regresses to the conditional mean of a multimodal pre-image set, which is itself *not* a valid pre-image; averaging pre-images is a category error, not a tuning problem. The correct formalism is the **Bayesian inverse problem**: `p(x | M(x)∈Z*) ∝ p(x)·P(M(x)∈Z*|x)`, with the prior `p(x)` carrying DoE structure (feasible box, mixture/sum-to-one, hard-to-change split-plot, nested variance) **enforced by construction** (§8). The payoff (Stuart 2010; Dashti & Stuart 2017): even when the classical inverse is ill-posed, the *Bayesian* inverse is **well-posed** — the posterior is Lipschitz-continuous in the data. We do not restore uniqueness; we restore **stability and a rigorous solution-as-distribution.** MFL's `R(Z*)` is the degenerate `posterior→point` collapse under the unstated assumption that the posterior is unimodal and the spec feasible.

### 14.2 Identifiability and conditioning

Characterize the local geometry cheaply (the sim supplies FD sensitivities; `E` is differentiable). **Jacobian `J=∂M/∂x` and its SVD:** near-zero singular values span the **unidentifiable ("sloppy") subspace** — `dim(pre-image)≈#{σᵢ≈0}` (sloppy-model analysis, Gutenkunst et al. 2007; Transtrum et al. 2015). This *replaces* MFL's Jacobian-norm-gated LR with a full conditioning report plus the pessimistic objective of §8. Full column rank ⇒ locally unique (inverse-function theorem) but **not globally injective** — folds remain; fall back to **profile likelihood** (Raue et al. 2009) where the linearized SVD is insufficient. **Deliverable per recipe:** an "identifiable directions" list and a "these knobs are interchangeable" list, uncertainty projected onto identifiable vs sloppy subspaces.

### 14.3 The SBI stack (concrete, SOTA) — matches D2

Solve the posterior with **simulation-based inference** (Cranmer et al. 2020). **Primary: Neural Posterior Estimation** with neural spline flows (Durkan et al. 2019; amortized SNPE-C/APT, Greenberg et al. 2019; `sbi`, Tejero-Cantero et al. 2020), deep-ensembled 5–10 members for epistemic spread. **Upgrade for arbitrary partial/masked conditioning (drop a subset of `Y` at test time): Simformer** (Gloeckler et al. 2024). **FMPE / flow-matching** (Wildberger et al. 2023) is a *separate scalability* upgrade of the density estimator — it does **not** provide arbitrary partial conditioning; route high-dimensional posteriors there, not masking. **Fallback: NRE** (Hermans et al. 2020) if the flow is unstable in high-D or you only need to score/accept; **NLE + MCMC** when an explicit likelihood is needed for physics-loss coupling. Switch NPE→NRE if SBC (§14.6) shows persistent miscalibration; →NLE if you need the likelihood.

### 14.4 Constraint handling on the coupling ladder

The headline failure — inverse optimizing the *learned* not the *physical* manifold — is defeated by climbing the ladder (weak→strong): **post-hoc screening → physics-loss regularization → simulator-in-the-loop → constraint-by-construction → certified deployment.** Default to the strongest feasible rung per constraint type (clipping is banned where a stronger rung exists). Box+mixture → constraint-by-construction; hard-to-change → structural; physics laws → physics-loss (principled loss-balancing: Wang 2021 / Maddu 2022 / McClenny 2023 / Liu 2021; adaptive collocation Nabian 2021, DAS Tang 2023) + simulator-in-the-loop (DiffTaichi-style; simulator-conditioned diffusion reaches targets ~3× fewer sims). **Physics-fidelity benchmark (required, replaces FID/likelihood):** fraction passing an **independent** solver (D7), violation magnitude, distance-to-feasible (LithoBench precedent). *Cut:* no full PINN forward model by default — loss-balancing is invoked only when a physics residual is actually used; pure data-driven deployments rely on the conformal gate.

### 14.5 Distribution-free guarantees and the surrogate-exploitation gate

**(a) Conformal forward model** (§5.6): `P(Y∈C(x))≥1−α` marginally under exchangeability; CQR for scalars, copula/Bonferroni for multi-output, functional conformal for fields; adaptive-width estimates noisy below n≈1000, so size the calibration set / use ACI + jackknife+ (D4). **(b) Acceptance gate:** accept `x'` **only if its conformal outcome set is contained in the spec:** `C(x')⊆Z*`. Because `C` is wide exactly where `E` is epistemically uncertain, a recipe in a surrogate-exploitation hole *usually* gets an interval that spills outside `Z*` and is rejected. **This gate is necessary but NOT sufficient:** §8.2 warns `σ_epi` (and hence the conformal band) can be *spuriously narrow* in a far-OOD hole where the ensemble wrongly agrees, letting a bad recipe pass — so the acceptance gate is valid **only inside the §8.2 typicality-AND-disagreement trust region**, as defense-in-depth with it and the mandatory real-tool refit, not a standalone "no-penalty-to-tune" fix. **(c) Risk control on the selected set (a genuine subtlety):** selecting recipes *because* their intervals are narrow is a **selection/multiplicity** problem — the marginal guarantee does not transfer for free. Control `P(realized ∉ Z*)≤δ` with **Learn-then-Test** (Angelopoulos et al. 2021) or **Risk-Controlling Prediction Sets** (Bates et al. 2021) on a *held-out* set, with **multiple-testing correction across the candidate pool** (default `δ=0.05`). **Campaign-level multiplicity (new):** the prospective campaign issues *many* targets, each consuming this guarantee, so add a **family-wise / FDR correction across the pre-registered target sequence** — the *aggregate* reliability over the whole list must be controlled at the advertised level, not just within one query's pool. **(d) Pessimistic search objective:** optimize a **conformal lower confidence bound on "meets spec"** (offline-RL pessimism; CQL Kumar et al. 2020; MOPO Yu et al. 2020) so the engine maximizes a *guaranteed-achievable* objective.

**Honest limitation (verbatim in the paper):** conformal coverage is **marginal and requires exchangeability**, which the inverse search *actively violates* by pushing `x` off-distribution and which drift violates over time. Repairs: weighted conformal (Tibshirani et al. 2019) + Mondrian for covariate shift; **ACI / conformal-PID** (Gibbs & Candès 2021; Angelopoulos, Candès & Tibshirani 2023) for temporal drift.

### 14.6 Is the posterior correct? SBC and TARP (a blocking gate)

Test it. **Primary: Simulation-Based Calibration** (Talts et al. 2018) — per-coordinate ranks of the true recipe must be uniform (∪-shaped = overconfident, the dangerous direction). For high-dim recipes, primary shifts to **TARP** (Lemos et al. 2023). Defaults `N≥200`, `L=100`. **Gate: no posterior ships to the inverse engine until SBC/TARP passes** — a failed SBC is a blocking defect. **Caveat 1: SBC/TARP certify the posterior only *relative to the surrogate's own generative model*, not against the real machine** — real-machine correctness rests solely on the prospective hit-rate + interval coverage (§12.5). **Caveat 2 (AL invalidates the pre-set gate):** the in-silico gate runs on *prior* draws, but the shipped amortized posterior is re-distilled (D6) on **AL-selected data** (BALD/EPIG/qLogNEHVI-concentrated at R's optima — a non-prior, non-i.i.d. distribution, with no APT-style proposal correction available), so the gate certifies a *different* object than ships and the AL loop can silently degrade coverage of alternative pre-image modes. **Re-run SBC/TARP on the AL-conditioned re-distilled posterior** (or importance-correct the re-distillation) before each deployment — this is distinct from the SNPE-C proposal-leakage already flagged. MFL never applies even the surrogate-relative standard to itself.

### 14.7 Convergence and OOD limitation-theorems

MFL's round-trip loop converges at best to a stationary point of a non-convex loss satisfiable by a degenerate `R` that ignores multimodality and says nothing about the real machine — the baseline to beat. Our forward training is ERM (no strong guarantee; ensemble disagreement is a *heuristic* epistemic estimate that is known to **under-estimate in far-OOD holes** where members spuriously agree — hence the typicality-AND-disagreement gate, not an upper bound). Our amortized inverse convergence is the flow/score objective's, validated by SBC not a loss value (watch SNPE-C proposal leakage, R̂, ESS). Our active loop is Bayesian optimization / experimental design — **warm-start mandatory** (Kanarik HF-CL), **DoE space-filling start**, qLogNEHVI/constrained-EI/TuRBO in BoTorch; GP-UCB sublinear regret (Srinivas et al. 2010) is *motivating not certifying* (its RKHS-norm assumption is essentially never verifiable); drift via Run-Indexed TV-BO. **Limitation-theorems:** (1) no guarantee off-manifold — the domain-adaptation bound (Ben-David et al. 2010), `ε_T(h) ≤ ε_S(h) + ½·d_{H∆H}(S,T) + λ`, contains the **adaptability term `λ`** (error of the ideal *joint* hypothesis) which is **not estimable without target labels** and can be large off-support; the H∆H-*divergence* is by contrast **bounded (in [0,2]) and estimable** from unlabeled samples — it merely *saturates* toward its maximum off-support. So off-support recipes are *by theorem* unguaranteed → detect and flag, never claim; (2) the inverse search *is* an OOD generator (optimizing a surrogate pushes `x` toward its optimistic regions) → detect via the **two-signal** composite score (typicality AND disagreement; §8.2).

### 14.8 Failure-mode register (each with an automatic detector)

| Failure mode | Detector | Mitigation |
|---|---|---|
| Sparse high-D data | SBC/TARP failure; large `#{σᵢ≈0}`; low effective samples/identifiable dim | reduce to identifiable subspace; cost-aware AL; report sloppy knobs |
| Drift faster than adaptation | online ACI miscoverage; CUSUM on residuals; run-indexed drift test; **alarm when running coverage error trends past α** | conformal-PID/ACI; weighted conformal; TV-BO; sliding-window recalibration |
| Hidden state / latent confounders | residual autocorrelation; unexplained batch variance (nested variance components) | **honest limit: no purely data-driven fix** — add proxy sensors; per-chamber random effects; label predictions "conditional on unobserved state" |
| Infeasible target | posterior mass near `Z*`≈0; conformal best-achievable min-distance > tolerance | report **feasibility gap** + **Pareto-closest achievable** — never a clipped recipe |
| **Surrogate exploitation (HEADLINE)** | gap between point prediction and (i) conformal width, (ii) ensemble disagreement, (iii) physics residual | **conformal acceptance gate `C(x')⊆Z*`** + selection-corrected risk control + pessimistic LCB + constraint-by-construction + independent qualification before deployment |

### 14.9 Certification path (always required)

The top rung is **certified deployment** and it is non-negotiable: no in-distribution coverage, risk bound, or SBC pass substitutes for **independent qualification of accepted recipes on the real tool** before production. Accepted recipes are re-run on an *independent* solver (D7) then hardware; qualification follows a reliability protocol (AEC-Q100-style) we treat as an *open, acknowledged gap*. This closes MFL weakness #1: we do not validate a network by inverting another network on synthetic Gaussian data; the emulator-vs-reality gap is a *measured, qualified* quantity.

### 14.10 What we can and cannot promise (put verbatim in the paper)

**Can guarantee:** (1) distribution-free marginal coverage of the forward model under exchangeability, with named repairs under shift/drift; (2) finite-sample, selection-corrected risk control `P(outcome∉Z*)≤δ` on accepted recipes over an exchangeable calibration set; (3) posterior correctness *tested* by SBC/TARP **relative to the surrogate's generative model** (real-machine correctness is *not* guaranteed — it rests on the prospective hardware metrics, §12.5); (4) well-posedness (stability) of the Bayesian inverse; (5) explicit identifiability geometry per recipe; (6) constraint satisfaction *by construction* for box/mixture/hard-to-change factors.
**Cannot guarantee:** (1) accuracy off the training support (Ben-David — impossible in principle; we flag); (2) coverage when drift outruns online adaptation (we alarm); (3) anything when unobserved confounders make the map non-functional (we mark predictions conditional); (4) that a feasible recipe exists for an arbitrary spec (we quantify the gap); (5) production readiness without independent qualification (never claimed, always required). Every "cannot" has a detector in §14.8.

---

## 15. Program Plan: Phases, Milestones & Go/No-Go Gates

This resolves the "no integrated schedule / critical path / MVP / kill criteria" gap. The layers are mutually dependent (inverse ⇐ forward ⇐ data; EPIG ⇐ inverse proposals), so the **build order breaks the dependency deliberately**: forward first, a cheap per-query inverse to unblock, amortized inverse and EPIG last.

### 15.1 MVP scope (the minimum publishable unit — resist scope creep)

**Paper 1 = one scalar-KPI process end-to-end.** Recommended: **plasma etch on a public level-set/feature-profile simulator** (head-to-head with MFL/BO on their turf) **plus one secured real dataset** (D1). Forward surrogate + inverse engine + one active-learning loop. **Defer to paper 2:** multi-process foundation/transfer *empirical* claim (D8), 1-D/2-D field modalities, full multi-process benchmark suite. The process-agnostic *interface* claim is demonstrated in paper 1 by running the same core on a second process via an adapter (MBE sim), which is cheap.

### 15.2 Phase 0 — MBE physics-simulator bootstrap (starts immediately, parallel to M0)

Concrete tasks using the existing simulator under `c:\Users\Jiaow\Documents\github\MBE sim` (Module B/D): (i) wrap it behind a `ProcessAdapter` with the input/output schema, cost model, and DoE hooks; (ii) generate labelled `recipe→outcome` data + **finite-difference sensitivities** for Sobolev distillation (§6.1); (iii) build the **in-silico stand-in "machine"** with injected pathologies (seasoning, first-wafer offset, metrology noise, a second perturbed "chamber") to exercise drift/hidden-state/transfer (§10) *before* real data; (iv) build the **different-physics reduced-order MBE verifier** required by D7 so a physics-fidelity claim is possible for MBE; (v) validate the *entire* closed loop and **lock hyperparameters** here (§12.5 Phase 0); (vi) build a **coherent-sim-bias detector/ablation on the forward path** — the one sim feeds *four* coupled forward channels (pretraining, MF low-fidelity, GP/BNN prior mean, gray-box backbone), so a single coherent bias (wrong sticking-coefficient regime, missing coupling) contaminates all four in the *same* direction, and at n≈100 the conformal/KOH layer cannot fully correct a coherent prior bias; hold out a real region and verify the sim-anchored gray-box does not *degrade* real held-out error vs a pure-data fit, and if it does, down-weight the sim across all four channels (the per-channel §6.5 stacking cannot see a global bias). Phase 0 is a data generator, an OOD-safe prior, and a rehearsal machine — **never a final claim.**

### 15.3 Milestones and gates

| Milestone | Deliverable | Go/No-Go gate |
|---|---|---|
| **M0 — Secure real data (D1, gated before build)** | A named real `recipe→outcome` dataset (fab/vendor agreement, the author's own MBE campaign, or a public run-to-run log) supporting a temporal + one-tool-held-out split | **GO** if ≥ one process with a real split is secured. **NO-GO** → de-scope the paper to explicitly-labelled in-silico and re-plan (this is the program's #1 risk) |
| **M1 — Calibrated forward on a real temporal split** | ForwardModel (backbone per D3) + shift-robust conformal (D4), evaluated on the real temporal + leave-one-tool-out splits | **GATE (power-aware — ~30 real points give a ~5–6% binomial SE on coverage, so a hard ±2% is decided by noise):** the **primary powered criterion is the *in-silico* version** of this gate (many seeds); the real split is a **directional** check — empirical coverage within the **binomial CI of nominal given `n_cal`** (not a fixed ±2%); leave-one-tool-out epistemic > in-distribution reported with its (wide, single-comparison) CI; CRPS beats a well-tuned GP baseline in-silico. Fail → revisit backbone / features / physics prior |
| **M2 — Inverse beats warm-started BO in-silico** | Per-query pessimistic inverse (§8) + feasibility/abstention, on the in-silico machine | **GATE:** lower in-silico cost-to-target than warm-started qLogNEHVI and re-implemented MFL at matched budget; false-success-on-infeasible ≈ 0; SBC passes. Also decide the human-comparator scope (§12.6) here |
| **M3 — Amortized posterior + closed-loop AL** | NPE flow (D2) re-distilled offline (D6); EPIG/qLogNEHVI blended AL (D5) | **GATE:** amortized proposal matches per-query quality after refinement; AL beats DoE/RSM on in-silico cost-to-target |
| **M4 — Prospective real campaign** | Pre-registered manifest (§12.5 Phase 2); batch-aware real runs | **GATE:** pre-declared in-tolerance hit-rate within pre-declared budget; realized outcomes in pre-declared intervals at nominal rate; cost-to-target beats warm-started BO / HF-CL baseline |
| **M5 — Certification + release** | QualificationGate pass on accepted recipes; reproducibility package | **GATE:** independent-verifier pass (D7); §18 checklist complete; export-control review clear (§17) |

### 15.4 Build-order DAG (resolves the circular-build risk)

`Phase 0 adapter + in-silico machine` → `M1 forward + conformal` → `M2 per-query inverse (unblocks without a generator)` → `M3 amortized generator + EPIG (needs inverse proposals for p*)` → `M4 real campaign` → `M5 certification`. Forward is on the critical path; the per-query inverse deliberately precedes the amortized generator so the AL loop and evaluation are not blocked on generator training.

### 15.5 Indicative schedule, staffing & the latency that actually dominates

Durations are *indicative and gate-conditioned* (the §15.3 gates govern, not the calendar); the point is to expose the real critical-path driver, which is **not** compute or model training.

- **Real-tool lot-turnaround latency dominates the hardware campaign.** A single wafer-lot turn (queue → run → metrology → data back) is typically **days to a few weeks** on a shared production/research tool. At ~4 recipes/batch, a ~100-run campaign is **~25 batches ⇒ many weeks–to–months of wall-clock**, even though the model work *between* batches is hours. This is precisely *why* the online/drift machinery (§10; RI-TVBO; online conformal) is mandatory, not optional — the process genuinely drifts across a campaign that long.
- **Indicative durations** (≈2–3 FTE: 1 ML lead, 1 research eng, ~0.5 process/fab liaison): Phase 0 bootstrap + M1 forward on real logged data ≈ **2–3 months**; M2 in-silico inverse + AL ≈ **1–2 months** (parallelizable with Phase 0); M3 amortized generator ≈ **~1 month**; **M4 prospective hardware campaign ≈ 3–6+ months, latency-bound** (the long pole); M5 certification/release ≈ **~1 month + external qualification lead time**.
- **Front-load everything not latency-bound** (Phase 0, M1–M3 all runnable before/while M0 data is secured) so the hardware campaign starts the moment tool access lands. **Staffing the campaign realistically (a "0.5 liaison" is a *coordinator*, not execution labor):** budget **fab-operator/technician time** to run ~25 batches; **metrology-technician time on a *separate* tool queue** — a *second* serial latency the growth-tool turn model omits; a **statistician / reliability engineer** for the survival/RMST/censoring/pre-registration statistics and Cpk qualification (or an explicit statement that the ML lead owns it); and **IRB lead time** for the human comparator (§12.6; can be months — start at M2, kept off the critical path). Without guaranteed tool time the M4 schedule is unbounded (tie to R1).

### 15.6 Execution prerequisites — the unscoped critical-path infrastructure (E1–E5)

The plan's polish hides that M0→M3 is gated by unglamorous engineering that is **not** a few one-line tasks. Scope these explicitly.

- **E1 — Ingestion / ETL behind the clean `RunRecord` (critical; blocks M0→M1).** §3.5 defines the *target* schema but not the path to produce it. Recipe setpoints live in a tool **MES/historian**; metrology (ellipsometry/XRD/thermawave/wafer maps/spectra) in *separate* systems; they must be **joined on lot/wafer/slot IDs across timestamp skew**, then reduced to KPIs by a **versioned reduction contract**. Build: the join/entity-resolution step; the metrology→KPI reduction with provenance; **ingest-time handling of rework/scrap/aborted/partial-metrology rows** (censoring must be *produced here*, not only modeled at §9.7); and a **data-readiness checklist** (exactly which fields the M0 source must contain, at what granularity, before M1 can start).
- **E2 — The existing MBE sim does NOT match the Phase-0 interface (critical — verified against the repo).** `optimize.py`'s knobs are machine **build/geometry** (`T_heater, heater_radius, gap, source_offset, source_height, aim_offset`; the code comment states "All knobs are machine-settable quantities") and `sensitivity()` returns `d(scalar-uniformity-objective)/d(knob)` — **not** the multi-KPI `d(outcome_vector)/d(recipe_vector)` Jacobian §6.1 Sobolev distillation needs (that harness is built from scratch). There is **no unified "recipe vector"**: per-run process knobs (cell fluxes/shutters/growth-rate/substrate-T, in `Layer`/`CellState`) live separately from machine config, and **recipe vs fixed-config vs hidden-state is undefined**. Output is a nested `snapshot()` dict, not an `OutcomeRecord` → a translation / DVC-array-ref layer is owed. The sim ships **two fidelities** (a fast mean-field/Arrhenius *regime* path and the slow seeded **kMC `ZoneEnsemble`**); the plan must **declare which is "the machine" vs "the physics prior."** Crucially the reduced-order path **shares physics lineage with the kMC**, so it does **NOT** satisfy the D7/R10 *independent*-verifier requirement — a genuinely different-physics ROM (or the real tool) is still owed.
- **E3 — The in-silico stand-in "machine" has no pathology hooks (major).** Everything before M4 — drift / hidden-state / leave-one-tool-out validation *and* the §12.1 non-circular exploitation figure — rests on an in-silico machine with injected seasoning / first-wafer-offset / metrology-noise / second-chamber. The kMC is **seeded and effectively memoryless** (no `time-since-clean` state object), there is **no `tool_id` / second-chamber parameter-perturbation mechanism**, and **no metrology-noise layer**. This injection substrate is a named dependency with no code today; it must be *built* first.
- **E4 — Program budget (money AND compute) (major).** "Cost is not an issue" is a posture, not a number. Name both: a **wafer/tool budget** (~\$50–150k for a ~100-run campaign at Kanarik defaults, *before* metrology/tool time — R1 covers data *existence*, not the money to generate it) and a **provisioned compute budget** (in-silico ≥50-seed ECT / ≥10-seed survival stats × K=10 ensembles × SBI + nightly re-distillation × EPIG nested MC × sweeps ⇒ a large GPU footprint — estimate GPU-hours and secure the cluster).
- **E5 — New-process onboarding runbook + adapter conformance gate (major).** "A new process is a plug-in via `ProcessAdapter`" has no operational path today. Ship: an **adapter-authoring runbook**, a **minimum-viable-adapter template**, a **schema-elicitation intake** (variables, bounds, change-cost classes, and the genuine *mixture-vs-independent-MFC* tags), and an **adapter conformance harness** (declared constraints consistent with data; ILR/box/mixture transforms round-trip; cost model complete with the fixed/variable split; DoE hooks emit *feasible* seeds; modality tags match the head). Without it the generality claim is unenforced and Paper 1's second-process demo is itself a research project.

---

## 16. Risk Register (risk → likelihood/impact → mitigation → detector → owner)

| # | Risk | L×I | Mitigation | Detector | Owner |
|---|---|---|---|---|---|
| R1 | **No real-data source** — whole thesis unfunded (D1) | High×Critical | M0 gate before build; enumerate real *recipe→outcome* paths (fab/vendor data-sharing agreement; the author's own tool campaign; a university self-driving-lab log). **No public recipe→ranged-outcome dataset exists — SECOM is sensor→pass/fail, usable only for a drift/anomaly sub-experiment, not inversion.** De-scope to labelled in-silico if all fail | M0 gate fails by its deadline | PI / lead |
| R2 | **Real campaign too small for any claim** (~100 runs) | Med×High | Budget allocation (§9.6); noise floor from sim/history; online ACI + jackknife+ small-n conformal (D4); report-all-trajectories with CIs, state n | forward calibration set < a few dozen; survival curves with wide CIs | ML lead |
| R3 | **Surrogate exploitation resurfaces on real data** | Med×Critical | Defense-in-depth (§8, §13.2); oracle refit every batch; conformal acceptance gate | widening realized-vs-predicted gap; optimism-gap plot > noise floor; reward-hacking canary test | ML lead |
| R4 | **Negative multi-process transfer sinks the foundation story** | Med×Med | D8 — defer empirical transfer to paper 2; keep only the interface claim in paper 1; PCGrad / task-affinity guards | shared-trunk worse than isolated on a process's forward split | research |
| R5 | **IP / export control blocks publication or release** (§17) | Med×High | Publish methods + models on public simulators fully open; real-data results as aggregate metrics only + synthetic stand-in; export-control review checkpoint before release | legal review flags EAR/ITAR relevance | PI + legal |
| R6 | **No fab partner for the human-expert comparator** | Med×Med | Scope or downgrade at M2 (§12.6); default headline = "beats warm-started BO / HF-CL" | M2 review: no engineers available | PI |
| R7 | **Drift outruns adaptation** in the real campaign | Low×High | Run-Indexed TV-BO; ACI/conformal-PID; sliding-window recalibration; alarm + defer | rolling ACI coverage trends past α; residual CUSUM | ML lead |
| R8 | **Compute posture masks an inner-loop cost blowup** | Low×Med | Ensemble distillation / SNGP single-model inner loop (§5.7); re-validate on full ensemble | inverse-loop wall-clock per proposal | eng |
| R9 | **Scope creep** (1-D/2-D, foundation model, 4-process suite in paper 1) | High×Med | MVP scope fixed at §15.1; defer explicitly; gate at M2 | milestone slip; DAG deviations | PI |
| R10 | **MBE verifier circularity** invalidates the physics-fidelity claim (D7) | Med×Med | Build the different-physics ROM verifier in Phase 0, or claim MBE fidelity only against the real tool | `independent_verifier == physics_prior` in CI | eng |

Tie each risk to the go/no-go gate that would surface it (§15.3).

---

## 17. Data Governance, IP & Export Control

Largely unowned across the source sections; make it an explicit checkpoint, because the open-release commitments (§13, §18) may be legally impossible for the very real data that makes the paper credible.

- **Trade-secret real recipes.** A model trained on fab recipes may be un-releasable. **Policy:** publish *methods + code + models trained on public simulators* fully open; report real-data results as **aggregate metrics only**, and release a **synthetic stand-in** generated by the versioned MBE sim, clearly labelled `physics_sim`. Data-sharing/NDA terms are agreed at M0 and recorded.
- **Export control.** Semiconductor process technology (especially MBE for RF/III-V and advanced nodes) is potentially **export-controlled (EAR/ITAR)**; publishing recipe-generation methods trained on such data has real legal constraints. **Add an export-control review checkpoint before any release (M5 gate).**
- **License audit.** BoTorch (MIT), `sbi`, DiffTaichi, GPyTorch, Lightning, Hydra, DVC — audit licenses before redistribution.
- **On-prem option.** If data cannot leave the fab, MLflow self-hosted + on-prem GitLab CI (§13.1 fallbacks), swapped via the `Logger` shim so no SDK is called directly.
- **Trade-secret egress guard — the default stack leaks recipe *values* (E6, major).** The default logger is **Weights & Biases cloud** (§13.1); normal training pushes recipe *values* — not just aggregate metrics — into configs/artifacts/sweeps, the exact trade-secret setpoints an NDA forbids leaving the fab. **Make on-prem / air-gapped the DEFAULT for any run touching real recipe values**; add an **enforced data-classification / no-egress guard** (a `Provenance.source==real_tool` tag *blocks* cloud logging of recipe values, checked in CI, not left to per-run configuration); a **secrets story** for historian/MES credentials and DVC-remote keys (a vault, not env files); and an explicit statement of **where compute runs** (cloud vs on-prem vs air-gapped) per data class. A single default-configured run must not be able to breach the M0 data agreement.
- **Dual-use / ethics.** Semiconductor recipe generation touches export-controlled and defense-relevant devices; include a one-line dual-use / broader-impacts statement (NeurIPS requires it anyway): the system accelerates legitimate process development and includes a certification gate; it is not a substitute for regulatory qualification and should not be used to circumvent export controls.

---

## 18. Reproducibility / Peer-Review Checklist (attach to submission)

1. **One command reproduces every table/figure:** `dvc repro && rig eval +experiment=paper`.
2. `uv.lock` + Docker digest pin the exact environment; container hash logged per run.
3. **Replication scoped honestly to cost:** *simulation/fixed-data* headline numbers use ≥10 seeds with bootstrap CIs; *real-hardware* cost-to-target reports **every available trajectory** with CIs and states `n` explicitly (we do not pretend 10 real re-runs per condition exist). Both beat MFL's single runs.
4. **Splits published and versioned:** random + OOD (leave-tool-out / scaffold) + temporal — never just random; normalization/conformal stats fit on the training window only; headline claims on `source==real_tool` only; the **sim-to-real gap** reported as a number.
5. Compute + wall-clock + hardware per experiment (W&B).
6. **Baselines with their proper priors**, identical budgets and cost accounting: warm-started GP-LogEI/qLogNEHVI/SCBO/TuRBO, CMA-ES, cINN/NPE, Sobol/RSM/Taguchi, and MFL reimplemented. Kanarik's <5% from-scratch finding respected — we warm-start and never claim from-scratch suffices.
7. **UQ reported:** interval score, PICP/MPIW, CRPS, NLL, quantile-calibration error / PIT, conformal coverage (marginal + per-tool + rolling); calibration is a gated regression test.
8. **Physics-fidelity benchmark** (fraction passing the *independent* verifier, violation magnitude, distance-to-feasible) — not FID/likelihood.
9. **Cost-to-target curves** (KM survival + RMST) as a first-class metric.
10. **SBC/TARP** posterior-correctness reports; feasibility/abstention calibration (false-success-on-infeasible ≈ 0).
11. Full config, code SHA, data hash, model artifact ID embedded in every checkpoint and run; **pre-registration manifest hash** for the prospective campaign.
12. **Model cards + limitations** (OOD/drift failure modes); the §14.10 "can/cannot promise" statement verbatim.
13. **A "negative results" subsection** (targets we failed to hit, coverage that degraded under drift, infeasible targets we abstained on) — its presence is itself a rigor signal and directly answers MFL's credibility gap.
14. NeurIPS-D&B datasheet + reproducibility-checklist compliance for the released benchmark; **dual-use / broader-impacts** statement (§17).

---

## 19. References (author year — topic)

**MFL & framing**
- MFL, arXiv:2505.16060 — few-shot test-time inverse recipe generation via round-trip loss (the work we improve on).
- Kanarik et al. 2023 (*Nature*, Lam Research) — human-machine collaboration; the little-data problem; cost-to-target; Human-First/Computer-Last; from-scratch BO beats experts in <5% of trajectories.
- Banad & Sharif 2026 — physics-informed generative AI; the coupling ladder; inverse-optimizes-the-learned-manifold as the headline failure; physics-fidelity benchmark over FID; certified deployment.
- Han, Taheri & Ko 2025 — PINNs for semiconductor film deposition; gray-box exemplars; PINN loss-balancing and adaptive collocation.
- Chen & Chen 2025 — experimental designs in wafer manufacturing (factorial, RSM/CCD/BBD, Taguchi, mixture, split-plot, nested, Plackett-Burman, Gage R&R).

**Uncertainty & calibration**
- Lakshminarayanan et al. 2017 — deep ensembles. Ovadia et al. 2019 — UQ under distribution shift. Nix & Weigend 1994; Kendall & Gal 2017 — heteroscedastic heads. Seitzer et al. 2022 — β-NLL. Gneiting & Raftery 2007; Gneiting et al. 2005 — CRPS / proper scores. Amini et al. 2020; Meinert et al. 2023; Bengs et al. 2022 — evidential regression and its pathologies. Liu et al. 2020 — SNGP. Daxberger et al. 2021 — Laplace Redux. Gal & Ghahramani 2016 — MC-dropout. van Amersfoort et al. 2020 — DUQ. Ober et al. 2021 — DKL feature collapse. Wilson et al. 2016 — DKL. Garnelo et al. 2018; Kim et al. 2019 — (Attentive) Neural Processes.
- Romano et al. 2019 — CQR. Kuleshov et al. 2018 — calibrated regression. Angelopoulos & Bates 2021/2023 — conformal. Gibbs & Candès 2021 — Adaptive Conformal Inference. Zaffran et al. 2022 — AgACI. Angelopoulos, Candès & Tibshirani 2023 — conformal-PID (NeurIPS 2023); Angelopoulos, Barber & Bates 2024 — online CP with decaying step sizes. Barber et al. 2021 — jackknife+. Barber et al. 2023 — nonexchangeable conformal. Tibshirani et al. 2019 — weighted conformal. Messoudi et al. 2021 — copula conformal. Feldman et al. 2023 — multi-output CQR. Diquigiovanni et al. 2021 — functional conformal. Angelopoulos, Bates, Candès, Jordan & Lei 2021 — Learn-then-Test. Bates et al. 2021 — RCPS.

**Physics hybrid & sim-to-real**
- Kennedy & O'Hagan 2001 — model calibration/discrepancy. Brynjarsdóttir & O'Hagan 2014 — discrepancy identifiability. Czarnecki et al. 2017 — Sobolev training. Li et al. 2020 — FNO. Lu et al. 2021 — DeepONet. Wang, Wang & Perdikaris 2021 — physics-informed DeepONet. Maddu et al. 2022; McClenny & Braga-Neto 2023; Wang et al. 2021; Liu & Wang 2021 — PINN loss-balancing. Nabian et al. 2021; Tang et al. 2023 (DAS) — adaptive collocation. Nolte et al. 2023 (LMN); Runje & Shankaranarayana 2023; You et al. 2017 (Deep Lattice) — monotone nets. Miyato et al. 2018 — spectral norm. Anil et al. 2019 — GroupSort/orthogonal. Meng & Karniadakis 2020; Perdikaris et al. 2017 (NARGP) — multi-fidelity. Le Gratiet 2014 — auto-regressive MF-GP. Tejero-Cantero et al. 2020 (`sbi`); Greenberg et al. 2019 (SNPE-C/APT); Ramos et al. 2019 (BayesSim); Chebotar et al. 2019 (SimOpt). Talts et al. 2018 (SBC); Hermans et al. 2022; Lemos et al. 2023 (TARP). Vehtari et al. 2017 (PSIS-LOO); Yao et al. 2018 (Bayesian stacking).

**Inverse / generative / SBI**
- Ardizzone et al. 2019 — invertible networks for inverse problems (cINN). Zhu et al. 2017 — cycle-consistency. Chu et al. 2017 — CycleGAN hides information. Sohn et al. 2015 — cVAE. Durkan et al. 2019 — neural spline flows. Papamakarios & Murray 2016; Cranmer, Brehmer & Louppe 2020 — SBI. Wildberger et al. 2023 — FMPE. Lipman et al. 2023 — flow matching. Gloeckler et al. 2024 — Simformer. Geffner et al. 2023 — score-based posterior. Hermans et al. 2020 (NRE); Miller et al. 2021 (TMNRE); Papamakarios et al. 2019 (NLE); Deistler et al. 2022 (TSNPE). Liu & Wang 2016 — SVGD. Chung et al. 2023 — DPS. Bengio et al. 2021/2023 — GFlowNet.

**Robust / constrained / offline MBO**
- Trabucco et al. 2021 (COMs), 2022 (Design-Bench); Fannjiang & Listgarten 2020 (autofocused oracles); Brookes et al. 2019 (CbAS); Kumar & Levine 2019. Kidambi et al. 2020 (MOReL); Yu et al. 2020 (MOPO); Kumar et al. 2020 (CQL). Sagawa et al. 2020 (group-DRO). D'Angelo & Fortuin 2021 (repulsive ensembles). Nalisnick et al. 2019 — deep generative models over-assign OOD likelihood; typicality test. Lee et al. 2018 — Mahalanobis OOD. Agrawal et al. 2019 (cvxpylayers); Amos & Kolter 2017 (OptNet); Donti et al. 2021 (DC3). Calafiore & Campi 2006 — scenario approach. Sui et al. 2015 (SafeOpt); Turchetta et al. 2019 (GoOSE).

**Active learning / BO / DoE**
- Houlsby et al. 2011 (BALD); Kirsch et al. 2019 (BatchBALD); Bickford Smith et al. 2023 (EPIG); Bect et al. 2012–2014; Gotovos et al. 2013 (SUR/level-set). Ash et al. 2020 (BADGE); Kulesza & Taskar 2012 (DPP). Daulton et al. 2021 (qNEHVI); Ament et al. 2023 (LogEI/qLogNEHVI); Eriksson et al. 2019 (TuRBO); Eriksson & Jankowiak 2021 (SAASBO); Eriksson & Poloczek 2021 (SCBO); Gardner et al. 2014 (cEI). Snoek et al. 2012 (EIpu); Lee et al. 2020 (CArBO); Frazier et al. 2008 (KG); Wu et al. 2019 (taKG, multi-fidelity KG; UAI 2019); Wu & Frazier 2016 (parallel/batch KG); Takeno et al. 2020 (MF-MES); Kandasamy et al. 2017 (BOCA), 2018 (async). Srinivas et al. 2010 (GP-UCB regret). Cho, Shao & Mesbah 2024 (Run-Indexed Time-Varying BO). Balandat et al. 2020 (BoTorch). Owen 2003 (scrambled Sobol); McKay 1979; Morris & Mitchell 1995 (LHS). Feurer et al. 2018 (RGPE); Wistuba & Grabocka 2021 (FSBO); Perrone et al. 2018 (ABLR).

**Drift / transfer / hidden state**
- Bifet & Gavaldà 2007 (ADWIN); Gama et al. 2004 (DDM/EDDM); Gretton et al. 2012 (MMD); Lopez-Paz & Oquab 2017 (C2ST). Li et al. 2018 (L2-SP); Kirkpatrick et al. 2017 (EWC). Bui, Nguyen & Turner 2017 (streaming sparse GP). Rangapuram et al. 2018; Krishnan, Shalit & Sontag 2017 (deep state-space). Perez et al. 2018 (FiLM); Houlsby et al. 2019 (adapters); Hu et al. 2021 (LoRA); Sun & Saenko 2016 (Deep CORAL); Bonilla et al. 2008 (multi-task GP). Fifty et al. 2021 (task affinity); Yu et al. 2020 (PCGrad); Chen et al. 2018 (GradNorm); Finn et al. 2017 (MAML); Nichol & Schulman 2018 (Reptile). Kendall et al. 2018 (multi-task uncertainty weighting).

**Theory / well-posedness / identifiability**
- Stuart 2010; Dashti & Stuart 2017 — Bayesian inverse problems, well-posedness. Gutenkunst et al. 2007; Transtrum et al. 2015 — sloppy models. Raue et al. 2009 — profile likelihood. Ben-David et al. 2010 — domain-adaptation bound. Malinin et al. 2020 — ensemble distribution distillation.

**Evaluation / statistics / interpretability**
- Demšar 2006; Benavoli et al. 2016, 2017 — multi-method comparison / CD diagrams. Säilynoja et al. 2022 — ECDF SBC bands. Friedman & Dieng 2023 — Vendi Score. Lundberg & Lee 2017 — SHAP. **S. Zheng et al. 2023 — LithoBench** (Su Zheng, Yang, Zhu, Yu, Wong, NeurIPS D&B; distinct from *C.* Zheng et al. 2023, Neural Lithography). Häse et al. (Olympus); Felton et al. (Summit) — self-driving-lab benchmarks. MacLeod et al. 2020 — self-driving lab.

---

## 20. SOTA 2024–2026 Research Update (refines the decisions above)

This section folds in a **dedicated 2024–2026 web-research pass** (8 topic scouts + adversarial consolidation) run after the sections above were written. It was cross-checked; **single-preprint or contested claims are flagged `[UNCERTAIN]`** and must not carry a load-bearing decision. Where an item here updates a choice above it supersedes on *method currency*; the **design decisions D1–D9 and the invariants (§2) stand**. Fully-sourced detail with URLs: `memory/implementation-plan_sota_2024_2026.md`.

### 20.1 Forward surrogate / UQ — refines §5, D3
- **Small-n backbone:** ADD **TabPFN v2** (Hollmann et al., *Nature* 2025) as a strong small-n *predictor* alongside the GP — but **conformal-wrap it and decompose epistemic explicitly; do NOT trust its native predictive interval as epistemic UQ** `[UNCERTAIN as an epistemic source]`. Keep **last-layer Laplace** (`laplace-torch`) as a cheap composable epistemic layer.
- **The "~300 runs/process" crossover (D3, §2.3 table, §5.2, M1) is a soft guideline, not a hard number `[UNCERTAIN — no benchmark pins a crossover]`.** Choose by *calibration behavior on the real split*, not a magic threshold: stay on GP / conformal-TabPFN while a heteroscedastic SNGP ensemble overfits (unreliable epistemic); switch to the ensemble+SNGP when its leave-one-tool-out epistemic exceeds in-distribution *and* its calibration holds — or for 1-D/2-D outputs where GPs scale poorly.
- **Evidential DL: drop is now double-sourced** (ICML 2024 + NeurIPS 2024 critiques) — keep it out of the epistemic/feasibility path (OOD-screen only). **Unconstrained DKL: demoted** (contested; feature-collapse) — make **plain/scalable GP the default**, DKL only under spectral-norm/lengthscale constraints (the plan's existing guard is correct).
- **Structured outputs:** promote **neural-operator (FNO/DeepONet) and (Conv)CNP heads to *primary*** for 1-D/2-D (not only when a PDE prior exists); plain tabular MLP/XGBoost is a *weak* baseline for structured outputs. Keep POD/PCA + GP/DKL as the low-data fallback.
- **New certification/OOD metric:** add the **distance–uncertainty correlation** (Pearson of predicted uncertainty vs. train–test distance — the "Distance-Aware Coefficient" of arXiv:2512.08499; NB "PG-SNGP" is not an established citable acronym, base method = SNGP, Liu et al. 2020) to §5.8/§5.9 and the certification harness — a direct check that epistemic tracks distance.

### 20.2 Conformal — refines §5.6, D4
- **cross-conformal / CV+ / jackknife+ is the small-n default** (Gasparin & Ramdas 2025 — reclaims calibration data at n≈25–100), **superseding naïve split conformal** at small n. Keep **CQR** as the score; add **localized CQR** if width-adaptivity matters.
- **Conformal PID / decaying-step online CP is the online/drift *endpoint*** (supersedes *bare* ACI, which becomes a component). Wrap CQR inside it.
- **Clustered / shift-class conditional CP** instead of raw Mondrian when per-tool groups have <~10 runs.
- **Standardized-residual / copula multi-output conformal** for profiles/fields (adds to the plan's copula/functional bands); per-pixel marginal intervals demoted.

### 20.3 Amortized inverse / SBI — refines §14.3, D2
- Keep **NPE-NSF** as the reproducible baseline; **FMPE is the default estimator** where its scalability helps — but state honestly its edge is **muted at ~100s of runs** (scalability, not accuracy). **Simformer** for ranged/partial targets is confirmed.
- ADD a **misspecification-robustness layer, mandatory for sim↔real: RNPE** (Ward et al., NeurIPS 2022) is the defensible primary; **FMPE-calibration / PRNPE** are the `[UNCERTAIN]` leading edge, not load-bearing.
- **Calibration gate** (§14.6): SBC + **TARP** are correct; add **L-C2ST** (local classifier-two-sample) and **Posterior-SBC** (Säilynoja et al. 2025) to the triad.

### 20.4 Diffusion inverse & hard-constraint generation — refines §8.3, §14.4
- Where a diffusion/score generator is used for a *calibrated* one-to-many posterior, **replace bare DPS with twisted-SMC (TDS / MCGDiff / DDSMC)** — asymptotically exact conditional sampling (**TDS** is the general/nonlinear result; **MCGDiff** is proven specifically for *linear-Gaussian* inverse problems, not arbitrary nonlinear forward models); DPS remains a fast approximate fallback.
- **Hard feasibility by construction (the biggest 2024–26 shift):** **mirror / reflected generation** — **Mirror Diffusion** (NeurIPS 2023; *analytical* mirror maps, **convex** feasible sets) and **Reflected Flow Matching** (ICML 2024; reflected/box domains) as the mature convex backbone, plus **Neural Approximate Mirror Maps (NAMM, ICLR 2025)** which *learns* approximate mirror maps for **general, possibly non-convex / coupled** constraints (approximate satisfaction, needs a bespoke trained map); and **PCFM-style projection-at-inference** for arbitrary nonlinear/coupled constraints with *exact, zero-shot* satisfaction on a pretrained model `[UNCERTAIN — adopt the *pattern*, cite the mature mirror/NAMM line]`. This *strengthens* §8.3's constraint-by-construction for the generative branch; penalty/soft enforcement is demoted to fallback. **Chance-constrained FM** for probabilistic spec limits `[UNCERTAIN]`.
- **Framing (state explicitly):** SBI (NPE/FMPE/Simformer) *amortizes* the target→recipe posterior; diffusion/FM projection + simulator feedback *enforces* hard feasibility — **one complementary stack, not competitors.** **Data-regime honesty:** standalone diffusion-inverse wins in the literature use **10k+ sims**; at ~100s real runs, pair generative inverse with GP/physics priors — do not expect it to generalize alone `[UNCERTAIN for our regime]`.

### 20.5 BO / active learning / self-driving lab — refines §9, §11, §2.3, §12.3
- **qLogNEHVI / qLogEI everywhere** — confirmed (already adopted; now the BoTorch/Ax default).
- **BO default GP uses a √D log-normal length-scale prior** (Hvarfner et al., ICML 2024) — this **demotes SAASBO from a default to a sparsity-only tool** and makes **TuRBO a local fallback rather than the high-dimensional default**. **SCBO is *not* demoted** — Hvarfner's result concerns *unconstrained* high-D BO and says nothing about constrained BO; **SCBO remains the primary for *unknown/black-box output constraints***, which the plan still needs. Update the §9.8 / §12.3 *unconstrained* high-D framing accordingly.
- **Adopt Atlas (BoTorch-based; Hickman et al., *Digital Discovery* 2025) as the primary closed-loop SDL planner** over a hand-rolled qLogNEHVI/TuRBO stack — it provides multi-objective + multi-fidelity + **unknown-constraint feasibility** + **meta-learning warm-starts from prior campaigns**, directly serving the ~100-run regime; keep qLogNEHVI/EPIG as acquisitions *within* it. **Olympus/Summit** stay as the cheap pre-deployment benchmark (already in the plan).
- **Multi-fidelity:** add **CAGES cost-per-information** acquisition (Tang & Paulson 2024, arXiv:2405.07760 — *local* multi-fidelity BO; the cost-aware *stopping* rule stays §11.1's multi-fidelity-KG logic, it is not part of CAGES) (augments §9.4 / §11.1).
- **A-Lab validation-oracle lesson:** automated characterization has produced false "novel" claims — **log provenance, flag OOD proposals, require confirmation runs before declaring a target hit** (reinforces §9.7 / §11.4; the plan's "physical wafer is the final arbiter" is exactly right).
- **LLM agents = orchestration / recipe-I/O only, never the numerical optimizer** (RAG / fine-tuned local models to mitigate hallucination).
- **Evaluation:** add a **cost-to-target, fixed-budget, distribution-shift campaign benchmark** (MADE-style *principle*; the specific benchmark is `[UNCERTAIN]`) reporting sample-efficiency curves vs. random/BO — augments §12.

### 20.6 Physics-hybrid & foundation/transfer — refines §6, §10.4, D8
- **Gray-box differentiable-forward-with-learnable-physics is the *default* surrogate wherever a differentiable stub exists** (Neural Lithography, SIGGRAPH Asia 2023; DiffTaichi) — promote §6.1's simulator-in-the-loop from a reserve rung to the default *when the stub exists*; it supersedes pure black-box heteroscedastic surrogates for that process.
- **Foundation / transfer:** **no validated cross-process shared-trunk model exists at ~100s of runs `[UNCERTAIN — do not promise one]`** — this *confirms* D8: defer the empirical transfer claim, keep only the interface claim + per-process adapters/FiLM. Confine LLMs to orchestration.

### 20.7 Baselines & datasets — refines §12.3, §12.7, M0/R1
- **MFL** remains the inverse target-to-beat; **Kanarik BO** the real-fab baseline; **Run-Indexed TV-BO** the drift baseline (all present). Add **GUIDe / Range-Aware BO** as generative-UQ-inverse baselines-of-record `[both verified to exist: GUIDe arXiv:2509.05641; Range-Aware BO arXiv:2606.11574 — still single preprints, keep them baselines not load-bearing]`.
- **Honest correction on the "public real-data floor": a genuinely public *recipe→ranged-outcome* dataset essentially does not exist — which is *why* D1/M0 is the program's #1 risk, not a solved problem.** **SECOM** (UCI — 1,567 rows of ~590 in-line *sensor* signals → a *binary pass/fail* label) has **no recipe setpoints and no ranged KPI**, so it **cannot** serve recipe inversion; scope it only to a **drift / anomaly / virtual-metrology sub-experiment**. A "Kanarik-style etch" set is *simulator*-based (the real Lam data is proprietary) — an in-silico bench, not real data. The realistic real-data paths remain D1's (fab/vendor agreement or the author's own tool campaign). **Build your own cross-process I/O physics-fidelity certification set** (LithoBench/MaskOpt template) — none exists; do not assume external coverage.

### 20.8 Do NOT over-claim (single-preprint / not-yet-mature — cite as *direction*, gate real decisions on older, multiply-sourced results)
Cross-process foundation / shared-trunk at ~100s runs; TabPFN-as-epistemic-UQ; foundation-model BO (GIT-BO); the specific robust-FMPE calibration papers (PRNPE, FMPE-calibration); standalone diffusion-inverse in the little-data regime; and the newest single-preprint benchmarks (MADE) and generative-inverse methods (GUIDe, Range-Aware BO, PCFM, chance-constrained FM).

### 20.9 New references (2024–2026)
- Hollmann et al. 2025 — TabPFN v2 (*Nature*). Ward et al. 2022 — RNPE ("Robust Neural Posterior Estimation and Statistical Model Criticism", NeurIPS 2022). Wu/Cardoso et al. 2023–24 — MCGDiff / TDS / DDSMC (twisted-SMC diffusion posteriors). Liu et al. 2023 — Mirror Diffusion (NeurIPS); Reflected Flow Matching (ICML 2024); Neural Approximate Mirror Maps (ICLR 2025); PCFM 2025 `[UNCERTAIN]`. Hvarfner et al. 2024 — vanilla-BO-in-high-dimensions / √D length-scale prior (ICML). Hickman et al. 2025 — Atlas (*Digital Discovery*). CAGES 2024 — cost-per-information multi-fidelity. Gasparin & Ramdas 2025 — efficient cross-conformal. Säilynoja et al. 2025 — Posterior-SBC (*Stat & Comput*). arXiv:2512.08499 (2025) — "Distance-Aware Coefficient" distance–uncertainty correlation (informally "PG-SNGP"; not an established acronym — the base method is SNGP, Liu et al. 2020). C. Zheng et al. 2023 — Neural Lithography (SIGGRAPH Asia; distinct from S. Zheng et al. LithoBench); Hu et al. 2020 — DiffTaichi. Boelts/Deistler et al. 2025 — sbi reloaded (JOSS).

---

*End of implementation-plan.md. This document is the authoritative integrated master. Provenance & archives live in the Claude project **memory** directory (NOT this repo — the relative `memory/…` names below will not resolve from the repo root): `C:\Users\Jiaow\.claude\projects\c--Users-Jiaow-Documents-github-MBE-sim\memory\` — `implementation-plan_sections_raw.md` (exhaustive per-section detail), `implementation-plan_sota_2024_2026.md` (full sourced SOTA update), `implementation-plan-project.md` (generation record).*
