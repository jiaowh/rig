# BUILD_LOG — append-only journal (newest at bottom; never edit past entries)

## 2026-07-15 — session 1, orchestrator

- Audited `implementation-plan.md` end-to-end (internal consistency + repo grounding + 3 parallel
  citation fact-check agents covering ~48 claims). Verdict: sound; 12 fixes applied
  directly to `implementation-plan.md`. Full report: `docs/audit-2026-07-15.md`.
- Verified E2/E3 against `c:\Users\Jiaow\Documents\github\MBE sim` (optimize.py knobs,
  scalar-objective sensitivity(), kMC-vs-regime fidelities, no pathology hooks).
- Created persistent orchestration state: `CLAUDE.md` (session bootstrap + protocol),
  `docs/BUILD_STATE.md` (source of truth), this log.
- Kicked off WP-A (foundation), then WP-B (MBE adapter + pathology machine) and WP-C
  (GP forward surrogate + conformal) as parallel subagents. Their entries follow.

## 2026-07-15 — WP-A agent

Built the process-agnostic foundation package (implementation-plan §3, §8.3, §13.1–13.3).

**Files created:**
- `pyproject.toml` — PEP 621 + hatchling, package `rig`, py>=3.12, src layout;
  runtime deps pydantic>=2/pint/numpy/scipy; `[dev]` extra pytest/hypothesis/ruff/
  import-linter/pandera; empty `rig.adapters` entry-point group; ruff config;
  `[tool.importlinter]` forbidden contract `rig -/-> rig_adapters` (verified KEPT).
  No torch/botorch (deferred to WP-E).
- `src/rig/__init__.py`, `src/rig_adapters/__init__.py` (empty, WP-B lands here).
- `src/rig/interfaces.py` — canonical `PredictiveDistribution(mean, aleatoric_sigma,
  epistemic_sigma, conformal_set)` (frozen dataclass, exact order test-enforced);
  `ForwardModel` / `InverseSolver` / `QualificationGate` as runtime-checkable
  Protocols; `ProcessAdapter` Protocol + `validate_adapter()` (D7 identity check:
  physics_plugin `is` independent_verifier ⇒ AdapterValidationError);
  `RecipeCandidate`, `Infeasible`, `type InverseResult = list[RecipeCandidate] |
  Infeasible` (tagged union, never a clipped point); typed variables
  (Continuous/Categorical/Compositional + `ChangeCost` enum, MFC-flows-are-not-a-
  simplex documented on CompositionalVariable); `OutputSpec` with modality tag;
  `CostModel(c_batch, c_recipe, batch_size)`; `sobol_seed_design()` helper
  (scipy.stats.qmc scrambled Sobol, seeded-deterministic).
- `src/rig/schema.py` — Pydantic v2 frozen models: `Quantity{magnitude, unit}`
  canonicalized to SI base units at validation (module-level pint registry `ureg`
  with `sccm`/`slm` defined; degC→K, sccm→m³/s verified); `CategoricalValue`
  (level-validated), `Fraction` (distinct type, [0,1]), `ArrayRef{hash, path}`
  placeholder (no DVC dep yet); `RecipeRecord.validate_against(input_schema)`
  compares bounds in SI (cross-unit slm-vs-sccm test passes); modality-tagged
  `OutcomeRecord` (curve_1d/field_2d must be ArrayRef); `Provenance.source`
  Literal{"physics_sim","real_tool"}; `RunRecord` with run_id/process_id/tool_id/
  timestamp.
- `src/rig/registry.py` — discovery ONLY via `importlib.metadata.entry_points(
  group="rig.adapters")`; `list_adapters()`, `get_adapter(process_id)` (runs
  `validate_adapter` incl. D7 on load), `register_adapter_for_testing()` +
  `clear_test_registry()`.
- `src/rig/constraints.py` — declarative `ConstraintSet(box, simplex, linear,
  monotone, change_cost)` + pointwise `validate(x) -> list[str]` /
  `is_satisfied(x)`; monotone declarations are consumed by the surrogate, not
  pointwise-checked (documented).
- `src/rig/transforms.py` — `BoxTransform` (lo+(hi−lo)·sigmoid(u), inputs clipped
  ±40), `SimplexTransform`, `RecipeTransform` (u-vector ↔ typed recipe dict;
  compositional keys flattened as `"<var>.<component>"`; categoricals rejected —
  they are conditioning per §8.3).
- `tests/` — 42 tests: Hypothesis property tests (box in-bounds ∀u, simplex
  non-neg/sum-to-1 atol 1e-9, u→x→u′→x″ round-trip idempotence), schema
  canonicalization/rejection tests, canonical-field-order test, registry tests
  (testing hook + monkeypatched fake entry point + static no-rig_adapters-import
  grep over src/rig), D7 same-object-fails test, constraints checker tests.
- `.github/workflows/ci.yml` — ruff + lint-imports + pytest, {ubuntu, windows} ×
  py3.12 matrix.

**Decisions:**
- Simplex parameterization: **ALR / fixed-gauge softmax** (x = softmax([u, 0]),
  inverse u_i = log(x_i/x_K)) chosen over ILR — exact bijection on K−1 dims,
  closed-form inverse; ILR's orthonormal basis only matters for compositional
  statistics, not for an optimizer's reparameterization. Documented in
  transforms.py docstring.
- Interfaces are `typing.Protocol` (structural), not ABCs — adapters need zero
  imports from rig base classes to conform; D7 enforced by a free function
  `validate_adapter()` that `registry.get_adapter()` always calls.
- `InverseResult` is `list[RecipeCandidate] | Infeasible` via PEP 695 `type`
  alias (ruff UP040 forces the `type` keyword on py3.12).
- Recipe dict flattening convention: compositional components appear as
  `"<variable>.<component>"` keys in both `RecipeRecord.values` and
  `RecipeTransform` dicts. WP-B/WP-D must follow this.
- Entry points must resolve to a zero-arg (or kwargs-only) **factory**
  returning a ProcessAdapter instance.

**Gotchas:**
- pint has no `sccm`/`slm` by default — defined in `rig.schema` on the shared
  module registry (`sccm = cm**3/min`); always use `rig.schema.ureg`, never a
  second `UnitRegistry` (pint quantities from different registries don't mix).
- Windows Store Python: console scripts (`lint-imports`, `ruff`) are not on
  PATH — invoke via `python -m ruff` and
  `python -c "from importlinter.cli import lint_imports; lint_imports()"`.
- `Quantity` is frozen; SI canonicalization uses `object.__setattr__` inside
  the model_validator.

**Test summary (verbatim):** `42 passed in 3.41s`
Also verified: `python -m ruff check src tests` → "All checks passed!";
import-linter → "Contracts: 1 kept, 0 broken."

## 2026-07-15 — WP-C agent

**Scope:** forward surrogate v0 (D3 little-data backbone) + D4 conformal layer
+ §5.8 UQ metrics. numpy/scipy only (torch stack stays WP-E).

**Files added:**
- `src/rig/forward/gp.py` — `GPForwardModel`: one exact GP per output,
  Matérn-5/2 + ARD, hyperparams (log ℓ_1..d, log σ_f², log σ_n²) by exact
  NLML minimization with ANALYTIC gradients (L-BFGS-B, 5 seeded restarts:
  1 deterministic start at unit-ℓ/unit-signal + 4 random; bounds
  ℓ∈[1e-3,1e4], σ_f²∈[1e-6,1e4], σ_n²∈[1e-8,1e2] in standardized space),
  Cholesky with escalating jitter (1e-10→1e-4 of mean diag). Inputs AND
  outputs standardized internally with train stats only (§5.3). Conforms to
  the `ForwardModel` protocol: `predict` → canonical
  `PredictiveDistribution` (epistemic = latent posterior std, aleatoric =
  fitted constant noise std per output — the §10.3-honest v0 floor,
  conformal_set=None when unwrapped); `support_score` = −Mahalanobis in
  standardized input space, regularized covariance (§8.2 cheap fallback);
  `jacobian` = closed-form Matérn-5/2 posterior-mean gradient (no 1/r
  singularity), chain-ruled back to raw units; `update(records)` = map via
  input/output keys, stack, full refit.
- `src/rig/forward/data.py` — `records_to_arrays(records, input_keys,
  output_keys)`: RunRecord → SI-magnitude matrices; rejects categorical
  inputs (conditioning per §8.3) and non-scalar outcomes.
- `src/rig/calibration/conformal.py` —
  `SplitConformalCalibrator` (score |y−μ|/σ_total, σ_total²=aleatoric²+
  epistemic²; band = μ ± κ·σ_total(x), κ = ceil((1−α)(n+1))-th order stat,
  +inf when that exceeds n — small-n honesty), `JackknifePlusCalibrator`
  (true jackknife+/CV+ per Barber et al. 2021: keeps LOO/fold models, band
  from order stats of μ_{−i}(x)±R_i; LOO for n≤40, K-fold CV+ (K=10 default)
  above), `ACIController` (α_{t+1}=clip(α_t+γ(α_target−err_t), 0.001, 0.5),
  γ=0.05, per-output α_t; `observe()` scores the pre-update interval, then
  updates α and appends the fresh score to the calibrator when it supports
  online scores — that append is what lets bands outgrow a stale calibration
  set under drift; `rolling_coverage` over a deque window = the §5.6 drift
  detector), `ConformalForwardModel` (fills `conformal_set` with (m,2) /
  (n,m,2) interval arrays; delegates support_score/jacobian/update to base).
- `src/rig/metrics/uq.py` — closed-form Gaussian CRPS (Gneiting 2005), PIT
  values (array returned, no plotting), quantile-calibration error (mean
  |emp−q| over q=0.05..0.95), Winkler interval score, PICP/MPIW,
  RMSE/MAE, `uq_report()` bundling all of it at 50/80/90/95% Gaussian bands.
  Convention: predictions-first arg order (μ, σ, y); (n,) or (n,m) inputs;
  aggregates always come back (m,).
- `tests/test_gp.py` (11), `tests/test_conformal.py` (9),
  `tests/test_uq_metrics.py` (7) — all synthetic, seeded. Includes the
  §5.9 disqualifying test (OOD epistemic > 3× in-range mean), Jacobian vs
  central FD at rtol 1e-4, 200-trial split + 200-trial CV+ coverage in
  [0.85, 0.97], ACI-recovers/static-degrades drift stream, CRPS vs
  scipy.integrate.quad at rtol 1e-3.

**Decisions (also added to BUILD_STATE standing decisions):**
- ACI update rule exact form: α_{t+1} = clip(α_t + γ(α_target − err_t),
  0.001, 0.5), γ=0.05 default, α maintained PER OUTPUT; interval is computed
  at the pre-update α_t (so err_t is honest), and the observed score is
  appended to the split calibrator's score buffer online.
- Jackknife+ ↔ CV+ switch point: LOO for n ≤ 40, K-fold CV+ (K=10) above.
- Aleatoric floor v0 = constant fitted GP noise std per output; hetero-
  scedastic aleatoric is deferred until replicates exist (§10.3 identifiability).
- `predict` shape contract: (d,) → fields (m,); (n,d) → fields (n,m);
  conformal_set (m,2) / (n,m,2). `jacobian` takes a single (d,) point →
  (m,d). `support_score` returns float for (d,), (n,) array for batch.
- Conformal quantile returns +inf when ceil((1−α)(n+1)) > n — an honest
  "no finite-sample claim possible", never a clamped finite band.

**Gotchas:**
- Single-output models that return (n,) means (not (n,1)) are normalized
  inside the calibrators via a `_mean_2d(mean, nq)` helper — np.atleast_2d
  is WRONG here (gives (1,n)).
- `np.broadcast_to` errors if α has MORE dims than (m,): ACI keeps α_t as a
  flat (m,) (or scalar) array; `observe()` flattens interval endpoints
  before comparing.
- GP fits are fast enough that the 400 coverage-trial fits (200 split +
  200×5 CV+ folds) run in ~9 s total with n_restarts=1 — analytic NLML
  gradients are what make this cheap; do not switch to finite-difference
  gradients.

**Test summary (verbatim):** `105 passed, 2 warnings in 12.73s`
(includes WP-A's 42 and concurrent WP-B's tests; the 2 warnings are WP-B's
Sobol' power-of-2 UserWarnings, not WP-C). Also verified:
`python -m ruff check src tests` → "All checks passed!"; import-linter →
"Contracts: 1 kept, 0 broken."

## 2026-07-15 — WP-B agent

**What:** Built `rig_adapters.mbe` — the Phase-0 MBE ProcessAdapter (resolves
E2) and the E3 in-silico pathology machine, plus Sobol data generation and a
checked-in smoke fixture. 36 new tests. Core (`src/rig/**`) untouched.

**Files:**
- `src/rig_adapters/mbe/simlink.py` — sole locator/importer of the external
  sim: `MBE_SIM_PATH` env var, default `c:\Users\Jiaow\Documents\github\MBE sim`,
  lazy `sys.path` insert, `MBESimNotFoundError` with an actionable message,
  `sim_available()` for test skips.
- `src/rig_adapters/mbe/adapter.py` — `MBEAdapter` + `make_adapter()` factory;
  entry point `mbe = "rig_adapters.mbe.adapter:make_adapter"` added to
  pyproject `[project.entry-points."rig.adapters"]` (editable install
  refreshed so `registry.get_adapter("mbe")` discovers it). Also
  `evaluate_physics()` — the one deterministic fast-Arrhenius evaluation
  path (never kMC/ZoneEnsemble), with keyword-only hidden parameters
  (emissivity, cosine_n, flux_eff) as the E3 injection surface.
- `src/rig_adapters/mbe/outcomes.py` — metric→OutcomeRecord translation layer.
- `src/rig_adapters/mbe/machine.py` — `InSilicoMachine` + `PathologyConfig`.
- `src/rig_adapters/mbe/generate.py` — `generate_dataset()` + JSONL CLI
  (`python -m rig_adapters.mbe.generate --n 64 --out data/mbe_silico_v0.jsonl`).
- `tests/fixtures/mbe_silico_smoke.jsonl` — 16 clean-machine runs, seed 0,
  fast path (24 nodes / 24 phi); a test asserts it is bit-identically
  regenerable.
- `tests/test_mbe_adapter.py`, `tests/test_mbe_machine.py`,
  `tests/test_mbe_generate.py`.

**Decision — the recipe-vs-config split (E2's "undefined recipe vector"):**
- RECIPE (ChangeCost.EASY, the inverse's search space): `T_heater` [K,
  1150–1500 from `optimize.DEFAULT_KNOBS`] and `film_thickness` [m,
  2e-7–5e-6] — the target-thickness/growth-time proxy the sim genuinely
  consumes (`UniformityProblem(film_thickness=...)`, drives the cooldown-bow
  metric) and the channel where flux pathologies become observable.
- MACHINE-CONFIG (ChangeCost.HARD_TO_CHANGE, split-plot whole-plot,
  conditioning-not-free per §8.3): `heater_radius, gap, source_offset,
  source_height, aim_offset`, bounds mirrored from `DEFAULT_KNOBS` into
  `MACHINE_CONFIG_BOUNDS` (hardcoded so import stays lazy; a test asserts the
  mirror matches the sim repo, so cross-repo drift fails loudly).
- `expert_ranges`/`seed_design` span RECIPE variables ONLY; machine config is
  held at `MACHINE_CONFIG_DEFAULTS` and recorded in `RunRecord.extra
  ["machine_config"]` so WP-D can condition on it.

**Decision — metric→OutcomeRecord mapping (outcomes.py, the E2 translation
layer; fast path's `UniformityProblem.evaluate()` plays snapshot()'s role):**
`combined_nonuniformity_pct→nonuniformity_pct` [percent],
`T_center→T_center` [K], `slip_max_ratio→slip_max_ratio` [-],
`bow_cooldown_um→bow_cooldown_um` [um], `thickness_grown_m→thickness_grown`
[m]. NB pint canonicalizes percent to a dimensionless FRACTION (5% → 0.05)
and um to m in the serialized records; the machine's pathology math happens
in declared engineering units before canonicalization.

**Decision — E3 machine semantics:** all pathologies OFF by default;
per-tool perturbation = fixed multiplicative vector (default ±3%) on hidden
(emissivity, cosine_n, flux_eff), derived from sha256(tool_id) so it is
process- and order-independent; seasoning multiplies flux_eff by
(1 − 0.004·runs_since_clean) per tool, reset by `clean()`; first-wafer =
additive offsets ({nonuniformity_pct: +0.5 %, T_center: −3 K}) on the run
right after clean (including the machine's very first run); metrology noise
σ(y) = a + 0.002·|y| per output from a per-run-index rng stream; optional
censoring saturates at the range bound and flags
`extra["censored"][output] = "low"|"high"`. Deterministic run_id (uuid5),
timestamp (base + 1 h·run_index) ⇒ bit-identical RunRecords (tested,
serialized-JSON equality). Hidden state is deliberately NOT in the records
(`state_snapshot()` gives ground truth for §12.1 figures);
`provenance.source="physics_sim"` always. Cost model: c_batch=$1000,
c_recipe=$1000, batch_size=4 (Kanarik, §11.1). `physics_plugin` = clean
fast-path evaluator; `independent_verifier=None` (honest D7 — the ROM
verifier is still owed, never point it at this sim).

**Gotchas:**
- `import mbe_sim` pulls in tkinter (via setup_menu/reportview) — fine on
  this box (~0.25 s), but headless CI without tk would need the sim import
  guarded; simlink already confines the import to one place.
- A pure flux-scale change is INVISIBLE in the sim's normalized uniformity
  outputs (flux_profile normalizes to centre) — that is why seasoning acts
  through achieved `thickness_grown` (= target × flux_eff, which also feeds
  the bow metric), not through nonuniformity.
- pint "percent" → 0.01 dimensionless at SI canonicalization: tests that
  compare declared-unit offsets against record magnitudes must convert
  (bit me once in test_first_wafer_offset_after_clean).
- WP-A follow-up (schema untouched per mandate): OutcomeRecord has no
  first-class `censored` field, so censoring flags live in `RunRecord.extra
  ["censored"]`. Consider promoting it in a future WP-A revision (E1 will
  need censoring-at-ingest anyway).
- Transient mid-session: tests/test_conformal.py failed while WP-C was
  mid-flight; green again by final run — concurrent-agent noise, not a real
  breakage.

**Test summary (verbatim):** `105 passed in 12.45s` (full suite: WP-A 42 +
WP-C + WP-B 36; Sobol power-of-2 warnings eliminated by using power-of-2 n
in tests). `python -m ruff check src tests` → "All checks passed!";
import-linter → "core must never import adapters … KEPT, Contracts: 1 kept,
0 broken."

## 2026-07-15 — session 1 close (orchestrator)

- Independently re-ran the full suite: `105 passed in 12.73s`; ruff/import-linter clean
  (per WP agents, re-verified suite only).
- Cross-package end-to-end smoke (not covered by any single WP's tests):
  `get_adapter("mbe")` → `generate_dataset(32 runs)` → `records_to_arrays` →
  `GPForwardModel.fit` → `SplitConformalCalibrator.fit` → `ConformalForwardModel.predict`
  ⇒ canonical shapes OK, conformal_set (2,2) filled, support_score −1.27 in-range vs
  −70.1 far-OOD, epistemic 3e-6 → 0.087 OOD (inflates ≫3×). PASS.
- Session-1 deliverables: audited+corrected implementation-plan.md; WP-A, WP-B, WP-C all DONE.
- NEXT SESSION START HERE → WP-D (per-query pessimistic inverse, implementation-plan §8): consume
  the WP-B/WP-C handoff notes in BUILD_STATE standing decisions + the WP agents' log
  entries above. Then WP-E (torch stack, needs install decision), WP-F (AL loop),
  WP-G (eval harness). M0 (real dataset) remains the user's action item.
- Nothing committed to git (awaiting user instruction).

## 2026-07-15 — WP-H agent

**What:** Built `rig_adapters.tabular` — the generic config-driven adapter +
file ingestion (E5 seed + E1 slice), so the system is not MBE-specific in
practice: any recipe→outcome process plugs in via a declarative spec + a flat
CSV, no Python needed for the schema. 60 new tests. Core (`src/rig/**`) and
the mbe adapter untouched.

**Files:**
- `src/rig_adapters/tabular/spec.py` — `ProcessSpec` + `load_spec`/`parse_spec`.
  **Format decision: TOML primary** (stdlib `tomllib`, zero new deps,
  comment-friendly for annotated specs, same dialect as pyproject); JSON
  accepted as a secondary machine-generated format (dispatch on suffix).
  Strict load-time validation (the E5 schema-elicitation intake): every error
  is a `SpecError` naming the offending key — unknown top-level/block keys
  (typos fail loudly), non-pint units, inverted bounds/specs, <2
  levels/components, bad modality, bad cost block, dots in names (reserved
  for the `"<var>.<component>"` flattening convention), input/output name
  overlap. The E5-mandated compositional tag is enforced: a `compositional`
  block declaring a non-dimensionless unit (e.g. `sccm`) is REJECTED with a
  message citing implementation-plan §3.1 and telling the author to declare independent
  flows as continuous variables. Variables reuse the WP-A
  `ContinuousVariable`/`CategoricalVariable`/`CompositionalVariable` types
  directly; units validated against the ONE shared `rig.schema.ureg`.
  `curve_1d`/`field_2d` modalities rejected with "not yet supported".
  Cost defaults $1000/$1000/4 (Kanarik, matching mbe).
- `src/rig_adapters/tabular/adapter.py` — `TabularAdapter` (structural
  ProcessAdapter; `validate_adapter` passes; D7-honest: physics_plugin=None,
  independent_verifier=None). `seed_design` = `sobol_seed_design` over
  continuous bounds (declared units) + compositional components sampled on
  [0,1]^K then renormalized onto the simplex ⇒ every seed FEASIBLE by
  construction (E5). Categoricals excluded from the design + encoders
  (conditioning, §8.3). Encoder order = `spec.numeric_input_names` (spec
  order, compositional flattened). Entry point `tabular =
  "rig_adapters.tabular.adapter:make_adapter"` added to pyproject (editable
  install refreshed).
- `src/rig_adapters/tabular/ingest.py` — `ingest_csv`, `ingest_jsonl`,
  `write_jsonl`, CLI `python -m rig_adapters.tabular.ingest --spec ... --csv
  ... --out runs.jsonl` (plus --tool-column/--timestamp-column/--source/
  --default-tool-id/--on-error). Values read in SPEC-DECLARED units,
  SI-canonicalized by the WP-A schema validators (shared ureg, never a second
  registry). Missing required columns = hard `IngestError` listing them;
  unmatched columns = one warning + per-record
  `extra["unmatched_columns"]`. Row failures (bounds, levels, non-sum-to-1
  composition atol 1e-6, bad numbers/timestamps) follow
  `on_error="raise"|"skip"`; skip returns `RejectedRow(row_index, reason)`
  entries. No timestamp column ⇒ deterministic monotone synthetic ladder
  (2000-01-01Z + 1 h/row) + `extra["synthetic_timestamp"]=True` flag
  (temporal splits meaningless, §12.4). Deterministic uuid5 run_ids;
  `provenance.data_hash` = sha256 of the CSV file.
- `examples/tabular_minimal.toml`, `examples/pecvd_example.toml` (annotated,
  illustrative-structure-only PECVD: degC/torr/W + genuine compositional
  precursor blend + thickness/nonuniformity/stress with spec semantics).
- `docs/new-process-onboarding.md` — the E5-seed runbook: write spec → lint
  (= load) → ingest CLI → 10-ish-line GP+conformal fit snippet (verified by
  actually running it end-to-end: CLI ingest of a 24-row synthetic CSV →
  conformal_set (3,2), support −0.24 in-range vs −128 far-OOD). States
  explicitly what is NOT covered (1-D/2-D modalities, physics plug-ins, full
  E5 harness, full E1 ETL).
- `tests/test_tabular_adapter.py` (36), `tests/test_tabular_ingest.py` (24) —
  spec happy paths (both examples + JSON), every validation error incl. the
  sccm-compositional rejection (asserts the message cites §3.1), adapter
  conformance/D7, feasible+deterministic seed design, encoder round-trip,
  factory kwarg/env-var/bare-error paths, SI canonicalization from non-SI CSV
  (degC→K, torr→Pa base, sccm→m³/s, nm/min→m/s, percent→fraction),
  unmatched/missing columns, on_error both modes, sum-to-1 both sides of
  atol, synthetic-timestamp determinism (bit-identical serialized records),
  ISO-8601 + bad-timestamp, JSONL round-trip equality, spec-mismatch on
  reload, CLI end-to-end, and the INTEGRATION proof of "not MBE-specific":
  30-row seeded synthetic PECVD-ish CSV → ingest → records_to_arrays →
  GPForwardModel.fit → canonical PredictiveDistribution + in-range
  support_score > far-OOD support_score.

**Decisions (also added to BUILD_STATE standing decisions):**
- Spec format: TOML primary via stdlib tomllib; JSON secondary. No new deps.
- Parameterized-factory pattern for config-driven adapters (binding template
  for future ones): factory takes the config as an optional kwarg
  (`spec_path=None`) → falls back to a documented env var
  (`RIG_TABULAR_SPEC`) → else raises an actionable `LookupError` naming both
  options and the runbook. Never a silent default process.
- `ingest_csv`/`ingest_jsonl` return an `IngestResult` (records + rejects +
  unmatched_columns + synthetic_timestamps) rather than a bare list — the
  "skip" policy needs a rejects report the caller gets back; `IngestResult`
  iterates over records so it drops into `records_to_arrays`/`write_jsonl`
  unchanged.
- DoE over a simplex: Sobol on [0,1]^K per compositional variable, then
  renormalize each draw to sum 1 (feasible-by-construction seeds; a proper
  simplex-uniform design can replace this later without API change).

**Gotchas:**
- pint formats base-unit pressure as `kg/(m·s²)` (not "Pa") after
  `to_base_units()`; tests compare magnitudes via
  `ureg.Quantity(x, "torr").to_base_units().magnitude`, not unit strings.
- A flattened-input-name collision is structurally impossible once dots are
  rejected in names/components — the collision check in `parse_spec` is
  defensive; the real guard is the dot rejection.
- With sum-to-1 compositions the full flattened component set is exactly
  collinear — the onboarding doc and the integration test drop one component
  per compositional variable before GP fitting.
- `csv.DictReader` + `utf-8-sig` handles Excel-BOM CSVs; empty cells arrive
  as "" (rejected per-row with the column name).
- Core follow-up (none blocking): a first-class `Infeasible`-style censoring
  field and Pandera frame validation at ingest remain E1 items; the tabular
  adapter records nothing in `extra["censored"]` yet.

**Test summary (verbatim):** `165 passed in 11.35s` (105 existing + 60 new).
`python -m ruff check src tests` → "All checks passed!"; import-linter →
"core must never import adapters … KEPT, Contracts: 1 kept, 0 broken."
Registry smoke: `list_adapters()` → `['mbe', 'tabular']`;
`get_adapter('tabular', spec_path=...)` and the `RIG_TABULAR_SPEC` path both
load and validate.

## 2026-07-15 — session 1 addendum (orchestrator, post WP-H)

- WP-H (generic tabular adapter) verified independently: full suite `165 passed in
  15.23s`; `list_adapters()` → ['mbe', 'tabular']; `examples/pecvd_example.toml`
  loads (6 flat inputs incl. a genuine compositional blend, 3 outputs).
- The "not MBE-specific" user requirement now has an executable proof:
  tests/test_tabular_ingest.py::test_end_to_end_generic_process_csv_to_calibrated_forward_model.
- Onboarding runbook for arbitrary processes: docs/new-process-onboarding.md.

## 2026-07-15 — WP-I agent

**What:** Tool-aware forward surrogate — the implementation-plan §10.4 level-(a) chamber-
matching path on the numpy GP backend (GP-era primary; per-tool FiLM/ANP is
the torch-era WP-E work). Motivated by the 2026-07-15 user signal: switching
machines/tools must not mean retraining from scratch. Validated entirely
in-silico (WP-B pathology machine); real-data claims stay gated on M0.

**Files:**
- `src/rig/forward/_gp_common.py` (NEW, private) — shared exact-GP machinery
  extracted from gp.py so multitask.py reuses it without copy-paste:
  `matern52`, `matern52_grad_x`, `cholesky_with_jitter`, `multistart_minimize`
  (the L-BFGS-B multi-start loop, byte-for-byte the WP-C behavior),
  `standardize_stats`, `regularized_cov_inv`, log-param bounds.
- `src/rig/forward/gp.py` — refactored onto `_gp_common` (imports aliased to
  the old private names). PUBLIC BEHAVIOR UNCHANGED: WP-C's tests pass
  untouched, fits remain bit-deterministic.
- `src/rig/forward/multitask.py` (NEW) — `MultiToolGPForwardModel` +
  `ToolBoundForwardModel` + internal `_ICMSingleOutputGP`. One ICM GP per
  output: k((x,s),(x',t)) = k_Matern52_ARD(x,x') · B[s,t]. Exact NLML with
  ANALYTIC gradients for all params (log ell, W, log v, log sn2), L-BFGS-B
  multi-start (deterministic first start encodes strong prior inter-tool
  correlation: W = 0.9/sqrt(rank), v = 0.1 ⇒ B off-diag ≈ 0.81 — tools are
  similar until the data disagrees). Constant per-output sn2 = the same
  §10.3-honest aleatoric floor v0 as WP-C, shared across tools.
- `src/rig/forward/data.py` — added `records_to_arrays_with_tools(records,
  input_keys, output_keys) -> (X, Y, tools)` reading `RunRecord.tool_id`;
  the existing function's signature is untouched.
- `src/rig/forward/__init__.py` — exports the new names.
- `tests/test_multitask_gp.py` (22, synthetic, no rig_adapters import),
  `tests/test_multitask_mbe.py` (1, skip-if sim unavailable like WP-B).
- `docs/new-process-onboarding.md` — new "Multiple tools / switching
  machines" section (ingest --tool-column, fit, few-shot onboarding, how to
  read the transfer alarms; states RGPE negative-transfer guard is future).

**Decisions (also in BUILD_STATE standing decisions):**
- B parameterization: B = W Wᵀ + diag(v), W ∈ R^{T×rank} unconstrained
  (bounds ±30 in standardized-y units), v log-parameterized (> 0), rank
  default 1 (1–2 recommended at small T; ctor validates rank ≥ 1). The
  Matérn factor is UNIT variance — sf2 would be redundant with B's scale.
  Gradients: grad_W[t,r] = Σ_{i: s_i=t} (M W[s])_{i,r} with M = A∘Kx;
  grad_{log v_t} = 0.5 v_t · (per-tool block sum of M); verified vs central
  FD to rtol 1e-5 in tests.
- Standardization is POOLED across tools (train stats only) — per-tool
  y-standardization would erase exactly the tool offsets B must learn.
- Unknown-tool fallback formula (documented in the module docstring):
  w_t ∝ Σ_s max(B[t,s],0); μ_u = Σ w_t μ_t;
  σ²_epi,u = max_t σ²_epi,t + Σ w_t (μ_t − μ_u)² + (1 − ρ̄²)·mean_t B[t,t],
  ρ̄ = mean pairwise tool correlation clipped to [0,1] (ρ̄ = 0 at T = 1).
  Deliberately conservative: unknown-tool epistemic dominates every known
  tool's elementwise, so the §5.8 LOTO check holds by construction.
  `tool_id=None` takes the same fallback (never a silent known-tool guess);
  `add_tool()`-declared but dataless tools also stay on the fallback.
- Tool→index map: first-appearance order, stable across refits; `update()`
  registers unseen `RunRecord.tool_id`s implicitly; `adapt_to_tool` = full
  refit (fine at this n) + INFO log of the tool's run count.
- support_score: per-tool Mahalanobis (tool mean + regularized covariance)
  when the tool has ≥ d+2 runs, else the global cloud; same negative-
  distance semantics as WP-C.
- Conformal wrapping: `ConformalForwardModel` stays tool-blind and UNMODIFIED;
  bind the tool first via `model.for_tool(tool_id)` (a ForwardModel-protocol
  view — cleaner than a lambda/partial and it delegates support_score/
  jacobian/update too). Tested: conformal_set (m,2)/(n,m,2) filled and finite.

**Few-shot results (measured, seeded — the numbers behind the claims):**
- Synthetic (tool B = f·(1+0.1) + 0.15, noise σ=0.05; 40 A runs + k B runs,
  held-out noise-free B truth, RMSE):
  k=4: multi-tool 0.089 < pooled-blind 0.142 < scratch-on-B 0.264;
  k=8: multi-tool 0.069 < pooled-blind 0.127 < scratch-on-B 0.519.
  Learned tool correlation B̂: 0.998 (k=4), 0.990 (k=8) — recovers the
  by-construction high tool similarity.
- LOTO epistemic (§5.8): single-tool fit, in-dist mean epistemic 0.022 vs
  unknown-tool 0.888 (the (1−ρ̄²)·b̄ term dominates at T=1, as designed).
- In-silico MBE (tool_perturbation ON, seed 0; A n=32 + B n=8 train, 8
  held-out B; `thickness_grown`, the flux-sensitive KPI per WP-B):
  multi-tool RMSE 4.0e-12 m ≪ pooled-blind 3.0e-8 m (≈7500× — pooled-blind
  averages two per-tool flux slopes); scratch-on-8 8.2e-12 m (also good
  because thickness_grown is near-linear in film_thickness; multi still
  wins). Unknown-tool "C" mean epistemic 3.2e-8 vs known A 6.3e-11 /
  known B 8.6e-11. Integration test wall time ≈ 1.5 s (well under 60 s).

**Gotchas:**
- Ruff (isort, combine-as-imports off) rewrites a multi-name aliased
  from-import into one block per alias in gp.py — cosmetic, left as ruff
  wants it.
- np.add.at is the correct scatter for grad_W (repeated tool indices);
  a plain fancy-index += silently drops duplicates.
- The MBE pooled-vs-multi gap is huge ONLY on thickness_grown; normalized
  uniformity outputs are blind to per-tool flux scale (WP-B handoff), so
  don't "fix" the integration test by switching KPI.
- Scratch-on-B can beat pooled-blind (MBE case): pooling is not just
  suboptimal under tool shift, it can be worse than 8 runs alone — worth
  keeping as a talking point for §10.4.

**For WP-D (tool-conditioned inversion, §8.3 split-plot conditioning):**
recipes are generated GIVEN a tool — the inverse should hold `tool_id` fixed
as conditioning (like machine_config), NOT search over it. Use
`model.for_tool(tool_id)` to get a plain ForwardModel for the existing
tool-blind solver/wrapper plumbing; `predict(x, tool_id=...)`,
`support_score(x, tool_id=...)`, `jacobian(x, tool_id=...)` are the explicit
forms. For a tool with zero/few runs the fallback's inflated epistemic will
(correctly) shrink the trust region — surface "collect k runs on this tool
first" rather than INFEASIBLE.

**Test summary (verbatim):** `188 passed in 13.13s` (165 existing + 23 new).
`python -m ruff check src tests` → "All checks passed!"; import-linter →
"core must never import adapters … KEPT, Contracts: 1 kept, 0 broken."

## 2026-07-15 — session 1 addendum (orchestrator, post WP-I)

- WP-I verified independently: full suite `188 passed in 13.97s`; fresh (non-test)
  few-shot smoke: tool B with 6 runs predicts within noise of ground truth
  (-0.227 vs -0.211), unknown-tool epistemic 0.075 > known-tool 0.022. PASS.
- Session-1 final state: WP-A/B/C/H/I DONE, 188 tests. Next: WP-D (pessimistic
  inverse, §8) — note WP-I's handoff: invert GIVEN a tool via model.for_tool();
  zero-run tools ⇒ "collect k runs first", not INFEASIBLE.

## 2026-07-15 — WP-D agent (session 2)

**What:** Per-query pessimistic inverse solver — the D2 canonical refiner (implementation-plan
§8) at the GP/numpy-scipy tier. Torch-only §8 pieces (deep-ensemble worst-of-K,
normalizing-flow typicality, PGD, qLogNEHVI, k-DPP, amortized generator) are
DELIBERATELY deferred to WP-E; WP-D realizes each §8 pessimism channel with the
plan's own blessed GP-tier fallbacks. Validated in-silico (WP-B machine); real
claims stay gated on M0.

**Files:**
- `src/rig/inverse/pessimistic.py` (NEW) — `PessimisticInverseSolver`
  (`InverseSolver` protocol), `SpecBox`, `parse_targets`. §8 → GP-tier map:
  * §8.1 objective: robust worst-case CREDITED interval must sit inside the box.
    Per constrained output j, standardized worst-cased margins
    `u_hi=(U−μ−s)/σ_ale`, `u_lo=(μ−L−s)/σ_ale`, with displacement
    `s = z_epi·σ_epi (epistemic worst-member proxy) + Σ|J_i|·Δ_i (§8.5 exact
    first-order Taylor of max over the ℓ∞ δ box — the analytic-Jacobian fallback
    the plan blesses; PGD is WP-E)`. Feasible iff `min_j min(u_hi,u_lo) ≥ κ`
    (§8.4 credited-band bar, default κ=2). Epistemic enters ONCE (no κ·U_epi
    double-count).
  * §8.2 anti-reward-hacking: soft `λ_m·support_score` reward + HARD reject below
    `support_floor` (default = 5th-pct train score). Fail-closed: ctor REQUIRES
    `support_floor` or `X_train`. support_score = negative Mahalanobis (the §8.2
    cheap fallback; flow typicality = WP-E).
  * §8.3 constraint-by-construction: box+simplex exact via `RecipeTransform`;
    tool/hard-to-change are conditioning — bind via `model.for_tool()` BEFORE the
    solver (never searched). Linear couplings = future (cvxpylayers/DC3 = WP-E).
  * §8.6 loop: Sobol multi-start (u=0 centre + Sobol) → L-BFGS-B on the smooth
    log-sigmoid objective (plan's Adam/512-restarts is the GPU-distilled budget;
    GP-tier CPU uses fewer). Confidence = Π_j (Φ(u_hi)+Φ(u_lo)−1) pessimistic
    spec-hit prob (independence approx; joint-residual-cov MC = WP-E).
  * §8.7 non-injectivity: farthest-point (max-min) diversity selection in
    normalized recipe space (k-DPP stand-in); empty feasible set ⇒ `Infeasible`
    with nearest point + distance + per-output relaxation (never a clipped point).
- `src/rig/inverse/__init__.py` (NEW) — exports.
- `tests/test_inverse.py` (NEW, 23) — analytic mock model driving each pessimism
  channel independently + GP + tool-bound integration.
- `tests/test_inverse_mbe.py` (NEW, 2, sim-gated) — end-to-end: GP on in-silico
  MBE runs → invert reachable `thickness_grown` spec → the pessimistic interval
  COVERS the real machine outcome (honest, not decorative — width ≥ 2κσ_ale
  asserted); unreachable spec ⇒ Infeasible.

**Adversarial review (2 rounds, workflow + focused agent) — findings fixed:**
The build ran a 5-lens adversarial review workflow (20 agents, per-finding
verify) then a focused re-check of the fixes. 13 confirmed findings → 3 real code
defects + 4 test gaps, ALL fixed, each now guarded by a test:
- **[HIGH] `epi_dominated` was trivially always-true** (compared epistemic-vs-δ,
  not vs the spec gap; with δ=0 it collapsed to `epi≥0.5·epi`), mislabeling
  genuinely-unreachable specs as "collect more runs". FIX: replaced the per-point
  bool with a proper **nominal-feasibility probe** — a second epistemic-free
  multi-start (`ignore_epi=True`), because the pessimistic search AVOIDS
  high-epistemic regions so the mean-feasible point is never in the primary
  restarts. Deficit now DECOMPOSED at the binding output (mean-in-box? epistemic
  share? aleatoric-narrow-box vs δ?) into four honest verdicts: off-manifold /
  epistemic-limited (collect runs) / partly-epistemic-tight-box (data cuts the
  needed relaxation) / hard conflict (relax spec). The probe also PROMOTES an
  on-support pessimistically-feasible point it finds that the primary missed
  (never return INFEASIBLE when a feasible recipe exists).
- **[LOW] relaxation used raw σ_ale not the floored scale** → "relax by 0" for a
  near-deterministic output. FIX: convert with `nearest.scale_spec` (the same
  floored σ the margins used).
- **[LOW] bare `{'target': t}` (zero-width box)** was always-infeasible silently.
  FIX: `parse_targets` rejects a zero-width box with an actionable "give a tol"
  error (fail-loud, matches WP-H "no silent default").
- Test gaps fixed: genuine two-pre-image twin test (was y=x with no real twin);
  non-vacuous epistemic-reason assertions; MBE interval-width (honest-not-
  decorative) assertion + all-covered; multi-output partial-violation test.

**Decisions (added to BUILD_STATE standing decisions):** GP-tier §8 realization
knobs (κ=2, z_epi=2, delta_frac=0.02, λ_m=0.3, support fail-closed); the
nominal-feasibility probe as the §8.8 diagnosis mechanism; the 4-way infeasibility
taxonomy; tool conditioning via `for_tool` (never searched).

**Test summary (verbatim):** `213 passed in 19.03s` (188 + 25 new). `ruff check
src tests` → "All checks passed!"; import-linter → "core must never import
adapters … KEPT, Contracts: 1 kept, 0 broken."

**For the next session (WP-F/WP-G):** the solver's Loop-B online hook (§8.6 step 6
— ACI update + realized-vs-predicted gap → adjust κ,τ) is the active-learning
boundary and lives in WP-F, not here; WP-D is the single-query solve. WP-G's
evaluation harness can drive cost-to-target through `solve()`. Multi-output
confidence uses the per-output independence approx (joint-residual-cov MC = WP-E).

## 2026-07-15 — WP-G agent (session 2)

**What:** Evaluation & benchmarking harness (implementation-plan §12) — the measurement layer
that makes MFL's failures un-hideable. numpy/scipy only (lifelines/torch absent).

**Files:** `src/rig/eval/` (NEW package):
- `survival.py` — cost-to-target as right-censored survival (§12.2): Kaplan-Meier
  (product-limit + Greenwood var), RMST = ∫_0^τ S + its KM-based variance,
  difference-in-RMST test (Uno 2014, the §12.2 PRIMARY comparator — log-rank is
  invalid under the expected curve crossing, deferred as a caveated secondary),
  `split_feasible` (infeasible targets EXCLUDED not censored, §12.2 (i)).
- `inverse_metrics.py` — target-hit-rate (top-1, not best-of-q), success-rate-at-
  budget, false_success (hit on infeasible ≈0), false_abstention (feasible-hard
  wrongly refused), feasibility-flag accuracy, constraint-satisfaction (BEFORE
  projection), robust-hit-rate.
- `exploitation.py` — the §12.1 surrogate-exploitation stress test (headline):
  optimism gap + interval-violation on inverse-proposed recipes run on an
  OOD/pathology machine; machine-agnostic (caller supplies realized_Y).
- `diversity.py` — Vendi score (Friedman & Dieng 2023) + mode-count + pairwise-L2.
- `calibration_gates.py` — SBC (Talts 2018) + TARP (Lemos 2023) posterior-recovery
  gates sharing a simulation-calibrated simultaneous ECDF uniformity band
  (Säilynoja-style). A HARNESS — the amortized posterior it consumes is WP-E;
  validated now on synthetic posteriors (calibrated passes, overconfident fails).
- Tests: `test_eval_survival.py` (hand-pinned KM/RMST/**RMST-variance**),
  `test_eval_inverse_metrics.py`, `test_eval_calibration.py`,
  `test_eval_exploitation.py`, sim-gated `test_eval_mbe.py` (the non-circular
  headline: optimism gap worse on the perturbed tool than the clean one).

**Adversarial review (5-lens workflow, per-finding verify) → 6 confirmed, fixed:**
- **[HIGH] TARP discreteness bug:** `tarp_test` fed discrete credibilities
  {0,1/M,…,1} into a CONTINUOUS-uniform null with no continuity correction, so a
  calibrated COARSE posterior (small M — the advertised GP-tier §8.7 feed) was
  rejected ~100% of the time. FIX: apply the SAME `(K+U)/(M+1)` jitter `sbc_test`
  uses (recover K=round(cred·M)); verified pass-rate 0.90+ at M=5/10/50 (was 0).
- **[MED] RMST variance untested** (code correct, `se=√(2/9)` for [1,2,3]) → pinned.
- **[LOW]** exploitation joint-violation ≈1−(1−α)^m not α for m>1 (doc fix; the
  per-output field is the ≈α check); log-rank deferral note added; diversity
  docstring's phantom `valid_mask` removed; the "no sim-gated exploitation test"
  finding was a false alarm (test_eval_mbe.py exists — I'd omitted it from the
  review's file list).

**Decisions:** survival "event" = spec hit, "censor" = budget-exhausted, SMALLER
RMST better; infeasible EXCLUDED from survival (feasibility metrics instead);
SBC/TARP simultaneous band by simulation (KS sup-deviation), continuity-corrected
before the band; Vendi standardized+median-bandwidth ⇒ scale-invariant.

**Test summary (verbatim):** `249 passed` at WP-G completion (later 268 with WP-F);
ruff clean; import-linter KEPT.

## 2026-07-15 — WP-F agent (session 2)

**What:** Active-learning loop (implementation-plan §9) — the numpy/GP-tier experiment
selector (MFL's missing "given enough (x,z) pairs" answer). qLogNEHVI Phase-II
and offline re-distillation are the torch WP-E (explicit NotImplementedError).

**Files:** `src/rig/active/` (NEW package):
- `acquisition.py` — §9.4 cost-cooled blend `[λ·EPIG + (1−λ)·BALD]/cost^β`.
  **BALD** = `0.5·log(1+σ_epi²/σ_ale²)` per output, summed (H[total]−E[H[ale]],
  never raw variance). **EPIG** (prediction-targeted): observing y(x) reduces
  latent var at the inverse's target point x* via the GP joint covariance —
  rewards runs that sharpen the surrogate WHERE the inverse proposes; EPIG≈0 for a
  far-OOD point even though its BALD is huge (the key §9.4 distinction). `anneal`
  (λ 0.2→0.9, β 1→0 CArBO). `qlognehvi_phase2` raises → WP-E.
- `batch.py` — §9.5 greedy submodular batch: top-acq then `acq − w_div·max_corr`
  (predictive correlation from `posterior_cov`, input-RBF fallback); split-plot =
  select within a shared-whole-plot group (caller groups).
- `loop.py` — the closed loop: Sobol DoE warm start → fit GP → per batch
  {re-solve inverse (WP-D), cost-cooled explore acquisition, 1 exploit + (q−1)
  diverse explore, query machine, refit} → stop on target-met (exploit in-spec on
  machine) / budget / acquisition-stall. Machine-agnostic (`machine(recipe)`,
  `in_spec(outcome)`); yields a `Trajectory` (cost-to-target) for WP-G's survival.
- `src/rig/forward/gp.py` — added `_SingleOutputGP.cov` + `GPForwardModel.
  posterior_cov(X1,X2)->(m,n1,n2)` (latent joint covariance for EPIG; diagonal ==
  predict epistemic², verified). Public GP behavior otherwise unchanged.
- Tests: `test_active_acquisition.py` (BALD closed-form pinned, EPIG-localizes-vs-
  BALD, cost-cooling, anneal, batch anti-duplication), `test_active_loop.py`
  (converges on reachable, budget-exhausts with NO false hit on unreachable,
  deterministic), sim-gated `test_active_mbe.py` (cost-to-target on the MBE
  machine, Kanarik $1k/recipe+$1k/batch).

**Decisions:** BALD/EPIG both in NATS (linearly blendable); cost enters by
DIVISION (CArBO), fixed c_batch NOT in the per-recipe ratio (it's the §11 stop
rule); β DECAYS 1→0 (cost-frugal early, cost-agnostic late); EPIG needs a GP-like
model exposing `posterior_cov` (BALD needs only public predict); batch = 1 exploit
+ (q−1) explore each lot; refit a fresh GP per batch (fine at this n).

**Test summary (verbatim):** `268 passed in 31.14s`; ruff "All checks passed!";
import-linter "Contracts: 1 kept, 0 broken."

**Adversarial review (3-lens workflow, per-finding verify) → 3 confirmed, fixed:**
- **[HIGH] hit-detection only checked the exploit recipe `Yb[0]`** — in-spec
  outcomes in the seed DoE and the (q−1) explore runs were measured but never
  credited, so cost-to-target (§9.1 PRIMARY, feeds WP-G survival) was OVERSTATED
  and `per_batch_hit` was wrong. Worse for M2: the BO baseline already credited
  `any(hits)`, so RIG's cost was systematically overstated RELATIVE to BO —
  biasing the comparison against RIG. FIX: `_any_in_spec(Y)` over ALL queried
  outcomes each lot (seed included); a hit is any in-spec run in the lot (the
  whole lot is fired + measured together), cost-to-target = cumulative cost of
  that lot. Now symmetric with the baseline.
- **[LOW×2] test gaps** (masking the above): the reachable test only asserted
  `cost_to_target ≤ 40`; the unreachable "no false hit" was non-discriminating
  (physics-impossible). FIX: `test_loop_cost_to_target_pins_first_in_spec_batch`
  (instrumented — pins cost to the exact lot of the first in-spec run, seed or
  explore) + `test_loop_credits_seed_doe_hit`.

## 2026-07-15 — M2 baseline + comparison (session 2)

**What:** the numpy-tier completion of M2's in-silico measurement — a fair
warm-started BO baseline + the head-to-head harness.

**Files:**
- `src/rig/baselines/warm_bo.py` (NEW) — `WarmStartedBO`: GP + Expected
  Improvement on the scalar objective `g(x)=‖relu(L−y, y−U)‖₂` (distance to the
  spec box, 0 iff in-spec), warm-started from the expert ranges, matched-budget,
  mirrors `ActiveLearningLoop`'s ctor + returns the same `Trajectory` so WP-G
  compares directly (§9.8 / §12.3, "the fair BO baseline MFL omitted"). EI closed
  form verified (`EI(μ=best)=σ·φ(0)`). BoTorch qLogEI/qLogNEHVI/TuRBO = WP-E.
- `tests/test_m2_comparison.py` (NEW) — RIG loop vs BO over 10 seeds on a
  non-monotone machine, cost-to-target → `rmst_difference_test`. Asserts the
  HARNESS is valid (finite RMSTs, p∈[0,1], signed effect) and both reach feasible
  targets — NOT a flaky "RIG always wins" (that's the empirical campaign result).

**Decision:** the loop/BO cost-to-target counts ANY in-spec queried run as the
"found a working recipe" event (a space-filling hit IS a hit); the stricter
§12.2 "top-ranked recipe only" deployment metric is a SEPARATE WP-G metric
(`TargetOutcome.hit`). Both methods use the identical seed DoE + budget = fair.

**Test summary (verbatim):** `272 passed in 39.05s`; ruff "All checks passed!";
import-linter "Contracts: 1 kept, 0 broken."

## 2026-07-16 — full-codebase audit (orchestrator + 14-agent workflow)

Ran a thorough multi-lens audit of the numpy-tier codebase (correctness/math,
spec-conformance, numerics, determinism, test-quality) — 14 per-module auditor
agents, each finding adversarially verified by a second agent that reproduced the
failure numerically. Full report: **`docs/audit-2026-07-16.md`**. Nothing fixed
(user asked to surface issues only); nothing committed.

**Outcome: 33 confirmed findings (0 critical, 0 currently-wrong-answer; ~14 medium,
~19 low), 2 refuted.** The mathematics is correct almost everywhere — every
load-bearing numeric check reproduced clean (NLML grads ~1e-9, kernel, posterior_cov
diagonal==epistemic², KM/RMST, BALD/EPIG closed forms, EI, §8 margins, LOTO
dominance). Determinism sweep clean (no unseeded RNG in src; guarded Cholesky).

**Top issues:**
- **P0 A1 (CI blind spot):** all 41 MBE sim-integration tests `skipif(not
  sim_available())`; `ci.yml` never sets `MBE_SIM_PATH` and never fetches the sim
  repo → in CI the run is `231 passed, 41 skipped`, exit 0. The "272 passed"
  headline is workstation-local. Reproduced via `MBE_SIM_PATH=/nonexistent`.
- **P1 defects (medium):** simplex sum-to-1 not enforced in `validate_against`
  (schema.py:154); MBE `RecipeRecord` path bypasses the E2 recipe/config split
  (machine.py:159); JSONL ingest uncaught `KeyError` + silent-incomplete accept
  (ingest.py:117); onboarding `input_keys[:-1]` collinearity-drop order-fragile →
  rank-deficient GP (spec.py:139 + doc); multitool model lacks `posterior_cov` so
  EPIG/batch AL crash on the chamber-matching model (multitask.py:542).
- **P1 test-gaps (medium, correct-but-unpinned):** single-output NLML gradient,
  `posterior_cov` invariant, jackknife ±inf tiny-n, SBC continuity correction,
  exploitation joint-vs-per-output, EPIG magnitude, λ blend direction, stall stop
  path, M2 seed-DoE masking, warm-BO EI, multi-output ICM values (C1–C11). Each
  proven by injecting the bug and watching the suite stay green.
- **P2 (low):** transform silent OOB clamp / exact-boundary, `max_candidates≤0`→[],
  `_margins` 7-vs-8 arity annotation, δ range-scaling untested, pessimistic interval
  unasserted, Sobol non-pow2 warning, jitter docstring, percent/ppm compositional
  unit, Greenwood var, ACI err-timing, c_batch, M2 direction (D1–D14).

**Two systemic blind spots** flagged for a future hardening pass: nearly all
UQ/eval tests are single-output (m=1), and the adaptive cores (BO EI, AL stall)
are only exercised where the seed DoE already wins. None of this blocks M0/WP-E.

Note: the audit workflow hit the account session limit near the end — 5 inverse +
4 other verifiers and the determinism auditor were killed; those 9 findings + the
determinism sweep were re-verified by hand (direct read + reproduction) and are
included above.

## 2026-07-16 — audit remediation (all 33 findings fixed)

Fixed every finding in `docs/audit-2026-07-16.md` (user: "fix all that you have
identified"). Suite **272 → 306 passed** (34 new guarding tests); ruff clean;
import-linter KEPT. CI-path (no sim) = 265 passed, 41 skipped. Nothing committed.

**Code changes (behavioral):**
- A1 CI blind spot — NEW `tests/conftest.py`: `RIG_REQUIRE_MBE_SIM=1` turns a
  missing sim into a hard error; a prominent terminal banner fires whenever the
  41 MBE tests skip. `ci.yml` now runs `-rs` + documents the gap.
- B1 `schema.py` — `validate_against` now enforces simplex sum-to-1 + component
  completeness for any compositional factor present.
- B2 `mbe/machine.py` — `_coerce_recipe` validates a `RecipeRecord` against
  RECIPE_VARIABLES only, rejecting machine-config keys on both input paths (E2).
- B3 `tabular/ingest.py` — `_validate_recipe` checks input completeness up front
  and raises a catchable `ValueError` (missing compositional component no longer a
  KeyError that escapes on_error='skip'; incomplete recipe no longer silently
  accepted).
- B4 `tabular/spec.py` — NEW `gp_input_keys` property (drops one component per
  compositional var, order-safe); onboarding doc now uses it instead of `[:-1]`.
- B5 `forward/multitask.py` — implemented `posterior_cov` on
  `MultiToolGPForwardModel` + `ToolBoundForwardModel` (+ `_ICMSingleOutputGP.cov_tool`);
  EPIG/batch AL now drive the chamber model (known-tool diagonal == epistemic²).
- D1 `transforms.py` — `BoxTransform.inverse` raises on out-of-bounds input
  instead of silently clamping. D2/D9 docstring accuracy.
- D3 `inverse/pessimistic.py` — `solve` rejects `max_candidates < 1`. D4 fixed the
  `_margins` 8-tuple annotation/docstring.
- D8 `interfaces.py` — `sobol_seed_design` suppresses the non-pow2 balance warning
  + guards empty ranges. D10 `tabular/spec.py` — compositional unit gate compares
  the unit itself (rejects percent/ppm).

**Test-only guards added (C1–C11, D5–D7, D11–D14):** FD test of the single-output
NLML gradient; posterior_cov invariant; jackknife ±inf tiny-n + multi-output
conformal; ACI pre-update err-timing; SBC small-M continuity; exploitation m≥3
joint-vs-per-output; KM Greenwood pin; EPIG closed-form magnitude; λ blend
direction; AL stall-stop path; c_batch accounting; δ range-scaling; pessimistic
interval endpoints; distinct-candidates ≥2; NEW `tests/test_warm_bo.py` (EI
known-answers); M2 adaptive-phase-contributes + direction-surfaceable. Each was
confirmed to fail against the described regression before landing.

The 2 refuted findings were intentionally NOT "fixed" (recipe completeness is
guarded by tested adapter encoders; independent-variance RMST is by-design per
§12.4). Determinism sweep stays clean.

---

## 2026-07-16 — Session 3: M2 empirical result (honest), BO scale-bug fix, M0 dataset hunt

**M2 empirical result produced, adversarially validated, and rebuilt honestly.**
Full write-up: `docs/M2-result-2026-07-16.md`; result JSON: `docs/m2-result.json`.

Files touched:
- NEW `src/rig/eval/m2_sweep.py` — method-agnostic RIG-vs-BO cost-to-target
  evaluator (difference-in-RMST primary, per-target + pooled, paired win-rate,
  paired bootstrap CI, both-hit split). Imports ONLY `rig.eval.survival` (import
  contract preserved). Exported from `rig.eval`.
- NEW `tests/test_m2_sweep.py` — deterministic direction check + well-formed-panel
  smoke (harness validity, not a flaky "RIG wins").
- NEW `examples/run_m2_sweep.py` — powered driver on the InSilicoMachine.
- CHANGED `src/rig/baselines/warm_bo.py` — **real baseline bug fix (BF-1):**
  `_distance_to_box` now normalizes each box residual by the per-output tolerance
  (scale-fair scalarization). Raw L2 was numerically blind to small-scale outputs
  (`bow_cooldown_um` ~1e-4 vs `T_center` ~1e3), so BO could never satisfy the
  small-scale box — a scalarization artifact, not a method gap. Fix makes BO
  STRONGER; existing `tests/test_warm_bo.py` unaffected.
- NEW `docs/M2-result-2026-07-16.md`, `docs/m0-dataset-candidates.md`.

What happened (the honest path):
1. A v1 config (single-output/separable-identity target, all pathologies OFF,
   un-anchored `tol_frac` knob) reported a big RIG win. A 6-lens adversarial
   validation workflow **REFUTED it**: 9 attacks survived, 1 FATAL (TS1: doubling
   the un-anchored tol → exact tie). The v1 *stats* were sound (11 stats attacks
   refuted); the *experimental config* was rigged-easy. Not reported as a result.
2. Rebuilt honestly: metrology_noise ON (stochastic machine, real seeds),
   coupled non-identity target `T_center × bow_cooldown_um` (a physics probe
   `scratchpad/probe_machine.py` showed bow is the only genuinely 2-D output;
   thermal KPIs collapse to T_heater, thickness_grown is a literal identity),
   metrology-anchored tol = tol_k·σ + a tol-sensitivity CURVE, scale-fixed BO,
   richer reporting (both-hit ΔRMST, pathology config serialized, support-score).
3. Verified before trusting: RIG hits are **100% from the inverse exploit**
   (`scratchpad/attribute_rig.py`); BO scale-fix lifts BO hit-rate 0.38→0.58.
4. Powered run: RIG ΔRMST=−2.55e4 (p≈1e-156, win 93%, hit 1.00 vs BO 0.42);
   both-hit ΔRMST=−1.4e4 (n=67); **RIG wins at every tol_k∈{2,3,4,6,8}** with a
   smooth robustness gradient — the decisive contrast with v1's knob artifact.

Suite 308 green, ruff clean, import-linter exit 0.

**Gotchas:** (a) Unicode Δ crashes cp1252 Windows stdout → ASCII "dRMST" in
prints. (b) `records_to_arrays` returns SI-canonical magnitudes (percent→fraction,
µm→m) so metrology σ must be measured empirically in the sweep's output space —
done via a 400-rep noisy probe, not analytics. (c) piping a live run through
`| tail` buffers all output until EOF — run background jobs WITHOUT the pipe to
see progress. (d) The pessimistic solver conservatively reports the tight box
INFEASIBLE on sparse noisy data but its `nearest_achievable` still hits — so
attribution to the exploit is legitimate (the solver's fail-closed verdict ≠ its
best-effort recipe).

**In flight at session end:** v2 6-lens re-validation workflow (attacks the NEW
decisions: tol-curve robustness, coupling adequacy, BO-fix fairness,
INFEASIBLE-fallback, pathology realism, stats); a general-purpose agent wiring the
magnetron-sputtering SDL dataset through the WP-H tabular adapter as an
M1-machinery-on-real-data proof (`examples/real_data/sputtering/`). NOTE: an
account session limit truncated the dataset-hunt completeness pass + the NREL HTEM
verifier (that entry is unverified in the shortlist).

**M0 dataset hunt** (`docs/m0-dataset-candidates.md`): 26 verified public
recipe→outcome datasets. No public MBE set exists; best real semiconductor/
thin-film = BOSCH plasma-etch (Zenodo) + NREL HTEM (unverified) + magnetron-
sputtering SDL; best machinery testbeds = Buchwald-Hartwig / Suzuki-Miyaura HTE,
Olympus, Summit; largest recipe→outcome = Perovskite Database (~42k).

Nothing committed.

## 2026-07-16 — Session 3 (cont.): M1 machinery proven on real public data

Real-data machinery proof (general-purpose agent, independently re-run & verified
by the lead before recording): RIG's forward+conformal+inverse stack works
end-to-end on GENUINE measured data via the WP-H tabular adapter — NOT the M1
program gate (that needs the real MBE target data = M0), framed as such.

- Dataset: magnetron-sputtering SDL (`jarlsanna/gps-for-magnetron-sputtering`,
  `Zr_grid.csv`) — actually a 15×15 power×pressure grid = 225 rows, of which 16
  are GP-augmented (`synthetic==1`); kept the **209 measured** rows. Inputs
  power[W]/pressure (declared `mtorr` — `mTorr` is not pint-parseable); outputs 3
  QCM mass-rate channels [ng/(cm²·s)] each with a per-point `_error` σ.
- Artifacts (all under `examples/real_data/sputtering/`, nothing shared touched,
  not committed): `sputtering.toml` spec, `run_m1_sputtering.py` (deterministic),
  `RESULTS.md`, local `Zr_grid.csv` (not redistributed — no dataset license).
- Verified numbers (seed=0, 125/42/42 split): per-output test RMSE
  {qcm_1 2.77 (6.8%), qcm_2 0.45 (11%), qcm_3 0.82 (5.5%)}; conformal mean
  **PICP 0.929** vs 0.90 nominal (slightly conservative); CRPS/QCE/PIT-KS
  consistent with an approximately-calibrated predictive. Fitted aleatoric floor
  is 4–10× the CSV counting-σ — expected: the v0 constant floor absorbs model
  misfit + run-to-run variation, not just QCM σ.
- §8 inverse demo: reachable target → FEASIBLE with 3 diverse on-support recipes
  (§8.7 non-injectivity: distinct power/pressure, same rate band); a tighter
  target → correct well-formed INFEASIBLE. Both paths exercised.

Gotchas: `mTorr`→`mtorr` for pint; cp1252 console crashes on κ/σ/§ in the §8
diagnostic strings → script forces `sys.stdout.reconfigure(encoding="utf-8")`;
dataset is 225 rows (15×15), not the ~625 the task guessed. Nothing committed.

## 2026-07-16 — Session 3 (cont.): M2 v2 re-validation → claims narrowed (not overturned)

A v2 6-lens adversarial re-validation (18 agents) attacked the honest M2 result on
its NEW decisions. **6 attacks survived, 6 refuted, 0 clean lenses.** The core
ΔRMST verdict (RIG's loop reaches spec cheaper in-silico) SURVIVED — none of the
six overturns it — but two CONFIRMED findings mean the write-up's *claims* were
overstated and have been corrected in `docs/M2-result-2026-07-16.md`:

- **BF-1a (major, CONFIRMED):** BO only searches a discrete 128-pt Sobol pool; the
  spec box is sub-pool-resolution (0/128 in-spec for all 4 targets), while RIG's
  winning exploit optimizes CONTINUOUSLY (L-BFGS multistart). So "n_pool identical"
  equalized only RIG's explore channel, not its hit channel. My earlier
  "continuous-EI doesn't help BO" note was a v1 result and does NOT apply to the v2
  (tight-box) config — dropped. Claim narrowed: "beats a FIXED-POOL GP-EI", not
  "beats a competent continuous BO". A continuous `optimize_acqf`/TuRBO BO is owed
  (WP-E) and could close part of the gap.
- **IF-1 (major→minor, CONFIRMED):** the pessimistic solver returns INFEASIBLE on
  100% of solves; every RIG hit is the `nearest_achievable` fallback → the
  κ/z_epi/δ feasibility guarantee NEVER binds (GP aleatoric σ for bow collapses to
  its 1e-9 floor → credited band degenerate). So M2 validates a margin-guided
  point-inverse, NOT the *robust* pessimistic inverse. Cost numbers unaffected;
  attribution prose corrected. Owed: feasibility-attribution column + nominal
  mean-inverse ablation, or recalibrate the aleatoric floor.
- **A1-1 (major→minor, PLAUSIBLE):** headline uses only the RIG-benign
  metrology_noise; adding §10 drift cuts RIG hit 1.00→0.71 and halves the margin
  (still significant). Quantified in Scope.
- **TS1-v2 (minor), BF-1b (minor), SV-3 (cosmetic):** σ-anchor makes "6σ"≈5.1-5.8σ
  (bounded by tol-curve, non-overturning); scalarized-GP BO vs constrained-BO/SCBO
  (owed); both-hit split is collider-conditioned (conservative for RIG). All noted.

LESSON reinforced: even the "honest" v2 had two overstated claims the fleet caught.
Build → validate → **correct the claims to match what survived**, don't defend the
first framing. BUILD_STATE M2 row updated to "PASS in-silico vs FIXED-POOL GP-EI
(claims narrowed)". Owed next: continuous/constrained BO baseline (WP-E) +
feasibility-attribution/floor recalibration + full-pathology companion. Nothing
committed. Workflow: wf_c7e68a82-1c6.

## 2026-07-16 — Session 3 (cont.): BF-1a crux test — continuous-acquisition BO

Resolved the major re-validation finding (BF-1a) by TEST, not caveat. Added a
continuous acquisition optimizer to the BO baseline: `WarmStartedBO(acq_optimize=
True)` runs L-BFGS multistart on the EI surface (the optimize_acqf pattern) instead
of only ranking a discrete Sobol pool. Default-off (existing tests + behavior
unchanged; test_warm_bo 4/4, full suite green). Files: `src/rig/baselines/warm_bo.py`
(+_ei_at/_neg_ei_u/_refine_u/_pick_batch helpers, acq_optimize/acq_restarts ctor
args); crux experiment `scratchpad/bo_continuous_crux.py`.

Crux head-to-head (honest config, CRN, 4 targets x 15 seeds = 60 pairs/arm):
  rig      hit 1.00  median-cost 14000  RMST 15417
  bo_fixed hit 0.45  median-cost 29000  RMST 40084
  bo_cont  hit 0.90  median-cost 31000  RMST 33267
  dRMST[rig-bo_fixed]=-24667 (p=6e-54); dRMST[rig-bo_cont]=-17850 (p=1.8e-31);
  dRMST[bo_cont-bo_fixed]=-6817 (p=1.6e-3).

Finding: continuous acquisition DOUBLES BO hit-rate (0.45->0.90) — the discrete
pool was a genuine RELIABILITY handicap (BF-1a correct). BUT RIG still reaches spec
~2x cheaper than the continuous BO (14k vs 31k; dRMST -17850, ~72% of the
fixed-pool gap retained) because its inverse converges in the FIRST adaptive batch
while EI-BO needs many. => M2 cost-to-target verdict SURVIVES a competent
continuous BO; the "100% vs 42% hit-rate" reliability contrast does NOT and was
dropped from the claim. docs/M2-result-2026-07-16.md updated (crux subsection +
verdict corrected). Still owed: constrained-BO/SCBO (BF-1b) + BoTorch/TuRBO (WP-E);
feasibility-attribution + aleatoric-floor recalibration (IF-1); full-pathology
companion (A1-1). Nothing committed.

---

## 2026-07-16 — Session 3 (cont.): IF-1 fully resolved (§8 attribution + binding study)

Closed the last open M2 re-validation finding, IF-1 ("the pessimistic feasibility
guarantee never binds → the win is the margin-guided fallback, not the robust
inverse"). Three pieces of work, all scratchpad + docs (no core-logic change beyond
one honesty readout):

1. **Mechanism diagnosis** (`scratchpad/if1_diagnose.py`) — OVERTURNED the earlier
   "aleatoric σ collapses to its 1e-9 floor → credited band vanishes" story that was
   written into the M2 doc. Real cause at the tight 6σ joint spec: (a) the sparse GP
   mean barely reaches the coupled box (0% of grid recipes put BOTH KPI means in-box
   at n=8), and (b) the robust margin is standardized by a near-floor
   σ_ale(bow)≈7.5e-9, so a modest ABSOLUTE epistemic band (~0.5·tol) reads as a large
   σ-unit deficit. INFEASIBLE is the solver being HONEST ("mean reaches spec,
   uncertainty too large — collect more runs"), not a degenerate test. Corrected the
   mechanism paragraph in docs/M2-result-2026-07-16.md.

2. **Ablation: is pessimism what wins M2?** (`scratchpad/if1_ablation.py`, 4 targets
   × 15 seeds, CRN-paired, RIG z_epi=1/κ=1 vs mean-inverse z_epi=0/κ=0 in the SAME
   loop.) Both hit 1.00 @ median cost 14,000; ΔRMST[pessimistic−mean] = −1,083,
   p=0.053, differ in 30/60 campaigns. → The M2 COST win is "inverse-guided loop
   beats BO," robust to turning pessimism OFF (mean-inverse RMST 16,500 still crushes
   continuous BO's 33,267). Pessimism reshapes the fallback exploit in half the
   campaigns but its edge is marginal + non-significant.

3. **Binding study: where does §8 bind, and does it pay?** (`scratchpad/if1_binding.py`.)
   FEASIBLE-certificate fraction is a step function of SPEC TOLERANCE, not data
   budget: 0% at tol_k=6, 100% at tol_k∈{15,30,60}, uniformly across n_seed∈
   {8,20,40,80}. Where it binds, re-measuring each certified recipe 200× under
   metrology noise: pessimistic cert gives 0.0% false-success (vs mean-in-box 7.8% at
   the tol=15σ boundary) and ~95–98% credited-interval coverage (vs mean's degenerate
   0%). → §8's distinct payoff is CALIBRATED FEASIBILITY, validated in-silico, on an
   axis orthogonal to the M2 cost race.

Files touched: examples/run_m2_sweep.py (feasibility-attribution readout —
`_inverse_readout` now returns `verdict ∈ {FEASIBLE_CERTIFIED, INFEASIBLE_FALLBACK}`
+ distance_to_feasible + §8.8 cause; added `Infeasible` import). docs/M2-result-
2026-07-16.md (corrected mechanism + added Attribution ablation/binding tables +
updated IF-1 row + top caveat + closing "owed"). docs/BUILD_STATE.md M2 row.
Suite 308 green, ruff + import-linter clean. Nothing committed. Gotcha: cp1252
Windows stdout crashes on κ/§/σ — scripts must `sys.stdout.reconfigure(encoding=
"utf-8")`; and buffered stdout hides interim progress on background runs — use
`python -u`. STILL OWED (deferred to WP-E): SCBO/constrained-BO (BF-1b) +
BoTorch/TuRBO slate; A1-1 full-pathology companion.

---

## 2026-07-17 — Session 4: WP-E unblocked (torch installed) + deep-ensemble forward tier (backend B)

**Torch stack installed & GPU-verified.** Per the user's "continue the build, install
torch": installed `torch 2.11.0+cu128` from the CUDA-12.8 index
(`--index-url https://download.pytorch.org/whl/cu128`) + `gpytorch 1.15.2` +
`botorch 0.18.1` from PyPI. RTX 5050 Laptop GPU detected at compute capability
**(12,0) = sm_120 (Blackwell)**, driver 577.05; confirmed real GPU compute — a
512×512 matmul + autograd backward runs on `cuda` (detection alone is not enough on
a brand-new arch). Added a `[torch]` optional-dependency extra to `pyproject.toml`
(torch≥2.7 / gpytorch≥1.13 / botorch≥0.12) with the cu128 install note. **WP-E is no
longer BLOCKED.**

**First WP-E slice — the D3 large-data forward backbone (implementation-plan §5.4 backend B).**
New `src/rig/forward/ensemble.py :: DeepEnsembleForwardModel`:
- K-member deep ensemble (default 5 dev / 10 final), **no bagging** (D3) — diversity
  = independent seeded inits + per-member input jitter (input-domain randomization).
- Each member: **spectral-normalized (≈bi-Lipschitz) ResMLP trunk** (`spectral_norm`
  on every linear + residual blocks) → **heteroscedastic β-NLL(β=0.5) aleatoric head**
  (`stopgrad(σ^{2β})` reweight, Seitzer 2022; §5.4) + a **RFF-GP SNGP last layer** for
  the mean (fixed random Fourier features of the trunk φ; trained β = Bayesian-linear
  mean; Laplace covariance computed post-hoc).
- **PredictiveDistribution (canonical, §3.2, verbatim field order):** mean = mixture
  mean; `aleatoric_sigma = sqrt(E_m[σ_m²])`; `epistemic_sigma = sqrt(Var_m[μ_m] +
  E_m[SNGP-Laplace var])` — ensemble spread PLUS the distance-aware RFF-GP term. The
  SNGP term is what makes epistemic inflate OOD (plain ensembles agree confidently far
  out — the plan's own §5.4 honest-qualification warning); verified ~14× inflation.
- `support_score` = negative Mahalanobis in the **spectral-normalized latent φ** of the
  reference member (§8.2/§11 gate space; upgrade over the GP's input-space Mahalanobis).
- `jacobian` = autograd d(mixture-mean)/dx, (m,d), raw units (chains both
  standardizations). `update(records)` refits on old+new (invariant 2d).
- Training defaults per §5.7: AdamW lr 1e-3 + cosine decay, wd 1e-4, batch 128, early
  stop on val β-NLL (patience 30). Nets float32 (GPU); RFF-GP Laplace algebra in
  float64 numpy/scipy (Cholesky, reusing `_gp_common` patterns). `device="cpu"` default
  is bit-deterministic; `device="cuda"` runs the sm_120 path.

**Design rationale for the epistemic mechanism** (so a future session doesn't "simplify"
it away): the OOD inflation is the RFF-GP Laplace variance `Φ(x*)ᵀ S⁻¹ Φ(x*)` with
`S = ridge·I + Φᵀ diag(1/σ²) Φ`. Far from training, Φ(x*) decorrelates from the trained
Gram → variance rises toward the prior `≈1/ridge`; the spectral (bi-Lipschitz) trunk is
what preserves input distance into φ-space so this is a *valid* distance signal. The
ensemble `Var[μ_m]` alone would NOT pass §5.9 invariant 1 — do not drop the SNGP term.

**Wiring:** `rig.forward.__init__` exposes `DeepEnsembleForwardModel` via a lazy
`__getattr__` so `import rig` / `import rig.forward` stays **torch-free** for the
numpy/scipy core + CI paths that never install `[torch]` (verified: torch not in
`sys.modules` after base import).

**Tests** — `tests/test_ensemble.py` (13, `pytest.importorskip("torch")` so absent-torch
CI just skips): ForwardModel-protocol + canonical-distribution field order; shape
contract ((d,)→(m,), (n,d)→(n,m)); beats predict-the-mean; **§5.4 β-NLL no-collapse**
(aleatoric stays in [0.03,0.3] around true 0.1, positive); **§5.9 invariant 1 OOD
epistemic >3× in-range** (got ~14×); support_score discriminates + typed;
jacobian≈finite-difference; determinism (same seed → bit-identical, CPU); not-fitted
raises; multi-output; **§5.6 conformal-wrapper PICP≈nominal** on a held-out split;
CUDA-path smoke (skipif no device).

**Suite:** 321 passed (was 308 + 13). One transient FAILED in the run —
`test_transforms.py::test_box_output_always_in_bounds`, a Hypothesis
`HealthCheck.too_slow` on *input generation* (7s) triggered by system load while the
CUDA tests ran; **passes in isolation in 2.25s** — environmental, not a logic
regression, and unrelated to this change. ruff clean; import-linter KEPT.

Gotchas: (1) base import must stay torch-free — never add `ensemble` to
`forward.__init__`'s eager imports; the lazy `__getattr__` is load-bearing. (2) The
RFF `sqrt(2/D)` normalization + per-output heteroscedastic `1/σ²` weights matter — an
unweighted or unnormalized Laplace mis-scales the epistemic band. (3) CUDA is not
bit-deterministic (float kernel nondeterminism); determinism test pins CPU only.

STILL OWED (WP-E, dependency-ordered): ensemble distillation → single distributional
net + SNGP-single-member inner loop (§5.7, so §8/§9 don't run K forwards); BoTorch
continuous/constrained BO slate (qLogEI/qLogNEHVI/TuRBO/SCBO — closes M2 BF-1b +
A1-1); amortized NPE flow generator + SBC/TARP gate (§14.3/§14.6 = the M3 gate);
normalizing-flow typicality + PGD δ-box for §8. Nothing committed.

---

## 2026-07-17 — Session 5: renames + WP-E slices 2-3 (inner-loop surrogate, BoTorch baseline) + adversarial verify

Continuation of Session 4 ("continue the build until a blocking decision; think and
verify before building"). Two housekeeping renames + two WP-E slices, each designed
against a recon workflow and adversarially verified before finalizing.

**RENAMES.** (1) `nnplan.md` → `implementation-plan.md`; global token replace across
79 files (all refs were prose/§-citations/labels — no functional strings; import-linter
contract name changed cosmetically). CLAUDE.md + docs + memory updated. `grep nnplan`
= 0. (2) Repo FOLDER `seminn` → `rig` — **ATTEMPTED, BLOCKED**: the folder is the live
workspace/CWD and is locked by the running session/editor ("The process cannot access
the file because it is being used by another process"). It CANNOT be renamed from inside
the session. TO DO IT: close this session/editor, then from `C:\Users\Jiaow\Documents\github`
run `Rename-Item seminn rig` (or `mv seminn rig`), and reopen the project at the new path.
SIDE EFFECT the user must know: the harness scratchpad + auto-memory are keyed to the
`...github-seminn` path and will be ORPHANED by the folder rename — migrate the memory dir
`C:\Users\Jiaow\.claude\projects\c--Users-Jiaow-Documents-github-seminn` to the
`...github-rig` equivalent (not auto-migrated). Package name is unaffected (always `rig`).

**RECON (workflow wf_61cc57d4-980, 4 agents).** Specced the next slices against the
plan + code seams and surfaced the real blocker (below). Confirmed the exact seams:
AL loop's `surrogate_factory` is the ensemble injection point; `PessimisticInverseSolver`
takes any ForwardModel (fast view drops in); m2_sweep methods need only `.run()->Trajectory`.

**WP-E SLICE — inner-loop surrogate + EPIG-capability (§5.7 / §8 budget).**
`src/rig/forward/ensemble.py`:
- `posterior_cov(X1,X2)` — per-output joint epistemic covariance (m,n1,n2) = ensemble-
  spread cov `Cov_m(μ_m)` (ddof=0) + ensemble-mean SNGP-Laplace joint cov. Its DIAGONAL
  equals `predict().epistemic_sigma²` to machine precision (verified 2.3e-16), so the §9.4
  EPIG (`acquisition.epig`) is self-consistent. This makes the deep ensemble a full
  `_JointModel` → the §9 AL loop now runs on backend B (previously EPIG would crash: no
  posterior_cov). DO NOT return only the SNGP term — the diagonal would then mismatch
  epistemic_sigma² and mis-scale EPIG's σ²(x*|x) reduction.
- `sngp_member_view(member=0)` / `inner_loop_surrogate(mode="sngp_member")` → a fast
  single-member ForwardModel view (`_SNGPMemberView`): predict = one member's mean +
  aleatoric + its own SNGP-Laplace epistemic (no ensemble-spread term); support_score
  delegates to the parent (shared spectral latent); jacobian = single-member autograd;
  posterior_cov = that member's SNGP joint cov. ~K× cheaper (measured 4.8× at K=5; ~K×
  at K=10). §5.7 option B; the distilled distributional net (option A, Malinin 2020, the
  ≥20× / /invert serving path) is a named follow-on.
`src/rig/active/loop.py`: guarded wiring — `inner = surrogate.inner_loop_surrogate() if
hasattr(...) else surrogate`, recomputed each batch (after the per-batch refit), used for
the §8 solver + EPIG/BALD acquisition + select_batch; the FULL model is still refit and
the exploit is fired on the real machine (oracle re-validation). GP tier has no such attr
→ byte-for-byte unchanged (verified: loop-wiring finder returned ZERO findings).
`src/rig/inverse/pessimistic.py`: opt-in `revalidation_model` (default None ⇒ behavior
byte-for-byte identical to the M2/WP-D path). When set, solve() re-scores the selected
set on the full model (+ conformal C(x')⊆Z* gate, §13.2) and drops what it does not
certify; `_margins`/`_evaluate` gained an optional `model=` param (default self.model).

**WP-E SLICE — production BoTorch BO baseline (§9.8 / §12.3), closes M2 BF-1b (continuous).**
`src/rig/baselines/botorch_bo.py :: BoTorchBO` — `SingleTaskGP` (Matérn-5/2 + §20.5
Hvarfner √D dim-scaled prior + input Normalize + outcome Standardize) + `qLogEI`/`qLCB`
optimized CONTINUOUSLY via `optimize_acqf`. Fair matched-budget comparator: IDENTICAL
Sobol warm-start to WarmStartedBO (bit-verified), same tol-normalized g(x)=‖relu(L−y,
y−U)/w‖₂ objective (models f=−g since BoTorch maximizes; best_f=−min g), same any-in-lot
hit rule, same Trajectory, budget = machine queries only, PLAIN posterior (no μ−κσ leak),
deterministic (torch.manual_seed per fit/propose). Continuous-only (raises on
compositional). Lazy-imported in `rig.baselines.__init__` (base stays torch-free, verified).
qLogNEHVI / SCBO / TuRBO = follow-on WP-E.

**ADVERSARIAL VERIFY (workflow wf_fca06e2d-8f3, 4 finders → per-finding verify).** The two
highest-risk dimensions — ensemble covariance MATH and loop wiring — returned ZERO
findings (clean); BoTorch sign-convention/fairness core clean. 4 findings, ALL LOW, ALL
confirmed, ALL fixed + guarded:
1. `_revalidate` gated the FULL model's support against the FAST model's floor (support_score
   is per-model). FIX: `_reval_support_floor()` derives the floor from the revalidation
   model (stored `_X_train`). Guard: `test_revalidation_floor_is_per_model`.
2. A conformal-only rejection returned `distance_to_feasible=0.0` (contradictory) + an
   epistemic "collect runs" reason (wrong cause — it's aleatoric/coverage). FIX:
   `_reval_infeasible` diagnoses the cause; conformal rejection → nonzero `_conformal_spill`
   distance + a "conformal band wider than spec box; reduce variation / relax κ" reason.
   Guard: `test_revalidation_conformal_rejection_diagnosed`.
3. BoTorch docstring said "Matérn-5/2" but `get_covar_module_with_dim_scaled_prior` defaults
   to RBF in botorch 0.18. FIX: `use_rbf_kernel=False` → real Matérn-5/2 (also matches the
   RIG GP tier's kernel family). Guard: `test_gp_uses_matern_kernel`.
4. BoTorch searched the full closed box while the reference arms search the box-sigmoid
   interior (u∈[−u_bound,u_bound]); the two "matched" BO arms had different domains
   (conservative-for-RIG, but real). FIX: clamped `_bounds` to the same sigmoid interior.
   Guard: `test_search_bounds_match_reference_interior`.

**Suite 342 passed** (was 308 at Session 3; +13 ensemble Session 4, +9 ensemble +8 botorch
+4 fix-guards Session 5), ruff + import-linter clean. Also hardened
`test_box_output_always_in_bounds` with `suppress_health_check=[too_slow]` — a Hypothesis
input-GENERATION timing flake that trips only under the CPU load of the torch tests (not a
property failure). Nothing committed.

**► BLOCKING DECISION (the stopping point the user asked to run up to): the M3 amortized
NPE generator (§14.3) needs a conditional-flow density estimator, and NONE is installed**
(sbi/pyro/zuko/nflows all absent; torch/gpytorch/botorch are the only torch libs). This is
the load-bearing dependency choice for the M3 gate. Options: (a) `pip install sbi` — fastest
to M3 (NPE-NSF + SNPE-C/APT + SBC/TARP built in, matches §14.3 verbatim) but a LARGE
transitive tree and torch-2.11-compat unverified (sbi/pyro may pin older torch); (b)
`pip install zuko` only (pure-torch neural-spline flows, dep = torch alone) + hand-wire the
NPE loop, reusing WP-G's existing SBC/TARP gates — tiny dep tree, on-prem-clean; (c)
hand-roll a conditional rational-quadratic spline flow in pure torch (zero new deps, most
eng effort). Everything up to this point (both slices) is done + green; M3 waits on this call.

---

## 2026-07-17 — Session 6: M3 amortized NPE generator (zuko) + SBC/TARP gate — DONE + adversarially verified

**Decision resolved:** the Session-5 blocking flow-library choice → **zuko** (user: "zuko is
good"). Pure-torch conditional neural spline flows, dep = torch alone (on-prem-clean); NPE
loop hand-wired, SBC/TARP reused from WP-G `rig.eval.calibration_gates`.

**Built `src/rig/inverse/amortized.py :: AmortizedInverseGenerator` (+ `CalibrationGate`).**
The M3 §14.3 "instant-answer" D2 proposal service:
- **Constraint-by-construction (§14.4):** K flows are trained + sampled in the UNCONSTRAINED
  `u`-space of `RecipeTransform`; every `u→recipe` map goes through box-sigmoid / simplex-
  softmax, so a proposal is ALWAYS feasible (box + simplex hold exactly). Verified: box
  samples ∈ [lo,hi], compositional samples sum to 1 (atol 1e-6).
- **Region-augmented box conditioning (D2):** each flow is a zuko `NSF(features=d_u,
  context=2m, ...)`; training draws a random standardized box `[y−hw, y+hw]`, `hw∈region_hw`,
  around each simulated outcome, so `q(recipe | y∈box)` is learned directly (a point target =
  a tight box). Verified: proposal for a box around y=2 recovers mean≈2.0 AND width≈0.3 (the
  true posterior spread — not collapsed).
- **Deep ensemble (§14.3):** K members (seed diversity); samples are the even MIXTURE
  (`_member_counts` split), `log_prob` is the mixture log-density (logsumexp − log K).
- **§14.6 SBC/TARP BLOCKING gate** `validate(simulator, prior_sampler=None, ...)`: draws
  prior recipes → simulates y → samples the posterior in u-space → `sbc_ranks`/`sbc_test` +
  `tarp_test`. Verified it PASSES a well-trained flow and BITES an undertrained (2-epoch) one.
- Lazy-imported by `rig.inverse.__init__` (`__getattr__`) so `import rig` stays torch-free.
- zuko added to the `pyproject [torch]` extra.

Files: `src/rig/inverse/amortized.py` (new), `src/rig/inverse/__init__.py` (lazy export),
`pyproject.toml` (`zuko>=1.4`), `tests/test_amortized.py` (new).

**ADVERSARIAL VERIFY (workflow wf_4e3547fa-a9f, 9 agents: 3 dimension finders — sbc-tarp-
correctness, flow-training-context, sampling-constraints-determinism — each with per-finding
adversarial verify).** 6 findings, ALL CONFIRMED, ALL fixed + guarded:
1. **HIGH — the gate drew ONE member per trial, not the shipped MIXTURE.** `validate` did
   `k=rng.integers(n_members); flows[k].sample((n_posterior,))` → all posterior samples from a
   single component, narrower than the mixture whenever members disagree (the §14.3 epistemic
   spread) → U-shaped SBC ranks → a calibrated mixture is FALSE-FAILED and a member-tight but
   over-wide mixture FALSE-PASSED. The blocking gate certified a DIFFERENT law than
   `sample_array` ships. FIX: `_draw_u_std_mixture(ctx, n, base_seed)` splits the draw across
   ALL members (mirrors `_member_counts`) — used by `sample_array` AND `validate`, so the gate
   now certifies exactly the shipped law. Guard: `test_gate_posterior_draws_the_mixture_not_one_member`.
2. **MED — default SBC prior was a moment-matched Gaussian.** `prior_sampler=None` drew
   `u ~ N(u_mean, u_scale)`, valid only if the training-u marginal is Gaussian; uniform-in-box
   DoE → logistic u, clustered ops → bimodal u, so a calibrated flow is tested against the
   wrong posterior (silent invalid PASS/FAIL) on the path-of-least-resistance default. FIX:
   bootstrap-resample the empirical training-u rows (retained `self._U_train` in `fit`) — the
   exact prior the flow trained under. Guard: `test_default_prior_bootstraps_training_recipes`.
3. **MED — `_spec_context` served unconstrained/one-sided outputs a 6σ (width-12) box** — ~3×
   the widest box the flow saw in training (region-augmentation widths `2·hw ∈ (2·hw_lo,
   2·hw_hi)`), far OOD, corrupting the joint-conditioned constrained outputs too; and
   `validate` (symmetric hw_std boxes) can't detect it. FIX: rewrote `_spec_context` to close
   an open side to the MAX trained box width `2·region_hw[1]` — an unconstrained output →
   widest box centered at the outcome mean; a finite one-sided bound anchors its edge and the
   open side extends by the max width. Guard: `test_spec_context_unconstrained_output_stays_in_trained_width`.
4. **LOW — inverted spec box** (lower>upper) for a finite one-sided bound beyond ±6σ (the old
   inf-clamp pinned the open side independent of the finite bound). Structurally fixed by the
   #3 rewrite (open side is always derived from the finite edge). Guard:
   `test_spec_context_extreme_one_sided_not_inverted`.
5. **LOW — `sample_array(spec, 0)` crashed** with `ValueError: need at least one array to
   concatenate` (all `_member_counts` zero → empty concat). FIX: `n==0` → clean empty
   `(0, d)`; `n<0` → loud `ValueError`. Guard: `test_sample_zero_returns_empty`,
   `test_sample_negative_raises`.
6. **LOW — `sample()` did a redundant `u→recipe→u→recipe` triple transform** (recomputed dicts
   `sample_array` had already built). FIX: shared `_draw_recipes` path returns `(recipes,
   matrix)` from one draw; `sample`/`sample_array` are thin views. Guard:
   `test_sample_and_sample_array_are_consistent` (dicts == matrix, exact).

**Determinism preserved:** `_draw_u_std_mixture(ctx, n, self.seed)` uses the identical per-
member seeding (`self.seed + 977*k + 1`) and `_member_counts` split as the old `sample_array`,
so `sample_array` output is byte-identical to pre-fix for all n≥1 (existing determinism/
posterior tests unchanged). `validate` per-trial seed = `seed + 100003*i` (non-overlapping for
any realistic K). The two gate tests (`_passes_calibrated_flow`, `_bites_miscalibrated_flow`)
still hold under the corrected mixture law.

**Suite 358 passed** (was 342 Session 5; +16 `test_amortized.py`, of which 7 are fix-guards),
ruff clean, import-linter **1 kept / 0 broken** (verified via `lint_imports` API — the CLI
output is swallowed by a Windows console-encoding quirk, but exit=0 and the API report both
confirm KEPT), `import rig` torch-free. Nothing committed.

**► NO HARD BLOCKER in code.** The M3 amortized gate — the Session-5 stopping point — is
resolved. Standing USER items: (1) M0 real recipe→outcome dataset (#1 program risk,
`docs/m0-dataset-candidates.md`); (2) repo-folder rename `seminn`→`rig` (PENDING the live-
session lock — `Rename-Item seminn rig` from the parent dir with the session closed, then
migrate the `...github-seminn` memory dir). Code follow-ons (not blockers): M3 D2 integration
(amortized proposal → `PessimisticInverseSolver` with the `revalidation_model` conformal gate)
+ an end-to-end M3 acceptance run on the in-silico machine; qLogNEHVI/SCBO/TuRBO slate +
powered M2 re-run; ensemble distillation (§5.7 option A).

---

## 2026-07-17 — Session 7: full-program AUDIT + remediation (10 defects fixed)

**Scope:** user-requested audit — "check for any errors or failures, check if it is able
to achieve the goal of predicting input from output, fix any problems". Four parallel
adversarial audit agents (forward+calibration, inverse, end-to-end/evidence, infra/docs)
plus direct verification. Every finding below was REPRODUCED before fixing and GUARDED by
a test verified to FAIL against the defect. Nothing committed.

### P0 — the suite did not run at all

**The editable install pointed at a directory that no longer exists.** The
`seminn`→`rig` folder rename (listed in BUILD_STATE as PENDING a session lock) had in
fact HAPPENED, but nobody re-ran the editable install, so `_editable_impl_rig.pth` still
mapped to `C:\Users\Jiaow\Documents\github\seminn`. Every `import rig_adapters` raised
ModuleNotFoundError and `pytest` died in conftest collection. Fixed by
`python -m pip install -e ".[dev]"` (which also re-registers the `mbe` entry point).
**Suite now 382 passed / 0 skipped locally** (sim + torch layers both really ran).

### P1 — verification tooling was reporting a constant

**The documented import-linter command always exits 0, even on a BROKEN contract.**
BUILD_STATE prescribed `python -c "from importlinter.cli import lint_imports; lint_imports()"`.
`lint_imports` is a plain function RETURNING an int; only the `lint-imports` console-script
wrapper calls `sys.exit`. The one-liner discards it. Reproduced by injecting
`rig.registry -> rig_adapters.tabular.spec`: report printed "0 kept, 1 broken", exit **0**.
Every "import-linter clean, exit=0" claim through Session 6 was evidence-free on the
exit-code half. Corrected form (now in BUILD_STATE) wraps it in `sys.exit(...)`. The
contract itself DOES hold (verified: 1 kept, 0 broken, exit 0; 45 files analyzed, so not
vacuous). NB a second trap documented there: redirecting stdout makes the `§` in the
contract name raise UnicodeEncodeError → exit 1, a FALSE FAILURE.

### P1 — EPIG was silently disabled on the chamber-onboarding path

**`multitask.posterior_cov` was inconsistent with `predict` on the UNKNOWN-tool branch.**
`predict` inflates (`max_t var_t + spread + (1−ρ̄²)·mean diag B`); `posterior_cov` returned
only the mixture `Σ_t w_t·Cov_t`. `epig()` takes `var_f_star` from `predict` but
`Cov(f(x*),f(x))` from `posterior_cov`, so the two laws broke Cauchy-Schwarz and the
log-ratio collapsed: **EPIG 1.01 vs BALD 19.06 (~19× under-report), and exactly 0.0 nats
across a 4-candidate batch** — i.e. `λ·EPIG+(1−λ)·BALD` degenerated to pure BALD precisely
as λ anneals 0.2→0.9 to let EPIG dominate, and precisely on the §10.4 new-chamber path
`posterior_cov`'s own docstring advertises. `test_gp.py` and `test_ensemble.py` both pin
"posterior_cov diag == epistemic²" for their tiers; multitask had no such test.
**Fix:** `_unknown_tool_cov` builds the mixture's total covariance (within + between +
constant unknown-tool offset — each a valid PSD kernel) then applies a **congruence
rescale** `diag(s)·base·diag(s)`, `s = sqrt(var_u/diag(base))`, which preserves PSD-ness
and the mixture's correlation structure while making the diagonal equal `predict`'s `var_u`
EXACTLY. `predict` was left alone deliberately: its `max_t` is the binding WP-I decision
that makes unknown-tool epistemic dominate every known tool (§5.8 LOTO), so
`posterior_cov` is what had to be reconciled. Guards: 4 tests incl. the sharp
`EPIG(x;{x}) ≡ BALD(x)` identity and a PSD check.

### P1 — a real-data example searched the wrong space (UNITS)

**`spec.continuous` bounds are in DECLARED units; ingested data is SI.** In
`examples/real_data/sputtering/`, pressure is declared `1..43 mtorr` but the data lands in
`0.133..5.73 Pa`. The solver was handed `variables=list(spec.continuous)` → it searched
pressure over **1..43 Pa = 7.5..322 mTorr**, a range whose LOWER bound sits ABOVE the
data's maximum. Only the §8.2 fail-closed support floor kept the answers sane; nothing
errored. The demo's `x_ref = [25.0, 13.0]`, commented *"25 W, 13 mtorr … (guaranteed
on-support)"*, was really **13 Pa ≈ 98 mTorr**, 2.3× beyond the measured max:
**support_score −5.761 vs floor −2.044 — the exact opposite of on-support**, making the
headline "reference" rate a GP extrapolation. **Fix:** new `ProcessSpec.continuous_si`
(SI-canonical bounds — the accessor that pairs with ingested data; `.continuous` must stay
declared-unit because ingest reads cells with it), example switched to it, `x_ref`
converted via the shared `ureg`. Reference now reads **support_score −0.802** (on-support)
and the inverse still returns 3 FEASIBLE candidates. Trap documented in CLAUDE.md.

### P2 — amortized generator (§14.3): three defects

1. **`log_prob` returned the u-space density while advertising `log q(recipe|box)`**,
   excusing the gap as "a monotone reparam of recipe space". A monotone reparam preserves
   the ordering of the VARIABLE, not the DENSITY. The box-sigmoid `|du/dx|` was never
   applied: values integrated to **3.36**, not 1, and — since `σ'(u)` spans ~38× across a
   box — **RE-ORDERED the posterior (3274 of 400×400 recipe pairs backwards)**. Anything
   ranking or importance-weighting proposals by density was silently wrong. **Fix:** new
   `log_abs_det_du_dx` on `BoxTransform` / `SimplexTransform` / `RecipeTransform` (ALR's
   `−Σ_{i=1}^{K} log x_i` via the matrix-determinant lemma; both verified against finite
   differences to ~1e-10), wired into `log_prob` with the `u_scale` term. Now integrates
   to **1.0000**.
2. **`_member_counts` shipped a skewed/truncated sub-mixture** — the small-n mirror of the
   Session-6 HIGH gate bug. `divmod`'s remainder went to the LOW-INDEX members every time:
   `n=3,K=5 → [1,1,1,0,0]` (members 3,4 unreachable); `n=8 → [2,2,2,1,1]` (weights
   .25/.25/.25/.125/.125, not .2). `log_prob` (logsumexp−log K) and the §14.6 gate
   (`n_posterior=100` → exact `[20]*5`) both assume the EVEN mixture — so the gate
   certified one law and D2 shipped another, and `AmortizedRefiner`'s default
   `n_proposals=8` sat exactly on the skewed case. **Fix:** leftovers go to `rem` DISTINCT
   members drawn from the seeded stream → each member's expected weight is exactly 1/K for
   every n, while `K|n` keeps the gate's stratified `[20]*5` bit-for-bit.
3. **`sample()` never advanced its RNG** — `_draw_recipes` passed `self.seed` every call,
   so a posterior SAMPLER returned bit-identical rows: `2× sample(spec,3)` was 3 duplicated
   pairs, collapsing any MC estimate or pooled proposal set built by looping. **Fix:**
   per-instance `_draw_index` advances the stream; §13.4 determinism is preserved in the
   sense that matters (a fresh same-seed generator replays the identical SEQUENCE; `fit`
   resets). NB `test_sample_and_sample_array_are_consistent` was PINNING this bug (it
   compared two separate calls); rewritten to assert its stated intent — the one-draw
   consistency of `_draw_recipes` — plus new tests for fresh draws and sequence replay.
   Also removed a dead `u_bound` kwarg (assigned, never read).

### P2 — §8 solver / AL loop

- **`ActiveLearningLoop.u_bound` was silently dropped**: declared, used for the loop's own
  Sobol pool, and not forwarded to `PessimisticInverseSolver` while every sibling knob was
  → the solver fell back to its own default 8.0. A user setting `u_bound=3.0` got a search
  over u∈[−8,8]; and it made the M2 comparison reach-asymmetric (`WarmStartedBO` uses 5.0:
  RIG reached T_heater∈[1150.117,1499.883] vs the BO arm's [1152.342,1497.658]). Now
  forwarded.
- **`revalidate` could return a FALSE INFEASIBLE** having tested only the `q` diverse picks
  while tens of pool survivors sat unexamined (`_greedy_diverse` cuts by SPREAD, not by
  revalidation merit). Now sweeps the remainder before abstaining — the fast path is
  unchanged; it costs extra only when we were about to abstain. (Latent: `revalidation_model`
  defaults to None.)
- **Linear constraints: docstring claimed a safety feature that does not exist.** It read
  "Linear couplings are enforced by a soft penalty + reject". There is no penalty and no
  reject; the solver takes no `ConstraintSet` and nothing under `rig/inverse/` references
  one. A process declaring a `LinearConstraint` would get violating recipes back with
  `feasibility_flag=True` — a silent wrong answer. Currently UNREACHABLE (no shipped
  adapter declares one). Docstring now states NOT IMPLEMENTED + the precondition for
  onboarding such a process. **Left as owed work, not silently "fixed".**

### P2 — CI blind spot A1, recurring one tier up

`ci.yml` installs only `[dev]`, so `importorskip("torch"/"zuko")` silently skipped **56 of
382 tests (15%) — the ENTIRE WP-E/M3 surface** (deep ensemble, BoTorchBO, amortized
generator, M3 gate) on every green run. The MBE layer got both A1 guards; the torch layer
had neither. Added `RIG_REQUIRE_TORCH=1` strict switch (mirrors `RIG_REQUIRE_MBE_SIM`).
With the ~41 sim tests, a green hosted CI badge covers ~74% of the suite.

### P2 — `run_m2_sweep.py` crashed before writing its own artifact

Missing the cp1252 `sys.stdout.reconfigure` guard both sibling example scripts have, it
raised UnicodeEncodeError printing the IF-1 readout (`reason` contains a sigma glyph)
**after all the compute and before `out_path.write_text`**. So the documented reproduce
command could not regenerate `docs/m2-result.json` — which is why the shipped artifact
still carries the **pre-IF-1 `inverse_readout` schema (no `verdict` key)** while the M2 doc
advertises the IF-1 attribution as wired in. Guard added. **The shipped m2-result.json is
still stale — a powered re-run is owed** (~40-60 min).

### Reproduced, honest, unchanged

- `run_m1_sputtering.py`: mean conformal PICP **0.929** vs 0.90 nominal — reproduces exactly.
- `run_m3_acceptance.py`: byte-identical to the shipped `m3-acceptance.json` (PASS, 5/5,
  cost 0.281×).
- M2 direction/magnitude reproduce at reduced power (ΔRMST −2.67e4 vs shipped −2.55e4).
- **M2 is genuinely NOT circular and the comparison IS fair** — verified in code: both arms
  draw a bit-identical Sobol warm start, same budget accounting, same hit rule on MACHINE
  output, CRN pairing. Ground truth is the simulator, symmetric to both arms. This is the
  strongest part of the program.

### Findings recorded but NOT code-fixed (framing/scope — see BUILD_STATE)

- **F9 (largest un-caveated gap): every result in this repo is 2-DIMENSIONAL.** MBE recipe
  = 2 vars, sputtering = 2, M3 toy = 2. No inverse has ever been run above 2 input dims,
  and inverse problems bite exactly where dimensionality/non-identifiability do. Nothing
  anywhere states this.
- **F7:** PICP 0.929 is bought with fitted aleatoric noise **10.5×/3.9×/5.2×** the CSV's
  measured error — the GP absorbs misfit into "noise" until bands cover. BUILD_STATE quotes
  the PICP with no mention. Also the direct cause of F6.
- **F6:** on real data the inverse is genuinely correct at the demo's ±6 tol (19/19 nearest
  measured runs in-tol) — but ±6 is 15% of the output range, ≈ the conformal MPIW, and 56%
  of all 209 runs already sit in that band. At ±2/±1/±0.5 it is **0/25 FEASIBLE**. Honest
  (it abstains rather than lying) but the certified real-data inverse only fires where the
  answer is nearly free.
- **F3/F2:** the M3 gate is near-tautological (pass rule `>= cold_heavy − 0.02` while all
  confidences saturate at 0.9997–1.0), its "amortization fills the gap" claim rests on
  **n=1** of 5 targets (cold_light already succeeds in 4/5), and it is scored against the
  GP that was fit on the same 220 samples that trained the generator (ground truth never
  called during eval — though 10/10 returned recipes were verified to DO hit ground truth,
  so it is harmless here, not fraud). It runs on a toy tanh, not the InSilicoMachine.
- **M2 runs RIG at `kappa=1.0, z_epi=1.0`** vs the binding §8 defaults of 2.0/2.0 —
  pessimism dialed down, not disclosed in the M2 doc.
- `ruff format --check` fails on 45/83 files, but the formatter has NEVER been adopted (no
  CI step, no config, zero repo references) — "ruff clean" has always meant `ruff check`
  only. Adopt-or-drop is a USER decision; not touched (a 45-file reformat would bury this
  audit's diff).

### VERDICT on the program goal (predicting inputs from outputs)

**The machinery is sound; the goal is NOT demonstrated.** What is real: a calibrated
forward surrogate + conformal layer + continuous pessimistic inverse that runs end-to-end,
reproduces exactly, abstains honestly instead of lying, and beats a matched-budget
scalarized GP-EI BO on a FAIR in-silico cost race. What is not: any evidence that this
predicts recipes from outcomes **in semiconductor manufacturing**. Every quantitative claim
is either in-silico on a simulator, or on a 2-knob public sputtering grid that is a dense
15×15 lookup table (invertible by nearest-neighbour — not a meaningful test of a learned
inverse). **M0 is not a schedule item; it is the entire scientific claim.** The repo's
honesty is well above average (the M2 doc self-refutes its own v1 and discloses a baseline
bug it fixed in the BASELINE's favour; the M1 script says "Never read the numbers below as
'M1 passed'") — the overstatement is in framing/staleness, not in the numbers.

**Files touched:** `src/rig/forward/multitask.py`, `src/rig/inverse/amortized.py`,
`src/rig/inverse/pessimistic.py`, `src/rig/transforms.py`, `src/rig/active/loop.py`,
`src/rig_adapters/tabular/spec.py`, `tests/conftest.py`, `tests/test_amortized.py`,
`tests/test_multitask_gp.py`, `examples/run_m2_sweep.py`,
`examples/real_data/sputtering/run_m1_sputtering.py`, `.github/workflows/ci.yml`,
`CLAUDE.md`, `docs/BUILD_STATE.md`, `docs/BUILD_LOG.md`.

### Also: Session 6.5 was never logged (doc drift)

`src/rig/inverse/d2.py` (`AmortizedRefiner`), `tests/test_d2_integration.py`,
`tests/test_m3_acceptance.py`, `examples/run_m3_acceptance.py`,
`docs/M3-acceptance-2026-07-17.md` + `m3-acceptance.json`, and a MODIFIED `pessimistic.py`
all post-date the last BUILD_LOG entry and appear in NO log entry. BUILD_STATE still listed
D2 integration + the M3 acceptance run as "owed" — they are DONE. That work also added a
binding, undocumented API: **`spec['warm_start_recipes']`** on the §8 solver, which
`d2.py` requires. Now recorded in BUILD_STATE's standing decisions.

## 2026-07-17 — explainer figure fix (docs/rig-explained.html)

All five canvas figures rendered as blank boxes with a broken-image icon on any HiDPI
display. Root cause was a runaway feedback loop in the shared `setup()` canvas helper:
it read the design height back out of `cv.getAttribute('height')`, but `cv.height` is an
IDL attribute that **reflects into that same content attribute**, so after the first paint
the attribute held the dpr-scaled backing store, not the CSS height. Each `ResizeObserver`
pass therefore multiplied the height by `dpr` — measured at 360 → 188,743,680 px (360·2^19)
before saturating Chrome's max canvas size, at which point the backing-store allocation
fails and the browser paints its broken-image placeholder.

Two things made it hide:
- **`dpr == 1` is a fixed point** (`h * 1 == h`), so the bug is invisible on a 1× display
  and only fires on HiDPI / Windows display scaling.
- Allocation failure throws **no exception**, so the file's own `fail()` isolation wrapper
  never tripped and never showed its red stack-trace box — exactly the silent-blank-rectangle
  failure mode the wrapper's comment claims to have eliminated. `fail()` catches throws, not
  a canvas that merely declines to paint.

**Fix:** latch the authored height once in a `WeakMap` (`DESIGN_H`) on first `setup()`, before
it can be clobbered, and read from that thereafter.

**Verified** with headless Chromium (Playwright) at dpr ∈ {1, 1.5, 2, 3}, including forced
viewport-resize passes to drive the ResizeObserver: for every canvas the backing store is
exactly `design_h · dpr`, CSS height is exactly the authored height, non-transparent pixel
count > 0 (i.e. it genuinely paints, not just sane numbers), zero page errors. Figure 0
screenshotted in both light and dark mode.

**Files touched:** `docs/rig-explained.html` (canvas `setup()` helper only; no figure logic).

**Gotcha for future work:** any new canvas added to this document must not re-read
`getAttribute('height')` after the first paint — go through `DESIGN_H`. Note also that the
per-figure `fail()` wrapper only surfaces *thrown* errors; a canvas that silently fails to
paint still looks identical to a crash.


---

## 2026-07-17 — Session 7 (cont.): F9 CLOSED — the inverse above 2 dimensions

**User ask:** "have the program handle inputs with more dimensions" — i.e. the audit's F9,
the largest un-caveated gap.

**First, the framing correction.** The core was ALREADY dimension-agnostic by construction
(the GP, `RecipeTransform`, and the §8 margin all take vectors). Nothing needed to be
"made" d-dimensional. What was missing was **evidence** — no result in the repo had ever
run the inverse above d=2, on the exact axis where inverse problems are hardest. So this
slice is a measurement + two real fixes the measurement exposed, not a rewrite.

**Method (`examples/run_dimensionality_study.py`, `docs/dimensionality-2026-07-17.md`):**
smooth d-dim 2-output process with EVERY input dim active; GP on 12·d Sobol runs; ask for
a reachable target; then **evaluate the TRUE function at the returned recipe**. Ground
truth is never consulted during the solve — deliberately avoiding the self-scoring
circularity the audit flagged in the M3 gate (F2).

**Result: it works.** d = 2, 4, 6, 8, 10, 15 → FEASIBLE, **3/3 returned recipes genuinely
in spec on the true function** at every d. d=20 @ 240 runs → INFEASIBLE with the §8.8
**epistemic-limited** diagnosis ("collect runs"); **doubling to 480 runs flipped it to
FEASIBLE with a genuine ground-truth hit.** So the 20-D abstention was HONEST, not a
failure: the solver diagnosed data sparsity, named the fix, and the fix worked — the §8.8
taxonomy earning its keep on a problem 10× larger than anything it had seen.

**Two real dimensional weaknesses found:**

1. **`n_restarts` was a FIXED 48 for every dimension — FIXED.** Dense in 2-D, vanishing in
   20-D. Not just slow: a starved multi-start degrades into a **FALSE INFEASIBLE** (we fail
   to FIND a recipe and report none EXISTS) — the same false-abstention class as the
   re-validation bug fixed earlier today, and the exact confusion §8.8 exists to prevent.
   Now `n_restarts=None` ⇒ `max(48, 24·dim)`. **`24·dim` == 48 at dim=2 exactly**, so M2,
   the AL loop and every 2-D result are bit-for-bit unchanged. Keys on `RecipeTransform.dim`
   (free u-coords, K−1 per simplex), not the recipe key count. Explicit int still wins.
2. **Cost ~O(d²), and it is the GRADIENT — DOCUMENTED, NOT FIXED.** `minimize` is called
   WITHOUT `jac`, so SciPy finite-differences: `d+1` objective evals per gradient step, each
   running `predict` AND `jacobian`. Times the growing restart budget: 2.6 s @ d=2 → ~150 s
   @ d=20. An analytic gradient needs `∂σ_epi/∂x` and `∂J/∂x` (a SECOND derivative of the GP
   mean) — real work, explicitly owed, not silently accepted. d ≳ 20 needs torch autograd
   or a cut budget.

**Tests (+4):** `test_inverse_returns_ground_truth_hits_above_two_dimensions[4,8]` (scores
against TRUTH, not the surrogate), `test_restart_budget_scales_with_search_dimension`
(pins dim=2 == 48 so 2-D results cannot silently move), and
`test_simplex_restart_budget_uses_u_space_dimension` (a 4-component simplex is 3 free
coords → 72, not 96).

**Honest limits (in the doc, not buried):** synthetic and smooth (friendly to a Matérn-5/2
GP); `12·d` runs is a choice, not a law, and real campaigns will not grant 240 runs for a
20-knob process; single seed per cell. **This closes "does the inverse work above 2-D"
(yes, with ground-truth evidence). It does NOT touch M0** — the program's actual claim
still needs a real recipe→outcome dataset.

**Files:** `src/rig/inverse/pessimistic.py` (dimension-scaled restart budget + the
measured dimensionality note in the module docstring), `tests/test_inverse.py` (+4),
`examples/run_dimensionality_study.py` (new), `docs/dimensionality-2026-07-17.md` (new).


---

## 2026-07-17 (evening) — Four new modules built, then adversarially verified; gradient red-tests resolved

Continuation of the same session. Two arcs: (A) four modules were built in parallel earlier
today (qualification, distillation, Phase-II qLogNEHVI acquisition, linear constraints) plus an
analytic objective gradient; (B) all of it was then adversarially verified by execution in
isolated sandboxes, and the two red tests the gradient work left behind were diagnosed and fixed.

### A. What was built (by parallel builder agents; each wrote its own tests)

- **`src/rig/qualification.py`** — first concrete `QualificationGate`: `ConfirmationBatchGate`
  with an exact one-sided **Clopper-Pearson** lower bound (`clopper_pearson_lower`,
  `min_runs_for_claim`). D7 honored (verifier INJECTED, no adapter import); fail-closed;
  provenance-gated (`physics_sim` ⇒ not headline-eligible).
- **`src/rig/forward/distill.py`** — §5.7 ensemble→student distillation preserving the
  aleatoric/epistemic split; out-of-transfer-box guard (`_box_excess`).
- **`src/rig/active/acquisition.py`** — Phase-II `qlognehvi_phase2` via real botorch qLogNEHVI.
- **`src/rig/inverse/pessimistic.py`** — linear-constraint support (`ConstraintSet` box+coupling)
  via a hard reject (`_admissible`) independent of a soft log-sigmoid barrier; AND an opt-in
  **analytic objective gradient** (`analytic_grad=True`, `_GPTermProvider`, `_dx_du`) replacing
  SciPy finite differencing (the F9-owed item).

### B. Adversarial verification (isolated sandboxes, mutation-tested, ground-truth-scored)

Every module was audited by an agent working in a disposable full-repo copy (PYTHONPATH
isolation, verified empirically — the real tree was proven byte-identical to a pre-run hash
manifest throughout). Verdicts:

- **acquisition → SOUND.** Real botorch qLogNEHVI (import chain traced, nothing faked); objective
  SENSE correct (tested with an identity model + hand-checkable ordering); 21 mutations, 18
  caught. Minors: `tests/conftest.py` `_TORCH_SKIP_MARKER` is dead (a promised torch skip-banner
  is never emitted — a CI run skipping all ~56 torch tests would announce nothing); a docstring
  cites pad numbers (13.8→11.4) that don't reproduce under the test's own fixtures (91.37→91.09).
  Both confirmed by 3/3 refuters. Not fixed (doc/CI-cosmetic, not source).
- **qualification → SOUND.** Clopper-Pearson is genuine Beta-quantile inversion, not Wald:
  matches scipy + a brute-force binomial inversion to ≤5e-13; correct one-sided α (the classic
  α-vs-α/2 bug is ABSENT — 8/10 → 0.4931 one-sided, not 0.4439 two-sided); published values
  reproduced (8/8→0.6877, 28/29→0.8466, 0/20→0.1391, min_runs(0.90,0.95)=29); k=0/k=n edges
  correct; fail-closed. 14-mutation battery, 11 caught.
- **distill → PARTIAL (core sound).** The aleatoric/epistemic split genuinely survives
  distillation (student tracks teacher aleatoric at corr 0.99999, epistemic at 0.9999; epistemic
  peaks in a data hole, aleatoric in the noisy region); canonical PredictiveDistribution
  returned; split-collapse mutation → 3 tests RED (real guards).
- **linear-constraints → SOUND.** THE critical check — does any certified recipe violate its own
  `A@x ≤ b`? — computed independently over adversarial specs whose UNCONSTRAINED optimum violates
  the coupling: worst excursion **0.0**. Safety is a hard reject independent of the soft barrier
  (no-op the reject → the exact violating recipe leaks and tests catch it; no-op the barrier →
  safety tests stay green, proving independence). Simplex+constraint `NotImplementedError` is
  honest fail-closed at CONSTRUCTION, not a mid-solve crash. No false abstention.

### C. The two RED tests the gradient work left — DIAGNOSED, source CORRECT, fixed

The analytic-gradient builder died mid-investigation leaving `tests/test_inverse.py` with 2
failures. Both were benign; the source is correct, proven three independent ways:

1. **1-vs-3 candidates** (`test_analytic_grad_and_fd_paths_agree_on_the_solution`): scored against
   ground truth, EVERY recipe from both paths is in-spec. The exact-gradient path explores the
   pre-image better and returns 3 distinct valid recipes where the FD path returns 1. The
   `len(fd)==len(an)` assertion encoded a false premise. **Real finding: the FD path
   UNDER-EXPLORES**, so `analytic_grad` (documented §8.6 speed-only) can change the returned set
   SIZE (never the top recipe). Rewrote to assert the true property (no presented recipe misses
   ground truth, on BOTH paths; top recipe valid; `len(an) >= len(fd)`).
2. **simplex gradient** (`test_analytic_gradient_covers_the_simplex_block`, 1.88e-4 vs 1e-6): the
   gradient is CORRECT. Its inline fixture (Dirichlet conc=1 → points at simplex vertices, 40
   pts) drove objective gradients to ~1e4, where central FD floors at ~2e-4 relative — the
   simplex twin of the near-linear trap the module already documents. An independent adversary
   confirmed with a **60-digit mpmath** FD at the exact failing point: rel err **5.38e-8**, no
   plateau. Rewrote with a well-conditioned fixture (2.4e-7) + an isolated softmax-Jacobian guard
   (matches to 5e-11, fixture-independent). The adversary also mutation-tested 6/8 other gradient
   terms as guarded; and flagged that the softmax branch was guarded ONLY by this (then-red)
   test, so the fix IMPROVES the reference rather than loosening the tolerance.

### D. Three test-coverage gaps found by verification, all FIXED and proven guards

Each shipped SOURCE was correct; each gap was a guard tested only through the thing it guards.
Each new/edited test was proven to catch its defect by mutating the source and watching it redden
(mutations run in scratch copies; the real source was never mutated):

1. **qualification** — the multi-output in-spec conjunction (`np.all` across outputs) was never
   tested (all specs single-output). Added a 2-output test; `np.all→np.any` → n_in_spec 0→29 RED.
2. **distill** — the `_box_excess` OOD guard was tested only on the UPPER box face; dropping the
   lower-face term left all tests green while a far-OOD point BELOW the box got zero guard
   inflation (§8.2 fail-closed hole, unsafe direction). Added a symmetric lower-face test; the
   drop → `0.0 > 0.0` RED.
3. **linear-constraints** — the barrier-off safety test asserted feasibility via the SAME
   `ConstraintSet.is_satisfied` the reject uses, so a regressed shared checker would only redden
   the attribution tests. Added an independent `A@x − b` re-derivation; regressing the checker to
   always-true → leaks `a=2.5,b=2.5` RED (previously green).

### Process notes (mistakes made and corrected, for the record)

- **Falsely reported "nothing is running, the workers are done"** at ~16:35 — the gradient agent
  ran until 17:09. Cause: inferred agent liveness from `Get-Process python` being empty, but
  subagents are LLM calls inside the `claude` process; python only appears while a test runs.
  Correct liveness check: poll the agent transcript for growth + `Get-Process claude | Sort CPU`.
- **Falsely reported the workflow journal/transcripts were "never created"** — I looked in
  `<session>/workflows/` (scripts only); the real path is `<session>/subagents/workflows/<runId>/`.
  All four builder reports were recoverable the whole time.
- **A mutation-verification silently didn't mutate** (used `mktemp`, whose `/tmp/...` path Windows
  Python can't resolve) and the test "passed" — a false green of exactly the kind this repo
  guards against. Redone with a real scratchpad path; both mutations then confirmed to fire.
- A first verification workflow was KILLED before any agent ran because it told agents to mutate
  source on the ONE shared tree and self-check with `git diff` — but `src/` is untracked here
  (single "Initial commit"), so `git diff` is blind. Replaced with per-agent disposable copies.

### Files

- Source (built earlier today, verified now, NOT modified during verification): `src/rig/qualification.py`,
  `src/rig/forward/distill.py`, `src/rig/active/acquisition.py`, `src/rig/inverse/pessimistic.py`
  (linear constraints + analytic gradient), `src/rig/forward/__init__.py`.
- Tests edited this evening: `tests/test_inverse.py` (2 gradient tests rewritten + isolated
  softmax-Jacobian guard added + independent A@x check in the barrier-off safety test),
  `tests/test_qualification.py` (+1 multi-output conjunction), `tests/test_distill.py` (+1
  symmetric lower-face OOD guard).
- Docs: `docs/prereg-mfl-bakeoff-2026-07-17.md` (new — pre-registered RIG-vs-MFL bake-off
  predictions, written before any bake-off code), `docs/RESUME-STATE-2026-07-17.md` (new — pause
  state + gradient resolution), `docs/rig-explained.html` (republished with the confirmed-real
  d=20 false-success correction).

**Suite: 511 passed / 0 failed / 0 skipped** (was 386 at the start of the session's audit; the delta
is the four modules' tests + the analytic-gradient block + the 3 new coverage tests). ruff clean;
import-linter contract KEPT (corrected `sys.exit(lint_imports())` form).

**Not done / still owed:** the MFL bake-off itself (only pre-registered); the owed experiments
(false-success RATE vs d across seeds and GP-fit restarts; whether conformal re-validation catches
the d=20 miss; powered M2/M3 re-runs); wiring `revalidation_model` into `active/loop.py`. M0 (real
data) remains the entire scientific claim and is blocked on a PI action, not code. USER decisions
still open: `ruff format` adopt-or-drop, pandera implement-or-drop, M0 dataset.


---

## 2026-07-17 (evening, addendum) — Two standing user decisions resolved: ruff format ADOPTED, pandera DROPPED

User asked me to decide + execute both deferred policy items.

- **`ruff format` — ADOPTED and applied.** `python -m ruff format .` reformatted 48/88 files
  (whitespace/wrapping only, behavior-preserving). `ruff format --check .` and `ruff check .`
  both clean; **full suite still 511 passed / 0 failed** post-format (proven, not assumed —
  formatting a 48-file diff got the same green as before). Made the adoption enforceable, not
  one-shot: added a `ruff format --check .` step to `.github/workflows/ci.yml` and recorded
  ruff format as "formatter of record" in CLAUDE.md. Deliberately NOT committed (per protocol);
  the reformat sits with the rest of the session's uncommitted work.
- **pandera — DROPPED.** Verified imported NOWHERE in src/tests/examples (genuinely dead), then
  removed `pandera>=0.19` from the `dev` extra in pyproject.toml. Rationale: a declared-but-unused
  validation dependency implies a frame-validation guarantee that does not exist — the exact
  false-signal class this project keeps having to walk back. DataFrame validation at ingest stays
  a real **E1** item, to be built against the actual data contract when M0 lands, not
  speculatively now. Updated the CLAUDE.md note from "aspirational, not implemented" to "dropped".
  pyproject still parses; the other four dev tools resolve; pandera remains installed in the local
  venv (harmless — just no longer declared, so a fresh `.[dev]` install won't pull it).

Files: `pyproject.toml`, `CLAUDE.md`, `.github/workflows/ci.yml`, + 48 files reformatted by ruff.


---

## 2026-07-18 (~00:30) — Session limit hit mid-tasking; state persisted; HTML MFL section shipped solo

User tasking: fresh audit + HTML update (MFL-comparison focus) + MFL follow-on research + M0
dataset hunt + persist-across-limits. Launched 2 of 3 workflows; ALL 14 subagents failed on
"session limit (resets 8:10pm SGT)"; 3rd workflow (bake-off build) never launched.

Done despite the outage (main loop still alive):
- `docs/RESUME-STATE-2026-07-18.md` — full execution plan + resume commands for both interrupted
  workflows (scripts on disk, run IDs wf_163b8a0a-9d2 / wf_677c9377-586). BUILD_STATE header now
  points at it.
- `docs/mfl-bakeoff-build-spec-2026-07-18.md` — the bake-off build spec (Alg-1 transcription,
  arms, ledger, metrics, tests, steelman pass), since its workflow script was never persisted.
- `docs/rig-explained.html` — NEW section 11 `#versus`: honest MFL comparison from the paper's own
  numbers (Table-1 margin diagram: their 4.45nm error vs 5.55nm remaining margin; Eq.-4 ∂M/∂x
  query-cost inversion at d=11; abstention; formulation table; explicit concessions on
  amortization/simplicity; both-in-silico caveat citing the pre-registered bake-off with predicted
  RIG losses). Glossary renumbered 11→12. Tag balance verified. Republished to the same artifact
  URL (label `mfl-comparison-section`).
Repo code untouched; still 511 passed. Next session: execute RESUME-STATE §2.


---

## 2026-07-19 (early AM) — Fresh audit: 11 confirmed findings, ALL FIXED same-session (orchestrated on sonnet/opus)

Six-lens adversarial audit (finders -> 3-refuter panels; correctness refuters on opus, others
sonnet) over the previously un-reaudited surface. 11 confirmed / 2 refuted. Then a 6-agent fix
fleet (2 opus for the majors, 4 sonnet) on disjoint file sets, every fix reproduced-first and
guarded by a test PROVEN red-then-green. Arbiter suite after: **527 passed / 0 failed** (was 511;
+16 guard tests). NOTE: ruff could not run — Windows Application Control began blocking ruff's
compiled binary mid-session (WinError 4551); formatting verified only up to the last successful
run. USER ACTION: allow ruff.exe in App Control or reinstall a signed build.

### The two majors
1. **`ProcessSpec.continuous_si` mis-converted OFFSET units (spec.py) — the SI trap recursed
   into its own fix.** It converted bounds via a multiplicative unit factor; degC 200..400 became
   5.5e4..1.1e5 K instead of 473.15..673.15 K — the shipped pecvd example spec had garbage SI
   bounds. Fixed via absolute Quantity conversion (`ureg.Quantity(v, unit).to_base_units()`);
   documented that bounds are POINTS not widths. Guards cover degC AND mtorr; pecvd spec pinned.
2. **`revalidation_model` never wired (active/loop.py)** — the §13.2 conformal re-validation gate
   was inert in every shipped campaign path; on the ensemble tier the solver ran on the fast SNGP
   member view with NO full-ensemble re-check (repro: fast view certified mean=1.5-in-spec while
   the full model said 5.0 — not caught). Fixed: when `inner is not surrogate` (a genuine
   fast/full split) the freshly-refit full surrogate is UNCONDITIONALLY the revalidation_model;
   GP tier (inner is surrogate) stays opt-in via a new ctor passthrough, default = historical
   behavior. Guard asserts identity AND that a constructed fast-certifies/full-rejects divergence
   is caught (returns Infeasible).

### The nine minors (all fixed, all guarded)
- ACI NaN-as-covered (conformal.py): non-finite observation now counts as a MISS (widening,
  fail-closed), not appended to the score buffer; dead-sensor stream now reads coverage 0.0.
- KM tie convention (survival.py CORRECT, tests couldn't tell): wrong-convention mutant passed
  all 14 survival+m2 tests. Added mutation-proven guards incl. the M2-shaped budget-tie case.
- Seed DoE vs budget (loop.py): budget < n_seed now raises at CONSTRUCTION; guard proves 0
  machine calls fire.
- Non-finite outcome cells (ingest.py): now rejected fail-closed naming row+column; skip-policy
  drops just the bad row.
- Mixed naive/aware timestamps (ingest.py): file-level IngestError; all-naive or all-aware OK.
- OutputSpec SI accessor (spec.py): `outputs_si` added, offset-safe, pointing at the CLAUDE.md
  SI-trap note (closes the latent consumer-less instance).
- BoxTransform.inverse NaN bypass (transforms.py): non-finite now trips the D1 fail-loud guard.
- SimplexTransform.inverse silent clamp (transforms.py): genuine out-of-simplex input now raises
  (float-drift tolerance retained); full suite verified unbroken.
- M3 acceptance artifacts stale: REGENERATED on the current tree; old->new raw_proposal_hit_rate
  0.969->1.0 and 0.703->0.609 (only those two fields changed; verdict PASS unchanged); the new
  artifact reproduces byte-identically (sha256-matched across two runs); doc claims corrected,
  near-tautological-gate caveat KEPT.

### Refuted (correctly)
- false_success_rate "vacuity" reclassified: a definitional nuance (the prereg's
  certified_miss_rate rename is the right headline), not a code defect.
- m2-result.json staleness: already-documented owed item, not a new finding.

Parallel workstreams still running at write time: MFL bake-off build (opus builder wf_fc1bfc4a),
M0 STRONG-dataset downloads into data/m0-candidates/. Research docs landed earlier
(m0-dataset-candidates-2026-07-18.md: 4 STRONG of 25; mfl-follow-on-research-2026-07-18.md:
MFL has ZERO citations). Nothing committed.

## 2026-07-19 — MFL bake-off BUILT (per docs/mfl-bakeoff-build-spec-2026-07-18.md)

Builder agent. Faithful Model Feedback Learning (Gu et al. 2025, arXiv:2505.16060, Alg. 1)
comparator + the full pre-registered bake-off (docs/prereg-mfl-bakeoff-2026-07-17.md).
Nothing committed.

Files created:
- `src/rig/baselines/mfl.py` — `ModelFeedbackLearning` (torch, lazy-imported; base stays
  torch-free — verified `import rig.baselines` loads no torch), `MFLLedger`, `fd_jacobian`,
  `spectral_norm`. Alg. 1 faithful: two loops, gradient THROUGH E in Loop A (Eq. 4, not a
  detached target), Loop B on M via FD `∂M/∂x`, domain randomization in Step 1, conservative-
  LR sensitivity gate (α2=0.99·α1, δ=0.9, counters exposed), ONE R conditioned on z' for the
  whole target set, Table-10 defaults as kwargs (α1=0.01, hidden 64, epochs 700, T=1200,
  T0=1150, τ=200, τ0=150), input-bound clip.
- `tests/test_mfl_baseline.py` — 6 tests (linear-inverse recovery; Loop B corrects a biased
  E; LR gate fires AND stays off at an unreachable δ; FD Jacobian vs analytic 1e-4; ledger
  exactness; scorer flags a PLANTED miss + counts RIG abstentions as non-presented). All pass.
- `examples/mfl_bakeoff/pre_register_targets.py` — dense noise-free search (Sobol 4096 + local
  refine) → 20 hash-FROZEN targets (10 feasible / 5 hard / 5 infeasible), each class verified
  by dense search, witness recipe stored.
- `examples/mfl_bakeoff/run_bakeoff.py` — 4 arms (rig, rig-reval, mfl-charitable/deployable),
  prereg §0 metrics VERBATIM, the charitable-vs-deployable ledger, --smoke/--full, JSON out.
  Scorer helpers are numpy-only + importable (test 6).
- `examples/mfl_bakeoff/README.md` — repro + every deviation, honestly.
- `examples/mfl_bakeoff/targets.json` — frozen (hash 52e66d3b141f...); runner refuses on
  mismatch.

Ground-truth (noise-free) mechanism found: `evaluate_physics(recipe, machine_config)` at
nominal keyword params (== `InSilicoMachine(PathologyConfig())`, all pathologies OFF), taken
through `metrics_to_outcomes` for SI parity with the noisy path. Deterministic; the seeded-
replicate fallback was NOT needed.

Two real findings during the build (both fixed, both documented as deviations):
1. **FD on a NOISY machine is noise-dominated.** `fd_step=1e-3` (paper-appropriate for a
   noiseless M) drove MFL target-recovery error 0.03 → 2.4 (pure-noise gradients destroy R);
   `fd_step=0.05` restores it. Adopted per the prereg §4.6 steelman ("if a cheap change
   materially strengthens MFL, ADOPT it"). Also sharpens the deployable-arm critique.
2. **A trivial one-sided bound is pathological for the MFL point rule.** `slip ≤ 1.0` (slip
   only reaches ~0.3) made the spec's one-sided point (`bound − σ ≈ 0.95`) chase an
   unreachable value → `certified_miss_rate → 1.0` artifact. Fixed by setting the one-sided
   bound to the 85th pct of reachable slip (a genuine constraint); the point rule itself is
   applied verbatim.

FULL RUN (wall 146s, n_seed=60, 20 targets, 200-replicate yield; results/full_20260719T023200Z):
| arm | miss | false_abst | yield | margin | q_charit | q_deploy |
| rig | 0.000 | 0.533 | 1.000 | +10.45 | 60 | 60 |
| rig-reval | 0.000 | 0.533 | 1.000 | +10.69 | 60 | 60 |
| mfl-charitable | 0.700 | 0.000 | 0.000 | −9.63 | 4060 | 12060 |
| mfl-deployable | 0.700 | 0.000 | 0.000 | −9.63 | 4060 | 12060 |
RIG presents 7/20 (0 misses, 0 false successes, abstains on all 5 infeasible); MFL presents
20/20 (14 misses, incl. all 5 infeasible → confident garbage, P4).

Prediction scoring (predictions FROZEN, not edited):
- **P2 HOLDS** (the real claim): RIG miss 0.0 vs MFL 0.70 — 70pp ≥ 15pp.
- **P3 HOLDS**: normalized_margin gap ~20σ ≥ 1σ (MFL median margin NEGATIVE — edge-hugging).
- **P5 HOLDS as the predicted LOSS**: RIG false_abstention 0.53 ≥ 10% (pessimism's price).
- **P1 REFUTED — and in the OPPOSITE direction** from the author's prediction: RIG 60 machine
  queries vs MFL charitable 4060 (68×). RIG's GP inverse is machine-free at solve time; MFL's
  on-machine Loop B is not. The P1 intuition ("MFL amortized ⇒ fewer queries") missed that
  RIG's surrogate inverse is ALSO amortized. A legitimate pre-registration finding.
- **P4** tautological, not counted (MFL 5/5 infeasible presented vs RIG 5/5 abstained).
- **VERDICT (prereg §2 rule): P2 AND P3 both hold ⇒ "RIG's formulation is better-posed" is
  SUPPORTED** — on THIS simulator only. Both venues in-silico; nothing about real hardware or
  MFL's own plasma-etch task. rig-reval == rig here (miss already 0), so the owed conformal-
  re-validation experiment's answer is "no change to the miss rate on this target set".

Gates: 533 tests collected (unbroken); test_mfl_baseline 6/6; baselines group (botorch/warm/
mfl) 20/20; import-linter 1 kept / 0 broken; base import torch-free. ruff format/check COULD
NOT RUN — `ruff.exe` is blocked by a machine Application Control policy (WinError 4551) in
this environment, every shell; code hand-conformed to line-length 100 + no unused imports.


---

## 2026-07-19 (~03:20) — ⚠ CAVEAT on the bake-off entry above: full run NON-CITABLE pending re-run

The steelman pass (prereg §4.6) rated the MFL implementation FAITHFUL and not under-tuned, BUT
found a REQUIRED harness defect the entry above does not reflect: **targets were LABELLED
feasible against the two-sided slip box while the STORED/SCORED spec overrides slip to one-sided
≤0.22.** Dense search proves hard_01/03/04 are actually INFEASIBLE (0 feasible recipes under the
stored spec) despite feasible_truth=True, and 6 stored witnesses violate the stored slip bound.

Impact on the numbers above: **P2 (70pp miss gap) and P3 (margin) stand per the steelman** (they
are computed over presented recipes and unaffected by the labels), but **P5's false_abstention
0.53 is INFLATED by mislabelled targets** (3 of RIG's "false" abstentions were on genuinely
infeasible targets — i.e. CORRECT abstentions), and any 'MFL missed 5 feasible-but-hard targets'
reading is wrong. A workflow conditional bug also SKIPPED the fix phase (the verdict began
'FAITHFUL', which my regex read as no-fixes-required despite the REQUIRED fix in the body).

In flight: label fix + Loop-A LR scaling fix (steelman-recommended, verified immaterial) +
independent per-target dense-search re-verification + targets.json RE-FREEZE (new hash) + full
re-run of both arms. **Do not cite any bake-off number until the re-run entry lands below.**
Lesson (again): the run that could refute was still pending when the entry above was written —
this time by a subagent; same rule applies to them.


---

## 2026-07-19 (~03:12) — MFL bake-off: steelman-REQUIRED fixes applied, RE-FROZEN, RE-RUN (CITABLE)

Resolves the caveat above. Predictions in `docs/prereg-mfl-bakeoff-2026-07-17.md` were NOT
touched. Files: `examples/mfl_bakeoff/pre_register_targets.py`, `src/rig/baselines/mfl.py`,
`examples/mfl_bakeoff/README.md`, `examples/mfl_bakeoff/targets.json` (regenerated).

**FIX 1 (blocker) — target-label / scored-spec mismatch.** `pre_register_targets.py` labelled
feasibility against the TWO-SIDED slip box while `_spec_json(one_sided_upper=True)` stored/scored
slip as one-sided `[None, slip_upper]`. Fix: new `_effective_bounds()` applies the one-sided-slip
override used for BOTH labelling and storage (single source of truth); `label_feasibility` now
receives the effective bounds; feasible + feasible-but-hard targets are anchored on the
SLIP-FEASIBLE reachable sub-cloud (`Z[:,slip] <= slip_upper` — the reachable set UNDER THE STORED
SPEC), so each witness genuinely meets the scored spec; hard targets = tiny boxes on the boundary
of THAT sub-cloud (extremes+shell in the two two-sided outputs). Added an in-generator assertion
(`in_spec(witness, stored_spec, tol=0)`) for every stored witness, plus a final assertion loop in
`main()`. Regenerated + re-froze targets.json.

**FIX 2 (recommended, immaterial) — Loop-A loss scaling.** `_loop_a`: `((ys-zt)**2).mean()` →
`((ys-zt)**2).sum(dim=1).mean()` = paper `L=(1/n')Σ‖z'-y'‖²`, consistent with Loop B's 2/n'
gradient. Differs only by 1/z_dim; steelman-verified immaterial (README faithful-list updated).

**VERIFY.** `test_mfl_baseline.py` 6/6 pass (before and after both fixes). Generator quotas met
(feasible 10 / hard 5 / infeasible 5); all in-generator witness assertions pass.
INDEPENDENT re-verification (own 20000-point Sobol, seed 987654321 ≠ generator's, through the
runner's `MachineHarness.ground_truth_outcome` + `in_spec`): feasible targets 3341–4757 in-spec
pts each; hard targets 385–398 each (genuinely feasible, ~12× tighter basin than the easy class,
on-boundary); infeasible 0 each; ALL 15 witnesses valid; file hash self-consistent.
HASH-FREEZE proven to FIRE: `load_frozen_targets` REFUSED both a stale-hash marker (the OLD
`52e66d3b…` hash) and a silently edited target; new hash `02603fc1…` ≠ old `52e66d3b…`, recorded.

**NEW targets.json hash:** `02603fc1421f8e53f5d46958975b5448713bbeec6856ca8fd5d39127f4fc674c`
(was `52e66d3b141fa190285814296b2b1b632c4c10c6f8b4dabced50efa4915960e4`).

FULL RUN (CITABLE; wall 149.3s; n_seed=60, 20 targets, 200-rep yield;
`results/full_20260719T031201Z/bakeoff_results.json`):
| arm | miss | false_abst | yield_med | margin_med | presented | q_charit | q_deploy |
| rig | 0.0000 | 0.3333 | 1.0000 | +10.5230 | 10 | 60 | 60 |
| rig-reval | 0.0000 | 0.3333 | 1.0000 | +10.9232 | 10 | 60 | 60 |
| mfl-charitable | 0.5000 | 0.0000 | 0.5000 | −1.4509 | 20 | 4060 | 12060 |
| mfl-deployable | 0.5000 | 0.0000 | 0.5000 | −1.4509 | 20 | 4060 | 12060 |

Corrected labels make the structure CLEAN: RIG presents+HITS all 10 clearly-feasible (0 misses),
ABSTAINS on all 5 feasible-but-hard (the 5 false abstentions → 5/15 = 0.3333) AND all 5 infeasible
(correct). MFL presents 20/20 and MISSES exactly the 10 boundary+infeasible targets (5 hard + 5
infeasible), hitting all 10 clearly-feasible → miss 0.5000. rig-reval == rig on miss/abstention
(margin +0.4 higher) — the owed conformal-re-validation experiment's answer stays "no change to
the miss rate on this target set". The previously-reported (mislabelled) run had RIG false_abst
0.533 / present 7/20; the corrected honest number is 0.333 / present 10/20 (the 5 infeasible
abstentions are now correctly OUTSIDE the false-abstention denominator). NOT scored against P1–P5
here (orchestrator's job); both venues in-silico. ruff BLOCKED (WinError 4551) — hand-conformed
(line-length ≤100, files parse). Nothing committed.


---

## 2026-07-19 (~03:40) — Bake-off RE-RUN with corrected labels: SCORED against the frozen predictions. Caveat above RESOLVED.

Corrected run `full_20260719T031201Z`, targets re-frozen `02603fc1…`, labels independently
re-verified (20,000-pt Sobol, different seed): 10 feasible (3341–4757 in-spec pts each),
5 hard (385–398 pts, ~12x tighter, on-boundary), 5 infeasible (0 pts each), all 15 witnesses
valid. Numbers verified by the orchestrator from the results JSON, not the agent's prose.

| arm | miss | false_abst | yield | margin | q_charitable | q_deployable |
|---|---|---|---|---|---|---|
| rig | 0.000 | 0.333 | 1.000 | +10.52σ | 60 | 60 |
| rig-reval | 0.000 | 0.333 | 1.000 | +10.92σ | 60 | 60 |
| mfl (both arms) | 0.500 | 0.000 | 0.500 | −1.45σ | 4,060 | 12,060 |

**Scored vs the FROZEN prereg predictions** (outcomes appended to the prereg, predictions
untouched): **P2 HOLDS** (50pp ≥ 15pp), **P3 HOLDS** (≈12σ ≥ 1σ; MFL's median presented recipe
is out-of-spec), **P5 HOLDS as the predicted loss** (33%: RIG refused ALL 5 hard targets —
pessimism's full price), **P1 REFUTED in the OPPOSITE direction** (RIG 60 vs MFL 4,060 — the
author's prediction that MFL wins queries was wrong; RIG's GP inverse is machine-free at solve
time), **P4 tautological, not counted.** **Frozen verdict rule (P2∧P3): "RIG's formulation is
better-posed" SUPPORTED — on this simulator only.** Perfect class separation: both methods hit
all 10 easy targets; at the boundary and beyond RIG refuses (5 correctly, 5 over-cautiously),
MFL presents all 10 and misses all 10. rig-reval == rig (miss already 0) ⇒ conformal
re-validation changed nothing on THIS set; the d=20 revalidation experiment stays open.

Explainer #versus updated with the scored table + republished (same URL). Fix-phase regex bug
in the workflow (verdict 'FAITHFUL' header skipped a REQUIRED body fix) noted as an
orchestration lesson: parse required-fix LISTS, not verdict headlines.


---

## 2026-07-19 (~04:00) — M0 candidate data PULLED and ground-truthed; repo-root debris cleaned

Sonnet download agent completed: all 4 STRONG candidates from the 2026-07-18 hunt are LOCAL in
`data/m0-candidates/` (382 MB total, per-file sha256/MD5 in `MANIFEST.md`), with every web-hunt
claim verified against the ACTUAL files. Corrections recorded at the top of
`docs/m0-dataset-candidates-2026-07-18.md`: empa n=3,150 (not "6-D flat" — two 5-D subspaces +
categoricals, MD5-verified); Ada 2022_09 n=177/180 with a LICENSE-DATA 404 (claimed CC-BY-4.0
unverifiable, only MIT present); Ada 2021_01 n=253 exact but NO wall-clock timestamps; NREL HTEM:
**nrel.gov is dead (DNS undelegated), successor nlr.gov**, live scale 1,891 libraries (well below
the doc's ~4,356+), per-sample API broken — a 440-entry live sputtering sample + library index
pulled as the working access path. BUILD_STATE M0 row updated: candidate data IN HAND, venue
choice = user/PI decision; none meets the full bar.

Also removed 4 stray files a fixer agent left at repo root (nan.csv, nan2.csv, rt.jsonl,
rt2.jsonl — its pre-fix NaN-ingest repro artifacts; inspected before deletion, confirmed debris).

---

## 2026-07-19 (~13:30) — HTML: new section 12 "The data" (inventory + sources), republished

User asked for the data and sources to be reflected in the explainer. `docs/rig-explained.html`:

- NEW section `#data` ("12 — the data / What it runs on today — and the real data now in hand"),
  inserted between `#versus` and the glossary (glossary renumbered 12→13, TOC updated):
  - The two current sources behind every published number: (1) `mbe_sim` + `InSilicoMachine`
    (provenance=physics_sim, barred from headline claims), (2) `Zr_grid.csv` machinery proof
    (209 rows, no license → on-prem only, NN-invertible grid).
  - Table of the 4 downloaded M0 candidates with source links (Zenodo DOI, berlinguette/ada,
    htem.nlr.gov + OpenEI), verified n, license status, and per-dataset catches — numbers taken
    from `data/m0-candidates/MANIFEST.md`, not the web claims.
  - Warn note on the nrel.gov→nlr.gov infrastructure rot (citation dead within one day).
  - "Why not just use all four?" — M0 is a per-process gate; disjoint processes can't pool;
    lead-venue + replications strategy; Empa HiPIMS lead recommendation. Venue choice stays a
    USER/PI decision — the HTML presents it as open, not decided.
- Hero tags refreshed to current truth: re-audited 07-19, **533 tests passing** (full suite
  re-run this session: 533 passed in 340 s), 21 defects fixed (10 original + 11 fresh-audit),
  "Inverse verified to 20-D" → "Stress-tested to 20 knobs" (the false-success finding makes
  "verified" the wrong word), new "Real M0 candidate data in hand" tag. §10 test count updated
  too. Cross-links added from §10 verdict and §11 closer to §12. Footer dates updated.
- Tag-balance + TOC/section-count checked (14/14); republished to the SAME artifact URL
  (label `data-inventory-section`).

Files touched: `docs/rig-explained.html` only. Nothing committed.

---

## 2026-07-19 (~afternoon) — Empa HiPIMS E1 data-prep slice (converter + 6 specs + ingest tests)

Subagent task: build the data-prep slice for the M0 lead candidate (Empa bipolar HiPIMS,
Zenodo 10.5281/zenodo.18495402, CC-BY-4.0) on the sputtering-example template. Nothing under
`src/` or `data/` touched; nothing committed.

- **`examples/real_data/empa_hipims/prepare_empa.py`** — deterministic (no-RNG) converter:
  6 raw `df_campaign_*.json` + `calibration.txt` → 6 tidy CSVs (`csv/<slug>.csv`, 3,150 rows
  total), stable-sorted by BatchNr, keeping raw `y1` AND calibrated
  `dep_rate_A_per_s = y1 * factor` (factor READ from each calibration.txt; asserted
  plausible 0.01–1000 A/s and cross-file-varying). Factors: Al 1.1684, Ti 0.722838 (per
  paper, Å/s). Summary table printed per campaign.
- **`examples/real_data/empa_hipims/specs/<slug>.toml`** ×6 — per-campaign WP-H specs;
  bounds transcribed VERBATIM from each campaign's OWN `Campaign.json` (they differ:
  Ti-120W's are full-precision data extents). Outputs: `dep_rate_A_per_s`
  (angstrom/second → SI m/s ×1e-10) + measured-not-set `Ipk (A)`. Headers carry the
  BO-sampling (BayBE, non-space-filling) caveat, provenance/license, and the rename note.
- **`tests/test_empa_ingest.py`** — 9 tests, all green: byte-identical determinism +
  checked-in-CSV freshness, exact row counts (601/651/401/495/601/401 = 3,150),
  calibrated column == y1×factor (factor re-read independently), TOML bounds ==
  Campaign.json bounds exactly (json.loads twice), BatchNr sort, real ingest of a PRR and
  a duty campaign (source=real_tool, PW ×1e-6 and dep-rate ×1e-10 SI spot-checks, BatchNr
  in extras), `continuous_si` trap respected (100 us → 1e-4 s).
- **Gotchas found in the DATA (all documented in the specs + pinned by tests):**
  (1) the adapter reserves `.` in variable names → the three `pos. *` columns are renamed
  dot-free in the tidy CSVs (`prepare_empa.RENAMES`);
  (2) **Ti - 120 W - short PW is degenerate**: ALL BatchNr==1, ALL FitNr null (the
  "BatchNr = run order" fact is FALSE there; raw file order kept), and its Campaign.json
  bounds are full-precision data extents while df values are rounded to 10 decimals, so
  **5 of 495 rows sit ~3e-11 outside bounds** — exact-inclusive ingest rejects them
  (`on_error="raise"` fails; `"skip"` → 490+5, asserted in tests). Data NOT clamped.
- Verification: `tests/test_empa_ingest.py` 9 passed; cross-check `-k empa` 9 passed /
  527 deselected; tabular-adapter regression 72 passed. ruff still blocked (WinError
  4551) — hand-conformed to ruff-format style (one long f-string table header matches the
  sputtering example's own pattern; E501 is in the ignore list).

---

## 2026-07-19 (~late afternoon) — Empa HiPIMS M1-gate-form runner (`run_m1_empa.py`)

Subagent task: build the M1-gate runner on the real Empa data, modeled on
`examples/real_data/sputtering/run_m1_sputtering.py` (same rig code paths: GPForwardModel →
SplitConformalCalibrator(α=0.1) → ConformalForwardModel → §8 PessimisticInverseSolver).
Nothing under `src/` or `data/` touched; nothing committed. Smoke-verified on ONE campaign
only (per task); the full 6-campaign run is the orchestrator's.

- **`examples/real_data/empa_hipims/run_m1_empa.py`** — per campaign: skip-ingest with
  PINNED reject expectations (only ti_120w may reject, exactly 5 — any other reject count
  aborts loudly); TEMPORAL split by BatchNr run order (train 60% / cal 20% / test 20%) +
  seeded RANDOM contrast split of the same sizes; per-output AND pooled conformal PICP with
  the **exact (Clopper-Pearson) binomial 95% CI** — the §15.3 M1 row's DIRECTIONAL gate
  ("nominal 0.90 inside the CI", never ±2%). Full runs add: 4-campaign PRR-space OOD check
  (random-split models; ID = own held-out rows, OOD = other campaign's recipes; directional
  "OOD epistemic > ID", 12 ordered pairs) and a 3-regime inverse demo on Al-shortPW.
  Machine-readable JSON → `results/m1_empa[.<slug>][.smoke].json` (no wall-clock in payload;
  deterministic apart from `wall_seconds` timing fields). `--campaign <slug>` and `--smoke`
  (GP restarts 5→2, solver 120→24) keep any single command well under 8 min (full ≈4–6 min).
- **`examples/real_data/empa_hipims/README.md`** — converter → tests → runner how-to.
- **Smoke run (al_120w_short_pw, --smoke, 42 s):** temporal pooled PICP 0.875 [0.826,0.914]
  PASS, random pooled 0.917 [0.874,0.948] PASS; per-output all PASS (dep temporal 0.858
  [0.783,0.915], Ipk temporal 0.892 [0.822,0.941]). Temporal coverage sits BELOW random —
  consistent with BO drift; the directional CI check absorbs it at this n.
- **Judgment calls:**
  (1) the task-suggested [q60,q90] inverse band is **arithmetically un-creditable**: width
  0.167 Å/s < 2κσ_ale = 0.454 Å/s, so NO recipe can pass §8 pessimism at κ=2 — kept as
  demo regime 1 (honest abstention, expectation computed from the arithmetic, not
  hardcoded), with [q10,q90] as the FEASIBLE regime (NN-verified: nearest measured run
  in-band) and beyond-max as regime 3 (INFEASIBLE, "genuinely unreachable", −10.8σ);
  (2) inverse demo + OOD check use the RANDOM-split model (temporal-train model never saw
  the late exploitation cluster — would conflate the machinery demo with the drift story);
  (3) pooled CI flagged optimistic (outputs share test rows);
  (4) ti_120w temporal split labeled `file_order_UNVERIFIED` in JSON + console (BatchNr
  degenerate), its gate row starred in the summary;
  (5) solver stays on the default FD path (`analytic_grad` untouched), sputtering-template
  consistency.
- Verification: `tests/test_empa_ingest.py` 9 passed after all edits; smoke JSON structure
  inspected (meta/campaigns/ood_check/inverse_demo; per-output picp/ci95/nominal_in_ci/mpiw).
  ruff still App-Control-blocked — style hand-conformed to the sputtering template.

## 2026-07-19 (evening) — ADVERSARIAL REVIEW of the Empa prep + M1-gate runner (lens: leakage/splits/gate/statistics)

Reviewed `examples/real_data/empa_hipims/` (prepare_empa.py, 6 specs, run_m1_empa.py,
tests/test_empa_ingest.py) by reading the code AND re-running everything. Nothing under
`src/`/`data/` touched; nothing committed. **No BLOCKER found — no published number is wrong.**

- **Leakage attacks (all came up clean, verified in source):** GP standardizes with TRAIN
  stats only (`src/rig/forward/gp.py` fit, §5.3 guard); conformal scores come ONLY from the
  cal slice (`SplitConformalCalibrator.fit`); temporal split is contiguous first-60/next-20/
  last-20 in BatchNr order with a monotone assertion at ingest; inverse-demo target bands use
  TRAIN-slice quantiles (`Y_all[fit_idx, 0]`); NN verification of FEASIBLE candidates uses
  only measured rows. nRMSE normalizes by test range (reporting only, not fitting).
- **Gate arithmetic independently recomputed** (scipy, all 6 smoke CIs): matches the checked-in
  JSON to 6 decimals; CP CI + "nominal inside" is the exact-binomial-test dual of §15.3's
  "coverage within the binomial CI of nominal" (directional, not ±2%). NB §15.3 says "given
  n_cal"; the runner uses n_test — numerically identical here (60/20/20 makes n_cal == n_test
  in all six campaigns).
- **Determinism verified by running:** two fresh `--campaign al_120w_short_pw --smoke` runs
  byte-identical AND equal to the checked-in results JSON modulo `wall_seconds`.
- **NEW EVIDENCE — full 6-campaign run executed for the first time (smoke restarts, 44 s),**
  results in the reviewer's scratchpad (not checked in; full-restart run still owed):
  1. **`ti_200w_high_pw` FAILS the gate on BOTH splits** at smoke restarts: temporal pooled
     PICP 0.817 [0.762,0.864] (under-coverage, both outputs FAIL) and random pooled 0.950
     [0.914,0.974] (OVER-coverage; Ipk 0.958 alone fails). All other 5 campaigns pass both
     splits. Do NOT generalize the al_120w smoke PASS to "the gate passes" — the full-restart
     run must be reported WITH its FAIL rows.
  2. **OOD check: 8/12 directional pass.** The 4 failures are EXACTLY the cross-material
     same-power-tier pairs (Al↔Ti at 120W-shortPW and at 200W-highPW): epistemic sigma AND
     support are numerically indistinguishable (e.g. 0.0315 vs 0.0307; −2.24 vs −2.22)
     because material is not a knob and those campaign pairs share ~the same knob box. The
     model is PROVABLY BLIND to material shift — frame the 4 failures as material-blindness
     (pool campaigns only with material as conditioning), not as noise.
- Minor doc nit: the runner BUILD_LOG entry above says "(no wall-clock in payload)" — the
  payload does carry `wall_seconds` (the only nondeterministic field), as its own parenthetical
  then admits.

---

## 2026-07-19 (~night) — M0 VENUE DECIDED (user: Empa HiPIMS); FULL 6-campaign M1 gate run on REAL data

USER decision landed: "continue with recommended" → **Empa bipolar HiPIMS is the M0 venue.**

Orchestrated build (workflow wf_32bbc728-683, 4 agents): prep (converter + 6 specs + 9 tests) →
runner (`run_m1_empa.py`) → 2 adversarial reviewers (units/calibration: CLEAN; leakage/stats:
APPROVE, no blockers, 1 MAJOR forward-obligation: report the full run WITH its FAIL rows).
Applied reviewer polish (3e-11..4e-11 rewording; n_cal==n_test duality comment in `binom_ci`).

**FULL RUN (gp_restarts=5, 168.6 s, seeded): 5/6 campaigns PASS the §15.3 directional gate on
BOTH splits; `ti_200w_high_pw` FAILS BOTH** — temporal by under-coverage on both outputs (dep
0.808 [0.726,0.874], Ipk 0.825 [0.745,0.888]; drift + static split-conformal, ACI §5.6/D4 not
wired here), random by over-coverage on Ipk (0.958 [0.905,0.986]). Same dual failure as the
reviewer's smoke pre-run → stable, not restart noise. OOD 8/12 with the 4 failures EXACTLY the
cross-material same-tier pairs (epi/support numerically indistinguishable from ID — material
blindness: screening cannot flag a shift in a non-input variable; per-campaign models are the
honest config). Inverse demo on real data: over-tight band → INFEASIBLE (width < 2κσ_ale floor,
§8.8 diagnosis), credit-wide band → FEASIBLE ×3 on-support, ALL 3 NN-verified in-band
(dists 0.092/0.201/0.237), beyond-data → INFEASIBLE −10.8σ "genuinely unreachable".

Verification: determinism double-run identical modulo wall_seconds; all 24 per-output
Clopper–Pearson CIs independently recomputed (scipy) — exact match. Suite: 9 empa tests green;
542 collected.

Files: `examples/real_data/empa_hipims/{RESULTS.md, results/m1_empa.json, results/m1_empa.rerun.json}`
+ the fix edits. BUILD_STATE M0/M1 rows updated. Nothing committed.

---

## 2026-07-19 (~late night) — D4/§5.6 ACI drift path WIRED into the Empa runner (subagent)

Extended `examples/real_data/empa_hipims/run_m1_empa.py` with the D4/§5.6 ONLINE ACI evaluation
as an ADDITIONAL path — the static split-conformal blocks are unchanged (verified byte-identical
vs `results/m1_empa.json` for ti_200w, wall_seconds aside). `src/` untouched: the library
`ACIController` + `SplitConformalCalibrator` already implement the protocol (reused UNCHANGED).

Design: per campaign, per split (temporal AND random — random is the exchangeable CONTROL), a
FRESH `SplitConformalCalibrator` is fitted on the SAME calibration slice (deterministic, identical
scores; isolates ACI's online appends from the static path), then `ACIController(cal,
alpha_target=0.1)` with **LIBRARY DEFAULTS ONLY** (gamma=0.05, window=50, clip (0.001,0.5),
update_scores=True — uniform everywhere, fixed before outcomes; no tuning-to-pass) streams the
test rows in split order: interval at CURRENT alpha_t scored FIRST (hit/miss + width), THEN
`observe(x, y)` (guard asserts observe's err == the pre-scored miss). Recorded per output:
realized coverage + exact binomial CI (same directional gate form as static), mean finite width +
n_infinite_width, alpha_t (used min/max/mean, final), rolling-window coverage min over FULL
windows (the §5.6 drift-detector statistic) + final. JSON: new `splits.<name>.aci` block + new
`meta.aci` key (all existing keys/values untouched); GATE SUMMARY gains temporal+ACI / random+ACI
rows; module docstring documents guarantee (asymptotic average coverage under arbitrary shift,
Gibbs & Candès 2021), non-guarantee (no finite-sample exactness → CI row is directional), default
hyperparams, and the §20.2 conformal-PID-supersedes-bare-ACI note.

**ti_200w_high_pw single-campaign run (full restarts): ACI PASSES the directional check on BOTH
splits with defaults.** Temporal: dep 0.892 [0.822,0.941], Ipk 0.883 [0.812,0.935], pooled 0.887
(static: 0.808/0.825/0.817 all FAIL); alpha_t driven down to 0.066/0.031, rolling min 0.840/0.860
(< 0.90 — the drift detector still flags the drift episode even though average coverage recovers).
Random: dep 0.917, Ipk 0.900, pooled 0.908 (static over-coverage 0.942/0.958/0.950); alpha_t rose
to 0.20/0.10 (tightening, as the control should). CAVEAT for the full-run report: on the temporal
stream 1 (dep) / 5 (Ipk) steps emitted INFINITE-width intervals (alpha_t below 1/(n_scores+1)) —
auto-hits by construction, disclosed as `n_infinite_width`; excluding them Ipk is 101/115 ≈ 0.878,
still inside its CI. Known ACI behavior (coverage recovered partly via trivial intervals), not
hidden. Determinism: double-run to scratchpad identical modulo wall_seconds. Full 6-campaign
re-run + RESULTS.md/BUILD_STATE updates are the ORCHESTRATOR's; `results/` not overwritten here.

---

## 2026-07-19 (~late night, cont.) — ACI drift path: FULL 6-campaign run + orchestrator adversarial verification

Ran the ACI-extended runner across all 6 campaigns (full restarts, 294 s) and a determinism
re-run. Promoted the verified result to `results/m1_empa.json` (a strict superset: static blocks
byte-identical to the prior baseline + new `splits.*.aci` blocks + `meta.aci`).

**RESULT: ACI repairs `ti_200w_high_pw` on BOTH splits with library defaults** — temporal pooled
0.817 FAIL → 0.887 PASS (α_t driven down: wider bands where it under-covered), random pooled 0.950
FAIL → 0.908 PASS (α_t driven up: tighter bands where it over-covered). The other 5 campaigns stay
PASS (control moves small). §5.6 rolling detector still fires (window-min 0.840/0.860 < 0.90) —
detector and repair independently visible. So the one static M1 failure was a calibrator-choice
artifact, not a modelling dead-end.

**The Fable-5 reviewer agent died on a usage limit before verifying — so the orchestrator (Opus)
ran the adversarial checks by hand** (`scratchpad/verify_aci.py`, reproduces): (A) BASELINE
INTEGRITY — ACI run's static blocks byte-identical to the pre-ACI baseline (ACI didn't perturb
it); (B) DETERMINISM — v1==v2 modulo wall_seconds; (C) NO TUNING-TO-PASS — all 12 campaign/split
hyperparameter blocks == library defaults (γ=0.05, window=50, clip (0.001,0.5), update_scores=True)
exactly; (D) INFINITE-WIDTH TRAP — ACI can "cover" via unbounded intervals when α_t < 1/(n+1); on
ti_200w temporal 1 dep / 5 Ipk of 120 steps did so. EXCLUDING them (num+denom): Ipk 0.878
[0.804,0.932], 0.90 still inside → the PASS is NOT hollow. Every ACI PASS survives the exclusion
(scriptE recomputes all 24 CIs, all match). VERIFICATION PASSED.

Honest limits kept in RESULTS.md: directional (not finite-sample-exact) CI row; `update_scores=True`
conflates α-adaptation with score-refresh (a non-default arm would separate them — excluded by the
cardinal no-tuning rule); §20.2 conformal-PID is the eventual online endpoint, this validates the
bare-ACI D4 component. `RESULTS.md` updated (static→+ACI table, adversarial-checks paragraph,
revised verdict). Nothing committed.

---

## 2026-07-21 — inverse-capability audit

Created root `audit.md` at the user's request; no product code was changed. Read the required
BUILD_STATE/BUILD_LOG and relevant plan sections, then inspected core inverse/forward/active/
qualification code, the real Empa runner and recorded results.

Verdict: RIG can fit forward models and propose constrained inverse candidates, but is not
deployment-ready for predicting inputs from desired outputs. Release blockers documented: solver
feasibility uses raw sigmas rather than mandatory `conformal_set`, and `ConfirmationBatchGate` is
not wired into inverse, active-loop or Empa execution. Also documented: 1.0/1.0 active-loop/M2
settings versus binding 2.0/2.0; incomplete robust-objective features; limited real-data evidence;
stale revalidation docs; and local ruff failures (14 diagnostics; four files need formatting).
The focused pytest-plus-quality bundle exceeded the 60-second command limit before a completed
fresh test result, so no current suite pass is claimed. BUILD_STATE updated. Nothing committed.

---

## 2026-07-22 — Audit remediation + the two remaining M1 items (orchestrated, 4 parallel subagents)

Orchestrator (Fable-5) validated every audit.md finding FIRST-HAND against source before
delegating (F1/F2/F3/F5/F6/F7 all CONFIRMED; F3 sharpened: the SOLVER already defaults to the
binding 2.0/2.0 — only the loop/M2 sat at 1.0/1.0; F7 sharpened: loop.py DOES auto-revalidate
on the ensemble fast/full split, but no default path conformal-wraps, so the conformal component
was inert everywhere). Four subagents ran in parallel with disjoint file ownership (opus x3 +
sonnet x1); orchestrator did unowned-file ruff cleanup, the full 6-campaign Empa run, promotion,
and these docs. Individual entries follow.

## 2026-07-22 — F1/F3/F5/F7 audit fixes (inverse feasibility calibration) [subagent A, opus]

Made the §13.2 `C(x)⊆Z*` conformal containment part of the DEFAULT `solve()` path
(`_conformal_screen`/`_conformal_infeasible` in `pessimistic.py`) whenever `self.model` is
conformal-wrapped, with the reval path's anti-false-abstention pool sweep and an
aleatoric/coverage-worded Infeasible; unwrapped models are byte-identical (F1). Added
`RecipeCandidate.calibration_status ∈ {model-feasible, conformal-checked, revalidated}`,
default `model-feasible` = raw-σ only, explicitly NOT a calibrated guarantee (F1/F5).
Aligned `ActiveLearningLoop` defaults to the binding §8 2.0/2.0/0.02 (were 1.0/1.0/0.01);
solver already correct; added a defaults-match pin test; `run_m2_sweep.py` keeps 1.0/1.0/0.01
but now labels it `FEASIBILITY_POLICY` (ablation) in JSON/stdout — binding-policy re-run owed
(F3). Added the §8 fidelity ledger to `pessimistic.py` docstrings: joint→per-output product,
worst-member→z_epi·σ_epi, PGD→first-order δ, flow-typicality→Mahalanobis, conformal→default-on
when wrapped (F5). Annotated the stale "active/loop.py never sets revalidation_model" claim in
`docs/dimensionality-2026-07-17.md` in place (F7). New `tests/test_conformal_feasibility.py` —
mechanism twin of the d=20 false success (overconfident raw σ + honest conformal band →
wrapped default path REJECTS what raw pessimism certifies); red-proof: disabling the gate
turned 2 tests red, restored green. `_conformal_in_box`/`_conformal_set`/`_conformal_spill`
generalized with `model` arg defaulting to `revalidation_model` (preserves test_ensemble's
3-arg calls). `tests/test_interfaces.py` canonical-fields test updated for the new field
(ownership extended mid-task after another agent's suite run caught it). Files: interfaces.py,
inverse/pessimistic.py, active/loop.py, examples/run_m2_sweep.py, docs/dimensionality doc,
tests/{test_active_loop,test_interfaces,test_conformal_feasibility}.py. Bar: 85+32+9+51 tests
green across inverse/loop/d2/interfaces/ensemble+distill; ruff clean on touched files. Empa
inverse demo confirmed UNAFFECTED (runs the raw GP → gate inert → FEASIBLE×3 stands, now
labeled model-feasible).

## 2026-07-22 — F2: confirmation-campaign orchestrator wired to the gate [subagent B, sonnet]

Built `src/rig/active/campaign.py` (`ConfirmationCampaign`, `CampaignResult`,
`CandidateCertification`, `NothingToQualify`) — audit F2: `ConfirmationBatchGate` existed but
no production path called it. `Infeasible` input → ZERO machine calls + typed
`NothingToQualify`; otherwise each `RecipeCandidate` goes through one shared gate
(`gate.certify` per candidate), every confirmation measurement reconstructed as a logged
`RunRecord` from the gate's own `evidence["observed_values"]` (no duplicated statistics),
certified/rejected partitioned solely by `qualification.passed`. Multiplicity surfaced:
`n_candidates` + `confidence_per_candidate` always reported; opt-in `bonferroni=True` applies
alpha/q (incompatible with a pre-built static gate — raises). Serial-correlation/Cpk/staged-
ladder caveats referenced by number from qualification.py's honest-limits docs via
`CampaignResult.caveats`; provenance caveat auto-added when source != real_tool (in-silico
rehearsal, not tool qualification). Deterministic: uuid5 run ids, synthetic clock, no own RNG
— two fresh campaigns byte-identical. 19 new tests incl. one sim-gated InSilicoMachine
integration test; red-proof: inverting the certified/rejected partition turned 5 tests red
(incl. the sim-gated one), restored. Bar: test_campaign+test_qualification 63 passed; ruff
clean; import-linter 1 kept / 0 broken. NOT yet auto-invoked from ActiveLearningLoop or the
Empa runner — the orchestration path now exists; the hookup into a real campaign remains a
deployment step (M4/M5 territory).

## 2026-07-22 — Conformal-PID (§20.2 online ENDPOINT) BUILT + wired + tested [subagent C, opus]

New `src/rig/calibration/pid.py::ConformalPIDController` — the §20.2 endpoint that supersedes
bare ACI (Angelopoulos–Candès–Tibshirani 2023 P+I; decaying-step variant Angelopoulos–Barber–
Bates 2024, `step="fixed"|"decaying"` default fixed; scorecaster hook OFF by default). Forms
verified against the paper's released `core/methods.py` (P = quantile tracker
q_{t+1}=q_t+η(err_t−α); I = saturation_fn_log = KI·tan(S·ln(t+1)/(Csat·(t+1))); combo
q = qts + integrator). Tracks the threshold q_t on the STANDARDIZED-residual scale, bands
`mean ± q_t·σ_total(x)` (reuses `SplitConformalCalibrator` UNCHANGED — conformal.py untouched).
FINITE BY CONSTRUCTION — the paper's mytan→±∞ replaced by a clamped-argument tan; q_0
warm-start has a finite fallback at tiny n; observe() appends NO scores (pure threshold
dynamics — no ACI update_scores conflation, no stale-buffer infinite quantile). Library
defaults η=0.1 / KI=2.0 / Csat=7.0 / window=50, justified from the paper rescaled to the
standardized score — NOT tuned on Empa; recorded as hyperparameters.provenance in the JSON.
`tests/test_pid.py` 10 tests (exchangeable coverage; static-undercovers-vs-PID-recovers REPAIR;
pre-update-scoring guard; all-miss finiteness with n_infinite_width==0; KI=0 integrator
mutation-proof; decaying-volatility; determinism; per-output independence; warm-start-finite;
scorecaster hook) — red-proof: flipping the P-update sign turned 7/10 red, restored. Wired as
the THIRD runner path in run_m1_empa.py (`splits.*.pid` + `meta.pid` + GATE SUMMARY +PID rows)
mirroring the ACI protocol exactly; runner made ruff-clean (E402 restructure + noqa, F541;
prepare_empa.py reformatted, AST-identical). Smoke: static+ACI blocks byte-identical to
pre-edit baseline; deterministic.

## 2026-07-22 — FULL 6-campaign run + verification + promotion [orchestrator]

Ran the PID-extended runner across all 6 campaigns twice (full restarts, 220.9 s each,
scratchpad first — results/ never written by an unverified run). Verification script
(scratchpad m1_full/verify_full.py): (A) DETERMINISM v1==v2 modulo wall_seconds — PASS;
(B) BASELINE the static+ACI+OOD+inverse content byte-identical to the recorded
results/m1_empa.json (the only inverse-demo delta being the new additive calibration_status
labels) — PASS; (C) PID: **all 12 campaign/split gates PASS with n_infinite_width = 0 on
every campaign/split/output** (ti_200w_high_pw temporal 0.875 [0.826,0.914], random 0.929
[0.889,0.958]; static FAILs unchanged; §5.6 rolling detector still fires on the temporal
drift episode, window-min 0.820/0.840); (D) demo-verdict spot-check was vacuous (schema
mismatch) — superseded by (B), which covers the full demo block; noted rather than counted.
Promoted v1→results/m1_empa.json, v2→results/m1_empa.rerun.json (superset discipline, same
as the ACI session). RESULTS.md: +§20.2 PID section (static→ACI→PID table, finiteness
argument, threshold traces), header/status/verdict updated, caveat 2 updated for the
default-on conformal gate. NET: of the two remaining M1 items, **conformal-PID is DONE**;
material-conditioned pooling is the subagent-D entry below. Also this session (orchestrator):
ruff cleanup of unowned files (mfl bakeoff I001+format, test_mfl_baseline E402-noqa+F841,
test_empa_ingest format; 15 tests green) and import-linter re-verified with the corrected
exit-code form (1 kept / 0 broken, exit 0) — the audit's "lint-imports unavailable" was the
Windows Store Python PATH quirk, not a broken contract.

## 2026-07-22 — Material-conditioned pooling BUILT + full run (M1 remainder; audit F4 control) [subagent D, opus]

Built `examples/real_data/empa_hipims/run_m1_empa_pooled.py` + `tests/test_empa_pooled.py`
(7 tests) reusing `MultiToolGPForwardModel` (ICM §10.4) with **material as the task**; pooling
within each parameterization subspace (PRR 4 campaigns / DUTY 2 — knob names differ; never
across). **No src edits needed.** Recorded `--full` (n_restarts=2, seed 0, ~9 min) →
`results/m1_empa_pooled.json`; deterministic by construction (seeded; verified zero wall-clock
keys in the payload) + determinism unit tests. Baseline comparisons read `results/m1_empa.json`
(valid — its static/ACI/OOD/inverse blocks stayed byte-identical through the PID promotion).
Split reconstruction asserted equal to the baseline's split_sizes for all 6 campaigns.

- **Block A (awareness — the F4 control):** fitted al↔ti task corr 0.9985. Directional pass
  6/12 (the 4 previously-blind pairs: 3/4; all cross-material: 6/8); unknown-material fallback
  epistemic dominates both materials 12/12 (§5.8); predicted MEAN shifts 0.78–1.67 Å/s on all 4
  blind pairs (tens of σ_epi — structurally impossible for a per-campaign model; baseline
  flagged 0/4). Honest catches: the epistemic flag is ASYMMETRIC (Al-conditioning carries
  intrinsically higher epistemic than Ti after pooled standardization — fires cleanly only when
  the cross material is Al); support stays ~flat (±0.06) — input-space screening remains blind
  to a same-box material shift. Campaign-as-task corroborates (1/4). Awareness = material as an
  explicit axis (mean + unknown-fallback), NOT auto-detection of a wrong-material query.
- **Block B (LOMO honest transfer):** zero-shot §5.8 domination holds both directions but the
  fallback mean is the trained material's surface (dep-RMSE 9–24× the full-data ceiling — no
  zero-shot mean transfer). Few-shot K=10/20: dep-RMSE 2–14× ceiling, raw predictive PICP
  0.79–0.94 (3/4 arms mis-calibrated), conformal coverage held only via near-vacuous MPIW
  0.8–1.8 Å/s. `transfer_claimable=False` → **cross-material transfer FORBIDDEN (F4)**.
- **Block C (pooling cost):** 0 PASS→FAIL flips across all 12 cells; pooling FIXES
  ti_200w_high_pw/random (0.950 FAIL over-cover → 0.896 PASS), nudges its temporal 0.817→0.833
  (still FAIL — the drift case belongs to ACI/PID); others within ±0.04, all PASS.
- Gotchas recorded: (1) pool per parameterization subspace (PRR/DUTY knob names differ);
  (2) don't over-read the 3/4 epistemic flag — the robust awareness is mean-shift +
  unknown-fallback (support and same-tier epistemic are confounded by the shared knob box);
  (3) the LOMO few-shot cal/test split MUST be exchangeable (random) — a first draft's
  front-to-back temporal split put ti_120w (120 W) in cal and ti_200w (200 W) in test and faked
  a 0.55 conformal PICP (tier shift masquerading as transfer failure); fixed before recording;
  (4) smoke (max_iter=60) vs full (max_iter=100) differ at float precision — determinism claims
  are same-config only.

Tests: 7 new + 9 empa-ingest = 16 green; ruff check + format clean on both files. RESULTS.md
pooling section + verdict update integrated by the orchestrator. NET: **both remaining M1 items
(conformal-PID §20.2; material-conditioned pooling) are now DONE** — awareness gained, transfer
honestly measured and NOT claimed, pooling coverage-safe.

## 2026-07-22 — F2 remainder: opt-in qualification hook wired into ActiveLearningLoop [subagent I, sonnet]

`src/rig/active/loop.py`: added an OPT-IN `qualification: ConfirmationCampaign | None = None`
constructor argument. Default `None` is byte-identical to every prior release (proven by
`test_loop_qualification_none_is_byte_identical_to_no_param`: two seeded Trajectories — param
omitted vs explicit None — compared via dataclass ==; structurally guaranteed besides, all ctor
params keyword-only). When set, BOTH stop points (seed-DoE early return, in-loop per-batch hit)
first wrap every in-spec recipe as a RecipeCandidate (`_hitting_candidates`; only `.recipe` read
downstream, D7-safe) and run them through `qualification.run(...)` (`_qualify_hit`) before the
hit stands: certified → hit stands, `Trajectory.qualification_outcome` (new additive field)
carries the CampaignResult, stop_reason gains a "(qualified)" suffix; rejected → hit NOT
declared, loop does not stop, CampaignResult appended to `Trajectory.qualification_rejections`
(new additive field), falls through to the ordinary non-hit path (the data still counts, only
the STOP is gated); budget-would-overspend → nothing fires, distinct stop_reason "unqualified
hit, budget exhausted". Confirmation runs are budget-honest: charged against the same
`n_queries`/`budget`, computed exactly via `_expected_qualification_calls` (n_hitting × n_runs,
reading the campaign's private gate config since no public cost accessor exists — documented
coupling). Caveat surfaced NOT fixed (outside ownership): repeated `.run()` calls on one
ConfirmationCampaign instance can produce colliding RunRecord ids across CampaignResults
(candidate/run indices restart at 0 per call) — noted in the class docstring; small follow-up
owed in campaign.py (e.g. a per-call salt).

Tests (tests/test_active_loop.py, 11→17): defaults-to-None; byte-identity; pass-path with exact
n_queries arithmetic (8 seed + 8×5 confirmation = 48); pass determinism; rejection
does-not-stop (RED-PROOFED: hand-inverted `n_certified > 0` to `>= 0`, exactly that test went
red, restored green); budget-exhaustion refuses to fire. Bar: test_active_loop + test_campaign
36 passed; extra diligence test_active_mbe + test_ensemble 23 passed (other loop consumers,
neither passes qualification=). ruff check + format clean. Nothing committed.

## 2026-07-22 — D7 different-physics ROM verifier BUILT (Phase-0 owed item closed for MBE) [subagent F, opus]

New `src/rig_adapters/mbe/verifier.py::GeometricDepositionVerifier` — purely-geometric Knudsen
line-of-sight deposition ROM (cosine emission (cos_e)^m·cos_i/D², m=1 Lambertian, rotation-
averaged ring flux → area-weighted wafer mean; ZERO Arrhenius/thermal/regime/kMC content;
imports only numpy+stdlib; shares no code or constants with the fast path — enforced by an AST
import/code-name-scan test and a predicts-with-sim-monkeypatched-to-raise test). Bounds the
flux-scale channel `thickness_grown` via `thickness = film_thickness × g`,
`g = Φ̄(config)/Φ̄(nominal)` (=1 at the nominal build; obeys ~1/H²; independently reproduces the
sim's internal flux_nonuniformity_pct=1.558 exactly without touching its code). Wired into
`MBEAdapter.independent_verifier` as a distinct object → `validate_adapter`'s D7 identity check
passes with a REAL verifier for the first time; NO core (`src/rig/`) edits needed — the
existing Callable slot sufficed.

Scope established EMPIRICALLY and stated honestly: the machine's combined nonuniformity is
~98% Arrhenius-thermal (thermal_nu≈78 vs geometric flux_nu≈1.56) → out of scope, exposed as a
diagnostic only; slip/T_center/bow are thermal/thermo-mechanical → cannot verify; scoped to
the nominal chamber build (this in-silico machine models thickness as geometry-independent, so
the catchable corruption is a hidden FLUX-SCALE pathology — seasoning/depletion/drift — not a
geometry rebuild). Agreement band 5%, derived from noise-vs-pathology separation (metrology
≤~0.7% rel; meaningful pathology ≥10%), NOT tuned to an observed gap (nominal gap is 0).
Detects flux_eff 0.85/0.7/0.5 (rel_error −0.15/−0.30/−0.50) and accumulated seasoning loss;
within-band 2% correctly not flagged.

Tests: new `tests/test_mbe_verifier.py`, 5 groups — (a) D7 identity (red-proofed: wiring the
physics_plugin as verifier makes validate_adapter raise), (b) nominal agreement, (c)
independent disagreement (red-proofed: band 0.05→0.9 turns all 5 red), (d) determinism,
(e) mechanical different-physics; a/d/e run un-gated in CI. One unavoidable single-assertion
update in `tests/test_mbe_adapter.py` (`independent_verifier` no longer None — now asserts
present + distinct from physics_plugin). Bar: 45 passed (verifier+registry+adapter), 162
broader sweep; ruff clean; import-linter 1 kept / 0 broken. Nothing committed.

## 2026-07-22 — WP-E §8 hardening: PGD δ-box + flow typicality [subagent G, opus]

Built the two owed §8 robust-objective features (WP-E remainder item 2 / audit F5's "implement
and benchmark"), both OPT-IN and default-off byte-identical. Files: src/rig/inverse/
pessimistic.py, NEW src/rig/inverse/typicality.py, inverse/__init__.py (lazy export), NEW
tests/test_inverse_hardening.py. Nothing committed.

- **PGD δ-box** (`delta_mode="pgd"`, default "taylor"; pgd_steps=10): `_pgd_delta` replaces the
  first-order Σ|J|·Δ term with the §8.5 max_{δ∈Δ} inner problem via projected ℓ∞ sign-gradient
  ascent (δ=0 start, step Δ/4, both directions, deterministic), driven by the model's own
  `jacobian` — GP tier, no torch. Reproduces Taylor exactly on a linear μ (pinned); catches the
  curvature Taylor misses: convex μ=1.5x² → Taylor 2.100 vs brute-force box max 2.284 = PGD
  2.284 (<1e-6); end-to-end, a spec that is +5σ FEASIBLE under Taylor is −3.1σ INFEASIBLE under
  PGD. Honest limits: a LOWER bound (few box corners, interior extrema/J=0 stalls missed), not
  a certificate; ≈2·pgd_steps·m extra jacobian calls; RAISES with analytic_grad=True (mixing
  objectives forbidden). Red-proofed: stubbing _pgd_delta to Taylor turns the benchmark red.
- **Flow typicality** (`typicality=`, default None): NEW `FlowTypicalityScore` — small
  unconditional zuko NSF over the standardized input marginal (~7 s CPU, seeded), score =
  −|log p − E_train[log p]| (the TYPICALITY-SET statistic, NOT raw log-likelihood — Nalisnick
  2019), floor = 5th pct of train scores. Wired as an ADDITIONAL hard §8.2 screen alongside the
  Mahalanobis floor (both must pass; the fail-closed cheap fallback is never replaced);
  screen-only, deliberately NOT in the soft λ_m reward (not autograd-composable; too costly in
  the hot loop). Closes the multimodal hole: bimodal 2-D gap point (nn-dist 2.06) — Mahalanobis
  wrongly ACCEPTS (−0.01 ≥ floor −2.14), typicality REJECTS (−3.80 < −2.23); solver end-to-end
  certifies the gap recipe FEASIBLE without the screen, INFEASIBLE with it (that flip is the
  red-proof). Nalisnick property shown via a sharp-spike case (raw log-lik would ADMIT +1.97;
  typicality rejects −4.23 < −3.36). Honest limit: the high-d Gaussian SHELL case is NOT
  robustly demonstrable with a small CPU flow (it under-fits the peak; measured at d=12–24) —
  documented rather than faked; the screen must be fitted on the same X_train as the floor.
- Torch-free `import rig` preserved (TYPE_CHECKING + duck-typing + lazy export;
  subprocess-verified). Ledger docstrings updated — PGD + flow typicality now
  "IMPLEMENTED, opt-in"; remaining GP-tier approximations: joint-MC spec-hit, worst-of-K
  epistemic. Bar: 89 passed (test_inverse 72 unchanged + hardening 15 + conformal_feasibility
  2); ruff check+format clean.

## 2026-07-22 — M3 acceptance v2: honest re-run DONE [subagent H, opus]

New: `examples/run_m3_acceptance_v2.py`, `docs/m3-acceptance-v2.json`,
`docs/M3-acceptance-v2-2026-07-22.md`, `tests/test_m3_acceptance_v2.py` (v1 left byte-for-byte
intact). Fixes all four v1 audit critiques: InSilicoMachine (not toy tanh; metrology noise ON;
outputs = the coupled T_center + bow_cooldown_um pair), NON-SATURATING ground-truth pass rule
(top-1 hit COUNTS on the noise-free physics; confidences are not even arguments — unit-tested
that a saturated-confidence flip changes nothing and a ground-truth flip changes the verdict),
PRE-REGISTERED targets via a cold_light-only pre-probe (3 genuinely cold_light-INFEASIBLE + 3
HIT controls; selection cannot manufacture a d2 win), scoring never touches the surrogate. §8
arms run the BINDING policy (κ=z_epi=2.0, δ=0.02). §14.6 SBC/TARP gate is BLOCKING and PASSED
before any arm ran (sbc_p=[0.90,0.39], tarp_err=0.025, N=1024 training runs).

VERDICT PASS, non-tautologically: ground-truth hits cold_heavy 6/6, cold_light 3/6 (honest
INFEASIBLEs at −225σ/−120σ/support-floor), d2_light 6/6 — at 0.1875× starts (9/48) and 0.229×
wall-time. **Load-bearing caveat from the agent's own adversarial control: 9 RANDOM restarts
also rescue 3/3** (cold restarts 1/2/3→0, 5→1, 9→3, 48→3) — so d2's edge over cold_light is
the restart BUDGET, not amortized-vs-random starts; the stronger claim "amortization beats
equal-budget random search" is NOT demonstrated on this smooth, low-dim, near-invertible map
and is not claimed. Scope narrowed to 2→2: `thickness_grown` excluded because a
near-deterministic identity output breaks SBC calibration (overconfident posterior) — a real
gotcha recorded for future gate configs. Determinism: smoke double-run byte-identical
(timing-stripped digest 6f75c98f…). Tests: 5 passed; ruff check+format clean. v1's PASS stays
recorded as the toy-tanh result it was. Nothing committed.

## 2026-07-22 — M2 binding-policy re-run (audit F3 owed item CLOSED) [subagent E, opus]

Added `--policy {ablation,binding}` to `examples/run_m2_sweep.py` (default ablation =
1.0/1.0/0.01, behaviorally identical to the published run, still writes docs/m2-result.json;
binding = 2.0/2.0/0.02 → docs/m2-result-binding.json). Threaded through BOTH RIG pin sites
(`_make_factories`→ActiveLearningLoop and `_inverse_readout`→PessimisticInverseSolver) + the
tol-curve; policy label written into JSON meta and printed at start. The BO comparator is
PROVABLY untouched: WarmStartedBO takes no such knobs and the bo factory closure does not even
capture policy_knobs (checked via co_freevars). ruff clean.

Ran `--policy binding` (50 seeds × 4 targets + tol-curve, InSilicoMachine + metrology noise,
~138 min): **the M2 cost-to-target win SURVIVES the binding §8 policy, attenuated ~35%.**
Pooled ΔRMST −25,530 (ablation@40, published) → **−16,480** (binding@50), CI [−18.2k, −14.7k],
p = 8.7e-70, P(rig better) = 1.00; win-rate 93%→82%; rig hit-rate 1.00→0.99; rig RMST
15.25k→24.58k (+61% — conservatism priced as cost, not reliability collapse); both-hit median
saving 15k→5k; the win holds at every tol_k ∈ {2,3,4,6,8}. Control: BO RMST 40.78k→41.05k and
hit 0.42 flat across runs despite 40→50 seeds → the RIG-side movement is the POLICY, not the
seed count. Feasibility: both policies abstain 100% at 6σ (§8 never binds there, per IF-1);
binding abstains ~9× harder (mean distance_to_feasible 1.60σ→14.84σ). Determinism: a
3-seed×4-target binding subset reproduces the full run's seeds 0–2 byte-identically (24/24).

Files: examples/run_m2_sweep.py (flag), docs/M2-result-2026-07-16.md (dated binding section
APPENDED; historical numbers labeled ablation-policy), docs/m2-result-binding.json (new).
docs/m2-result.json deliberately NOT refreshed (optional ablation@50 refresh remains an
optional owed item; BO-invariance already isolates the policy effect). The audit's question is
answered: **the M2 cost win is not an artifact of the permissive ablation policy.** Nothing
committed.

## 2026-07-22 — campaign.py multi-fire run_id collision FIXED [orchestrator]

The caveat agent I surfaced (repeated `.run()` on one `ConfirmationCampaign` restarts
candidate/run indices at 0 → colliding RunRecord ids across CampaignResults) is fixed: a
per-instance `_invocation` counter now salts `_deterministic_run_id` (uuid5 over
`seed:invocation:candidate_index:run_index`) and strides the synthetic clock index
(invocation×1e6 + run_index); every `run()` call — including Infeasible/empty early returns —
consumes the next slot, so call N's ids depend only on call COUNT, never earlier calls'
content. Invocation 0 is byte-identical to the old derivation (existing determinism tests
pass unchanged); a replayed instance reproduces the same sequence call-for-call. New
regression test `test_repeated_run_on_one_instance_never_collides_but_replays_exactly`
(disjoint ids+timestamps across calls, exact two-call replay, Infeasible-consumes-a-slot);
RED-PROOFED by hard-pinning invocation to 0 in the hash (test fails), restored. loop.py's
documented caveat replaced with the multi-fire-safety note. Bar: test_campaign +
test_active_loop + test_qualification 81 passed; ruff clean. Nothing committed.

## 2026-07-23 — E1 frame validation built against the real Empa contract [subagent J, sonnet]

Built the E1 DataFrame-validation item deferred 2026-07-17 (pandera dropped), now against the
ACTUAL M0/Empa ingest contract. New `src/rig_adapters/tabular/validation.py` — `Frame`/
`frame_from_csv`, typed `Violation`/`ValidationReport` (machine-readable), `validate_frame(
frame, spec, *, order_key=, strict=)`. Six checks: `missing_columns`/`dtype`/`nan_inf`
BLOCKING (frame-wide versions of what ingest already enforces per-row), `bounds`/`order_key`/
`duplicate_rows` ADVISORY (reports what skip-mode already tolerates or was never checked). NO
pandas/pandera dependency added — "frame" = header + row dicts from csv.DictReader; a
heavyweight dataframe lib would repeat the exact mistake pandera was dropped for.

SI trap respected: the bounds check reads the RAW un-ingested cell (declared unit) against
`spec.continuous`/`.categorical`/`.compositional` — `continuous_si` never referenced —
reproducing ingest's SI-round-trip comparison bit-for-bit on the real Ti-120W 3e-11 rounding
edge. Empa agreement PINNED row-for-row (not count-for-count): the report's flagged ti_120w
rows (441/225/260/215/357 across 4 columns) == `ingest_csv(on_error="skip")`'s actual 5
rejects; degenerate `BatchNr` flagged `UNVERIFIED-ORDER` (all 495 rows), with a non-degenerate
al_120w control proving the check doesn't fire unconditionally.

Wiring, non-breaking: `ingest_csv` runs validation as a pre-pass and attaches
`IngestResult.frame_report` (additive field; opt-in `order_key`/`strict` kwargs; strict raises
ONLY on blocking violations — a bounds-only real-data quirk does not trip it, proven by test).
`prepare_empa.py` prints the per-campaign report summary; all 6 output CSVs verified
byte-identical before/after. Tests: 34 new (`tests/test_frame_validation.py`) + 9 empa-ingest
green; 72 pre-existing tabular tests green (refactor changed nothing observable). RED-PROOFED
twice (bounds block and dtype block disabled in turn — dependent tests incl. the real-data
agreement test went red; restored, 43/43 green). ruff clean. Nothing committed.

## 2026-07-23 — Conditional / per-region conformal coverage on the recorded Empa M1 results [subagent M, opus]

The owed §20 group-conditional study, disjoint ownership: NEW run_conditional_coverage.py +
results/m1_empa_conditional.json + tests/test_conditional_coverage.py; recorded
results/m1_empa.json NOT touched (read-only re-analysis). Four PRE-STATED groups fixed in the
docstring before computing any coverage (anti-p-hacking): density near/mid/far (k=5-NN
distance to TRAINING recipes, train-standardized, no test-point leakage), outcome-magnitude
low/mid/high per output, temporal stream early/mid/late, Mondrian per-output. Indicators
reproduced by IMPORTING the runner's own ingest/split/fit_and_eval + the calibration
controllers (no forked logic; online indicator = controller.observe() return). FIDELITY GATE:
reproduced pooled AND per-output k_covered byte-equal to m1_empa.json on ALL 12 campaign/split
cells (ACI n_infinite_width verified); double-run byte-identical.

FINDING: **pooled PASS hides regional under-coverage at the HIGH-outcome tail** — the
high-magnitude tertile under-covers 8/24 campaign/split/output cells (5 of 6 campaigns, both
outputs, both splits; 6 of the 8 hidden behind a PASSING marginal), the low tertile
over-covers (1.000) in 3 — the marginal 0.90 is bought by over-covering the low end and
under-covering the high tail. Aggregate under-covering tertile-cells of 180: static 14 →
ACI 9 / PID 9. The online endpoints repair ti_200w's DRIFT-conditional regional failures
(far-density, mid-stream, magnitude tails all back to nominal) but 7 of 8 high-tail failures
are NOT repaired — ACI/PID adapt over stream/time and do not condition on magnitude. Named
owed remedy: a group-conditional (Mondrian-by-magnitude) calibrator. Honest negatives
recorded: far-from-data (1 cell) and late-drift-phase (0 cells) are NOT the dominant hidden
modes; ti_200w's temporal under-coverage concentrates mid/early stream, not late; per-tertile
CIs are wide (n≈27-43) — the finding is the repeated direction, not one cell. Tests: 13 pass
(determinism, no-leakage density reference, binom_ci reuse `is` runner's, underpowered
flagging, one-campaign fidelity pin, red-proofed fidelity gate across all three paths). ruff
clean. RESULTS.md section added by the orchestrator. Nothing committed.

## 2026-07-23 — In-silico multi-tool M4 DRESS REHEARSAL built + full run [subagent K, opus]

New examples/run_multitool_rehearsal.py + tests/test_multitool_rehearsal.py +
docs/multitool-rehearsal-2026-07-23.md + docs/multitool-rehearsal.json (provenance=physics_sim,
labeled REHEARSAL throughout — machinery proof, never headline evidence). 3-tool fleet on
InSilicoMachine: ±3% hidden emissivity/cosine/flux via the sim's tool_perturbation (standing
for §10.2 run-invisible chamber differences) + ±5-8% build geometry via machine_config, offsets
fixed before outcomes were measured; outputs {thickness_grown, T_center}, tool signal
19.5×/6.3× noise. Four phases, all green, full run 14.7 s:
- (P2) pooled ICM + §5.8 LOTO zero-shot epistemic domination 3/3 folds; few-shot pooled vs
  from-scratch at equal n → **pooling HELPS on this fleet** (K=10: 3.78e-9 vs 4.55e-9; K=20:
  2.39e-9 vs 3.28e-9 thickness-RMSE; smoke config agrees, not a config artifact).
- (P3) EPIG-driven new-chamber onboarding: **EPIG = 2.97 nats > 0 on the unknown-tool path** —
  the 2026-07-17 EPIG-collapse fix holds live on its historically fragile seam
  (posterior_cov unknown-tool branch feeding epig()). Runs-to-loose-threshold TIE at 8 (honest:
  an easy smooth 2-D problem — a good Sobol seed clears it), but the warm start converges ~4.5×
  sharper (7.7e-10 vs 3.5e-9 m, below the single-tool ceiling) by borrowing fleet data.
- (P4) **runner-level qualification auto-invocation crumb CLOSED (in-silico form)**: direct
  `ConfirmationCampaign.run(solver.solve(spec))` — reachable spec → 2/2 certified (58 runs,
  CP bound 0.9019 ≥ 0.90, headline_eligible=False); unreachable spec → Infeasible →
  NothingToQualify with 0 machine calls (asserted); AND the ActiveLearningLoop
  `qualification=` hook fires on a solve-driven in-loop hit charging 10 confirmation runs to
  budget (n_queries=22).
No src bug found. Honest observation recorded: onboarded-tool candidates carry
calibration_status="model-feasible" — the multi-tool view is not conformal-wrapped, so the
§13.2 default gate is correctly inert there; natural M4 enhancement (owed, optional): wrap the
onboarded tool view in ConformalForwardModel (per-tool calibration split) to upgrade to
"conformal-checked". Tests: 6 (4 sim-gated end-to-end/determinism/CLI/budget-charge + 2
ungated numpy guards: EPIG>0 unknown-tool, EPIG(x;{x})==BALD(x)). Smoke AND full double-run
byte-identical minus timings; ruff clean; full-suite collection 702 tests, no errors. Nothing
committed.

## 2026-07-23 — PID decaying-step side study (labeled; NOT the path of record) [subagent N, sonnet]

New examples/real_data/empa_hipims/run_pid_step_study.py + results/m1_empa_pid_step.json +
tests/test_pid_step_study.py. Reuses run_m1_empa.py's ingest/split/fit/pid_eval BY IMPORT
(never edited); fixed-step reproduction verified against results/m1_empa.json EXACTLY on all
12 campaign/split cells (two independent cross-checks), n_infinite_width=0 on all 24
output-rows × 2 modes, double full-run byte-identical. Hypotheses PRE-REGISTERED in the
docstring before any run.

Finding: decaying-step (eta_t = eta·t^-0.6, Angelopoulos-Barber-Bates 2024) cuts late-stream
threshold volatility to 12-36% (median ~22%) of fixed-step's on all 24 output-rows, uniformly
— the predicted win, confirmed. BUT ti_200w_high_pw flips PASS→FAIL on BOTH splits: temporal
0.875→0.842 (drift UNDER-correction, exactly as pre-registered) and, unexpectedly, random
0.929→0.942 (over-coverage — the CI shifts entirely above 0.90). Both flips are one mechanism:
at these ~100-row Empa streams eta_t has already decayed to ~17% (t=20) / ~6% (t=120) of the
fixed rate, so the controller spends its whole budget early and freezes with too little runway
to converge in EITHER direction. Rolling-detector fire status unchanged everywhere (0/24
flips; it already fires under the recorded fixed path on 21/24 output-rows — a pre-existing
window-statistic sensitivity, not a decaying effect).

RECOMMENDATION: step="decaying" stays OPT-IN, not default; actively discouraged on drifting
tools / short online streams (n≈100) where the asymptotic Robbins-Monro guarantee has no room
to bite. Fixed-step remains the path of record (recorded artifacts untouched). Tests: 8 new —
real-data fidelity gate, volatility-metric unit tests incl. edge cases, synthetic
decaying-vs-fixed direction, determinism, red-proofed fidelity gate (single k_covered
perturbation flagged, restored). ruff clean. Nothing committed.

## 2026-07-23 — Optional ablation@50 M2 refresh DONE (stale pre-IF-1 JSON replaced) [orchestrator]

Ran the default-policy sweep (`python examples/run_m2_sweep.py`, 50 seeds, detached, ~78 min
main block + curve) → docs/m2-result.json REFRESHED with the post-IF-1 schema (verdict keys
present; policy="ablation" label; 50 seeds). Verified: pooled ΔRMST CI95 [−27,376, −24,126]
(point ≈ −25,750) vs the published ablation@40 −25,530 — the published numbers REPRODUCE
post-crash-fix to <1%; both-hit n=84, ΔRMST −15,000, median saving 15,000 (published: −14,030 /
15k); tol-curve wins at every tol_k 2–8 (hit rig=1.00 throughout; win 0.88–0.94). This also
completes the airtight apples-to-apples at EQUAL seed count: ablation@50 ≈ −25.75k vs
binding@50 −16.48k → the binding-policy attenuation is ~36% at matched n_seeds, confirming the
BO-invariance-based conclusion of the 2026-07-22 binding entry. The "stale pre-IF-1
docs/m2-result.json" caveat is CLOSED. Nothing committed.

## 2026-07-23 — False-success-rate vs dimension study (dimensionality owed item #1) [subagent L, opus]

Built examples/run_false_success_study.py + tests/test_false_success_study.py +
docs/false-success-study.json + docs/false-success-study-2026-07-23.md. Grid: d∈{2,8,15,20} at
12·d runs + the d=20/n800 CRIME SCENE × 2 arms (raw unwrapped GP = gate inert vs
conformal-wrapped = §13.2 default-on, n_cal=n/3 carved from the SAME budget — the honest cost)
× 20 seeds, binding 2.0/2.0/0.02, 48 restarts; grid uses analytic_grad (timed: FD 222 s vs
analytic 16 s per d=20 solve, same verdict), crime scene runs the exact original FD path.
92.9 min total, deterministic (smoke byte-identical + pinned by test).

**Crime-scene verdict: the original miss reproduces byte-exactly on the raw arm (1 false
success, excursion 0.2461 = 1.046 vs the ±0.8 box) and the default-on §13.2 gate KILLS it**
(rejects both band-spilling candidates, returns 1 conformal-checked survivor that genuinely
hits, 0 FS). New fragility: the same cell under the analytic optimizer path gives 3/3 hits —
the d=20 raw-σ margin is fragile to GP fit AND search path, which is exactly why the
calibrated gate must be the acceptance test at high d.

**Powered-grid honesty:** false successes are too rare at 12·d density to separate the arms —
raw 0/122 certified candidates (FSR ≤ 3.0% at 95% CP), wrapped 1/84 (1.2% [0.03, 6.4]). The
solver's own epistemic abstention (5-85%) dominates; beyond d=2 the gate adds only ~0-20 pp
abstention and costs no genuine hits at d≥15. d=2 wrapped is the degenerate small-budget case
(n_cal=8 → infinite conformal band → 100% abstention; flagged, excluded). **And the gate is
NOT a certificate: the wrapped arm produced its OWN d=8 selected-point miss** — split
conformal is marginal, not conditional, coverage, and the solver hands it SELECTED points; at
fixed budget wrapping is not strictly safer (cal carve-out weakens the surrogate).
Conditional/Mondrian conformal (or a selection-inflation term) is the real fix — now motivated
independently by this study AND the Empa high-outcome-tail conditional-coverage study.

Tests: 11 (scorer incl. certified-but-missing, inclusive boundary, reason bucketing,
Clopper-Pearson 0-count guard, smoke determinism); red-proof: inverting the scorer polarity
turned 4 red incl. the certified-miss test, restored, and a pinning test keeps the polarity
honest. ruff clean. Nothing committed.

## 2026-07-23 — Multi-tool rehearsal Phase 4b: onboarded-tool conformal wrap [subagent R, sonnet]

Extended examples/run_multitool_rehearsal.py with Phase 4b: conformal-wraps the onboarded tool
(ConformalForwardModel + SplitConformalCalibrator, α=0.1) on a held-out calibration split
carved from tool C's own onboarding runs (chronological trailing 1/3: fit 16 / cal 8 at full
config), re-solves the same reachable spec, and demonstrates the calibration_status upgrade
model-feasible → conformal-checked. BOTH branches ran naturally, no config-shrinking: the
natural split (n_cal=8) is honestly ONE run short of the minimum (9) for a finite α=0.1
quantile → infinite band → **the §13.2 gate rejects the raw-margin-admitted candidate
(Infeasible → NothingToQualify, 0 machine calls) — fail-closed working as designed and
reported plainly**; a labeled extra-collection variant charges exactly 1 more run (plain
Sobol-seeded, NOT EPIG-selected, preserving calibration exchangeability) to reach n_cal=9 →
finite band kappa=[1.462, 2.000], a conformal-checked FEASIBLE candidate (0/1 gate rejections
— agrees with raw margins there; calibrated band 1.98e-8 m / 7.92 K, tighter than the raw
worst-case 2.24e-7 m / 20.83 K), and a certified 29-run confirmation campaign. Zero src edits
— ConformalForwardModel composed around ToolBoundForwardModel as-is and the solver's default
gate consumed it unchanged. Phases 1-4 verified byte-identical to the recorded artifact
(programmatic sorted-JSON diff, before AND after ruff format, across two full runs); full run
37.8 s. Tests 6→10 green (min-n_cal boundary pin, synthetic infinite-band unit, sim-gated
upgrade + wiring); upgrade assertion red-proofed (skipping the wrap fails hard); ruff clean.
Nothing committed.

## 2026-07-23 — Mondrian / group-conditional conformal calibrator (top code priority) [subagent O, opus]

Added src/rig/calibration/mondrian.py: `MondrianConformalCalibrator` (per-group split conformal
on the standardized-residual score, exact ceil((1-α)(n_g+1)) rule per group incl. the honest
+inf branch; `min_group_n` POOLED fallback — never silent-shrink, never useless-infinite —
default = the finite-quantile floor, 9 at α=0.1), `predicted_magnitude_group_fn`, and
`MondrianConformalForwardModel` satisfying the SAME interface as ConformalForwardModel →
**the solver's default §13.2 gate consumes it with ZERO solver edits (proven by test)**.
Design constraint honored: grouping keys on the PREDICTED mean at fit AND predict (no true y
exists at predict time; consequence documented and MEASURED — assignment inherits model
error).

Empa study (run_mondrian_coverage.py + results/m1_empa_mondrian.json; static fidelity
byte-equal to m1_empa.json on all 12 cells; double-run byte-identical): **6 of the 8
high-observed-tertile cells that static conformal under-covered move to nominal**; the 2 that
stay under are exactly the two lowest predicted/observed assignment-agreement cells (0.66,
0.48) — when the GP mean can't predict which points are high-magnitude, predicted-grouping
cannot isolate the observed tail. Honest mechanistic limit, not a bug. Cost: high-tertile MPIW
1.0-6.0× (low tertile NARROWS — it was over-covering; the intended redistribution). Honest
negative: Mondrian broke ti_120w's marginal pooled PASS on both splits by OVER-covering
(0.939→0.959, 0.913→0.944 — the safe direction, a width cost not a safety cost).

Selected-point mechanism test (the false-success-study motivation): a solver-selected point in
a high-predicted-magnitude region with 16× wider residuals — the POOLED gate admits the tight
box (the d=8-style marginal miss slipping through), the MONDRIAN gate returns Infeasible with
a conformal-cause reason and nonzero spill. Red-proofs: forcing internal pooled-fallback flips
both the selected-point and +inf tests red; restored. Bar: test_mondrian + test_conformal +
test_conformal_feasibility = 25 passed; ruff clean. Limits recorded: coverage conditional on
the DECLARED predicted-magnitude group only — not per-point, not per-true-magnitude;
underpowered groups borrow pooled (safe, un-conditioned). Next candidates: investigate the two
low-agreement tail cells (predicted magnitude is a weak proxy where the GP mean is
tail-biased); magnitude-group support in the amortized path. Nothing committed.

## 2026-07-23 — WP-E BoTorch comparator slate: SCBO + TuRBO [subagent P, opus]

Added `SCBOBaseline` + `TuRBOBaseline` in NEW src/rig/baselines/trust_region_bo.py
(lazy-loaded via baselines/__init__; import-linter 1 kept / 0 broken) — faithful to the
canonical BoTorch TuRBO-1 (arXiv:1910.01739: TurboState machine verbatim, lengthscale-shaped
Sobol candidates + Thompson selection, restart-on-collapse) and SCBO (arXiv:2002.08526:
spec box as 2m outcome constraints via ModelListGP + ConstrainedMaxPosteriorSampling,
feasible-first). Declared simplifications: TuRBO-1 not -m; restart keeps history (helps the
comparator); SCBO objective = shared box-distance scalarization. Both inherit BoTorchBO's
fairness contract exactly (bit-identical warm start, budget/cost/hit-rule/domain/GP tier).
NEW examples/run_m2_botorch_slate.py reuses run_m2_sweep machinery by import, CRN pairing,
imported rmst_difference_test + paired bootstrap.

RESULT (50 seeds × 2 joint targets, tol=6σ, ablation policy, in-silico MBE): **the M2
"~2× cheaper than BO" claim HOLDS against the full slate** — RIG RMST 16,400 / hit 1.00 vs
BoTorchBO 27,050 / 0.97 (ΔRMST −10,650, p=5.7e-30, **1.65×**), TuRBO 40,350 / 0.68 (−23,950,
p=7e-118, 2.46×), SCBO 47,401 / 0.13 (−31,001, 2.89×); no comparator wins pooled or
per-target. Honest reads: "~2×" is arm-dependent — 1.65× against the strongest arm; SCBO is
the weakest arm because the ~4e-7-scale bow constraint defeats constrained-Thompson within 40
queries — a genuine SCBO-vs-problem finding, NOT a broken comparator (proven by the
discriminating bowl sanity: TuRBO 8/8, SCBO 8/8, pure random 0/8, hits from the optimization
loop not the seed DoE); hit = single noisy in-spec observation for ALL arms equally.

Tests: tests/test_botorch_slate.py 18 passed (warm-start bit-identity, determinism
byte-identical, budget exactness, known-answer sanity, compositional rejection, sweep
drop-in); red-proof: a hidden extra machine call inside _query turned the budget tests red for
both arms, removed. ruff clean. Artifacts: docs/m2-botorch-slate.json + doc. Still in-silico
(real headline gated on M0); optional follow-on: binding-policy re-run of the slate. Nothing
committed.

## 2026-07-23 — Analytic-gradient vs FD parity study BUILT + RUN [subagent Q, sonnet]

New examples/run_gradient_parity_study.py + tests/test_gradient_parity.py +
docs/gradient-parity.json + docs/gradient-parity-2026-07-23.md. Grid: d∈{2,4,8,15,20} × 15
seeds × 2 target classes (reachable = the false-success study's ±0.8 box, imported; hard =
±0.3 boundary box) × FD-vs-analytic, 150 pairs, SAME shared reduced budget both arms
(n_restarts=16, max_iter=40), binding policy; truth family reused by import. 42.7 min.

RESULTS: overall verdict agreement **146/150 = 97.3%** (CI [93.3%, 99.3%]; 100% at d=2/4/15,
86.7% at d=8/20). **All 4 disagreements are the same shape: FD → INFEASIBLE, analytic →
FEASIBLE, and analytic's certified recipe genuinely hits ground truth every time** — FD false
abstentions, zero false successes from either arm, zero cases favoring keep-FD. Among
agreeing-FEASIBLE pairs ground-truth agreement is 100% in every cell (recipes/margins differ,
top picks always hit together). Speedup monotone: 1.90× (d=2) → 5.91× (d=8) → 11.43× (d=20).

RECOMMENDATION (evidence gathered, DECISION NOT MADE): flip `analytic_grad=True` as the
default for d≳8; keep FD below (win <3.2× there, and every currently-published number rides
the FD default). Not unconditional: n=150 cannot rule out a rare false success (0 observed =
upper bound, not zero), and the hard/boundary class showed little live disagreement surface —
the marginal-boundary scenario nearest the original d=20/800 false success is not directly
probed. Migration checklist enumerated: dimensionality doc, M2 (doc + binding JSON + sweep via
loop.py), M3 v1/v2, test_active_loop.py, test_inverse.py's FD-pin test (needs restructuring),
mfl_bakeoff, Empa/sputtering real-data demos, multitool rehearsal — all ride today's FD
default and would need re-verification on a flip. (The false-success study's main grid is
already analytic by its own CLI default; its crime-scene arm hardcodes FD regardless.)

Tests: 13/13 green (margin-formula units, all 4 disagreement shapes, gt_split case, aggregate
tallying, the analytic+pgd construction-time raise, smoke determinism byte-identical);
red-proof live: ground-truth-blind scorer → 3 tests red → restored. ruff clean. No src edits.
Nothing committed.

## 2026-07-24 — Explainer HTML brought current with the full build; page de-timestamped (user directive)

User: "update but note that i dont want time stamped updates on the html, it should serve as a
comprehensive explainer. include any detail it might have missed too." Verified first that
`docs/rig-explained.html` reflected only the 2026-07-22 second-wave state (footer self-dated
07-22; zero mentions of Mondrian/TuRBO/SCBO/conditional-coverage/rehearsal/false-success/E1;
the M2 bullet still called the binding-policy re-run "the current work item" though it was DONE
07-22 evening). Published artifact confirmed byte-identical to git HEAD modulo the injected
mermaid runtime (WebFetch + diff), so the edits applied cleanly on top.

Edits (all woven in as standing explanation, NO dates anywhere on the page):
- §04: new "When the average hides the failure" subsection — marginal-vs-conditional coverage,
  the Mondrian group-conditional calibrator, predicted-value grouping limit, pooled fallback,
  interface parity with the §13.2 gate.
- §05: opt-in support-gate hardenings (flow typicality, PGD drift-box probe; default-off,
  byte-identical off).
- §08: qualification hook — both loop success exits gateable by a ConfirmationCampaign, budget-
  charged, Infeasible ⇒ zero machine calls.
- §10 real data: conditional-coverage finding (high-magnitude-tail under-coverage hidden behind
  pooled PASS; online endpoints repair drift- not magnitude-conditional) + Mondrian result
  (6/8 hidden cells → nominal; 2 unfixable = low prediction-agreement cells; width cost stated).
- §10 in-silico: M2 bullet rewritten to the full comparator slate (1.65× vs BoTorchBO strongest
  arm, ~2× TuRBO, further vs SCBO; win survives binding policy attenuated ~⅓); new bullet for
  the false-success study (crime-scene kill by the default-on gate; marginal-not-conditional
  caveat; d=8 wrapped-arm miss; analytic-gradient 97% parity / 11× at d=20, stays opt-in); M3
  bullet updated to the v2 acceptance run (6/6 at 0.19× budget, restart-budget caveat claimed
  narrowly); new fleet dress-rehearsal bullet (pooling wins, 4.5× sharper onboarding,
  auto-qualification incl. 0-call Infeasible, fail-closed conformal wrap of the onboarded tool).
- §11: stale "toy problem" concession about the amortized tier updated to the v2 state.
- §12: D7 independent-physics ROM cross-check of the sim wrapper; new ingest-validation
  paragraph (E1 — blocking vs advisory checks, declared-unit cells, Empa quirks pinned).
- §13: glossary rows for marginal-vs-conditional coverage and Mondrian conformal.
- Footer: "as of 2026-07-22" removed entirely per the no-timestamps directive (the only date
  left on the page is the MFL paper's citation date, which dates the paper, not the program).

Republished to the SAME artifact URL (a1ab0c0e-…), label comprehensive-current-build. Files
touched: docs/rig-explained.html only. Nothing committed.
