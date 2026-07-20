# Onboarding a new process from flat files (E5 seed)

RIG is not MBE-specific. Any recipe→outcome process — etch, CVD, litho,
anything tabular — plugs in through the generic `tabular` adapter: you write a
small declarative spec, ingest your CSV, and get validated, SI-canonical
`RunRecord`s that the whole forward/conformal stack consumes. No Python is
needed to describe the process.

## 1. Write a process spec (TOML)

Copy `examples/pecvd_example.toml` (fully annotated) or start from
`examples/tabular_minimal.toml`. The format:

```toml
process_id = "my_etch"                 # required, non-empty
description = "optional free text"

# --- one [inputs.<name>] block per recipe variable -------------------------
[inputs.temperature]
kind = "continuous"                    # continuous | categorical | compositional
unit = "degC"                          # any pint-parseable unit; CSV values are
lower = 20.0                           #   read in THIS unit, canonicalized to SI
upper = 120.0                          #   at ingest (degC->K, torr->Pa, sccm->m^3/s)
change_cost = "easy"                   # "easy" (default) | "hard" (split-plot
                                       #   whole-plot factor, implementation-plan §3.1/§8.3)

[inputs.cf4_flow]
kind = "continuous"                    # independent MFC flows are CONTINUOUS
unit = "sccm"                          #   variables — see the warning below
lower = 5.0
upper = 100.0

[inputs.chuck]
kind = "categorical"
levels = ["low", "high"]

[inputs.blend]                         # a GENUINE sum-to-1 composition only
kind = "compositional"
components = ["a", "b", "c"]           # CSV columns: blend.a, blend.b, blend.c

# --- one [outputs.<name>] block per measured KPI ----------------------------
[outputs.etch_rate]
unit = "nm/min"
# modality defaults to "scalar_vector" (the only supported one in v0)
target = 100.0                         # optional spec semantics, declared unit
lower_spec = 80.0
upper_spec = 120.0

[cost]                                 # optional; defaults $1000/$1000/4
c_batch = 1000.0
c_recipe = 1000.0
batch_size = 4
```

**Mixture vs independent flows (implementation-plan §3.1, enforced):** a `compositional`
variable is a *true* simplex — fractions that sum to 1 by definition (alloy
mole fractions, a blend defined as fractions). Independent MFC gas setpoints
in sccm are **not** a simplex (their total is not fixed); declare each flow as
its own `continuous` variable. A compositional block declaring a flow unit
like `sccm` is rejected at load with an error explaining exactly this.

JSON is accepted as a secondary format (same structure, `.json` suffix) for
machine-generated specs.

## 2. Lint the spec

Loading *is* the lint — validation is strict and every error names the
offending key:

```powershell
python -c "from rig_adapters.tabular.spec import load_spec; s = load_spec('my_etch.toml'); print(s.process_id, s.flat_input_names, s.output_names)"
```

## 3. Ingest your CSV

CSV columns must match the spec's input names (compositional components as
`<variable>.<component>`) and output names. Values are read in the
spec-declared units and canonicalized to SI.

```powershell
python -m rig_adapters.tabular.ingest --spec my_etch.toml --csv myruns.csv --out runs.jsonl
```

Useful flags:

- `--tool-column <col>` — CSV column holding the chamber/tool id (enables
  leave-tool-out splits). Default: every row gets `tool_id="unknown"`.
- `--timestamp-column <col>` — ISO-8601 timestamps. **Without it, timestamps
  are synthesized** (deterministic 1-hour ladder) and every record is flagged
  `extra["synthetic_timestamp"] = true` — temporal splits/drift monitoring
  over synthetic order are meaningless (implementation-plan §12.4).
- `--source real_tool|physics_sim` — provenance tag (default `real_tool`;
  headline metrics are only ever computed on `real_tool` rows, implementation-plan §3.5).
- `--on-error raise|skip` — bad rows (out-of-bounds, unknown categorical
  level, composition not summing to 1 within 1e-6) abort by default; `skip`
  drops them and prints a rejects report.

Missing required columns are always a hard error listing them; extra columns
are kept in `RunRecord.extra["unmatched_columns"]` with a warning.

## 4. Fit a calibrated forward model on the records

```python
from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.forward import GPForwardModel, records_to_arrays
from rig_adapters.tabular.ingest import ingest_jsonl
from rig_adapters.tabular.spec import load_spec

spec = load_spec("my_etch.toml")
records = ingest_jsonl("runs.jsonl", spec).records
input_keys = list(spec.gp_input_keys)                 # numeric inputs (categoricals
                                                      #   are conditioning, §8.3) with ONE
                                                      #   component dropped per compositional
                                                      #   variable — sum-to-1 is exactly
                                                      #   collinear, so keeping all components
                                                      #   makes the GP design rank-deficient
X, Y = records_to_arrays(records, input_keys, list(spec.output_names))
n_fit = int(0.75 * len(X))                            # held-out calibration split (§5.3)
model = GPForwardModel(input_keys=input_keys, output_keys=list(spec.output_names)).fit(
    X[:n_fit], Y[:n_fit]
)
calibrator = SplitConformalCalibrator(alpha=0.1)
calibrator.fit(model, X[n_fit:], Y[n_fit:])
dist = ConformalForwardModel(model, calibrator).predict(X[0])  # PredictiveDistribution
```

`dist` is the canonical `PredictiveDistribution(mean, aleatoric_sigma,
epistemic_sigma, conformal_set)`; `model.support_score(x)` gates OOD queries.

(`spec.gp_input_keys` drops exactly one component per compositional variable —
the reference component — because a factor's sum-to-1 makes its full component
set exactly collinear (a rank-deficient GP design). Prefer it over
`numeric_input_names[:-1]`, which only works when a single blend is declared
last.)

The adapter itself is available through the registry for anything that needs
schema/cost/DoE hooks (e.g. Sobol seed designs for a first campaign):

```python
from rig import registry
adapter = registry.get_adapter("tabular", spec_path="my_etch.toml")
seeds = adapter.seed_design(16, seed=0)   # feasible by construction (simplex renormalized)
```

(`RIG_TABULAR_SPEC=<path>` is the env-var alternative to the `spec_path`
kwarg; a bare `get_adapter("tabular")` with neither raises an actionable
error.)

## Multiple tools / switching machines (implementation-plan §10.4 level (a))

Switching chambers/machines of the SAME process is an expected pattern: you
should not retrain from scratch per tool. The tool-aware surrogate is
`MultiToolGPForwardModel` — an ICM multi-task GP (Bonilla et al. 2008) whose
kernel is `k_Matern52(x,x') · B[s,t]` with a learned PSD tool-covariance
`B = W Wᵀ + diag(v)`: all tools share one response surface and each tool only
has to learn its own row of B, so a handful of runs on a new machine is
enough to specialize.

**1. Ingest with tool identity.** Pass `--tool-column <col>` so every
`RunRecord` carries its real `tool_id` (without it every row is
`"unknown"` and there is nothing to pool over).

**2. Fit the tool-aware model** (drop-in next to the snippet above):

```python
from rig.forward import MultiToolGPForwardModel, records_to_arrays_with_tools

X, Y, tools = records_to_arrays_with_tools(records, input_keys, list(spec.output_names))
model = MultiToolGPForwardModel(
    input_keys=input_keys, output_keys=list(spec.output_names)
).fit(X, Y, tools)
dist = model.predict(x, tool_id="chamber-1")     # tool-conditioned prediction
```

**3. Onboard a new machine with a handful of runs:**

```python
model.adapt_to_tool("chamber-2", X_new, Y_new)   # few-shot refit (logs run count)
dist = model.predict(x, tool_id="chamber-2")
```

Before any runs exist, `model.predict(x, tool_id="chamber-2")` does NOT
pretend the tool is known: it returns the B-weighted population average with
deliberately inflated epistemic (at least the worst known tool's, plus the
between-tool disagreement, plus an irreducible new-tool variance term — see
the `rig.forward.multitask` module docstring for the exact formula). In-silico
few-shot result (WP-I tests): with 40 runs of tool A and only 4 runs of tool
B, the multi-tool model's held-out-B RMSE beats both a tool-blind pooled GP
and a from-scratch GP on B's 4 runs alone.

**4. Reading the "this tool doesn't transfer" alarms.** Two signals, same
semantics as everywhere else in RIG:

- `model.support_score(x, tool_id=...)` — negative Mahalanobis distance,
  per-tool once the tool has ≥ d+2 runs (global cloud before that). Scores
  far below the tool's own training scores mean you are extrapolating for
  that tool regardless of what the fleet knows.
- ACI rolling coverage — wrap the tool-bound view
  (`ConformalForwardModel(model.for_tool("chamber-2"), calibrator, controller)`;
  the wrapper is tool-blind, so bind the tool with `for_tool()` first) and
  feed realized runs to `observe()`. Sustained `rolling_coverage` below
  nominal on the new tool while other tools hold coverage = the shared model
  does not transfer to this chamber; fall back to a per-tool fit.
- `model.tool_correlation_` is the direct readout: a new tool whose learned
  correlation to the fleet stays low after ~10 runs is genuinely different.

**Honest limitation:** there is no automatic negative-transfer guard yet. If
a "new tool" is actually a different process, the shared kernel can pull its
predictions the wrong way; today your protection is the alarms above plus a
per-tool fallback fit. The plan's RGPE (rank-weighted GP ensemble, robust to
negative transfer) and the torch-era per-tool FiLM/ANP adapters are future
work packages.

## What the tabular adapter does NOT yet cover

- **Modalities:** `scalar_vector` only. `curve_1d`/`field_2d` outputs
  (spectra, wafer maps) need the ArrayRef/DVC pipeline and are rejected at
  spec load with "not yet supported".
- **Physics plug-ins / independent verifiers:** none — `physics_plugin` and
  `independent_verifier` are `None` (honest D7 default). A process with a
  physics prior needs a dedicated adapter (see `rig_adapters.mbe`).
- **Full E5 conformance harness:** this runbook plus spec-load validation is
  the E5 *seed*; the complete adapter conformance harness (declared
  constraints checked against data, transform round-trips, modality/head
  consistency) is still owed.
- **Full E1 ETL:** ingestion assumes you already joined MES/metrology into
  one flat file. Entity resolution across systems, metrology→KPI reduction
  contracts, and rework/scrap/censoring handling are the E1 work package.
- **Nonlinear/linear constraints beyond boxes and the simplex**, monotone
  declarations, and change-over penalties in the cost model.
