# RIG — Recipe Inverse Generator

Calibrated, uncertainty-aware learned process simulator + inverse recipe generation for
semiconductor manufacturing. **The authoritative spec is [implementation-plan.md](implementation-plan.md)** (audited
2026-07-15, see [docs/audit-2026-07-15.md](docs/audit-2026-07-15.md)). Design decisions
D1–D9 (§2.2) and the invariants (§2.1) are binding; §20 supersedes on method currency.

## Session bootstrap (READ FIRST, every session)

1. Read `docs/BUILD_STATE.md` — the single source of truth for what exists, what is in
   progress, and what is next. Do not re-derive the build order from scratch.
2. Skim the tail of `docs/BUILD_LOG.md` — append-only journal of what each work
   session/agent actually did.
3. `implementation-plan.md` is large (~780 lines); read only the sections your task cites.

## Non-negotiable conventions

- **Canonical interface names (implementation-plan §3.2–§3.4, verbatim):**
  `ForwardModel.predict(x) -> PredictiveDistribution(mean, aleatoric_sigma,
  epistemic_sigma, conformal_set)`; `ForwardModel.support_score(x)`;
  `ForwardModel.jacobian(x)`; `InverseSolver.solve(spec) -> list[RecipeCandidate]`
  (with feasibility flags / explicit INFEASIBLE); `QualificationGate.certify(recipe)`.
  Never `OutcomeDist`, never bare scalars, never `_var` tuples.
- **Hard boundary (§3, §13):** `src/rig/` (process-agnostic core) must NEVER import
  from `src/rig_adapters/` (per-process adapters). Adapters self-register via the
  `rig.adapters` entry-point group; `rig.registry` discovers them with
  `importlib.metadata` — no static adapter imports anywhere in core. Enforced by
  import-linter (`pyproject.toml` [tool.importlinter]) in CI.
- **Data contract (§3.5):** every run is a `RunRecord` (Pydantic v2 + Pint units,
  SI-canonicalized at ingest). `Provenance.source ∈ {physics_sim, real_tool}`;
  headline metrics only ever on `real_tool`.
  - **SI is the contract, and it is a live trap:** ingest canonicalizes VALUES to
    SI, but a process spec's declared variable BOUNDS stay in the declared unit
    (ingest needs them to read cells). Pair SI data with declared bounds and you
    silently search the wrong space — invisible whenever the declared unit is
    already SI. Use `ProcessSpec.continuous_si` for fitting/inverse/support, never
    `.continuous`. (This was a real defect in the sputtering example, fixed
    2026-07-17; see BUILD_LOG.)
  - **Frame validation is NOT implemented, and `pandera` has been DROPPED** (decision
    2026-07-17). It was declared in pyproject but imported nowhere, implying a
    frame-validation guarantee that did not exist; the dependency was removed rather
    than left as a false signal. DataFrame validation at ingest remains a real **E1**
    item, to be built against the actual data contract when M0 lands — not
    speculatively. Do not cite pandera or "frame validation" as an existing guarantee.
  - **`ruff format` is the formatter of record** (adopted 2026-07-17; the whole tree
    is formatted). Run `python -m ruff format .` before finishing, and keep
    `python -m ruff format --check .` green alongside `ruff check`.
- **Determinism (§13.4):** seeded everything; tests must be reproducible.
- All work follows the plan's build-order DAG (§15.4): Phase-0 adapter/in-silico
  machine → M1 forward+conformal → M2 per-query inverse → M3 amortized generator+EPIG.

## Environment

- Windows 11, PowerShell primary; Python 3.12 on PATH; **uv not installed** (pyproject
  targets it; use `python -m pip install -e .[dev]` until uv lands).
- The MBE physics simulator lives in the SIBLING repo
  `c:\Users\Jiaow\Documents\github\MBE sim` (package `mbe_sim`, not pip-installable —
  the adapter locates it via the `MBE_SIM_PATH` env var, defaulting to that path).
- Run tests: `python -m pytest tests/ -q` from repo root.

## Agent / session protocol (applies to every subagent and future session)

- Before working: read `docs/BUILD_STATE.md`.
- After working: (1) APPEND a dated entry to `docs/BUILD_LOG.md` — what you did, files
  touched, decisions made, gotchas; (2) UPDATE the affected rows of
  `docs/BUILD_STATE.md` (status + "next steps"). Never rewrite history in the log.
- Do not commit unless the user asks.
- Do not log real-recipe values to any cloud service (implementation-plan §17 egress guard) — moot
  until real data lands, but the default stays on-prem.
