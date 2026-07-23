# RIG inverse-capability audit — 2026-07-21

## Verdict

**Partially capable, not deployment-ready.** RIG has a credible prototype for inverse recipe generation: it fits a probabilistic forward model, returns multiple bounded candidates or an explicit infeasible result, applies constraints/support screening, and has run on real 5-input/2-output HiPIMS data. It cannot yet be claimed to reliably predict production input recipes from desired outputs. Its default feasibility decision is not backed by mandatory calibrated outcome acceptance or independent real-tool qualification.

This is an implementation-and-evidence audit, not a new experimental result. I read the binding plan, build state/log, core inverse/forward/active/qualification code, real-data runner and recorded results. No product code was changed.

## Evidence of capability

- Typed SI-aware ingest, GP and ensemble forward models, conformal wrappers, a constraint-aware inverse solver, and a confirmation-batch gate exist.
- The solver fails closed without a support threshold, enforces box/simplex transforms and declared linear couplings, returns explicit infeasibility, and returns diverse pre-images.
- The Empa run uses 3,150 real-tool rows across six BO-sampled campaigns, with five pulse controls and two measured outputs per campaign. Its online ACI path reports 6/6 passing the project’s directional coverage check.
- The repository records a deterministic synthetic d=20 false success: one of three certified recipes misses the actual target. This is useful negative evidence.

## Findings

### F1 — Critical: default feasibility ignores conformal intervals

`PessimisticInverseSolver._margins()` reads raw `aleatoric_sigma` and `epistemic_sigma`; it does not consume `PredictiveDistribution.conformal_set`. A conformal containment check only occurs through optional `revalidation_model`. Direct solver calls default it to `None`; the GP active-learning path also lacks it unless supplied by the caller. Therefore a result can be called FEASIBLE despite failing the calibrated interval criterion the plan assigns to safety.

`docs/dimensionality-2026-07-17.md` documents a deterministic d=20 false success and identifies this raw-uncertainty mechanism. A FEASIBLE verdict is not currently a calibrated guarantee.

**Remediation:** require conformal full-model revalidation for every emitted FEASIBLE candidate (or label it merely “model-feasible”); measure false-success rate across seeds, dimensions and process splits; add a regression test that the known bad candidate is rejected by the default path.

### F2 — High: independent qualification is not wired into any production path

`ConfirmationBatchGate` is thoughtfully implemented, but outside its module it is only referenced by interfaces/tests. The inverse solver, active loop and Empa runner do not call it. The Empa inverse demo checks nearest existing measurements, not a new independent execution of the proposed recipe. The plan requires independent qualification before production.

**Impact:** a successful solve is a recommendation, not a validated recipe.

**Remediation:** add a campaign orchestration path that runs each candidate on the real tool, logs provenance, blocks promotion on failure, and addresses the gate’s documented multiplicity, serial-correlation and Cpk limitations.

### F3 — High: operational defaults are less conservative than the binding policy

The plan specifies `kappa=2.0` and `z_epi=2.0`. `ActiveLearningLoop` defaults both to `1.0`, and `examples/run_m2_sweep.py` explicitly uses 1.0/1.0. These settings directly change FEASIBLE versus INFEASIBLE, so comparison and closed-loop results cannot be presented as binding-policy results.

**Remediation:** align defaults/comparisons to 2.0/2.0 or version and label a separate ablation policy; rerun cost-to-target and false-success claims under the declared policy.

### F4 — High: real-data evidence is encouraging but insufficient for general recipe inversion

The Empa data demonstrate machinery on a real process, not production inverse success: campaigns are isolated BO-sampled 5-D subspaces, there is one tool, two outputs, no prospective candidate confirmations, and one campaign lacks verified temporal order. Its own OOD check proves support/epistemic screening is blind to a material shift when material is absent from the inputs. ACI restores average coverage but includes some infinite-width intervals and is not a finite-sample conditional guarantee.

**Remediation:** condition on material/tool/state, obtain independently time-ordered multi-tool data with replicate recipes, and prospectively qualify generated recipes. Do not claim cross-material pooling before those controls exist.

### F5 — Medium: the implemented objective is an MVP approximation, not the plan’s full robust objective

The solver uses scalar margins, `z_epi * sigma_epi`, a first-order Jacobian tolerance term and Mahalanobis support. The plan calls for joint spec-hit probability, worst ensemble member, input-box PGD and flow typicality. BUILD_STATE still lists flow typicality and PGD as TODO. The present solver may be a reasonable MVP, but cannot inherit the stronger guarantees for correlated/tight multi-output specifications.

**Remediation:** implement and benchmark those elements, or make the approximation explicit in APIs/results and validate it on correlated multi-output holdouts.

### F6 — Medium: current repository quality gates are red

On 2026-07-21, `python -m ruff check .` reported 14 diagnostics; `python -m ruff format --check .` would reformat four files. The `lint-imports` command was unavailable in this environment. The focused pytest-plus-quality bundle exceeded the 60-second tool limit before completing, so this audit does not claim a fresh full-suite pass; recorded green runs are historical evidence only.

**Remediation:** restore green formatting/lint/import-boundary CI, document the exact import-linter invocation, and publish a current full test result with locked environment information.

### F7 — Medium: source and state documentation conflict on revalidation

Older BUILD_STATE/dimensionality wording says `ActiveLearningLoop` never sets `revalidation_model`. Current source automatically uses the full surrogate when a fast ensemble view differs. The direct and GP paths still need an injected conformal revalidator, but the blanket documentation is stale and obscures the actual risk boundary.

**Remediation:** reconcile source, BUILD_STATE and dimensionality notes; explicitly identify direct/GP paths as still unprotected by default.

## Decision table

| Claim | Audit status |
|---|---|
| Fit output-from-input predictors | Yes, for implemented scalar/tabular cases. |
| Invert target outputs into bounded input candidates | Yes, as surrogate recommendations. |
| Abstain when unsupported | Often, with useful mechanisms; not a proven guarantee. |
| Reliably promise returned inputs achieve outputs | No — F1/F2 block this. |
| Ready for production recipe release | No — independent prospective qualification is absent. |

## Priority path

1. Make conformal full-model revalidation mandatory and quantify false success/abstention.
2. Wire the qualification gate into a real-tool campaign workflow.
3. Restore 2.0/2.0 defaults or transparently establish a different policy.
4. Run a prospective, time-ordered, multi-tool/material evaluation with held-out recipe confirmations.
5. Restore green CI and reconcile stale state documentation.

Until then, the accurate product statement is: **RIG is an uncertainty-aware inverse-design research prototype that can propose and abstain; it has not demonstrated reliable output-to-input prediction for deployment.**
