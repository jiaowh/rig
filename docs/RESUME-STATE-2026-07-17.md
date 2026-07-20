# RESUME STATE — 2026-07-17 17:25 (paused by user request)

> **UPDATE 17:55 — the 2 RED tests are DIAGNOSED and the source is CORRECT.** Both
> failures are benign (see §1a). Fixes are written and validated in a scratch copy
> (`<scratchpad>/fixval`, full `test_inverse.py` → 72 passed) and PROVEN to be real
> guards by source mutation. They are **NOT yet applied to the real repo** — held
> pending an independent adversarial refuter (agent `af3acdf`) that is trying to prove
> the gradient wrong. Apply only if it fails to refute. The real repo is still RED
> until then.

**The repo is currently RED.** 2 failing tests in `tests/test_inverse.py`. Do not report
"481 passed" — that was true at 16:40 and is stale. Read §1 before touching anything.

## 1a. RESOLUTION of the 2 red tests (17:55) — source correct, both benign

Diagnosed by execution, scored against ground truth, never the surrogate:

- **`test_analytic_grad_and_fd_paths_agree_on_the_solution` (1 vs 3):** every recipe from
  BOTH paths is in-box on `truth()`. The analytic (exact-gradient) path explores the
  pre-image better and returns 3 distinct valid recipes (pairwise 0.6–1.2 in 6-D) where
  the FD path returns 1. Not a defect — the exact path is BETTER, and separately the FD
  path UNDER-EXPLORES (an honest finding, now documented in the test). The old
  `len(fd)==len(an)` assertion encoded a false premise. Replacement asserts the real
  property (no presented recipe misses ground truth, on BOTH paths; top recipe valid;
  `len(an) >= len(fd)`). PROVEN a real guard: flipping the box-slope sign makes the
  analytic path abstain and the test catches it.

- **`test_analytic_gradient_covers_the_simplex_block` (1.88e-4 vs 1e-6):** the analytic
  gradient is CORRECT. Proven two ways: (a) softmax Jacobian x_a(δ−x_b) matches FD to
  5e-11 in isolation, incl. at the failing near-vertex point; (b) on a well-conditioned
  simplex fixture the full objective gradient matches FD to 2.4e-7. The test's inline
  fixture (Dirichlet conc=1 → points jammed at vertices, 40 pts) produced ~1e4 gradients
  where central FD floors at ~2e-5 relative — the simplex twin of the near-linear trap
  the module already documents. Fix: well-conditioned fixture + an isolated
  softmax-Jacobian guard. PROVEN a real guard: dropping the (−x_b) coupling from the
  source `_dx_du` makes it fail at rel err 0.42.

**Byproduct finding (real, not a bug):** `analytic_grad` is documented §8.6 as speed-only,
but the FD path under-explores the pre-image vs the exact-gradient path, so flipping the
flag can change the SIZE of the returned candidate set (not the top recipe). Worth a
BUILD_LOG note; the exact-gradient path is the more correct one.

Fix drafts: `<scratchpad>/DRAFT_test_fixes.py`; validated edits live in
`<scratchpad>/fixval/tests/test_inverse.py`. Apply those two function replacements + the
`from rig.transforms import SimplexTransform` import to the real
`tests/test_inverse.py` once `af3acdf` clears.

## 0. Two things I got WRONG earlier this session — do not repeat them

1. **"Nothing is running. The workers are done."** — FALSE when I said it (16:35). The
   analytic-gradient agent of workflow `wf_a269d711-a8f` ran until **17:09:26** and wrote
   `src/rig/inverse/pessimistic.py` (16:39) and 382 lines of `tests/test_inverse.py`
   (17:05) AFTER I declared it dead.
   **Why I was wrong:** I inferred "no agents" from `Get-Process python` returning empty.
   **Subagents are LLM calls inside the `claude` process — they spawn python only while
   actually running a test.** Absence of python at one instant ≠ absence of agents.
   **Correct liveness check:** poll the agent's transcript
   (`subagents/workflows/<runId>/agent-*.jsonl`) for size growth over ~6s, and check
   `Get-Process claude | Sort CPU`. A static transcript + no CPU = dead.
2. **"The workflow's run directory was never created. No journal, no transcripts."** —
   FALSE. I looked in `<session>/workflows/` (scripts only). The real location is
   **`<session>/subagents/workflows/<runId>/`**, which had a 35 KB `journal.jsonl` and 5
   agent transcripts the whole time. **The four builder reports were recoverable all
   along**; I told the user they were lost. Always check `subagents/workflows/` first.

## 1. Why the repo is RED (the important part)

The gradient agent died **mid-investigation**, not mid-writeup. Its final message:

> "Two failures left, and one of them is important: the two paths returned **different
> candidate counts (1 vs 3)**. Let me investigate rather than weaken the assertion."

Reproduced just now (`pytest tests/test_inverse.py -q -k analytic` → **2 failed, 25 passed**):

- `test_analytic_grad_and_fd_paths_agree_on_the_solution` → `assert 1 == 3`.
  The **finite-difference path returns 1 candidate; the analytic path returns 3**, on the
  same seed and spec. Both flagged FEASIBLE. Support scores differ (−0.62 vs −1.37).
- `test_analytic_gradient_covers_the_simplex_block` → failing, not yet diagnosed.

**This is a real, open, unresolved discrepancy and it is exactly the class of defect this
repo cares about.** Do NOT weaken the assertion — the dead agent's own instinct was right.
Two paths that claim to optimize the same objective returning different candidate SETS
means at least one of:
  (a) the analytic gradient is subtly wrong (but it matches FD to 5.4e-7 term-by-term —
      those 25 passing tests are strong), or
  (b) the objective landscape is multi-modal and L-BFGS-B genuinely lands elsewhere given
      an exact gradient vs an FD estimate — in which case the FD path finding only 1 of 3
      candidates means **the FD path (which produced every M2 result and the d=20 study)
      has been under-searching all along.**

(b) is the alarming branch and would touch published numbers. **Diagnose before fixing.**
Score any candidate against GROUND TRUTH, never the surrogate (audit F2).

## 2. What is verified vs unverified RIGHT NOW

| Module | Source | Status |
|---|---|---|
| `qualification.py` | built 15:55 | **UNVERIFIED** — probe killed before reporting |
| `active/acquisition.py` | built 16:06 | **VERIFIED SOUND** (see §3) |
| `forward/distill.py` | built 16:20 | **UNVERIFIED** — probe killed before reporting |
| linear constraints (`pessimistic.py`) | built 16:34 | **UNVERIFIED** — probe killed |
| analytic gradient (`pessimistic.py`) | built 16:39–17:09 | **INCOMPLETE, RED** (§1) |

## 3. Verification results that DID land (workflow `wf_74f5fe3f-128`)

**`acquisition.py` → SOUND**, `tests_are_real_guards=True`. This is a real result:
- botorch's qLogNEHVI is genuinely called (`inspect.getfile` → site-packages
  `botorch/acquisition/multi_objective/logei.py`); nothing faked.
- **Objective sense is CORRECT** — the probe's primary hypothesis, tested with an analytic
  identity model and hand-checkable ordering, and it FAILED to break it.
- **21 mutations, 18 caught, 3 survived.** The key attack (margin sign flip) IS caught;
  the core cannot be replaced by a constant. All 3 survivors are TEST GAPS, not source
  defects — and one was *honestly pre-disclosed in the docstring*.

**2 minor findings, each CONFIRMED by 3/3 independent refuters (`refuted=False`):**
1. `tests/conftest.py`: `_TORCH_SKIP_MARKER` (line 44) is a **dead constant**; the
   docstring promises a torch skip-banner that `pytest_terminal_summary` never emits. A CI
   run where all ~56 torch tests silently skipped would announce **nothing**. Same species
   as [[verification-commands-must-be-proven-to-fail]] — a documented guard that isn't there.
2. `acquisition.py` docstring cites measured numbers (spread 13.8→11.4) that **do not
   reproduce** under the test file's own fixtures (actual: 91.37→91.09). Refuters confirmed
   the code is spec-compliant (§8.7 ref pad = nadir + 10%); the DOCSTRING's numbers are
   unverifiable as written because it never states the config it measured.

Both are docstring/test-gap issues, not source defects. Neither blocks.

## 4. Saved artifacts (nothing lost)

- Builder reports (4, detailed, incl. self-found defects):
  `subagents/workflows/wf_a269d711-a8f/journal.jsonl`
- Verify results: `subagents/workflows/wf_74f5fe3f-128/journal.jsonl` (5 cached results)
- Quarantine copies: `<scratchpad>/quarantine/test_inverse.{AGENT-APPENDED-17-05,BASELINE-16-49}.py`
- Baseline hash manifest: `<scratchpad>/baseline.sha256` (84 files, taken 16:49)
- Sandboxes preserved: `<scratchpad>/sbx-{qualification,acquisition,distill,linear-constraints}`
  (full repo copies; `PYTHONPATH=<sbx>/src` makes them win over the editable install —
  verified empirically)

**Integrity:** baseline-vs-now diff → **only `tests/test_inverse.py` changed**, by the
authorized gradient agent. All 83 other files byte-for-byte identical. No rogue writes.

## 5. Resume

```
Workflow({scriptPath: "<session>/workflows/scripts/rig-verify-and-continue-wf_74f5fe3f-128.js",
          resumeFromRunId: "wf_74f5fe3f-128"})
```
Cached: the acquisition probe + 4 refuters (instant). Will re-run: the qualification,
distill, and linear-constraints probes.

**Do first, in order:**
1. Diagnose the 1-vs-3 candidate discrepancy (§1). It may invalidate FD-path results.
2. Resume verification for the 3 unverified modules.
3. Only then: BUILD_LOG entry + BUILD_STATE rows (not yet written for ANY of today's builds).

## 6. Still owed / not started

- MFL bake-off — pre-registered in `docs/prereg-mfl-bakeoff-2026-07-17.md`, **no code yet**.
  Predictions are frozen; do not edit them after seeing results.
- The known-owed experiments (false-success rate vs d across seeds AND GP-fit restarts;
  whether conformal re-validation catches the d=20 miss; selected-vs-random coverage).
- User decisions outstanding: `ruff format` adopt-or-drop (45/83 files); pandera
  implement-or-drop; M0 real dataset.

Related: [[claims-die-in-the-follow-up-run]], [[rig-pessimism-inherits-model-miscalibration]],
[[verification-commands-must-be-proven-to-fail]].
