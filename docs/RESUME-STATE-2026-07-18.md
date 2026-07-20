# RESUME STATE — 2026-07-18 ~00:15 SGT (SESSION LIMIT HIT; resets 8:10pm Asia/Singapore)

> **UPDATE 2 (2026-07-19 ~01:40, orchestrator mode — delegating to sonnet/opus per user):**
> RESEARCH/M0 DONE — `docs/m0-dataset-candidates-2026-07-18.md` (25 candidates, 4 STRONG; best:
> Empa bipolar HiPIMS, Zenodo 10.5281/zenodo.18495401, real tool, 6-D knobs, n>3000, CC-BY-4.0;
> NO candidate meets the FULL bar) + `docs/mfl-follow-on-research-2026-07-18.md` (MFL has ZERO
> citations; v1-only, unpublished venue-wise). AUDIT finders all DONE (11 findings; key: eval
> lens answered the prereg open question — false_success_rate IS structurally vacuous under
> deterministic scoring; KM/RMST tie convention UNGUARDED by tests); refuters re-running with
> model overrides (correctness→opus, others→sonnet), resume wf_163b8a0a-9d2. BAKE-OFF rebuilt as
> FRESH run wf_fc1bfc4a-b02 (opus build/steelman, sonnet fix/smoke) — old wf_d22a4091-362 dead
> (0 cached; its script file trips a control-char validator bug, don't reuse). A sonnet agent is
> downloading the 4 STRONG datasets into data/m0-candidates/ + MANIFEST.md with claim
> verification. NOTE: session limits keep cutting runs (~2h cadence); everything above survives
> on disk; resume pattern unchanged.
>
> **UPDATE (post-reset): all three workflows RELAUNCHED and confirmed alive** (transcripts
> growing). Audit = run wf_163b8a0a-9d2 (task wfisv3c5r); research/M0 = run wf_677c9377-586
> (task wfvdevk05); bake-off build = run wf_d22a4091-362 (task w52w7ptsw, script now ON DISK:
> `...\workflows\scripts\mfl-bakeoff-build-wf_d22a4091-362.js`). If cut again: resume each via
> `Workflow({scriptPath, resumeFromRunId})`; completed agents replay from cache. Then §2 items
> 4-6 (full bake-off run → score vs frozen predictions → splice into #versus + republish).

Supersedes `RESUME-STATE-2026-07-17.md` (whose items are all DONE — see BUILD_LOG tail).
Read this + `docs/BUILD_STATE.md` header, then execute §2.

## 0. What just happened

The user issued a large tasking (§1). I launched 2 of 3 planned background workflows; **all 14
subagents failed instantly on "You've hit your session limit · resets 8:10pm (Asia/Singapore)"**,
and the 3rd workflow (bake-off build) never launched (model classifier unavailable). Nothing was
built by them; nothing is running. The repo is UNTOUCHED by this tasking so far and remains at the
clean state: **511 passed / ruff clean / ruff-format clean (adopted) / import-linter KEPT /
pandera dropped**. Nothing committed.

## 1. The user's tasking (verbatim intent, 2026-07-18)

1. **Audit the current program, fix any problems found.**
2. **Update `docs/rig-explained.html`** to visually guide through what the system does, focused on
   the comparison with MFL (Gu et al., arXiv:2505.16060, "Few-Shot Test-Time Optimization Without
   Retraining..."). User asked to "surpass their research in every capacity" — the HONESTY STANCE
   (already told to the user, do not regress from it): we do NOT write claims into truth; MFL
   genuinely wins amortization/query-count in their framing (prereg P1/P5 predict RIG losses);
   RIG's defensible wins are margin-awareness, calibrated uncertainty, abstention, constraint
   guarantees, and the fact that NEITHER method has real data — M0 is the real "surpass" lever.
   The HTML claims what the evidence supports, concedes what it doesn't, and cites the bake-off.
3. **Search for research building ON MFL** (citations/follow-ons) → `docs/mfl-follow-on-research-2026-07-18.md`.
4. **Hunt online for M0 datasets** → `docs/m0-dataset-candidates-2026-07-18.md` (do NOT overwrite
   the older `m0-dataset-candidates.md` if present).
5. **Persist state across session limits** (this file).

## 2. EXECUTION PLAN after the limit resets (in order)

1. **Relaunch the audit workflow** — script ON DISK:
   `C:\Users\Jiaow\.claude\projects\c--Users-Jiaow-Documents-github-rig\563c1d8d-0281-4820-9db7-9dd2d6189a23\workflows\scripts\rig-fresh-audit-wf_163b8a0a-9d2.js`
   Relaunch: `Workflow({scriptPath: <that path>, resumeFromRunId: "wf_163b8a0a-9d2"})` (all agents
   errored → nothing cached → full re-run; that is fine). 6 lenses: conformal math, eval metrics
   (incl. the pre-registered OPEN question: is `false_success_rate` structurally vacuous?),
   active-loop wiring (CONFIRM `revalidation_model` never set by `active/loop.py`, then WIRE it),
   schema/ingest SI traps, examples end-to-end, gp-core numerics. Findings → 3-refuter verify →
   orchestrator applies fixes (agents report only; mutation only in private sandbox copies —
   git diff is BLIND, src/ untracked).
2. **Relaunch the research/M0 workflow** — script ON DISK:
   `...\workflows\scripts\rig-research-and-m0-hunt-wf_677c9377-586.js`
   Relaunch: `Workflow({scriptPath: <that path>, resumeFromRunId: "wf_677c9377-586"})`.
   6 web sweeps + completeness critic + doc writer. RULES baked in: never list an unfetched URL;
   M0 bar = real-tool, knobs→continuous outcomes, n≥50-100, split metadata, license.
3. **Launch the bake-off build** — the workflow script was NEVER persisted; its full build spec is
   saved as **`docs/mfl-bakeoff-build-spec-2026-07-18.md`** (same directory as this file). Either
   re-author the workflow from that spec, or hand the spec to a single strong builder agent +
   steelman adversary + smoke runner (the spec contains all three prompts' content). Prereg
   (`docs/prereg-mfl-bakeoff-2026-07-17.md`) is BINDING: metric names/definitions frozen,
   predictions untouchable.
4. **Full bake-off run** (after smoke passes): launch as a background Bash (NOT inside an agent —
   10-min agent command cap), then analyze against the frozen predictions. Score honestly: P4 is
   tautological, P1/P5 are predicted LOSSES; verdict rule = "better-posed" only if P2 AND P3 hold.
5. **HTML update — PARTIALLY DONE (00:30 SGT, solo while agents were limit-blocked).**
   New section 11 `#versus` ("How this compares to the published alternative") is WRITTEN and
   PUBLISHED to the artifact URL: the formulation table, their-own-Table-1 margin diagram
   (4.45 nm error vs 5.55 nm margin = 80%), the ∂M/∂x Loop-B query-cost analysis, honest
   concessions (amortization/simplicity/two-loop idea), and the both-in-silico caveat noting the
   pre-registered bake-off with predicted RIG losses. Glossary renumbered to 12; tag balance
   verified; label `mfl-comparison-section`. STILL OWED for the HTML: splice bake-off empirics
   into #versus when they land (the note-warn block says "will replace this sentence" — replace
   it), plus any walkthrough improvements informed by the audit. Original plan follows:
   `docs/rig-explained.html` → republish to the SAME artifact URL
   (https://claude.ai/code/artifact/a1ab0c0e-d63b-49ec-baee-ba272b3fd59b — pass it as `url` if a
   fresh conversation). Planned new content: (a) a visual system walkthrough (spec box → forward
   GP+conformal → §8 pessimistic margins → candidates/INFEASIBLE → qualification gate);
   (b) "RIG vs MFL" section: side-by-side pipeline diagram; the margin analysis FROM THEIR OWN
   TABLE 1 (target 2255.55 vs floor 2250 — their residual 4.45 nm is 80% of the 5.55 nm margin;
   MSE-to-a-point has no margin concept); the Loop-B ∂M/∂x query-cost analysis (their Eq. 4 —
   "5 iterations" becomes ~60+ real queries at d=11 under finite differences); abstention
   (MFL cannot); uncertainty split + conformal (MFL has none); AND the honest concessions:
   MFL amortizes (one 7kB reverse model, forward-pass inference) — note RIG's M3
   AmortizedInverseGenerator is the counterpart WITH an SBC/TARP calibration gate; bake-off
   empirics once they land. (c) Keep ALL existing honest-failure content (d=20 false success).
6. **BUILD_LOG + BUILD_STATE** after each lands; memory updates for durable findings.

## 3. Standing state (unchanged by this tasking)

- Tree: 511 passed; all 4 new modules verified (acquisition/qualification/linear-constraints
  SOUND, distill PARTIAL-core-sound); 3 test-coverage gaps fixed + mutation-proven; ruff format
  adopted (CI gate added); pandera dropped; analytic gradient proven correct (mpmath 5.38e-8).
- Owed experiments (unchanged): d=20 false-success RATE across seeds+GP-restarts; conformal
  revalidation vs the d=20 miss; honest M3 re-run on InSilicoMachine; powered M2 re-run
  (BoTorchBO comparator); selected-vs-random coverage.
- USER decisions still open: M0 dataset (the entire scientific claim); 5am-schedule question
  (cloud routine can't see this local session — user never answered).

## 4. Known traps for the next session (read the memory dir too)

- Subagent liveness ≠ `Get-Process python`; transcripts under `<session>/subagents/[workflows/<runId>/]`.
- `git diff` is blind (src/ untracked); mutation-test only in scratchpad copies; assert the
  mutation text is present before running; never `mktemp` (MSYS /tmp breaks Windows Python).
- PYTHONIOENCODING=utf-8 for every pytest/py run (cp1252 console).
- Do not report a claim while its refuting run is executing; replicate EXACT configs.
