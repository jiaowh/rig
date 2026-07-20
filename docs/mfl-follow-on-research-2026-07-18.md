# MFL follow-on trace + 2025–26 inverse-recipe SOTA landscape (2026-07-18)

Two questions answered here: (A) what, if anything, has built on MFL (arXiv 2505.16060) since
May 2025; (B) what the 2025–26 state of the art in inverse process-recipe generation and
adjacent uncertainty-aware inverse design actually looks like. Findings are stated
conservatively; the final section draws positioning implications for RIG without claiming any
empirical superiority — the pre-registered comparison lives at
[docs/prereg-mfl-bakeoff-2026-07-17.md](prereg-mfl-bakeoff-2026-07-17.md) and remains the only
legitimate vehicle for such claims.

## A. MFL citation / follow-up trace

### A.1 The paper itself — status as of 2026-07-18

Gu, Ying, Jin, Lu, Wang, Lavaei, Spanos — "Few-Shot Test-Time Optimization Without Retraining
for Semiconductor Recipe Generation and Beyond" (Model Feedback Learning), UC Berkeley /
Virginia Tech / Lam Research / UCL. https://arxiv.org/abs/2505.16060

Still **v1 only** (submitted 2025-05-21; single entry in the submission history, no v2/v3).
No journal/conference comment on the abs page; no OpenReview submission found via
api.openreview.net; no venue version surfaced in searches for NeurIPS/ICML/IEEE TSM; not
listed on the first author's Berkeley homepage (people.eecs.berkeley.edu/~shangding.gu/).
An unpublished, unrevised preprint, 14 months old.

Method recap (from https://arxiv.org/html/2505.16060v1): test-time optimization — a
lightweight learned reverse model iteratively refined against a frozen forward "machine
model" (Loop A: emulator pretrain; Loop B: refine on target model). Reaches target recipes in
~5 iterations vs 20+ for BO and 84 for human experts. **Simulated data only** (Gaussian-sampled
surrogate mimicking the Kanarik 11-knob → 6-outcome space); the authors explicitly defer
real-fab validation. Uncertainty is handled heuristically ("conservative learning" =
sensitivity-scaled learning rates, plus domain randomization) — no conformal coverage, no
support/OOD score, no explicit INFEASIBLE verdict.

### A.2 Citing works: NONE FOUND (verified across three independent indices)

https://api.openalex.org/works/doi:10.48550/arXiv.2505.16060 — OpenAlex (W4416447665)
reports cited_by_count = 0. Semantic Scholar's citations endpoint
(api.semanticscholar.org/graph/v1/paper/arXiv:2505.16060/citations) returns empty data [].
Google Scholar was captcha-blocked, so a small Scholar-only tail cannot be 100% excluded, but
no citing work was found by any method. As of 2026-07-18 there is no verifiable published or
preprint work that cites or builds on MFL. Relevance to RIG: MFL's 5-iteration efficiency
claim stands uncorroborated by any third party — RIG cannot point to independent replication
of the baseline it plans to compare against, which is itself a fact worth stating in any
bake-off writeup.

### A.3 Verified NON-citers (checked because they were the most plausible citers)

- **Yang et al., "Self-Improvement of Large Language Models: A Technical Overview and Future
  Outlook"** (arXiv:2603.25681) — https://arxiv.org/pdf/2603.25681. Broad 2026 survey of LLM
  self-improvement/test-time methods; full-text pypdf search of the downloaded PDF found no
  reference to 2505.16060/MFL (the keyword hit was TextGrad's "language model feedback").
  Relevance: the most plausible surveying citer does not cite MFL — evidence MFL is not yet
  on the survey radar.
- **Yuksekgonul et al., "Learning to Discover at Test Time"** (arXiv:2601.16175, v2) —
  https://arxiv.org/pdf/2601.16175. Stanford/UCSD/NVIDIA test-time-training/discovery paper
  (Zou, Guestrin, Sun among authors); full-text search found no citation of MFL. Relevance:
  the closest-topic recent test-time-optimization work, different group, no uptake.
- **Semiconductor-manufacturing ML non-citers**: arXiv:2511.12788 (Physics-Constrained
  Adaptive NNs for real-time semiconductor manufacturing optimization,
  https://arxiv.org/pdf/2511.12788) and arXiv:2606.11247 (Physics-informed generative AI for
  semiconductor manufacturing, https://arxiv.org/pdf/2606.11247). Both PDFs downloaded and
  full-text searched for "16060", "Model Feedback", "Shangding", "test-time optimization
  without retraining" — zero hits in either (32 and 14 pages). Relevance: the two most
  on-topic 2025–26 semiconductor-process-ML preprints don't cite MFL — no uptake even within
  the immediate application niche.

### A.4 Trace dead ends

arXiv full-text API query for "Model Feedback Learning": returns only the MFL paper itself —
no follow-up reuses the term. Web search for "2505.16060" excluding arxiv.org: only hardware
part numbers. OpenReview: no matching submission. Google Scholar: citation search and profile
fetch both captcha-blocked; the visible top-20 of Shangding Gu's profile
(scholar.google.com/citations?user=E1GCDXUAAAAJ) does not include MFL. Berkeley homepage: MFL
absent; the author's 2025–26 output is safe RL, MMLU-ProX, agentic web, robust RL benchmarks —
no semiconductor/test-time follow-ups. arxiv.org/a/gu_s_1: 404. Same-group extension / journal
version searches: nothing. Net: no v2, no venue version, no citing or follow-up work found
anywhere verifiable.

## B. 2025–26 inverse-recipe and adjacent SOTA landscape

### B.1 Semiconductor / fab process

**Model Feedback Learning** (Gu et al., arXiv 2505.16060, May 2025) —
https://arxiv.org/html/2505.16060v1 — see A.1. The closest published competitor to RIG's
per-query inverse (M2). Its conservatism is a learning-rate heuristic, not calibrated: no
conformal coverage, no support/OOD score, no explicit INFEASIBLE verdict. A
calibrated-pessimistic inverse addresses a safety story MFL lacks, while MFL's iteration-count
efficiency at query time is a strength RIG has not demonstrated — that comparison is exactly
what the pre-registered bake-off exists to test, in both directions.

**Automated Discovery of Laser Dicing Processes with Bayesian Optimization** (Leeftink et al.,
arXiv 2511.23141, Nov 2025) — https://arxiv.org/pdf/2511.23141 — GP-based BO discovering
laser-dicing recipes on REAL manufacturing experiments (industrial collaboration),
multi-objective (cut quality / throughput / material integrity). The clearest 2025
Kanarik-lineage result: sequential BO on a real semiconductor tool. Relevance: it has the
real-tool validation RIG lacks; but it is per-target sequential optimization with GP-marginal
uncertainty only — no amortized generator, no distribution-free calibration, and infeasibility
is implicit (BO fails to converge) rather than certified.

**Robust and Reliable AI for Predictive Quality in Semiconductor Materials Manufacturing with
MLOps and UQ** (Gao et al., Merck/Versum, arXiv 2605.07752, 2026) —
https://arxiv.org/html/2605.07752v1 — conformal prediction + retraining-cadence benchmarking
on REAL data: 5 years of high-volume production (~1,200 batches) including a 30% supply-chain
shift; conformal intervals raise out-of-control batch detection from 1.9% to >80% (~40×) and
hold ~90% coverage under drift when paired with 5-batch retraining. Relevance: the strongest
2026 evidence that conformal process models pay off on real fab-adjacent data — direct
external validation of RIG's M1 conformal layer. But it is forward-monitoring only: the
calibration half of RIG without the recipe-generation half. It also points at RIG's open gap:
drift-aware reconformalization.

**Simulation-guided AI-driven digital twin for plasma etching** (ScienceDirect, 2025) —
https://www.sciencedirect.com/science/article/abs/pii/S2666998625003394 — digital twin
coupling plasma-etch simulation with AI models and BO for recipe optimization; explicitly
cites Kanarik et al. Nature 2023. (Abstract only; Elsevier paywall.) Relevance:
representative of the Kanarik-follow-on cluster — sim-anchored forward twins + BO, like RIG's
Phase-0 in spirit, but forward-twin-centric with no conformal calibration and no amortized
inverse.

**Hybrid meta-learning + metaheuristic recipe-setting optimization** (Information Sciences,
2025) — https://www.sciencedirect.com/science/article/abs/pii/S0020025525001306 — hybrid
meta-learning + metaheuristic search benchmarked against MetaBO-style methods on black-box
benchmarks and a real CVD process. (Fetch returned 403; details from search snippets —
treat specifics as unverified.) Relevance: the non-generative optimization mainstream —
point-solution search with no uncertainty calibration at all.

**SandBox Studio AI** (SandBox Semiconductor) — https://sandboxsemiconductor.com/ —
the commercial incumbent: physics-based models + ML for etch/deposition recipe development,
claimed ~8× reduction in recipe-creation cost vs DOE, in use at leading chipmakers; 2025
additions include hybrid metrology (Weave) and SPIE 2025 talks on physics-based AI for
RF-pulsing optimization. Claims, no public benchmarks. Relevance: physics-model-first with ML
assist; no published conformal/coverage guarantees or explicit infeasibility certification.
RIG's differentiator against it would be auditable calibrated pessimism, not physics fidelity
— a differentiation claim, not a performance claim.

**MBE-specific closed-loop cluster** — in-situ QD-emission self-optimization via
RHEED-video feedback (arXiv 2411.00332, https://arxiv.org/abs/2411.00332 /
https://arxiv.org/pdf/2411.00332) and multimodal ML (RHEED + XRD + AFM) for GaSe MBE growth
(arXiv 2606.13900 / ACS AMI 2026, https://arxiv.org/abs/2606.13900); NAMBE 2025 has a
dedicated AI/ML-for-MBE session. REAL tool data. Relevance: real-MBE closed-loop ML control
exists in 2025 — but it is in-run feedback control, not pre-run recipe generation from a
spec, and none of it emits calibrated predictive distributions or feasibility verdicts.
Inverse-from-spec with conformal pessimism appears to be unoccupied territory in the MBE
literature.

**Run-indexed time-varying BO for plasma-assisted deposition** (Computers & Chemical
Engineering, 2024) — https://www.sciencedirect.com/science/article/abs/pii/S0098135424000711
— BO with positional encoding over run index to handle run-to-run drift. (Not fetched;
Elsevier paywall.) Relevance: names the failure mode that will eventually hit any static
learned simulator, RIG included — tool drift invalidates both the forward model and its
conformal calibration set. RIG currently has no drift-aware recalibration story; this plus
the Merck paper suggest scheduled reconformalization as the fix.

### B.2 Adjacent uncertainty-aware inverse design (methods RIG's M3 draws on)

**How well do generative models solve inverse problems? A benchmark study** (Krüger et al.,
arXiv 2601.23238, 2026) — https://arxiv.org/pdf/2601.23238 — head-to-head benchmark of cINN,
conditional diffusion, NPE, and cGAN on inverse problems, explicitly evaluating whether
generated solutions carry calibrated posterior uncertainty vs mere point estimates
(simulation benchmarks). Relevance: the method-selection evidence base for RIG's M3
amortized generator — and independent confirmation that calibration of generative inverses is
an open, measured weakness, which RIG addresses externally (conformal wrapper) rather than
trusting the generator's posterior.

**Generative Inverse Design with Abstention via Diagonal Flow Matching** (de Campos et al.,
arXiv 2603.15925, 2026) — https://arxiv.org/pdf/2603.15925 — conditional flow-matching
inverse design that can ABSTAIN when a requested spec is infeasible or OOD, formalized via UQ
of when the conditional distribution becomes ill-defined; demos on airfoils, photonics,
materials, turbine blades (simulation). Relevance: philosophically the nearest neighbor to
RIG's explicit-INFEASIBLE + support_score design — independent 2026 confirmation that
refusing to hallucinate designs for unreachable specs is the frontier. Differences: its
abstention is learned inside the generator (no finite-sample guarantee) where RIG's pessimism
is conformal/distribution-free; and it has no process-tool domain. **The most important
follow-on-adjacent finding of this sweep.**

**GUIDe: Generative and Uncertainty-Informed Inverse Design** (Mu et al., arXiv 2509.05641,
Sep 2025) — https://arxiv.org/abs/2509.05641 — deliberately AVOIDS learning response→design
mappings: a probabilistic forward model scores each candidate's confidence under a user
tolerance and MCMC samples the design space through that filter; validated on nacre-inspired
composites (simulation); handles OOD targets. Relevance: the same architectural bet as RIG
(trustworthy forward model + search, not a trusted inverse map) — but its confidence filter
is a probabilistic tolerance check, not calibrated coverage, and there is no qualification
gate or conformal band.

**Conditional diffusion for inverse prediction of process parameters and dendritic
microstructures** (PMC, Oct 2025) —
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12549910/ — conditional diffusion mapping target
mechanical properties → (process parameters + microstructure) for polymer / CFRT materials;
the clearest 2025 example of diffusion-based inverse PROCESS-parameter (not geometry)
generation. (URL seen in search results, not fetched.) Relevance: demonstrates the pattern
RIG's M3 targets in an adjacent domain — with no feasibility abstention and no calibrated
uncertainty on generated recipes, i.e., exactly the failure mode RIG's calibrated pessimism
guards against.

### B.3 Landscape dead ends (verified gaps, not search failures)

- **cINN / NPE / simulation-based inference applied to semiconductor process recipes:
  nothing exists.** Searches returned only thin-film optics (arXiv 2210.04629 / IOP acb48d),
  photonics (2208.14212), phononics, 2D-materials inverse design; NPE searches returned
  astrophysics, UHECR, epidemiology — zero fab-process SBI papers. A genuine open gap that
  RIG's M3 would occupy.
- **Conformal prediction + inverse recipe/design as a single method: no hit.** All conformal
  work is forward/monitoring (Merck 2605.07752; distribution-free process monitoring
  2512.23602). The abstention-flow-matching paper (2603.15925) is closest, and its guarantee
  is learned, not distribution-free.
- **No published direct Lam-authored sequel** to Kanarik Nature 2023 — follow-ons are citing
  works (digital twins, MFL with a Lam coauthor, human-vs-algorithm studies).
- **No public benchmark dataset for recipe inverse generation** surfaced anywhere; every
  real-tool paper uses proprietary data (see docs/m0-dataset-candidates-2026-07-18.md).
- Verification limits: ScienceDirect 403'd two Elsevier items (search-snippet detail only);
  USPTO patent hits (e.g. 11836429, "Determination of recipes for manufacturing semiconductor
  devices") note that inverting complex forward models lands OOD — corroborating
  support_score's rationale — but patent PDFs were not fetched in depth.

## Implications for RIG's positioning (conservative)

1. **No performance claim against anything is warranted today.** RIG has zero real-tool
   results (M0 open) and its only MFL comparison is the pre-registered, not-yet-run bake-off
   (docs/prereg-mfl-bakeoff-2026-07-17.md). Until that runs, the honest statement is: "RIG
   differs from MFL in what it guarantees, not — so far as anyone has measured — in how well
   it performs." MFL's own numbers are simulation-only and uncorroborated (zero citations,
   no venue, no v2), which cuts both ways: the baseline is soft, and beating a soft baseline
   proves little.
2. **The differentiation that IS defensible is structural, not empirical**: as of 2026-07-18
   no found work combines (a) inverse recipe generation with (b) distribution-free conformal
   calibration and (c) certified infeasibility/abstention in (d) a semiconductor-process
   domain. Each pairwise neighbor exists — abstention-in-generator (2603.15925, learned
   guarantee only), conformal-on-fab-data (2605.07752, forward only), inverse-on-real-tool
   (2511.23141, no calibration) — but the intersection is empty. This is a claim about the
   literature, verifiable from the citations above, not a claim about performance.
3. **Independent work is converging on RIG's design bets**, which is validating but also
   means the window is closing: abstention-aware generative inverse design (2603.15925) and
   forward-model-plus-search architectures (GUIDe) appeared independently in 2025–26. RIG's
   distinct contribution narrows to the distribution-free guarantee + process-tool domain +
   qualification gate; positioning should lean on that, not on generative-inverse novelty.
4. **Two externally-evidenced gaps to acknowledge, not hide**: (a) drift — Merck 2605.07752
   and run-indexed BO show real processes drift and that static conformal sets decay; RIG has
   no reconformalization cadence yet and should say so (it inherits this as future work with
   external evidence for the fix's shape); (b) real-tool validation — laser-dicing BO and the
   MBE closed-loop cluster have it, RIG does not until M0 lands.
5. **If the bake-off runs and RIG loses on iteration efficiency, that is a publishable,
   pre-registered result** — the positioning survives because it never rested on beating MFL
   at its own metric, but on delivering calibrated coverage and honest INFEASIBLE verdicts
   that MFL's conservative-learning heuristic does not attempt. Conversely, any RIG win must
   be reported with the caveat that MFL was reimplemented from an unrevised preprint with no
   reference implementation to validate against.
