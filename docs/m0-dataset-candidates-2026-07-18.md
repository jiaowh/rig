# M0 dataset candidates — 2026-07-18 hunt (real-tool recipe→outcome)

> **GROUND-TRUTHED 2026-07-19: all 4 STRONG candidates DOWNLOADED and verified against the
> actual files — see `data/m0-candidates/MANIFEST.md` (382 MB local). Corrections found by the
> pull (this doc's web-sourced claims stand EXCEPT):**
> 1. **empa-hipims**: n = 3,150 (doc said ">3000" — holds, thinner margin than implied); the
>    "6-D" space is actually TWO 5-D continuous subspaces (PRR vs Duty-Cycle) + categorical
>    material/power selects, not one flat 6-D box. MD5-verified full zip.
> 2. **ada-2022-09**: n = 177 compiled / 180 raw (doc said 179). **License caveat: repo README
>    claims "Data: CC-BY-4.0" but the linked LICENSE-DATA file 404s; only MIT files exist.**
>    Files must come via the GitHub LFS media endpoint (raw.githubusercontent serves stubs).
> 3. **ada-2021-01**: n = 253 exact. The "timestamped campaign CSVs" claim does NOT hold at the
>    processed level (ordinal index only, no wall-clock). Same license caveat as 2.
> 4. **nrel-htem**: **nrel.gov is DEAD (DNS undelegated); live successor is nlr.gov /
>    NatLabRockies GitHub org.** Live scale is 1,891 libraries, well below the doc's ~4,356+;
>    the per-sample API endpoint is broken; the OpenEI "bulk copy" contains no downloadable
>    file. Rating should be read as PARTIAL-at-best on access, pending the successor site.
> A 440-entry live sputtering sample + full library index were pulled as the working access path.

> Supersedes-in-part: this is a NEW dated hunt. The earlier shortlist
> [docs/m0-dataset-candidates.md](m0-dataset-candidates.md) (2026-07-16) remains valid as the
> broader adjacent-domain survey (perovskite DB, HTE chemistry, LPBF, injection molding, etc.);
> this file is a deeper, adversarially-verified pass focused on the strict M0 bar below. Where
> the two disagree (e.g. BOSCH DRIE), this file's per-criterion notes are the more careful read.

## The M0 bar (restated)

M0 = secure a real recipe→outcome dataset (implementation-plan D1; #1 program risk; owner: user/PI).
A candidate must satisfy all five criteria:

1. **real_tool provenance** — physical process tool / physical experiments, not simulation. Under
   the §3.5 data contract, headline metrics may only ever be computed on `real_tool` runs;
   simulation-derived data is `physics_sim` and cannot carry the scientific claim.
2. **Settable knobs → continuous measured outcomes** — deliberately varied recipe inputs
   (powers, pressures, flows, temperatures, times) mapped to continuous metrology outcomes
   (rate, thickness, depth, resistivity...). Sensor-trace→fault-label data (the SECOM failure
   mode) does not qualify.
3. **n ≥ 50–100 runs per process** — enough to fit + conformal-calibrate + test.
4. **Split metadata** — run order / timestamps / tool or campaign IDs enabling temporal and
   leave-one-group-out splits (needed for honest drift-aware evaluation).
5. **Open access** — explicit license, anonymous bulk download, no request-gating.

Ratings below are preserved verbatim from the hunt agents (STRONG / PARTIAL / WEAK / N/A).
Where independent sweeps rated the same source differently, both ratings are shown.

---

## STRONG candidates

### 1. Empa bipolar HiPIMS autonomous-sampling deposition-rate dataset — STRONG
Wieczorek et al., Digital Discovery 2026. https://zenodo.org/doi/10.5281/zenodo.18495401
(resolves to versioned record 18495402); companion code 10.5281/zenodo.18495504; paper
https://pubs.rsc.org/en/content/articlehtml/2026/dd/d6dd00063k (arXiv 2601.05287).

Raw data zip (396 MB compressed / 792 MB) covering **>3000 real sputter-deposition conditions**
autonomously sampled (Bayesian) over a 6-D pulse-parameter space (negative pulse width,
frequency, positive pulse voltage/width/delay, power density; peak current measured) →
QCM-measured deposition rate, plus logged process parameters and per-run oscilloscope
waveforms; 6 sub-datasets (Al/Ti × pulse-width ranges).

Per-criterion: (1) real custom Empa sputter tool — YES. (2) genuinely settable continuous
knobs → continuous outcome — YES, though the only outcome is deposition rate via QCM, not
wafer film metrology. (3) n>3000 — YES, far beyond needs. (4) sequential BO sampling gives
inherent run ordering, and the 6 material/power-density sub-datasets enable
leave-one-dataset-out; explicit timestamps unverified until the zip is opened. (5) CC-BY-4.0,
direct Zenodo download — YES. **Best M0 candidate found in this hunt.**

### 2. NREL HTEM-DB (High-Throughput Experimental Materials Database) — STRONG (two sweeps) / PARTIAL (one sweep)
https://htem.nrel.gov/ · API https://htem-api.nrel.gov · Scientific Data paper
https://pmc.ncbi.nlm.nih.gov/articles/PMC5881410/ · data.gov listing
https://catalog.data.gov/dataset/high-throughput-experimental-materials-database-51e02/resource/8f3bff7e-6df8-4b2f-9756-4be4f29a3e20
· API examples https://github.com/NREL/htem-api-examples · bulk copy https://data.openei.org/submissions/8168

141,574 thin-film sample entries in ~4,356 combinatorial sputtering libraries (~100 materials
systems; one sweep cites >300k samples / ~7,327 libraries from a later count). Per-library
synthesis knobs: target power, gas flows, substrate temperature, pressure, deposition time,
plus chamber ID, date, operator. Continuous outcomes: composition+thickness (72,952 entries),
XRD (100,848), optical absorption/band gap (55,352), electrical conductivity/sheet resistance
(32,912); synthesis conditions on ~83,600.

Per-criterion: (1) real NREL PVD chambers — YES. (2) YES with a real caveat that drove the
PARTIAL rating: libraries are combinatorial, so intra-library variation is partly positional
(composition/temperature gradient across the plate) — some "knobs" are implicit in sample
position, not settable inputs; effective distinct-recipe count per material system is far
below the headline n. (3) YES, tens of thousands with both conditions and outcomes.
(4) YES — chamber IDs + dates enable temporal and leave-one-chamber-out splits. (5) CC-BY-4.0,
web UI + REST API, no registration. **Live trap: htem.nrel.gov / nrel.gov did not DNS-resolve
from the hunt sandbox (curl exit 6 / ENOTFOUND)** — all facts rest on the Scientific Data
paper and data.gov listing, not a live fetch. Verify reachability and API contents from the
user's own network before promotion. (This also finally resolves the ⚠️unverified flag on the
2026-07-16 shortlist's HTEM row — still not fetched live, but its documentation is verified.)

### 3. Ada SDL (Berlinguette, UBC) — 2022_09 spray-combustion Pd films campaign — STRONG
https://github.com/berlinguette/ada/tree/master/2022_09%20A%20self-driving%20laboratory%20optimizes%20a%20scalable%20materials%20manufacturing%20process
(Cell Rep. Phys. Sci. 2023; repo root https://github.com/berlinguette/ada)

7 knobs (precursor concentration, DMSO content, combustion temp, air flow, spray flow, spray
height, number of spray passes) → continuous outcomes (conductivity avg+std, conductance,
thickness, sheet resistance, resistivity). 91 unique experiments in duplicate = 179 Pd film
samples across 5 BO campaigns, with per-step _START/_FINISH timestamps for every stage.

Per-criterion: (1) real robotic spray-coater — YES. (2) 7 settable knobs → continuous
outcomes — YES, best-in-class here. (3) n=179 (91 unique) — adequate, borderline for held-out
per campaign. (4) full timestamps + 5 campaigns → temporal / leave-one-campaign-out — YES
(single tool). (5) GitHub (Git LFS), data CC-BY-4.0, code MIT — YES. Caveat:
chemistry-adjacent Pd metallization film, not a semiconductor fab tool.

### 4. Ada SDL — 2021_01 Pd combustion-synthesis Pareto-front campaigns — STRONG
https://github.com/berlinguette/ada/tree/master/2021_01%20Self-driving%20laboratories%20can%20advance%20the%20Pareto%20front%20for%20thin-film%20materials
(Nat. Commun. 13, s41467-022-28580-6)

4 continuous knobs (fuel-to-oxidizer ratio, acetylacetone fraction, precursor concentration,
annealing temp 180–280 °C) → conductance mean/std, XRF-normalized conductance, conductivity.
253 sequential qEHVI runs across 4 replicate campaigns; timestamped campaign CSVs + raw
per-sample folders + campaign_log.log.

Per-criterion: (1) real_tool robotic SDL — YES. (2) YES. (3) n=253 (~60/campaign) — YES.
(4) 4 replicate timestamped campaigns + run order + event logs — YES; the replicate campaigns
are a ready-made drift/transfer testbed. (5) CC-BY-4.0 — YES. Caveats: only 4-D;
single-objective outcome; adjacent domain, not fab. (A third sweep saw only the repo landing
page and rated the Ada repository PARTIAL pending per-campaign counts; the two folder-level
verifications above supersede that.)

---

## PARTIAL candidates

### 5. BOSCH DRIE Multi-Model Dataset (ZFM Chemnitz) — PARTIAL
https://zenodo.org/records/17122442

10 daily NetCDF files (476–833 MB each, Jul 2–Aug 22 2024) with OES spectra (25 Hz) and 31
machine-parameter traces (5 Hz), plus CSVs of pre/post-etch oxide thickness and step height at
9 and 89 wafer points (etch depth, selectivity, uniformity), Lot_status.xlsx.

Per-criterion: (1) real ICP DRIE tool — YES. (2) outcomes are excellent continuous wafer
metrology, but the deliberately varied input is chamber CONDITIONING (O2 vs O2/SF6 plasma,
1/3/9 repeats, on chuck/Si/SiO2) — categorical treatments, not a continuous recipe-knob sweep;
the 31 "process parameters" are 5 Hz sensor traces, edging toward the SECOM failure mode.
(3) run count not stated on the record; an external paper (arXiv 2603.23576, seen in search
results) describes fewer than ~100 wafer-level runs — marginal. (4) date-named daily files +
Lot_status.xlsx give run ordering — YES. (5) CC-BY-4.0 — YES, but ~5–8 GB total. NOTE: the
2026-07-16 shortlist rated this HIGH / "best true-fab fit"; this hunt's closer read of the
experimental design (conditioning treatments, not knob sweeps) downgrades it to PARTIAL.

### 6. PHM Society 2016 Data Challenge — CMP material removal rate — PARTIAL (both sweeps)
https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2016/phm-data-challenge-4/

Per-wafer time-series CSVs (25 columns: wafer ID, stage A/B, timestamp + 19 process variables
incl. pressures, rotation, slurry flow, usage counters) → AVG_REMOVAL_RATE (continuous);
1,981 training wafers, 185 files, ~673k rows, plus validation/test sets.

Per-criterion: (1) real industrial CMP tool — YES (provenance not fully documented).
(2) continuous outcome YES, but inputs are mostly usage/sensor trajectories rather than
settable recipe knobs — partially the SECOM trap, though pressures/flows are recipe-adjacent
and summarizable per wafer. (3) n=1,981 — YES. (4) timestamps + wafer/stage IDs enable
temporal splits — YES. (5) free zips linked from the page but **no explicit license stated** —
the weak point.

### 7. Polybot (Argonne CNM) — PEDOT:PSS blade-coating campaign — PARTIAL
https://github.com/polybot-nexus/PEDOT_PSS_supporting_data (Nat. Commun. 2025,
s41467-024-55655-3; verified CSV at
raw.githubusercontent.com/polybot-nexus/PEDOT_PSS_supporting_data/main/PEDOT_experiment.csv)

75 rows (30 train + 45 sequential BO): DMSO vol%, EG vol%, coating speed, coating temp,
post-processing solvent (categorical), post coating speed/temp → avg conductivity
0.035–1988.7 S/cm + std, avg coverage % + std; index 0–74 gives execution order.

Per-criterion: (1) real blade-coater robot — YES. (2) 7 knobs (6 continuous + 1 categorical)
→ 2 continuous outcomes with replicate stds — YES, exactly the RIG shape,
transparent-electrode / semiconductor-adjacent. (3) n=75 — marginal: enough to fit +
conformal-calibrate, thin for a test split. (4) run-order index only, no wall-clock
timestamps, single tool — PARTIAL. (5) GitHub, MIT license, no README/data dictionary — YES
with caveat.

### 8. a-C:H:W reactive sputtering central-composite DOE (Coatings 2014) — PARTIAL
https://doi.org/10.3390/coatings4040772

Central composite design, ~25 combinations + center point ×7 (~31 runs); knobs: sputtering
power, bias voltage, Ar flow, C2H2 flow; outcomes: deposition rate + mechanical properties.
Per-criterion: (1) real industrial-style coater — YES per abstract. (2) YES, 4 knobs, proper
CCD. (3) ~31 runs — below the 50–100 bar; calibration/test split thin. (4) run order likely
unpublished — assume NO. (5) open access CC-BY, **but mdpi.com hard-403s automated fetch
(4 routes tried)** — table columns/row count unconfirmed; verify manually in a browser.

### 9. ML modeling of magnetron-sputtered Pt coatings (Coatings, Dec 2025) — PARTIAL (provisional)
https://www.mdpi.com/2079-6412/16/1/8

Experimental Pt sputtering under varied discharge current, pressure, deposition time →
thickness and deposition rate; GPR best model. (1) real_tool YES per abstract. (2) YES but
only 3 knobs, low-dimensional. (3) n unknown — MDPI 403-blocked every fetch. (4) unknown,
likely none. (5) open access CC-BY but needs a human browser to confirm whether the full run
table ships. Could drop to WEAK if n is small or the table is absent.

### 10. Penn State 2DCC-MIP LiST — MBE/MOCVD growth database — PARTIAL
https://www.mri.psu.edu/2d-crystal-consortium/user-facilities/data-management-list
(public portal https://list.2dccmip.org/list/data/ — JS app, contents not enumerable without
an account; contact 2dcc-datamanagement@psu.edu)

18,000+ bulk crystals and thin-film samples from real MBE/MOCVD/CVT tools, with growth
recipes + characterization. (1) real_tool YES — and directly relevant to RIG's MBE adapter,
the only MBE-domain candidate found. (2) recipe→characterization mapping exists but
per-sample characterization is heterogeneous, not a uniform outcome table. (3) headline n
large but the knob→outcome-complete subset is unverified. (4) plausible, unverified.
(5) **the blocker: request-gated, no anonymous bulk download, license unstated.**

---

## WEAK candidates

### 11. Chopped vs standard HiPIMS deposition-rate dataset (Univ. of West Bohemia) — WEAK
https://zenodo.org/records/17294981 — real HiPIMS experiments, pulse configurations →
deposition rate + substrate energy flux; n undisclosed, likely tens of conditions; no split
metadata visible; CC-BY-4.0. Worth a 30-minute zip inspection only if Empa disappoints.

### 12. 84-condition SiO2 CF4/O2/Ar ICP etch study (Kim et al.) — WEAK (on access alone)
https://arxiv.org/abs/2505.03826 — 84 real etch conditions on a 6-inch ICP etcher (CF4 flow
5–20 sccm, pressure 20/40 mTorr, top power 50–110 W) → 9-point ellipsometry etch depth + OES.
Near-perfect shape for M0 (real tool, settable knobs, continuous outcome, n=84) — **but no
public dataset**: PDF grepped locally, no data-availability statement, no repo link. Data
exists only as figures/tables. Would require emailing the authors; the highest-value
author-contact lead in this hunt.

### 13. Zenodo PLASMAI/Merck "plasma etch rates via transfer learning" — WEAK
https://zenodo.org/records/15343316 — power/pressure → etch rate at 10 reactor positions +
PyTorch code, CC-BY-4.0. **FAILS real_tool: generated from physics-based models**
(`physics_sim` provenance). Only useful as extra pretraining material, which mbe_sim covers.

### 14. Mendeley surrogate-assisted CVD dataset — WEAK
https://data.mendeley.com/datasets/9p8jb3zrnb/1 — susceptor temp + inlet gas velocity →
deposition rate/uniformity, but values are XGBoost predictions on CFD cases — doubly
synthetic; FAILS real_tool. CC-BY-4.0.

### 15. ZnO chemical etching DOE — WEAK
https://zenodo.org/records/5741397 — NH4Cl concentration/temp/agitation → etch rate,
selectivity; wet bench not a process tool; almost certainly far below 50 runs (19 KB raw
sheet); CC0.

### 16. IEEE DataPort HARC/Bosch etch OES datasets — WEAK
https://ieee-dataport.org/documents/correction-oes-signal-based-viewport-contamination-levels-harc-etch-process
— real 6-inch ICP tool but OES sensor traces for signal correction, not a recipe→outcome
table (SECOM-adjacent); subscription-gated. Not worth pursuing.

### 17. Taguchi L16 sputtering of CoCrFeNi films (Materials 2022) — WEAK
https://pmc.ncbi.nlm.nih.gov/articles/PMC9693865/ — full 19-row run table printed in-paper:
5 knobs → 9 continuous outcomes. Real tool, right shape, open CC-BY — but n=19 and no run
order. Representative of the entire Taguchi/orthogonal-array family (n=9–27): every member
fails criterion 3; useful pooled or as sanity/transfer probes only.

### 18. Zenodo DLC:Ag reactive sputtering + ellipsometry — WEAK
https://zenodo.org/records/7341684 — process settings → ellipsometry optical constants;
outcomes are spectra not scalar metrics; n likely far below 50; open download. One click,
not a base dataset.

### 19. NREL autonomous sputter synthesis of nitrides (Zn-Ti-N) — WEAK (both sweeps)
https://arxiv.org/pdf/2305.11122 (APL Materials 11, 071119, 2023) — real autonomous
BO-driven reactive sputtering, target powers → XRF composition. DAS extracted verbatim from
the PDF: "available from the corresponding author upon reasonable request" — FAILS access.
If HTEM is adopted, check whether these libraries were back-deposited there.

### 20. Self-driving sputter epitaxy of β-Ga2O3 (Feb 2026) — WEAK
https://arxiv.org/abs/2602.22531 — RF magnetron sputter epitaxy SDL, 4-D knobs → continuous
Urbach energy; the best topical match found (genuinely semiconductor). No data link on the
abstract page as of today; n unknown. Watch / email authors.

### 21. PV-Lab (MIT) SPProC perovskite stability — WEAK
https://github.com/PV-Lab/SPProC — real experiments, Apache-2.0, but inputs are material
compositions, not settable tool knobs — wrong shape for recipes.

### 22. NIST CAMEO closed-loop discovery dataset — WEAK
https://catalog.data.gov/dataset/closed-loop-autonomous-materials-exploration-and-optimization-1-0
— real measurements but the "knob" is position on a pre-fabricated composition-spread wafer
and the primary output is phase labels; access good, shape wrong.

### 23. UCSB NanoFab Wiki ICP etching recipes — WEAK
https://wiki.nanofab.ucsb.edu/wiki/ICP_Etching_Recipes — real fab tools, exactly the right
shape per-recipe (knobs → etch rate/selectivity/uniformity), but each recipe is ~one point in
knob space across heterogeneous tools; well below the runs bar. Same class as Stanford
SNF/Cornell CNF/MIT.nano wikis: priors and sanity anchors, not an M0 dataset.

### 24. Eigenvector Research LAM 9600 metal etch — WEAK
https://eigenvector.com/resources/data-sets/metal-etch-data-for-fault-detection-evaluation/
— 129 wafers from a real LAM 9600 etcher (1995), but sensor traces → fault labels: the SECOM
failure mode exactly. Documents that the best-known "real etch tool" public dataset is the
wrong shape for M0.

### 25. NIST/SEMATECH e-Handbook case-study datasets — WEAK
https://www.itl.nist.gov/div898/handbook/pri/section6/pri6.htm — genuine SEMATECH/NIST DOE
data printed in-page, fully open, but ~8–100 runs per case study and few are front-end
process-tool recipes. Unit-test / toy-calibration material.

---

## Verified non-candidates (N/A)

- **Kanarik et al., Nature 616, 707 (2023)** — https://pmc.ncbi.nlm.nih.gov/articles/PMC10132970/
  Checked specifically as a data source: **dead.** All runs were on a "sophisticated virtual
  platform" (proprietary Lam simulator) — provenance would be `physics_sim` even if data
  shipped. DAS verbatim: source data for Figs. 2–3 only (cost/progress-tracker trajectories,
  11 KB / 39 KB XLSX — no recipe values); the 11-knob → 6-outcome run tables and the simulator
  are "available on reasonable request" from Lam. Useful only as a benchmark-design reference.
- **A-Lab (LBNL/Ceder), Nature 2023** — https://www.nature.com/articles/s41586-023-06734-w —
  real robotic lab but outcomes are phase identification / target yield per discrete target
  (largely categorical) and precursors are categorical: not a continuous knob→outcome surface.
- **Intel fab discrete-event benchmark** — https://arxiv.org/abs/2408.09307 — simulation
  output (factory-logistics DEVS model), fails real_tool outright.

## Dead ends (searched, yielded nothing)

- **Zenodo API** `"etch rate" AND "process parameters"` (type=dataset): 0 hits; useful records
  surface only under looser queries. **Kaggle**: nothing beyond SECOM and WM-811K wafer maps
  (classification). **UCI**: nothing besides SECOM. **Figshare / OSF**: sputtering/PECVD/RIE
  process-parameter searches returned papers and patents, no data records. **data.gov**
  general search: unproductive.
- **NIST**: Plasma Process Metrology program page exists
  (nist.gov/programs-projects/plasma-process-metrology) but no downloadable recipe→outcome
  dataset; nothing in MIDAS/SRD via search.
- **ALD**: no per-run public datasets anywhere; Zenodo ALD records are characterization dumps
  (XRD/XPS); the ML-ALD literature (arXiv 2205.08378, arXiv 2602.18565, ScienceDirect
  S2452414X25001025) is simulation-trained; the ACS Chem. Mater. ML-in-ALD review explicitly
  laments ALD data are "stored locally... unstructured, or otherwise inaccessible."
- **Virtual metrology with published tables**: every hit (Lynn thesis, IEEE VM-for-plasma-etch)
  uses proprietary industrial data; no public CSVs — and VM is sensor-trace-shaped anyway.
- **Self-driving-lab community**: AccelerationConsortium/awesome-self-driving-labs README has
  NO datasets section (papers/software/hardware only); Acceleration Consortium releases are
  simulation benchmarks; Abolhasani/NC State fluidic labs — no dataset links surfaced;
  NIMS-OS is orchestration software only; arXiv 2506.05999 (Empa co-sputter composition
  mapping) has no data links and is composition-mapping, not recipe→outcome.
- **Access blockers to note**: mdpi.com hard-403s all automated fetches (two Coatings
  candidates partially unverified); nrel.gov/htem.nrel.gov DNS-failed from the sandbox;
  nature.com redirects automated fetches to an auth flow (PMC mirrors used); ScienceDirect
  abstracts 403'd; Springer J. Korean Phys. Soc. sputtering-ML paper
  (10.1007/s40042-026-01596-7) auth-blocked, data availability unverifiable.
- **arXiv 2505.03826** (84-condition etch): PDF grepped locally — zero hits for
  github/zenodo/osf/availability; confirmed no public data. **arXiv 2601.05287** (Empa HiPIMS
  preprint): the Zenodo DOIs appear only in the published RSC version's DAS, not the preprint.
- **Net structural finding**: no public benchmark dataset for recipe inverse generation exists
  anywhere — every real-fab paper uses proprietary data. The only verified sequential,
  timestamped, multi-campaign real-tool releases are the Ada folders; polybot's 75-run CSV is
  the only other verified knob→continuous-outcome table from an SDL.

## Bottom line (honest)

**No candidate meets the full M0 bar within the semiconductor-fab domain.** The bar's joint
requirement — real fab-class tool + settable knobs → wafer-level continuous metrology +
n≥50–100 + split metadata + open license — is met by nothing public; every real-fab dataset is
either proprietary, request-gated, sensor-trace-shaped (SECOM class), or tiny (Taguchi class).

**Best available compromise, in order:**

1. **Empa bipolar HiPIMS (Zenodo 18495401)** — the primary recommendation. Meets criteria 1,
   2, 3, 5 outright and probably 4 (BO ordering; confirm timestamps on unzip). Compromises:
   deposition-rate-only outcome (QCM, not wafer metrology) and a deposition (not etch/fab)
   process. It is a real vacuum plasma tool with a 6-D continuous recipe space and n>3000 —
   more than an order of magnitude beyond anything else verified downloadable.
2. **Ada 2021_01 + 2022_09** as the drift/transfer testbed (replicate timestamped campaigns,
   full per-step timestamps) — domain-adjacent, but the only verified data that can exercise
   temporal-split conformal validity honestly.
3. **NREL HTEM-DB** as the scale play IF (a) it is reachable from the user's network and
   (b) a material system with genuine knob-driven (not position-driven) variation can be
   carved out — a hands-on API session is required before counting it.
4. **BOSCH DRIE (Zenodo 17122442)** if wafer-level etch metrology on a true fab tool is
   non-negotiable — accepting that the varied input is conditioning treatments, so it tests a
   narrower claim than free recipe inversion.

The MBE-specific version of M0 remains empty: the only MBE-domain source (2DCC LiST) is
request-gated. Highest-value manual follow-ups: email Kim et al. (arXiv 2505.03826, 84-run
etch table) and the β-Ga2O3 SDL group (arXiv 2602.22531); open the two MDPI Coatings papers
in a browser; probe HTEM API reachability.
