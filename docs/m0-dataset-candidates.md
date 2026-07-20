# M0 dataset candidates — verified public recipe→outcome datasets (2026-07-16)

**M0** (secure a real recipe→outcome dataset) is the #1 program risk. The audit's
prior read holds: **no large public *MBE* recipe→outcome dataset exists.** This is
the verified shortlist of real, accessible alternatives — the closest real
semiconductor/thin-film process data, plus adjacent recipe→outcome datasets the
generic tabular adapter (WP-H) can ingest.

Produced by an adversarial dataset-hunt workflow (6 source angles × per-candidate
fetch-and-verify + completeness critic): 27 candidates checked, 26 verified real &
recipe→outcome-usable (11 high fit). Two verifier agents were cut off by an account
session limit — the **NREL HTEM** entry is therefore *unverified* (flagged below)
and the completeness-critic pass did not run, so this list is thorough but not
exhaustive.

Every URL below was fetched-and-confirmed by a verifier agent unless marked ⚠️.

## A. Closest to the domain — real semiconductor / thin-film process data

| dataset | fit | what it is | size | license | link |
|---|---|---|---|---|---|
| **BOSCH Plasma-Etching** | HIGH | real semiconductor etch: 31 process params + in-situ OES + wafer metrology (etch depth/uniformity), designed experiment w/ drift & replicates. **Best true-fab fit.** Needs time-series→per-run aggregation. | ~7.9 GB, 10 NetCDF + per-run CSVs | CC-BY-4.0 | https://zenodo.org/records/17122442 |
| **NREL HTEM** ⚠️unverified | (likely HIGH) | combinatorial thin-film libraries (sputter/PLD synthesis conditions → structural/optoelectronic properties). Closest analog to MBE. **Verifier hit the session limit — confirm directly.** | large | check | https://htem.nrel.gov/ |
| **Magnetron-sputtering SDL** (Uppsala) | MEDIUM | power×pressure → QCM deposition rate *with per-point σ*, gathered via GP active learning. Small but *exactly* RIG's uncertainty-aware AL shape. | ~625 rows × 15 cols (Zr) | none stated | https://github.com/jarlsanna/gps-for-magnetron-sputtering |
| Gr-ResQ (graphene CVD) | MEDIUM | crowd-sourced graphene-synthesis recipes → quality; ~200-300 params/sample. Request access. | crowd-sourced | software MIT; data request | https://nanohub.org/resources/gresq |
| RF-sputtered WC-Co thin films | LOW | 8 samples (temp/power-varied) → structure/morphology. Tiny. | 8 samples | CC-BY | https://pmc.ncbi.nlm.nih.gov/articles/PMC6728266/ |

## B. Largest recipe→outcome table (materials fabrication, adjacent but rich)

| dataset | fit | what it is | size | license | link |
|---|---|---|---|---|---|
| **Perovskite Database Project** | HIGH | ~42,400 PV devices; fabrication params (deposition method, anneal T/time/atmosphere, composition, spin-coat) → PCE/Voc/Jsc/FF + stability. Multi-lab → real drift/batch signal. Device-stack fab (not single-film), literature-mined & messy (heavy parsing). | ~42k rows × up to ~100 cols | CC-BY-4.0 | https://www.perovskitedatabase.com/ · CSV: https://github.com/Jesperkemist/perovskitedatabase_data |

## C. Best for *exercising/validating the RIG machinery* (HTE / BO benchmarks — clean, combinatorial, replicated; reaction chemistry, MIT)

Not semiconductor, but the fastest path to a real-data forward+inverse+active-loop
proof through the tabular adapter.

| dataset | fit | what it is | size | link |
|---|---|---|---|---|
| **Buchwald–Hartwig HTE** | HIGH | categorical recipe (ligand/additive/base/aryl-halide) → % yield. Fully-crossed, canonical BO benchmark. | ~3,955 rxns | https://github.com/rxn4chemistry/rxn_yields |
| **Suzuki–Miyaura HTE** | HIGH | categorical recipe (electrophile/nucleophile/ligand/base/solvent) → yield. | ~5,760 rxns | (same repo, `data/Suzuki-Miyaura/`) |
| **Olympus** benchmark datasets | HIGH | ~10+ real self-driving-lab campaigns, continuous params → objective; purpose-built for BO/AL. | tens–hundreds/campaign | https://aspuru-guzik-group.github.io/olympus/ |
| **Summit** benchmarks | HIGH | Reizman-Suzuki / Baumgartner flow-chemistry; continuous+categorical params → yield/TON/STY, **cost-aware & multi-objective** (mirrors cost-to-target). | ~50-100/campaign | https://github.com/sustainable-processes/summit |

## D. Adjacent process→quality (drop-in via tabular adapter)

| dataset | fit | what it is | size | license | link |
|---|---|---|---|---|---|
| **LPBF Ti-6Al-4V** | HIGH shape | laser power/scan-speed/hatch/layer → porosity, grain, hardness, tensile props. *Replicate specimens* per set. | 42 param-sets | CC-BY-4.0 | https://zenodo.org/records/6587905 |
| **Plastic injection molding** (AIRTLab) | HIGH shape | 13 machine params → quality label; multi-day drift. | 1,451 rows | none formal | https://github.com/airtlab/machine-learning-for-quality-prediction-in-plastic-injection-molding |
| 3D-printer (Kaggle) / FDM sets | MEDIUM | print params → mechanical/quality. Small. | ~50–500 rows | varies/request | https://www.kaggle.com/datasets/afumetto/3dprinter |

## E. Materials-synthesis recipe corpora (recipe *side* only — outcome = compound identity, no metrology KPI)

Good for the inverse / recipe-space-coverage side; not a forward metrology target.

| dataset | fit | size | license | link |
|---|---|---|---|---|
| Ceder solid-state synthesis | MEDIUM | ~30k reactions (T/time/atmosphere → target) | CC-BY 4.0 (paper) | https://github.com/CederGroupHub/text-mined-synthesis_public |
| Ceder solution-based synthesis | MEDIUM | ~35,675 procedures | CC-BY 4.0 | https://figshare.com/articles/dataset/16583387 |
| NIMS StarryData2 | MEDIUM | ~82k samples, ~194k property curves | open | https://starrydata.nims.go.jp/starrydata2/ |

## F. Semiconductor fab, but NOT clean recipe→outcome (down-ranked)

Sensor/yield/fault data with no tunable recipe columns — weak for inverse design.

| dataset | fit | note | link |
|---|---|---|---|
| SECOM | LOW | 1,567 × 591 anonymized sensors → pass/fail; no recipe knobs | https://archive.ics.uci.edu/dataset/179/secom |
| LAM 9600 metal-etch (SEMATECH) | LOW | OES/sensor fault-detection; not recipe→outcome | https://eigenvector.com/resources/data-sets/metal-etch-data-for-fault-detection-evaluation/ |
| PHM-2016 CMP | MEDIUM | removal-rate; per-wafer time-series (needs aggregation); request access | https://phmsociety.org/ (data.phmsociety.org) |

## Recommendation

- **For the real MBE/thin-film goal:** chase **NREL HTEM** (verify directly) and
  the **BOSCH plasma-etch** set — real semiconductor process data.
- **To get RIG running on real data now** (forward + conformal + inverse + active
  loop end-to-end via WP-H): start with the **magnetron-sputtering SDL** set
  (continuous inputs, per-point σ, thin-film domain) or **Buchwald–Hartwig**
  (large, clean, combinatorial). ← a machinery proof on the sputtering set is in
  progress (see `examples/real_data/sputtering/`).

> None of these is the project's own real MBE recipe→outcome data — M0 (a
> fab/vendor agreement, an owned MBE campaign, or a university SDL log) remains the
> real unlock. These de-risk everything upstream of it.
