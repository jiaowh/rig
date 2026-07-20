# M0 dataset candidates — download & verification manifest

Generated 2026-07-19. Pulled the 4 "STRONG" candidates from
[docs/m0-dataset-candidates-2026-07-18.md](../../docs/m0-dataset-candidates-2026-07-18.md)
into `data/m0-candidates/<slug>/`. Ingress only — nothing in this tree was uploaded anywhere.

**Total disk used: ~382 MB** (well under the 2 GB budget). Breakdown:
`empa-hipims` 380 MB, `ada-2022-09` 360 KB, `ada-2021-01` 60 KB, `nrel-htem` 1.4 MB.

**Headline honesty note:** three of the four candidates check out close to the doc's claims.
The fourth (NREL HTEM) surfaced a major, independently-confirmed live-infrastructure change
the doc could not have known about — see §3 below — plus a real access-path limitation (no
working per-sample endpoint) that changes what "ingest via the API" would actually mean today.

---

## 1. empa-hipims — Empa bipolar HiPIMS deposition-rate dataset

- **Source**: Zenodo record `10.5281/zenodo.18495402` (resolves from the concept DOI
  `10.5281/zenodo.18495401`). Fetched via `https://zenodo.org/api/records/18495402`.
- **License**: verified on the record — `cc-by-4.0` (matches doc claim). `access_right: open`.
- **Files downloaded**:
  - `hipims_raw.zip` — 395,942,394 bytes (377.6 MiB / ~396 MB, matches doc's "~396 MB
    compressed"). MD5 `681fa5ee5c1e842bb30f4f7e0c5cb39c` — **verified: matches the Zenodo
    API's listed checksum exactly.** SHA256 (recorded for our own future-integrity checks,
    not a Zenodo-published value): `d947090585e49f12056c7294727b06d046e65cca286796a9ab98828b41dfa285`.
  - `record.json` — Zenodo API metadata (5,059 bytes).
  - `extracted_sample/` (2.4 MB) — the 6 `df_campaign_*.json` outcome tables, 6
    `Campaign.json` BayBE search-space definitions, 6 `calibration.txt` files, and ONE
    sample oscilloscope waveform log, extracted from the zip for inspection. **The zip
    itself was kept intact, not deleted, per instructions.**
- **VERIFIED facts** (from opening the zip, not from the doc):
  - Zip contains 6,345 entries total. Top-level layout: `Clean Datasets Used for
    Publication/` (6 sub-dataset folders, each with `calibration.txt`, `Campaign.json`,
    one `df_campaign_*.json`, and a `Logfile - Oscilloscope/` folder of per-run raw
    waveform `.txt` files) + a `Raw Data .Json Files/` folder (duplicate copies of the
    same 6 campaign JSONs).
  - **Row counts per sub-dataset** (exact, counted from the JSON): Al-120W-shortPW 601,
    Al-200W-highPW 651, Al-250W-lowDuty 401, Ti-120W-shortPW 495, Ti-200W-highPW 601,
    Ti-250W-lowDuty 401. **Total = 3,150 conditions.** This DOES clear the doc's ">3000"
    claim but the true number is 3,150, not the ~3,550 a naive add might suggest — worth
    recording precisely since the margin over 3,000 is thinner than "far beyond needs"
    implies (~5% headroom, not an order of magnitude).
  - **Columns** (4 of 6 sub-datasets): `PRR (Hz)`, `PW (us)`, `pos. Delay (us)`,
    `pos. PW (us)`, `pos. Setpoint (V)`, `Ipk (A)`, `y1`, `BatchNr`, `FitNr`. The 2
    "duty cycle series" sub-datasets swap `PRR (Hz)` for `Duty Cycle (ratio)` instead —
    so the "6-D" space is a UNION of two different 5-parameter continuous subspaces
    (5 knobs + material Al/Ti + power-density/duty tier selects the 6th-and-7th
    dimensions categorically), not one flat 6-D continuous space every row lives in.
    This is a real nuance the doc's one-line "6-D pulse-parameter space" glosses over.
  - `Campaign.json` confirms a genuine Bayesian-optimization (BayBE) `NumericalContinuousParameter`
    search space over exactly those 5 continuous knobs per sub-dataset — confirms
    "autonomously sampled (Bayesian)" claim.
  - `y1` is the deposition-rate target the BO optimized; `calibration.txt` gives a
    per-sub-dataset `(us-window, dep-rate calibration)` scalar (e.g. `220, 1.1684`)
    needed to convert `y1` to physical deposition rate — this calibration step is
    real and will need to be reproduced at ingest time, it's not already baked into `y1`.
  - `Ipk (A)` = peak current, confirmed measured-not-set (matches doc).
  - **Run ordering**: confirmed. Each sub-dataset has exactly one oscilloscope
    logfile per row (601 files for 601 rows in Al-shortPW, checked explicitly), and the
    logfile serial numbers in the filenames (e.g. `Logfile_003841033179.txt` →
    `..._058363.txt`) are monotonically increasing — a genuine per-run sequence, though
    NOT a wall-clock timestamp (doc flagged this as "unverified until the zip is
    opened" — now resolved: ordering exists, explicit timestamps do not).
- **Mismatches vs doc**: none material. The "6-D" framing is looser than the doc implies
  (see above) and the exact n is 3,150 not "~3,550"/an unspecified larger number, but both
  are minor precision issues, not wrong claims.
- **Next step to ingest via `rig_adapters.tabular`**: write a loader that (1) reads each
  `df_campaign_*.json` as a `RunRecord` batch keyed by sub-dataset (material × power tier),
  (2) applies the `calibration.txt` scalar to convert `y1` → physical deposition rate in
  SI, (3) treats `PRR/Duty Cycle` as a categorical-select of which 5-knob subspace is
  active rather than pretending all 6 knobs are simultaneously free, (4) tags
  `Provenance.source = real_tool` (genuine Empa sputter tool) and uses oscilloscope
  filename serials as the ordering key for temporal-style splits.

---

## 2. ada-2022-09 — Ada SDL spray-combustion Pd films campaign

- **Source**: GitHub `berlinguette/ada`, folder `2022_09 A self-driving laboratory
  optimizes a scalable materials manufacturing process`. Downloaded via GitHub API
  (directory listing) + `media.githubusercontent.com` (the repo uses Git LFS; the CSVs
  are LFS pointers under `raw.githubusercontent.com`, so the real content requires the
  LFS media endpoint, not the raw endpoint — a real gotcha, noted below).
- **License — MISMATCH FROM DOC, flagged prominently**: the repo root `README.md`
  states "Data: CC BY 4.0 / Code: MIT" but **links to `LICENSE-DATA` and `LICENSE-CODE`
  files that do not exist in the repository (both 404 on
  `raw.githubusercontent.com/berlinguette/ada/master/LICENSE-{CODE,DATA}`)**. The only
  license text actually present anywhere in the repo is **MIT** (root `LICENSE` file,
  copyright 2025 The Berlinguette Group; and a second `LICENSE.txt` inside the 2021_01
  folder, MIT, copyright 2020). **The doc's "data CC-BY-4.0, code MIT" claim rests on a
  broken link in the source repo — treat the data license as MIT-by-default-file /
  CC-BY-4.0-by-README-intent-only until the authors fix the dead links or you get
  written confirmation.** This applies to both ada-2022-09 and ada-2021-01.
- **Files downloaded**:
  - `compiled_optimization_data.csv` — 166,809 bytes. SHA256
    `854fd9c997a26c6b9f92733f8d1143079531593250546230b8081134ea578f8f` — **verified: matches
    the Git LFS pointer's declared `oid sha256:...` exactly**, confirming the real LFS
    object (not the 131-byte pointer stub) was retrieved correctly.
  - `raw_per_campaign/*.csv` (5 files, one per campaign) — the per-physical-sample
    `consolidated_data_processor_output.csv` from each of the 5 raw campaign folders,
    30,038 / 55,502 / 29,716 / 39,146 / 16,431 bytes.
  - `README.md` (7,482 bytes) — the folder's own data dictionary.
- **VERIFIED facts**:
  - `compiled_optimization_data.csv`: **177 rows, 55 columns**, 5 distinct
    `campaign_ID` values (`2022-07-11_12-55-37`: 28, `2022-07-12_10-41-55`: 57,
    `2022-07-13_09-53-56`: 29, `2022-07-13_23-11-10`: 46, `2022-07-14_14-54-21`: 17).
  - The 5 `raw_per_campaign/*_consolidated.csv` files (per-physical-sample level, 74
    columns each) sum to **180 rows** across the same 5 campaigns.
  - Columns confirm the doc's 7 knobs (`concentration`, `DMSO_content`,
    `combustion_temp`, `air_flow_rate`, `spray_flow_rate`, `spray_height`, `num_passes`,
    each with `_requested`/`_realized` pairs) and the outcome set (`conductance_mean/std`,
    `thickness_avg/std`, `sheet_conductance_avg/std`, `sheet_resistance_avg/std`,
    `conductivity_avg/std`, `resistivity_avg/std`, `conductive_fraction`).
  - Full per-stage timestamps confirmed present: `SAMPLE_START/FINISH`,
    `MIX_CHEMICALS_START/FINISH`, `SPRAY_COAT_START/FINISH`, `XRF_START/FINISH`,
    `MICROSCOPE_START/FINISH`, `CONDUCTIVITY_START/FINISH`, `FLIR_CAMERA_START/FINISH` —
    real wall-clock timestamps, e.g. `"2022-07-11 13:00:33,782"`.
  - The folder's own `README.md` states verbatim: "The self-driving lab performed 91
    unique experiments in duplicate (except for some samples that failed) and created
    179 individual Pd film samples" — this is the doc's source for "179," and it is a
    genuine source claim, not a hunt-agent fabrication.
- **Mismatch vs doc**: the doc says "179 Pd film samples." The files we actually hold say
  **177** (compiled table) and **180** (raw per-campaign sum) — neither is exactly 179.
  Likely explanation: 2 of the 179 attempted samples failed and were dropped from the
  compiled table (177 = 179 − 2), while the raw per-campaign consolidated CSVs may
  include 1 extra non-sample control/calibration row (180 = 179 + 1) — but this is
  inference, not confirmed from a data dictionary. **Use 177 as the safe "usable rows"
  number for M0 fitting purposes**, and treat "179" and "180" as both approximately but
  not exactly right.
- **Next step to ingest via `rig_adapters.tabular`**: load `compiled_optimization_data.csv`
  directly (177 rows × 7 `_realized` knob columns → the `_avg`/`_mean` outcome columns),
  tag `Provenance.source = real_tool`, use `campaign_ID` for leave-one-campaign-out splits
  and the `*_START` timestamp columns for temporal splits — this file alone is already
  ingest-ready without touching the raw per-sample folders.

---

## 3. ada-2021-01 — Ada SDL Pd combustion-synthesis Pareto-front campaigns

- **Source**: GitHub `berlinguette/ada`, folder `2021_01 Self-driving laboratories can
  advance the Pareto front for thin-film materials`. Same LFS-pointer gotcha as above;
  fetched processed CSVs via `media.githubusercontent.com`.
- **License**: same CC-BY-4.0-claimed-but-dead-link issue as §2 (repo-wide). This folder
  additionally ships its OWN `LICENSE.txt`, which is **MIT**, copyright 2020 — so at the
  folder level there is a concrete, present MIT license file, plus the (broken-linked)
  repo README claim of CC-BY-4.0 for data. Same caveat: verify data terms with the
  authors before treating CC-BY-4.0 as certain.
- **Files downloaded** (`processed_data/`, 4 campaign CSVs):
  - `campaign 2020-12-18_17-38-40.csv` — 11,208 bytes,
    SHA256 `4944552a7ccc533d9d2c2f449169263a67fba81f81ee62d585e50ddf8ea545e4`
  - `campaign 2020-12-23_17-06-50.csv` — 10,676 bytes,
    SHA256 `82bc70894dec60ffa77fa4f4173283e41e12e02cbc9441688e8561fe1a3cf053`
  - `campaign 2021-01-04_08-37-39.csv` — 8,665 bytes,
    SHA256 `42431d285d025e66d8f51548d4e9d3a47469e0ba3274b400115bd95a945013f3`
  - `campaign 2021-01-12_16-26-56.csv` — 11,654 bytes,
    SHA256 `29bb0bb9a3120c455e7a161bc051ab096bf132ba57ab0d7d189b8edd2e45a8b4`
  - `README.md` (3,463 bytes), `LICENSE.txt` (1,079 bytes).
- **VERIFIED facts**:
  - Row counts: 65 + 63 + 53 + 72 = **253 rows across 4 campaigns — exactly matches the
    doc's "253 sequential qEHVI runs across 4 replicate campaigns."**
  - Columns confirmed: `sample`, `x0: fuel to oxidizer ratio`, `x1: acac amount`,
    `x2: total concentration`, `x3: temperature`, `conductance - mean`,
    `conductance - std`, `XRF-normalized conductance - mean`,
    `XRF-normalized conductance - std`, `Conductivity - mean` — 4 continuous knobs, 5
    outcome columns (matches doc's "conductance mean/std, XRF-normalized conductance,
    conductivity"; doc doesn't separately call out that there's no `Conductivity - std`
    column — there isn't one, `Conductivity` is mean-only).
  - `x3: temperature` empirically ranges **180.0–278.2 °C**, matching the doc's
    "180–280 °C" claim almost exactly.
  - **No wall-clock timestamps in the processed CSVs** — only an ordinal `sample`
    column. The doc's summary line calls these "timestamped campaign CSVs"; that is
    **not accurate at the processed-data level** we can verify without downloading the
    (large, per-sample-image-heavy) `raw_data/` tree. Wall-clock time only exists
    implicitly in the campaign folder names themselves (e.g.
    `2020-12-18_17-38-40` = campaign start time) and, per the folder's README, inside a
    `sample_log.log` per raw sample directory that we did not download (kept out of
    scope to respect the disk budget and because per-sample raw folders triggered
    Windows path-length failures during an initial `git clone` attempt — see below).
- **Mismatch vs doc**: "timestamped campaign CSVs" overstates what's in
  `processed_data/*.csv` — those are ordinal, not wall-clock-timestamped. Row count
  (253) and knob/outcome shape are otherwise exactly right.
- **Gotcha for future agents**: a plain `git clone --depth 1 --filter=blob:none --sparse`
  of this repo **fails on Windows** with `Filename too long` inside the raw per-sample
  folders (paths exceed 260 chars) — the GitHub Contents API + `media.githubusercontent.com`
  route used here avoids that entirely and is the cleaner path on Windows regardless of
  repo size.
- **Next step to ingest via `rig_adapters.tabular`**: load the 4 `processed_data/*.csv`
  files directly (already tidy, 4 knobs → 5 outcomes, 253 rows), tag
  `Provenance.source = real_tool`, use the campaign filename/folder timestamp as a
  per-campaign (not per-row) split key, and treat the 4 campaigns as the
  leave-one-campaign-out drift testbed the doc recommends.

---

## 4. nrel-htem — NREL HTEM-DB (sample pull + bulk-copy pointer, no bulk download)

**This candidate surfaced the largest, most consequential mismatch of the four — read
this section in full before trusting anything else written about HTEM elsewhere.**

### 4a. nrel.gov is dead; the live successor is nlr.gov / "NatLabRockies"

- Confirmed via **two independent network paths** (this sandbox's `curl` with system DNS,
  and Anthropic's `WebFetch` infrastructure running on a different network) that
  `nrel.gov`, `www.nrel.gov`, `htem.nrel.gov`, and `htem-api.nrel.gov` **do not resolve
  at all** — not a firewall/ENOTFOUND-from-a-restricted-sandbox issue (which is what the
  doc assumed), but genuinely **no DNS delegation for the `nrel.gov` zone** as seen from
  two public resolvers (Google `8.8.8.8` and Cloudflare `1.1.1.1` DNS-over-HTTPS both
  return only the `.gov` TLD's SOA record with no delegation — i.e. the zone is currently
  undelegated, a global condition, not a local one). This finally root-causes the doc's
  "live trap: htem.nrel.gov / nrel.gov did not DNS-resolve from the hunt sandbox" note —
  it wasn't the sandbox, the domain is actually down for everyone right now.
- The OpenEI submission page (`data.openei.org/submissions/8168`, which itself resolves
  and loads fine) **currently links to `https://htem.nlr.gov` and
  `https://htem-api.nlr.gov/`** — a different domain, `nlr.gov`, not `nrel.gov` — as the
  live website and API endpoints. Both resolve and respond:
  - `https://htem.nlr.gov` → HTTP 200, a JS single-page app shell (title "Htem app";
    matches the doc's description of the portal as "JS app, contents not enumerable
    without [running JS]" — consistent with this being the same product, just re-hosted).
  - `https://htem-api.nlr.gov/api/...` → live, responds to real queries with real data
    (see 4b).
- The GitHub org that owns `htem-api-examples` has also moved: `NREL/htem-api-examples`
  redirects (HTTP 301 at the API level, `repositories/106030629`) to
  **`NatLabRockies/htem-api-examples`**. The README and source code inside that repo
  still say "National Renewable Energy Laboratory" / `nrel.gov` throughout (documentation
  has not caught up to the rename), but the live org name is `NatLabRockies`.
- **Read on this literally**: this is either a very recent (post my knowledge, and
  apparently post the 2026-07-18 doc) renaming of NREL to something like "National
  Laboratory Rockies" with a `nlr.gov` domain, or some other infrastructure migration
  producing the same externally-visible symptoms. I have not found an announcement page
  confirming a name change (didn't chase that further — out of scope for a data-pull
  task), but the DNS + GitHub-org evidence is solid and independently corroborated. **Flag
  this to the user/PI explicitly: any future work citing "nrel.gov" needs a live check —
  the doc, written one day before this pull, was already citing a dead domain.**

### 4b. Live API pull — succeeded, but not the way the doc/example code describe

- `GET https://htem-api.nlr.gov/api/sample_library` (no ID) returns the **full library
  index**: HTTP 200, 949,459 bytes, **1,891 libraries** — saved as
  `nrel-htem/all_libraries_index.json`. Each entry has `id`, `elements`, `quality`,
  `has_xrd/has_xrf/has_opt/has_ele` (per-technique sample counts), and (for ~1,800 of the
  1,891) real deposition recipe fields: `deposition_power`, `deposition_gases`,
  `deposition_gas_flow_sccm`, `deposition_sample_time_min`,
  `deposition_base_pressure_mtorr`, `deposition_initial_temp_c`,
  `deposition_substrate_material`.
- **The official example code's per-sample endpoint (`api/sample/{id}`) is broken for
  every ID tried** — including IDs taken directly from a live library's own
  `sample_ids` field (e.g. library 9833's first `sample_ids` entry, `310329`, plus
  small IDs 5/50/.../2,000,000) all return HTTP 400
  `{"success":false,"error":{"name":"Not Found",...}}`. This is a **real, current
  limitation of the migrated API**, not a mistake on our end — the route exists (it
  returns a structured "not found" business error, not a generic 404 router miss) but no
  ID resolves.
- **What DOES work**: `GET /api/sample_library/{id}` (single-library detail, includes
  `sample_ids`, `owner_name`, `xrf_elements`, full deposition recipe) and
  `GET /api/sample_library/{id}/xrf` (per-position XRF composition, keyed
  `"{sample_id}-{position}"` → `{compound: pct, ...}`). We could NOT get `.../opt`
  (optical) or `.../ele` (electrical/4-point-probe) sub-resources to resolve under any
  name tried (`opt`, `optical`, `ele`, `electrical`, `elec`, `4pp`) — all 404 — even for
  libraries whose metadata says `has_opt`/`has_ele` > 0. Only `xrf` and `xrd`
  sub-resources are live.
- **Working sample pull saved**: `nrel-htem/htem_sample_pull_10libraries.json` — **440
  entries**, combining `/api/sample_library/{id}` recipe metadata with
  `/api/sample_library/{id}/xrf` per-position composition outcomes, across **10 real
  sputtering libraries** (IDs 9833, 10225, 10420, 10579, 12586, 12584, 8554, 8552,
  12865, 10421 — chosen as the top-10 libraries with both `deposition_power` and
  `deposition_gases` set and `has_xrf > 0`). Per-record fields: `library_id`,
  `position_key`, `elements`, `deposition_compounds`, `deposition_power_W`,
  `deposition_gases`, `deposition_gas_flow_sccm`, `deposition_base_pressure_mtorr`,
  `deposition_growth_pressure_mtorr`, `deposition_sample_time_min`,
  `deposition_initial_temp_c`, `deposition_substrate_material`, `sample_date`,
  `xrf_composition_pct`. Raw per-library metadata + XRF responses also kept individually
  under `nrel-htem/sample_pull/`.
  - Note this required **combining 10 libraries**, not "one," to reach "a few hundred
    entries" — every library caps at exactly 44 positions (a fixed combinatorial-wafer
    grid size), so a single library alone only ever gives 44 rows. This is itself
    confirmation of the doc's own caveat: "intra-library variation is partly
    positional... effective distinct-recipe count per material system is far below the
    headline n" — 44 fixed positions per library, recipe knobs vary only AT THE LIBRARY
    level (one recipe per library, 44 measured positions per recipe), not per-row.
  - Also saved `nrel-htem/api_docs/` (README.md, `library.py`, `sample.py` from the
    `NatLabRockies/htem-api-examples` repo) as the access-path documentation artifact.
- **API docs / examples repo used**: `https://github.com/NatLabRockies/htem-api-examples`
  (redirected from `NREL/htem-api-examples`), MIT-style license per its `LICENSE` file
  (not independently re-verified in detail — low priority given the code, not the data,
  carries that license).

### 4c. Scale mismatch — the live API currently exposes far fewer libraries/samples than documented

- **Doc claims**: "141,574 thin-film sample entries in ~4,356 combinatorial sputtering
  libraries (one sweep cites >300k samples / ~7,327 libraries)."
- **Live API, verified just now**: `api/sample_library` returns **1,891 libraries total**
  (confirmed not a pagination artifact — passing `?limit=5000` returns the identical
  1,891; library `id` values range 2,952–13,514, so the ID space is sparse). Summing the
  per-library `has_xrf`/`has_xrd`/`has_opt`/`has_ele` counts across all 1,891 libraries
  gives **55,067 XRF / 65,779 XRD / 46,905 optical / 19,797 electrical** sample-level
  entries — all substantially below the doc's cited 72,952 / 100,848 / 55,352 / 32,912.
- **This is a large, well-evidenced discrepancy** (roughly 25–40% of the previously
  documented library count, and correspondingly less sample-level data across every
  measurement technique). Given the `nrel.gov → nlr.gov` migration evidence above, the
  most likely explanation is that the migration either (a) has not yet re-exposed the
  full historical dataset, or (b) the currently-public API subset is smaller than what
  was reported in the original Scientific Data paper / data.gov listing (possibly by
  design — an access-control change). **Do not treat 141,574 / 4,356 as current facts.
  Re-check `all_libraries_index.json`'s count against a fresh API pull before relying on
  HTEM for scale.**

### 4d. OpenEI "bulk copy" — not a downloadable file; corrected

- **Task instruction assumed** a discrete bulk-copy file existed at
  `data.openei.org/submissions/8168` with a reportable size. **This is not accurate.**
  The submission's structured metadata (JSON-LD `Dataset` schema, extracted from the
  page) lists exactly **two `distribution` entries**, both `encodingFormat: "website"`:
  the interactive site (`https://htem.nlr.gov`) and the API
  (`https://htem-api.nlr.gov/`). **There is no third distribution with a `contentSize`
  or a static archive/zip.** "Bulk copy" appears to be OpenEI's framing of "you can get
  everything via the API," not an actual pre-packaged download — so there is no file
  size to record, and none was downloaded (correctly matching the instruction not to
  bulk-download, but for a different reason than expected: there is nothing to bulk-download).
  - Confirmed license on the same page: **CC BY 4.0** (`creativecommons.org/licenses/by/4.0/`,
    both as a footer link and in the JSON-LD `"license"` field) — this part of the doc's
    claim holds.
  - Submission metadata also confirms real NREL authorship/affiliation (Andriy
    Zakutayev, John Perkins, Marcus Schwarting, et al., all "National Renewable Energy
    Laboratory") — i.e. this is genuinely the HTEM dataset's canonical listing, not a
    wrong/stale page, despite the domain-name churn described above.
- **Files kept**: none beyond `all_libraries_index.json` and the sample pull above — no
  HTML page was retained (facts were extracted and are recorded here instead).

### Next step to ingest via `rig_adapters.tabular`

Do a fresh, larger pull of `api/sample_library` (already have all 1,891 as
`all_libraries_index.json`) filtered to libraries with populated
`deposition_power`/`deposition_gases`/`deposition_*_pressure_mtorr` (≈1,100–1,800 of
1,891, per the counts in §4b), then batch-pull `.../xrf` (and `.../xrd` where
`has_xrd>0`) for each — this is the only outcome sub-resource confirmed live. Treat each
LIBRARY (not sample) as the unit with a settable recipe, and each of its ≤44 XRF
POSITIONS as a combinatorial replicate with partly positional (not purely knob-driven)
variation, exactly as the doc's own PARTIAL-leaning caveat says — this materially affects
how `n` should be counted for the M0 bar. Tag `Provenance.source = real_tool` (confirmed
real NREL PVD chambers via the Scientific Data paper's authorship match). Do **not** cite
141,574 or the 4,356/7,327 library counts in any future write-up without re-verifying
against a fresh `all_libraries_index.json` pull first.

---

## Summary table

| Candidate | Downloaded | Verified against doc | Material mismatches |
|---|---|---|---|
| empa-hipims | Full raw zip (396 MB) + extracted sample | n=3,150 (doc: >3000, holds) | "6-D" is really 2 different 5-D subspaces + categorical selects; minor |
| ada-2022-09 | Full compiled CSV + 5 raw per-campaign CSVs | n=177/180 (doc: 179) | License CC-BY-4.0 claim rests on a dead link in the source repo (only MIT files actually present) |
| ada-2021-01 | Full 4 processed campaign CSVs | n=253 (doc: 253, exact match) | Same license caveat; "timestamped campaign CSVs" overstated — processed CSVs are ordinal-only |
| nrel-htem | Library index (1,891 libs) + 440-entry sample pull, no bulk file | Scale claim (141,574 / 4,356) does NOT hold live (55,067/65,779/46,905/19,797 across techniques, 1,891 libs) | **nrel.gov is dead; live host is nlr.gov ("NatLabRockies")**; per-sample API endpoint from official examples is broken; "bulk copy" link is not a downloadable file |

All four downloads/pulls succeeded (no outright failures). The nrel-htem candidate is the
one that most needs a human decision before further investment: the underlying
institution/domain appears to be mid-migration, the previously-documented scale is not
currently reproducible live, and reconciling how much of that is "not yet re-exposed"
vs. "permanently smaller" requires either waiting or contacting NREL/NatLabRockies
directly.
