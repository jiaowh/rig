# Empa bipolar-HiPIMS example (M0 lead candidate)

Real measured `recipe -> outcome` data from a real Empa sputter tool: six
BayBE-driven campaigns (3,150 rows total), Zenodo record
`10.5281/zenodo.18495402` (CC-BY-4.0; Wieczorek et al., Digital Discovery
2026, DOI 10.1039/D6DD00063K). Raw files live under
`data/m0-candidates/empa-hipims/extracted_sample/` (see
`data/m0-candidates/MANIFEST.md` section 1 for download verification).

**Honest framing:** this is the M1 gate FORM (implementation-plan §15.3 M1
row, directional binomial-CI coverage check on a real temporal split) on the
lead M0 *candidate* — it is NOT the signed M1 program gate. The M0 venue
decision is the user/PI's, and this is not the project's MBE target process.
Standing caveats: BO-driven sampling clusters rows near optima (conformal
exchangeability is an approximation; the temporal split doubles as a drift
stress test); `Ipk (A)` is measured, never set; `ti_120w_short_pw` has a
degenerate BatchNr (its "temporal" order is unverified file order) and 5 rows
3e-11 to 4e-11 outside its full-precision bounds (skip-ingested, documented).

## Files

- `prepare_empa.py` — deterministic (no-RNG) converter: raw campaign JSONs ->
  tidy per-campaign CSVs (`csv/<slug>.csv`), calibrated
  `dep_rate_A_per_s = y1 * factor` with the factor read from each campaign's
  own `calibration.txt`.
- `specs/<slug>.toml` — per-campaign WP-H process specs; bounds VERBATIM from
  each campaign's own `Campaign.json`.
- `run_m1_empa.py` — the M1-gate-form runner (this file's subject; see below).
- `results/m1_empa*.json` — machine-readable runner output.
- `tests/test_empa_ingest.py` (repo root `tests/`) — 9 tests pinning the
  converter, spec bounds, ingest behavior, and the Ti-120W edge cases.

## How to run (converter -> tests -> runner)

All commands from the repo root, on Windows with a cp1252 console — always set
`PYTHONIOENCODING=utf-8`. Add `PYTHONPATH=src` if `rig` is not pip-installed.

1. **Converter** (only needed after a raw-data or converter change; the tidy
   CSVs are checked in and byte-pinned by the tests):

   ```
   PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/prepare_empa.py
   ```

2. **Tests** (converter determinism + spec bounds + adapter ingest):

   ```
   python -m pytest tests/test_empa_ingest.py -q
   ```

3. **Runner** — the M1 gate form. Per campaign: tabular-adapter ingest ->
   temporal (BatchNr-order 60/20/20) + seeded random splits -> GP forward
   model -> split-conformal at alpha=0.10 -> per-output + pooled coverage with
   the exact binomial 95% CI (gate: nominal 0.90 inside the CI). Full runs add
   the 4-campaign PRR-space OOD check and the §8 pessimistic-inverse demo on
   `al_120w_short_pw`.

   ```
   # full 6-campaign gate (~4-6 min) -> results/m1_empa.json
   PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_m1_empa.py

   # one campaign (-> results/m1_empa.<slug>.json; OOD check skipped)
   PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_m1_empa.py --campaign al_120w_short_pw

   # fast shape check (reduced restarts; -> *.smoke.json)
   PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_m1_empa.py --campaign al_120w_short_pw --smoke
   ```

   Everything is seeded (SEED=0); recipe values in solver output are SI
   (§3.5): the `(us)` knobs are in seconds post-ingest. Bounds fed to the
   solver come from `ProcessSpec.continuous_si` — never `.continuous`
   (CLAUDE.md SI-vs-declared-bounds trap).
