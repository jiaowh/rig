"""Conformal-PID DECAYING-STEP evaluation on real Empa HiPIMS data.

LABELED SIDE STUDY (implementation-plan section 20.2 currency note): the M1
gate runner (``run_m1_empa.py``) records the online conformal-PID endpoint at
its ``step="fixed"`` library default -- that recorded run is, and stays, the
path of record (``results/m1_empa.json``, untouched by this script). This
script asks a narrower, DIRECTIONAL, side question: does the decaying-step
schedule (Angelopoulos, Barber & Bates, "Online conformal prediction with
decaying step sizes", ICML 2024 -- ``eta_t = eta * t^-(0.5+eps)``, eps=0.1
i.e. exponent 0.6, satisfying the Robbins-Monro conditions Sum(eta_t)=inf,
Sum(eta_t^2)<inf that give Polyak-Ruppert-style convergence to a fixed
population quantile under distributional stability) deliver lower
STEADY-STATE threshold volatility than the recorded fixed-step schedule,
WITHOUT losing coverage -- i.e. should ``step="decaying"`` ever become the
default?

PRE-STATED HYPOTHESIS (written BEFORE this script was run against real data;
whatever comes out below is the finding, not a target to tune towards):

  1. On the two STABLE / approximately-exchangeable streams per campaign (the
     seeded RANDOM 60/20/20 split, the exchangeable CONTROL throughout the M1
     work), decaying-step should REDUCE late-stream threshold volatility
     relative to fixed-step (a shrinking step size stops chasing per-step
     sampling noise once q_t nears its stationary value) WITHOUT pushing any
     currently-PASSing directional coverage gate to FAIL.
  2. On the one KNOWN-DRIFTING stream (`ti_200w_high_pw` TEMPORAL split -- the
     sole static-split-conformal FAILURE in the M1 gate, and the campaign
     whose fixed-step PID rolling-coverage detector already fires:
     min_full_window 0.82/0.84 < nominal 0.90, per RESULTS.md), a decaying
     step size is expected to be a LIABILITY: eta_t -> 0 makes the tracker
     adapt ever more slowly, so under a genuine, sustained BO-driven
     distribution shift decaying-step coverage may degrade relative to
     fixed-step, and/or the section 5.6 rolling-coverage detector may fire
     harder (a lower minimum, or fire where fixed-step recovers). This run
     measures whether that risk materializes on this stream -- other
     TEMPORAL streams may or may not show the same effect; they are reported
     but this is the pre-registered stress case.

Protocol (identical to run_m1_empa.py's pid_eval online path, for all 6
campaigns x both splits x both outputs): a FRESH SplitConformalCalibrator on
the SAME calibration slice as the recorded run; test rows streamed in split
order; each row's interval taken at the CURRENT controller state and scored
BEFORE observe(x, y) (an observation never influences its own interval). The
ONLY difference between the "fixed" and "decaying" runs below is the
ConformalPIDController ``step`` argument -- every other hyperparameter is the
library default (alpha_target=0.10, eta=0.10, KI=2.0, Csat=7.0, window=50,
decay_eps=0.1), uniform across all campaigns/splits, fixed before any
per-campaign outcome was seen here -- no tuning-to-pass.

FIDELITY GATE (must pass before anything is written): the "fixed" run in this
script must reproduce ``results/m1_empa.json``'s recorded PID numbers EXACTLY
(same model fit, same calibration slice, same controller, same protocol -> no
degrees of freedom differ). Checked two ways for every campaign/split: (a)
this script's own online tracer (``pid_stream``, needed because it captures
the FULL per-step threshold trace that ``run_m1_empa.pid_eval`` does not
expose, for the volatility endpoints) against the recorded JSON; (b) THE
RECORDED RUNNER'S OWN ``pid_eval`` function, imported and called directly
(not copied) on the same fitted model, against the recorded JSON -- a true
reuse-by-import cross-check. Any mismatch anywhere aborts before writing
``results/m1_empa_pid_step.json``.

Endpoints recorded per (campaign, split, output) cell, for BOTH step modes:
pooled + per-output PICP, exact-binomial 95% CI, nominal-in-CI gate flag;
n_infinite_width (asserted 0 -- finite by construction, per pid.py); the
threshold trace's steady-state volatility over the LAST 50% of the stream
(std of q_t, AND mean |delta q_t|, both stated); the section 5.6
rolling-coverage trailing-window minimum and whether it is below nominal (the
"detector fires" convention RESULTS.md already uses); the final q_t.

FILE OWNERSHIP / boundary: this script only ever READS
``results/m1_empa.json`` (the recorded reference) and ``prepare_empa.py`` /
``run_m1_empa.py`` (imported, never edited). It writes ONLY
``results/m1_empa_pid_step.json`` (new file). ``run_m1_empa.py`` and
``results/m1_empa.json`` are owned by other concurrent work and are never
modified here.

Run (Windows cp1252 console -> force UTF-8; a few minutes, one GP fit per
campaign/split, shared between the fixed and decaying traces):

    PYTHONIOENCODING=utf-8 python examples/real_data/empa_hipims/run_pid_step_study.py
        [--campaign <slug>] [--out <path.json>]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from rig.calibration.conformal import SplitConformalCalibrator
from rig.calibration.pid import ConformalPIDController
from rig.forward import records_to_arrays
from rig.schema import ureg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # run_m1_empa.py / prepare_empa.py live beside this script

# Imported, never copied: the recorded runner's own constants + functions. This
# is the mechanism that guarantees the fidelity gate has zero free parameters --
# every hyperparameter and every piece of split/ingest/fit logic below is the
# SAME object the recorded run used, not a re-typed duplicate that could drift.
from run_m1_empa import (  # noqa: E402
    ALPHA,
    CAMPAIGNS,
    NOMINAL,
    SEED,
    binom_ci,
    fit_and_eval,
    ingest_campaign,
    pid_eval,
    split_indices,
)

RESULTS_DIR = HERE / "results"
RECORDED_PATH = RESULTS_DIR / "m1_empa.json"  # READ-ONLY reference; never written
OUT_DEFAULT = RESULTS_DIR / "m1_empa_pid_step.json"

# The recorded run used gp_restarts=5 (meta.gp_restarts in m1_empa.json, "full"
# mode, not --smoke). The fitted GP is the ONE thing upstream of the PID
# controllers that involves any optimization at all; matching this exactly is
# required for the fixed-step fidelity gate to have a chance of passing.
GP_RESTARTS = 5

LATE_FRAC = 0.5  # "last 50% of the stream" per the tasking
STEPS = ("fixed", "decaying")


class FidelityGateError(RuntimeError):
    """Raised (and fatal -- no JSON written) when the fixed-step reproduction
    disagrees with results/m1_empa.json anywhere."""


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------------------
# data plumbing -- mirrors run_m1_empa.run_campaign's array/split prep exactly
# (byte for byte) so the fit/calibration/test slices are IDENTICAL to the
# recorded run. This is glue, not PID logic; the PID logic itself is either
# imported (pid_eval, for the fixed-step cross-check) or shared verbatim via
# ConformalPIDController(..., step=...) below.
# ---------------------------------------------------------------------------


def prepare_campaign_arrays(campaign):
    """(spec, X, Y, input_keys, output_keys, units, n, degenerate) for one campaign,
    outcomes in the declared (readable) unit -- see run_m1_empa.run_campaign."""
    spec, result, degenerate = ingest_campaign(campaign)
    records = result.records
    input_keys = list(spec.gp_input_keys)
    output_keys = list(spec.output_names)
    units = [o.unit for o in spec.outputs]
    if [v.name for v in spec.continuous_si] != input_keys:
        raise RuntimeError(f"{campaign.slug}: continuous_si order != gp_input_keys")
    X, Y_si = records_to_arrays(records, input_keys, output_keys)
    si_per_raw = np.array([float(ureg.Quantity(1.0, u).to_base_units().magnitude) for u in units])
    Y = Y_si / si_per_raw
    return spec, X, Y, input_keys, output_keys, units, len(records), degenerate


def campaign_splits(n: int, idx: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(fit, cal, test) index triples for "temporal" and "random" -- exactly
    run_m1_empa.run_campaign's split arithmetic (SEED + idx keys the random
    permutation, idx = the campaign's position in CAMPAIGNS)."""
    n_train, n_cal = split_indices(n)
    idx_all = np.arange(n)
    temporal = (idx_all[:n_train], idx_all[n_train : n_train + n_cal], idx_all[n_train + n_cal :])
    rng = np.random.default_rng(SEED + idx)
    perm = rng.permutation(n)
    random_split = (perm[:n_train], perm[n_train : n_train + n_cal], perm[n_train + n_cal :])
    return {"temporal": temporal, "random": random_split}


# ---------------------------------------------------------------------------
# the volatility metric (independently unit-testable on a hand-built trace)
# ---------------------------------------------------------------------------


def volatility_stats(q_trace: np.ndarray, tail_frac: float = LATE_FRAC) -> tuple[float, float]:
    """(std, mean |delta|) of ``q_trace`` over its trailing ``tail_frac`` window.

    Deterministic, pure function of the trace -- no model, no RNG. ``std`` uses
    ddof=0 (population std of the realized late-stream values, not a sample
    estimate of anything). ``mean |delta|`` is the mean absolute step-to-step
    change within that same window; a single-point tail (or empty) reports 0.0
    step-changes, not NaN.
    """
    q_trace = np.asarray(q_trace, dtype=float)
    n = q_trace.shape[0]
    start = n - int(round(n * tail_frac))
    tail = q_trace[start:]
    std = float(tail.std(ddof=0)) if tail.size else 0.0
    mad = float(np.abs(np.diff(tail)).mean()) if tail.size > 1 else 0.0
    return std, mad


# ---------------------------------------------------------------------------
# the online PID tracer -- SAME protocol as run_m1_empa.pid_eval (fresh
# calibrator on the same calibration slice; interval scored BEFORE observe),
# parameterized on ``step`` (which pid_eval hardcodes to the library default
# "fixed"), and additionally capturing the FULL per-step threshold trace that
# pid_eval's summary stats (used_min/used_max/used_mean) do not expose -- the
# volatility endpoints need the trace itself, not just its extremes/mean.
# ---------------------------------------------------------------------------


def pid_stream(
    model,
    Xc: np.ndarray,
    Yc: np.ndarray,
    Xt: np.ndarray,
    Yt: np.ndarray,
    output_keys: list[str],
    units: list[str],
    step: str,
) -> dict:
    cal = SplitConformalCalibrator(alpha=ALPHA)
    cal.fit(model, Xc, Yc)
    controller = ConformalPIDController(cal, alpha_target=ALPHA, step=step)  # library defaults else

    n_test, m = Yt.shape
    hits = np.zeros((n_test, m), dtype=bool)
    widths = np.empty((n_test, m))
    q_used = np.empty((n_test, m))
    rolling = np.empty((n_test, m))
    window = controller.window
    for t in range(n_test):
        x, y = Xt[t], Yt[t]
        q_used[t] = controller.q_t  # threshold at the CURRENT (pre-update) state
        itv = controller.interval(x)  # (m, 2) at that same pre-update threshold
        lo, hi = itv[..., 0].reshape(-1), itv[..., 1].reshape(-1)
        widths[t] = hi - lo
        miss = ((y < lo) | (y > hi)).astype(float)
        err = controller.observe(x, y)  # scores the SAME pre-update interval
        if not np.array_equal(err, miss):  # bookkeeping guard -- must never fire
            raise RuntimeError(f"PID hit/miss bookkeeping mismatch (step={step!r}) at t={t}")
        hits[t] = err == 0.0
        rolling[t] = controller.rolling_coverage

    per_output: dict[str, dict] = {}
    full = rolling[window - 1 :]  # rolling coverage over FULL windows only
    for j, key in enumerate(output_keys):
        k = int(hits[:, j].sum())
        ci_lo, ci_hi = binom_ci(k, n_test)
        finite_w = widths[np.isfinite(widths[:, j]), j]
        n_inf = int(n_test - finite_w.size)
        # finite by construction (pid.py): q_t is a tracked real threshold,
        # never a quantile index that can overflow -- this must be impossible.
        assert n_inf == 0, f"{key} (step={step!r}): n_infinite_width={n_inf}, expected 0"
        late_std, late_mad = volatility_stats(q_used[:, j])
        early_std, early_mad = volatility_stats(q_used[:, j], tail_frac=1.0 - LATE_FRAC)
        min_full = float(full[:, j].min()) if full.size else None
        per_output[key] = {
            "unit": units[j],
            "picp": float(k / n_test),
            "k_covered": k,
            "n_test": n_test,
            "ci95": [ci_lo, ci_hi],
            "nominal_in_ci": bool(ci_lo <= NOMINAL <= ci_hi),
            "mean_width": float(finite_w.mean()) if finite_w.size else None,
            "n_infinite_width": n_inf,
            "q_final": float(controller.q_t[j]),
            "q_late_stream_std": late_std,
            "q_late_stream_mean_abs_delta": late_mad,
            # informational (not a required endpoint): the FIRST half, for
            # narrating whether volatility falls over the stream at all.
            "q_first_half_std": early_std,
            "q_first_half_mean_abs_delta": early_mad,
            "rolling_coverage": {
                "window": window,
                "min_full_window": min_full,
                "final": float(rolling[-1, j]),
                # the section 5.6 concrete drift-detector convention already
                # used in RESULTS.md: trailing-window minimum below nominal.
                "detector_fires": bool(min_full is not None and min_full < NOMINAL),
            },
        }
    k_pool = int(hits.sum())
    n_pool = int(hits.size)
    p_lo, p_hi = binom_ci(k_pool, n_pool)
    return {
        "step": step,
        "hyperparameters": {
            "alpha_target": float(controller.alpha_target),
            "eta": float(controller.eta),
            "KI": float(controller.KI),
            "Csat": float(controller.Csat),
            "window": int(window),
            "step": controller.step,
            "decay_eps": float(controller.decay_eps),
        },
        "per_output": per_output,
        "pooled": {
            "picp": float(k_pool / n_pool),
            "k_covered": k_pool,
            "n_trials": n_pool,
            "ci95": [p_lo, p_hi],
            "nominal_in_ci": bool(p_lo <= NOMINAL <= p_hi),
        },
    }


# ---------------------------------------------------------------------------
# fidelity gate
# ---------------------------------------------------------------------------


def check_fidelity(slug: str, split_name: str, recorded: dict, candidate: dict) -> list[tuple]:
    """Compare ``candidate`` (this study's step="fixed" result, either the
    imported pid_eval's own return value or this script's pid_stream) against
    the recorded results/m1_empa.json pid block for (slug, split_name).

    Returns a list of (field, recorded_value, candidate_value) mismatches;
    empty means fidelity holds. Integer counts (k_covered, n_test) compare
    exactly; PICP/CI floats allow a tiny (1e-9) tolerance for JSON round-trip.
    """
    ref = recorded["campaigns"][slug]["splits"][split_name]["pid"]
    mismatches: list[tuple] = []

    def cmp_float(name, a, b, tol=1e-9):
        if abs(float(a) - float(b)) > tol:
            mismatches.append((name, a, b))

    def cmp_exact(name, a, b):
        if a != b:
            mismatches.append((name, a, b))

    cmp_exact("pooled.k_covered", ref["pooled"]["k_covered"], candidate["pooled"]["k_covered"])
    cmp_exact("pooled.n_trials", ref["pooled"]["n_trials"], candidate["pooled"]["n_trials"])
    cmp_float("pooled.picp", ref["pooled"]["picp"], candidate["pooled"]["picp"])
    for key, ref_po in ref["per_output"].items():
        cand_po = candidate["per_output"][key]
        cmp_exact(f"{key}.k_covered", ref_po["k_covered"], cand_po["k_covered"])
        cmp_exact(f"{key}.n_test", ref_po["n_test"], cand_po["n_test"])
        cmp_float(f"{key}.picp", ref_po["picp"], cand_po["picp"])
    return mismatches


def load_recorded() -> dict:
    if not RECORDED_PATH.exists():
        raise FileNotFoundError(
            f"recorded reference {RECORDED_PATH} not found -- run run_m1_empa.py first "
            "(this study never writes it, only reads it)"
        )
    return json.loads(RECORDED_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# per-cell / per-campaign driver
# ---------------------------------------------------------------------------


def cell_delta(fixed: dict, decaying: dict, output_keys: list[str]) -> dict:
    """Decaying-minus-fixed deltas the console table / recommendation reads."""
    per_output = {}
    for key in output_keys:
        f, d = fixed["per_output"][key], decaying["per_output"][key]
        per_output[key] = {
            "picp_delta": d["picp"] - f["picp"],
            "late_std_delta": d["q_late_stream_std"] - f["q_late_stream_std"],
            "late_mad_delta": d["q_late_stream_mean_abs_delta"] - f["q_late_stream_mean_abs_delta"],
            "gate_flip": f["nominal_in_ci"] != d["nominal_in_ci"],
            "detector_flip": (
                f["rolling_coverage"]["detector_fires"] != d["rolling_coverage"]["detector_fires"]
            ),
        }
    return {
        "pooled_picp_delta": decaying["pooled"]["picp"] - fixed["pooled"]["picp"],
        "pooled_gate_flip": fixed["pooled"]["nominal_in_ci"] != decaying["pooled"]["nominal_in_ci"],
        "per_output": per_output,
    }


def run_cell(campaign, idx: int, gp_restarts: int = GP_RESTARTS) -> dict:
    """Ingest + both splits for ONE campaign: one GP fit per split, shared by
    the recorded-form cross-check and both step-mode traces."""
    spec, X, Y, input_keys, output_keys, units, n, degenerate = prepare_campaign_arrays(campaign)
    splits = campaign_splits(n, idx)
    out: dict = {}
    for split_name, (fit_idx, cal_idx, test_idx) in splits.items():
        Xf, Yf = X[fit_idx], Y[fit_idx]
        Xc, Yc = X[cal_idx], Y[cal_idx]
        Xt, Yt = X[test_idx], Y[test_idx]
        _, model, _ = fit_and_eval(
            Xf, Yf, Xc, Yc, Xt, Yt, input_keys, output_keys, units, gp_restarts
        )
        # (a) the recorded runner's OWN pid_eval, imported and called directly
        # -- the true reuse-by-import fixed-step reference. Its own
        # "wall_seconds" field is wall-clock timing noise, not a computed
        # statistic (check_fidelity never reads it) -- drop it so the written
        # JSON is bit-for-bit deterministic across runs, not just numerically
        # equivalent modulo a timer.
        recorded_form = pid_eval(model, Xc, Yc, Xt, Yt, output_keys, units)
        recorded_form.pop("wall_seconds", None)
        # (b) this study's tracer, run identically for both step modes.
        traced = {
            step: pid_stream(model, Xc, Yc, Xt, Yt, output_keys, units, step) for step in STEPS
        }
        out[split_name] = {
            "degenerate_order": degenerate,
            "n_test": int(len(test_idx)),
            "output_keys": output_keys,
            "units": units,
            "recorded_form_fixed": recorded_form,
            "fixed": traced["fixed"],
            "decaying": traced["decaying"],
            "delta": cell_delta(traced["fixed"], traced["decaying"], output_keys),
        }
    return out


# ---------------------------------------------------------------------------
# console reporting
# ---------------------------------------------------------------------------


def print_cell_table(slug: str, split_name: str, cell: dict) -> None:
    label = f"{slug} / {split_name}"
    if split_name == "temporal" and cell["degenerate_order"]:
        label += " (FILE-ORDER, order key unverified)"
    print(f"\n{label}  (n_test={cell['n_test']})")
    hdr = (
        f"{'output':<18}{'PICP fix':>9}{'PICP dec':>9}{'gate f/d':>10}"
        f"{'lateSTD f':>10}{'lateSTD d':>10}{'lateMAD f':>10}{'lateMAD d':>10}"
        f"{'detect f/d':>11}{'q_final f':>10}{'q_final d':>10}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key in cell["output_keys"]:
        f = cell["fixed"]["per_output"][key]
        d = cell["decaying"]["per_output"][key]
        gate = f"{'P' if f['nominal_in_ci'] else 'F'}/{'P' if d['nominal_in_ci'] else 'F'}"
        det = (
            f"{'FIRE' if f['rolling_coverage']['detector_fires'] else 'ok'}/"
            f"{'FIRE' if d['rolling_coverage']['detector_fires'] else 'ok'}"
        )
        print(
            f"{key:<18}{f['picp']:>9.3f}{d['picp']:>9.3f}{gate:>10}"
            f"{f['q_late_stream_std']:>10.4f}{d['q_late_stream_std']:>10.4f}"
            f"{f['q_late_stream_mean_abs_delta']:>10.4f}{d['q_late_stream_mean_abs_delta']:>10.4f}"
            f"{det:>11}{f['q_final']:>10.4f}{d['q_final']:>10.4f}"
        )
    fp, dp = cell["fixed"]["pooled"], cell["decaying"]["pooled"]
    gate = f"{'P' if fp['nominal_in_ci'] else 'F'}/{'P' if dp['nominal_in_ci'] else 'F'}"
    print(f"{'POOLED':<18}{fp['picp']:>9.3f}{dp['picp']:>9.3f}{gate:>10}")


# ---------------------------------------------------------------------------
# top-level study driver
# ---------------------------------------------------------------------------


def run_study(
    campaign_slug: str | None = None, gp_restarts: int = GP_RESTARTS
) -> tuple[dict, dict]:
    """Run all selected campaigns; enforce the fidelity gate; return
    (results-by-campaign, recorded-reference-json). Raises FidelityGateError
    (nothing written) on any mismatch."""
    recorded = load_recorded()
    selected = [c for c in CAMPAIGNS if campaign_slug in (None, c.slug)]
    results: dict[str, dict] = {}
    failures: list[str] = []

    for idx, campaign in enumerate(CAMPAIGNS):
        if campaign not in selected:
            continue
        banner(f"CAMPAIGN {campaign.slug}")
        cell = run_cell(campaign, idx, gp_restarts)
        for split_name, data in cell.items():
            # cross-check (a): imported pid_eval vs the recorded JSON
            bad_a = check_fidelity(campaign.slug, split_name, recorded, data["recorded_form_fixed"])
            # cross-check (b): this study's own fixed-step tracer vs the same
            # recorded JSON (independent code path, same expected numbers)
            bad_b = check_fidelity(campaign.slug, split_name, recorded, data["fixed"])
            if bad_a:
                failures.append(f"{campaign.slug}/{split_name} pid_eval-vs-recorded: {bad_a}")
            if bad_b:
                failures.append(f"{campaign.slug}/{split_name} pid_stream-vs-recorded: {bad_b}")
            print_cell_table(campaign.slug, split_name, data)
        results[campaign.slug] = cell

    if failures:
        raise FidelityGateError(
            "FIDELITY GATE FAILED -- results/m1_empa_pid_step.json NOT written:\n"
            + "\n".join(failures)
        )
    return results, recorded


def summarize(results: dict) -> dict:
    """Roll the per-cell deltas up into the plain-language recommendation
    inputs: how many cells got quieter without losing coverage, any flips.

    Gate flips are counted at BOTH granularities and kept separate: a
    per-output flip (n=120 trials) and a POOLED flip (n=240, the tighter CI)
    are not the same event -- the pooled CI can flip even when neither
    marginal output does (it did, on this run: see the flip_details), which
    is itself a consequence of the documented "pooled CI is optimistic"
    caveat cutting the other way (a tighter CI is also more sensitive)."""
    n_cells = 0
    n_quieter_pooled = 0
    n_gate_flips = 0
    n_pooled_gate_flips = 0
    n_detector_flips = 0
    flips: list[str] = []
    for slug, camp in results.items():
        for split_name, cell in camp.items():
            n_cells += 1
            delta = cell["delta"]
            for key, po_delta in delta["per_output"].items():
                if po_delta["late_std_delta"] < 0:
                    n_quieter_pooled += 1
                if po_delta["gate_flip"]:
                    n_gate_flips += 1
                    flips.append(f"{slug}/{split_name}/{key}: per-output gate flipped")
                if po_delta["detector_flip"]:
                    n_detector_flips += 1
                    flips.append(f"{slug}/{split_name}/{key}: detector-fire status flipped")
            if delta["pooled_gate_flip"]:
                n_pooled_gate_flips += 1
                flips.append(f"{slug}/{split_name}: POOLED gate flipped")
    return {
        "n_cells": n_cells,
        "n_output_rows_quieter_late_stream": n_quieter_pooled,
        "n_gate_flips": n_gate_flips,
        "n_pooled_gate_flips": n_pooled_gate_flips,
        "n_detector_status_flips": n_detector_flips,
        "flip_details": flips,
    }


def build_payload(results: dict, recorded: dict, gp_restarts: int) -> dict:
    summary = summarize(results)
    ti_200 = results.get("ti_200w_high_pw", {}).get("temporal")
    drift_case = None
    if ti_200 is not None:
        drift_case = {
            "fixed_detector_fires": {
                k: v["rolling_coverage"]["detector_fires"]
                for k, v in ti_200["fixed"]["per_output"].items()
            },
            "decaying_detector_fires": {
                k: v["rolling_coverage"]["detector_fires"]
                for k, v in ti_200["decaying"]["per_output"].items()
            },
            "fixed_pooled_picp": ti_200["fixed"]["pooled"]["picp"],
            "decaying_pooled_picp": ti_200["decaying"]["pooled"]["picp"],
            "fixed_pooled_gate": ti_200["fixed"]["pooled"]["nominal_in_ci"],
            "decaying_pooled_gate": ti_200["decaying"]["pooled"]["nominal_in_ci"],
        }
    return {
        "meta": {
            "study": (
                "LABELED SIDE STUDY: PID decaying-step vs the recorded fixed-step path "
                "of record (results/m1_empa.json, untouched). Not itself a program gate."
            ),
            "pre_stated_hypothesis": (
                "(1) decaying-step reduces late-stream threshold volatility on the "
                "exchangeable RANDOM-split streams without flipping any PASS to FAIL; "
                "(2) on the known-drifting ti_200w_high_pw TEMPORAL stream, decaying-step "
                "risks slower adaptation and possibly worse coverage / a harder-firing "
                "section 5.6 rolling-coverage detector, because eta_t -> 0 over the stream"
            ),
            "seed": SEED,
            "alpha": ALPHA,
            "nominal_coverage": NOMINAL,
            "gp_restarts": gp_restarts,
            "late_stream_window_frac": LATE_FRAC,
            "detector_rule": (
                "trailing rolling-coverage window minimum < nominal_coverage "
                "(the convention already used in RESULTS.md for the fixed-step PID path)"
            ),
            "fidelity_gate": {
                "status": "PASS",
                "cells_checked": sum(len(camp) for camp in results.values()),
                "reference_file": "results/m1_empa.json",
                "checked": "pooled + per-output PICP/k_covered/n_test, both an imported "
                "pid_eval call and this study's own tracer, for every campaign x split",
            },
            "hyperparameters_common": {
                "alpha_target": ALPHA,
                "eta": 0.1,
                "KI": 2.0,
                "Csat": 7.0,
                "window": 50,
                "decay_eps": 0.1,
                "provenance": (
                    "ConformalPIDController library defaults, identical across both step "
                    "modes and all campaigns/splits, fixed before any outcome was seen"
                ),
            },
            "caveats": [
                "REAL measured data (real_tool), same six Empa campaigns as run_m1_empa.py; "
                "BO-driven (BayBE) sampling means split-conformal exchangeability is an "
                "approximation and the temporal split doubles as a drift stress test",
                "pooled coverage CI treats 2*n_test trials as independent though outputs "
                "share test rows (optimistic, same caveat as the recorded runs)",
                "online realized-coverage binomial CI is DIRECTIONAL (asymptotic guarantee "
                "only), same status as the recorded static/ACI/PID gates",
                "ti_120w_short_pw: temporal order key is unverified file order",
            ],
        },
        "summary": summary,
        "ti_200w_high_pw_drift_case": drift_case,
        "campaigns": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--campaign",
        choices=[c.slug for c in CAMPAIGNS],
        default=None,
        help="run ONE campaign (default: all six)",
    )
    parser.add_argument("--out", type=Path, default=None, help="results JSON path")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    banner("PID DECAYING-STEP SIDE STUDY on real Empa HiPIMS data (labeled; not the M1 gate)")
    print(__doc__.split("PRE-STATED HYPOTHESIS", 1)[1].split("Protocol", 1)[0].strip())

    t0 = time.perf_counter()
    try:
        results, recorded = run_study(args.campaign, GP_RESTARTS)
    except FidelityGateError as exc:
        print("\n" + str(exc), file=sys.stderr)
        return 1

    payload = build_payload(results, recorded, GP_RESTARTS)

    banner("SUMMARY")
    s = payload["summary"]
    print(f"cells (campaign x split): {s['n_cells']}")
    print(
        f"output-rows quieter late-stream under decaying: "
        f"{s['n_output_rows_quieter_late_stream']} / {2 * s['n_cells']}"
    )
    print(f"PASS/FAIL gate flips, per-output (fixed vs decaying): {s['n_gate_flips']}")
    print(f"PASS/FAIL gate flips, POOLED     (fixed vs decaying): {s['n_pooled_gate_flips']}")
    print(f"section 5.6 detector-fire status flips               : {s['n_detector_status_flips']}")
    for line in s["flip_details"]:
        print("  - " + line)
    if payload["ti_200w_high_pw_drift_case"] is not None:
        dc = payload["ti_200w_high_pw_drift_case"]
        print(
            f"\nti_200w_high_pw TEMPORAL (drift stress case): pooled PICP "
            f"fixed={dc['fixed_pooled_picp']:.3f} (gate "
            f"{'PASS' if dc['fixed_pooled_gate'] else 'FAIL'}) vs decaying="
            f"{dc['decaying_pooled_picp']:.3f} (gate "
            f"{'PASS' if dc['decaying_pooled_gate'] else 'FAIL'})"
        )
        print(f"  detector fires (fixed)   : {dc['fixed_detector_fires']}")
        print(f"  detector fires (decaying): {dc['decaying_detector_fires']}")

    out_path = args.out if args.out is not None else OUT_DEFAULT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nresults written -> {out_path}")
    print(f"total wall time : {time.perf_counter() - t0:.1f} s")
    banner("DONE (PID decaying-step side study)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
