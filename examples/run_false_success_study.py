"""False-success rate vs input dimension — the gate OFF vs ON, measured (2026-07-23).

Safety-central study. RIG's central promise is "no false successes": a CERTIFIED
recipe must really land in spec on the TRUE function, not merely on the surrogate's
opinion of itself. The known crack (docs/dimensionality-2026-07-17.md): at d=20 with
800 runs, ONE certified recipe missed ground truth (worst_err 1.046 vs a +-0.8 box) —
a real, deterministic FALSE SUCCESS whose mechanism is that the §8 pessimistic κ·σ
margins read the GP's RAW σ, which at high d was optimistic (one candidate's true error
was 3.12× its own claimed σ). Since 2026-07-22 the §13.2 conformal containment gate
C(x) ⊆ Z* runs BY DEFAULT whenever the solver's model is conformal-wrapped
(`_conformal_screen` in src/rig/inverse/pessimistic.py), and every candidate carries a
`calibration_status`. This study MEASURES what that fix buys and what it costs.

============================================================================
PRE-REGISTERED QUESTION (stated before any result exists)
============================================================================
Does conformal-wrapping the surrogate (arm B: the §13.2 C(x) ⊆ Z* gate default-on)
reduce the rate of CERTIFIED false successes relative to the raw-σ solver (arm A: an
unwrapped GP, gate structurally inert), and at what COST in abstention and genuine-hit
rate?

Directional hypotheses, fixed in advance (either direction is a finding):
  H1  arm B's certified false-success rate <= arm A's, at every d (the gate can only
      REMOVE candidates whose calibrated band spills the spec box, never add a miss).
  H2  arm B's abstention rate >= arm A's (removing candidates can only push solves
      toward INFEASIBLE) — this is the price of the gate.
  H3  arm B's genuine-hit rate per certified candidate >= arm A's (the survivors are a
      conformally-screened subset).

Honesty guards written in before results:
  * The wrapped arm pays a REAL cost: its calibration split is carved out of the SAME
    run budget (n_cal from n), so its surrogate is fit on FEWER points (effective train
    n reported per cell). This is the honest cost of calibrated acceptance, not a free
    lunch on extra data.
  * If certified false successes are too RARE to measure at this NSEEDS (0 observed),
    the reported result is the 95% Clopper-Pearson UPPER BOUND on the rate, explicitly
    NOT a claim of zero (lesson: infinite-width / hollow passes and n=1 anecdotes).
  * A wrapped cell whose calibration block is too small to form a FINITE conformal
    quantile (ceil((1-α)(n_cal+1)) > n_cal) has an infinite band: the containment gate
    then rejects EVERYTHING (over-abstains). Detected and reported, excluded from the
    headline gate-effect claim rather than silently scored.

The crime scene (d=20, n=800, the exact 2026-07-17 config: FD search, 48 restarts,
GP n_restarts=3, seed 0) is reproduced separately (`crime_scene_reproduction`) to answer
directly: does the original miss still reproduce, and does arm B kill it?

Reproduce (full grid, detached, ~2h): python examples/run_false_success_study.py --full
Smoke + determinism (seconds):        python examples/run_false_success_study.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
from scipy.stats import beta, qmc

# reuse the house ground-truth family + design pattern; do NOT duplicate it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_dimensionality_study import make_truth  # noqa: E402

from rig.calibration.conformal import (  # noqa: E402
    DEFAULT_ALPHA,
    ConformalForwardModel,
    SplitConformalCalibrator,
)
from rig.forward import GPForwardModel  # noqa: E402
from rig.interfaces import ContinuousVariable, Infeasible, RecipeCandidate  # noqa: E402
from rig.inverse.pessimistic import PessimisticInverseSolver  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console guard

# --------------------------------------------------------------------------
# fixed experimental design
# --------------------------------------------------------------------------

LO, HI = -2.0, 2.0
OUT_KEYS = ("y0", "y1")
TOL = np.array([0.8, 0.8])  # the +-0.8 spec box half-width (house choice)
NOISE_SIGMA = 0.05  # training-label noise (house choice)
GP_FIT_RESTARTS = 3  # GPForwardModel hyperparameter-fit multi-start (house choice)
CAL_FRAC = 1.0 / 3.0  # wrapped arm: calibration block carved from the SAME budget

# binding solver policy (§8.4/§8.1/§8.5): kappa / z_epi / delta_frac = 2.0 / 2.0 / 0.02.
SOLVER_KW = dict(kappa=2.0, z_epi=2.0, delta_frac=0.02)

# The powered grid. n = 12·d for the first four (the established scaling); the fifth is
# the d=20 / n=800 CRIME SCENE that produced the original deterministic false success.
CELLS = (
    {"d": 2, "n": 24, "tag": "12d"},
    {"d": 8, "n": 96, "tag": "12d"},
    {"d": 15, "n": 180, "tag": "12d"},
    {"d": 20, "n": 240, "tag": "12d"},
    {"d": 20, "n": 800, "tag": "crime-scene"},
)


# --------------------------------------------------------------------------
# ground-truth scorer — the safety-central, unit-tested core
# --------------------------------------------------------------------------


def evaluate_on_truth(
    recipe: dict, truth, dim: int, lower: np.ndarray, upper: np.ndarray
) -> tuple[bool, float, np.ndarray]:
    """Score ONE recipe against the TRUE function (never the surrogate).

    Returns ``(inbox, excursion, y_true)``:
      * ``inbox`` — the true outcome lies inside the spec box on every constrained
        output. For a CERTIFIED candidate (``feasibility_flag=True``), ``inbox=False``
        is a FALSE SUCCESS: the solver certified a recipe the true function misses.
      * ``excursion`` — max distance (raw output units) the true outcome falls OUTSIDE
        the box over the constrained outputs; 0.0 exactly when ``inbox``.
      * ``y_true`` — the true outcome vector, for the record.

    Pure and deterministic: ``truth`` is called here and ONLY here on the returned
    recipe — the model's own opinion never enters scoring (the circularity guard).
    """
    x = np.array([recipe[f"x{i}"] for i in range(dim)], dtype=float)
    y = np.atleast_2d(np.asarray(truth(x), dtype=float))[0]  # (m,)
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    excursion = float(np.max(np.maximum(below, above)))
    inbox = excursion <= 0.0
    return inbox, excursion, y


def score_result(result, truth, dim: int, lower: np.ndarray, upper: np.ndarray) -> dict:
    """Classify a solver result against ground truth.

    A returned candidate is CERTIFIED (``feasibility_flag=True`` by construction of the
    §8 solver, asserted here). Among certified candidates, one that misses the spec box
    on truth is a FALSE SUCCESS; one that lands in it is a GENUINE HIT. An INFEASIBLE
    verdict is an abstention (no candidate, no false success possible).

    Returns a per-solve dict with the counts and the worst certified-miss excursion.
    """
    if isinstance(result, Infeasible):
        return {
            "status": "INFEASIBLE",
            "n_cand": 0,
            "n_hit": 0,
            "n_false_success": 0,
            "worst_miss_excursion": 0.0,
            "reason_category": _reason_category(result.reason),
            "reason": result.reason[:100],
        }
    n_hit = 0
    n_fs = 0
    worst_miss = 0.0
    per_cand = []
    for c in result:
        assert isinstance(c, RecipeCandidate)
        # every returned candidate is certified — a non-certified one would make
        # "false success" meaningless. Fail loud if the contract ever changes.
        assert c.feasibility_flag is True, "solver returned a non-certified candidate"
        inbox, exc, _y = evaluate_on_truth(c.recipe, truth, dim, lower, upper)
        if inbox:
            n_hit += 1
        else:
            n_fs += 1
            worst_miss = max(worst_miss, exc)
        per_cand.append({"inbox": inbox, "excursion": exc, "status": c.calibration_status})
    return {
        "status": "FEASIBLE",
        "n_cand": len(result),
        "n_hit": n_hit,
        "n_false_success": n_fs,
        "worst_miss_excursion": worst_miss,
        "calibration_status": result[0].calibration_status,
        "per_cand": per_cand,
    }


def _reason_category(reason: str) -> str:
    """Bucket an Infeasible reason so abstention COST can be attributed."""
    r = reason.lower()
    if "conformal" in r or "c(x)" in reason:
        return "conformal"
    if "epistemic" in r or "collect more runs" in r:
        return "epistemic"
    if "genuinely unreachable" in r:
        return "unreachable"
    if "manifold" in r or "support floor" in r:
        return "off-manifold"
    return "other"


# --------------------------------------------------------------------------
# exact (Clopper-Pearson) binomial CI
# --------------------------------------------------------------------------


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial confidence interval for ``k`` of ``n`` at confidence ``1-alpha``.

    Handles the k=0 and k=n edges (a 0 count returns a genuine upper bound, NEVER a
    point claim of zero — the pre-registered honesty guard). Returns (lo, hi) in [0, 1];
    (nan, nan) when n == 0.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


# --------------------------------------------------------------------------
# cell construction + arms
# --------------------------------------------------------------------------


def build_cell(d: int, n_train: int, seed: int):
    """Truth family + training design + reachable spec box for one (d, n, seed).

    Mirrors the house pattern (run_dimensionality_study.probe): a smooth d-dim,
    2-output process with every input dim active; Sobol training design over the box +
    N(0, 0.05) label noise; a REACHABLE target = the true function at a held-out
    on-support point, +-0.8 on both outputs. The truth FAMILY varies per seed
    (``make_truth(d, seed)``), so each seed is an independent draw from the process
    family AND its data noise — the false-success rate is then a population estimate
    over the family, not an artifact of one fixed function.
    """
    truth = make_truth(d, seed)
    variables = [ContinuousVariable(f"x{i}", LO, HI) for i in range(d)]
    sob = qmc.Sobol(d=d, scramble=True, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = qmc.scale(sob.random(n_train), [LO] * d, [HI] * d)
    rng = np.random.default_rng(seed + 1)
    Y = truth(X) + rng.normal(0, NOISE_SIGMA, size=(n_train, 2))
    x_ref = qmc.scale(qmc.Sobol(d=d, scramble=True, seed=seed + 99).random(1), [LO] * d, [HI] * d)[
        0
    ]
    y_ref = truth(x_ref)[0]
    lower = y_ref - TOL
    upper = y_ref + TOL
    spec = {
        "targets": {
            "y0": (float(lower[0]), float(upper[0])),
            "y1": (float(lower[1]), float(upper[1])),
        },
        "max_candidates": 3,
    }
    return truth, variables, X, Y, spec, lower, upper


def make_solver(model, variables, X_train, *, n_restarts, analytic_grad, seed):
    return PessimisticInverseSolver(
        model,
        variables=variables,
        output_keys=list(OUT_KEYS),
        X_train=X_train,
        n_restarts=n_restarts,
        analytic_grad=analytic_grad,
        seed=seed,
        **SOLVER_KW,
    )


def run_arm_raw(cell, seed, *, n_restarts, analytic_grad) -> dict:
    """Arm A — RAW: solver on the UNWRAPPED GP fit on all n runs. The §13.2 gate is
    structurally inert (no conformal_set), so candidates are 'model-feasible': the raw-σ
    κ pessimism is the only acceptance test — the mechanism under study."""
    d, n = cell["d"], cell["n"]
    truth, variables, X, Y, spec, lower, upper = build_cell(d, n, seed)
    t0 = time.time()
    gp = GPForwardModel(n_restarts=GP_FIT_RESTARTS, seed=seed).fit(X, Y)
    t_fit = time.time() - t0
    solver = make_solver(
        gp, variables, X, n_restarts=n_restarts, analytic_grad=analytic_grad, seed=seed
    )
    t0 = time.time()
    try:
        res = solver.solve(spec)
    except Exception as e:  # noqa: BLE001
        return {
            "status": f"CRASH: {type(e).__name__}: {e}",
            "n_cand": 0,
            "n_hit": 0,
            "n_false_success": 0,
            "t_fit": t_fit,
            "t_solve": time.time() - t0,
        }
    rec = score_result(res, truth, d, lower, upper)
    rec.update(t_fit=t_fit, t_solve=time.time() - t0, eff_train_n=int(X.shape[0]), n_cal=0)
    return rec


def run_arm_wrapped(cell, seed, *, n_restarts, analytic_grad) -> dict:
    """Arm B — WRAPPED: the SAME GP, wrapped by the REAL split-conformal stack
    (SplitConformalCalibrator + ConformalForwardModel) on a held-out calibration block
    carved from the SAME budget (n_cal = round(n/3)); the surrogate is therefore fit on
    n - n_cal points (eff_train_n, reported). The §13.2 C(x) ⊆ Z* gate runs default-on,
    so survivors are 'conformal-checked'. An infinite conformal quantile (cal too small)
    is flagged: the gate then rejects everything (over-abstains)."""
    d, n = cell["d"], cell["n"]
    truth, variables, X, Y, spec, lower, upper = build_cell(d, n, seed)
    n_cal = round(n * CAL_FRAC)
    # deterministic held-out split: last n_cal Sobol rows for calibration.
    Xtr, Ytr = X[:-n_cal], Y[:-n_cal]
    Xcal, Ycal = X[-n_cal:], Y[-n_cal:]
    t0 = time.time()
    gp = GPForwardModel(n_restarts=GP_FIT_RESTARTS, seed=seed).fit(Xtr, Ytr)
    cal = SplitConformalCalibrator(alpha=DEFAULT_ALPHA)
    cal.fit(gp, Xcal, Ycal)
    band_finite = bool(np.all(np.isfinite(cal.kappa())))
    model = ConformalForwardModel(gp, cal)
    t_fit = time.time() - t0
    solver = make_solver(
        model, variables, Xtr, n_restarts=n_restarts, analytic_grad=analytic_grad, seed=seed
    )
    t0 = time.time()
    try:
        res = solver.solve(spec)
    except Exception as e:  # noqa: BLE001
        return {
            "status": f"CRASH: {type(e).__name__}: {e}",
            "n_cand": 0,
            "n_hit": 0,
            "n_false_success": 0,
            "t_fit": t_fit,
            "t_solve": time.time() - t0,
            "eff_train_n": int(Xtr.shape[0]),
            "n_cal": int(n_cal),
            "band_finite": band_finite,
        }
    rec = score_result(res, truth, d, lower, upper)
    rec.update(
        t_fit=t_fit,
        t_solve=time.time() - t0,
        eff_train_n=int(Xtr.shape[0]),
        n_cal=int(n_cal),
        band_finite=band_finite,
    )
    return rec


ARMS = {"raw": run_arm_raw, "wrapped": run_arm_wrapped}


# --------------------------------------------------------------------------
# grid aggregation
# --------------------------------------------------------------------------


def aggregate_cell(records: list[dict]) -> dict:
    """Roll up per-seed records for one (cell, arm) into the reported endpoints."""
    n_seeds = len(records)
    feasible = [r for r in records if r["status"] == "FEASIBLE"]
    infeasible = [r for r in records if r["status"] == "INFEASIBLE"]
    crashes = [r for r in records if str(r["status"]).startswith("CRASH")]
    n_cand = sum(r["n_cand"] for r in feasible)
    n_hit = sum(r["n_hit"] for r in feasible)
    n_fs = sum(r["n_false_success"] for r in feasible)
    n_seeds_with_fs = sum(1 for r in feasible if r["n_false_success"] > 0)
    worst_miss = max((r["worst_miss_excursion"] for r in feasible), default=0.0)
    reason_counts: dict[str, int] = {}
    for r in infeasible:
        c = r.get("reason_category", "other")
        reason_counts[c] = reason_counts.get(c, 0) + 1
    fs_lo_c, fs_hi_c = clopper_pearson(n_fs, n_cand) if n_cand else (float("nan"), float("nan"))
    fs_lo_s, fs_hi_s = (
        clopper_pearson(n_seeds_with_fs, n_seeds) if n_seeds else (float("nan"), float("nan"))
    )
    return {
        "n_seeds": n_seeds,
        "n_feasible": len(feasible),
        "n_infeasible": len(infeasible),
        "n_crash": len(crashes),
        "abstention_rate": (len(infeasible) / n_seeds) if n_seeds else float("nan"),
        "n_candidates": n_cand,
        "n_hits": n_hit,
        "n_false_success": n_fs,
        "n_seeds_with_false_success": n_seeds_with_fs,
        "false_success_rate_per_candidate": (n_fs / n_cand) if n_cand else float("nan"),
        "fsr_per_candidate_ci95": [fs_lo_c, fs_hi_c],
        "false_success_rate_per_seed": (n_seeds_with_fs / n_seeds) if n_seeds else float("nan"),
        "fsr_per_seed_ci95": [fs_lo_s, fs_hi_s],
        "genuine_hit_rate_per_candidate": (n_hit / n_cand) if n_cand else float("nan"),
        "worst_miss_excursion": worst_miss,
        "abstention_reasons": reason_counts,
        "eff_train_n": int(np.median([r.get("eff_train_n", 0) for r in records])) if records else 0,
        "n_cal": int(np.median([r.get("n_cal", 0) for r in records])) if records else 0,
        "n_band_infinite": sum(1 for r in records if r.get("band_finite") is False),
        "mean_t_solve": float(np.mean([r.get("t_solve", 0.0) for r in records]))
        if records
        else 0.0,
    }


def run_grid(
    nseeds: int,
    *,
    n_restarts,
    analytic_grad,
    base_seed: int = 0,
    cells=CELLS,
    verbose: bool = True,
    checkpoint_path: str | None = None,
) -> dict:
    """Run the full (cell × arm × seed) grid. Deterministic given (base_seed, nseeds,
    n_restarts, analytic_grad): every sub-seed derives from base_seed + seed_index.

    When ``checkpoint_path`` is set, the partial grid is dumped after EVERY (cell, arm)
    so a late crash in a multi-hour detached run never loses completed work."""
    results: dict[str, dict] = {}
    per_seed_store: dict[str, list] = {}
    for cell in cells:
        key = f"d{cell['d']}_n{cell['n']}"
        for arm, fn in ARMS.items():
            recs = []
            for si in range(nseeds):
                seed = base_seed + si
                r = fn(cell, seed, n_restarts=n_restarts, analytic_grad=analytic_grad)
                r["seed"] = seed
                recs.append(r)
                if verbose:
                    print(
                        f"  {key:12s} {arm:8s} seed={seed:2d} {r['status']:>11} "
                        f"cand={r.get('n_cand', 0)} hit={r.get('n_hit', 0)} "
                        f"fs={r.get('n_false_success', 0)} "
                        f"t={r.get('t_solve', 0.0):.0f}s",
                        flush=True,
                    )
            results.setdefault(key, {})[arm] = aggregate_cell(recs)
            per_seed_store.setdefault(key, {})[arm] = [_strip_timing(r) for r in recs]
            if checkpoint_path is not None:
                with open(checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"partial": True, "results": results, "per_seed": per_seed_store},
                        f,
                        indent=2,
                        default=_json_default,
                    )
    return {"results": results, "per_seed": per_seed_store}


def _strip_timing(r: dict) -> dict:
    """A deterministic view of a per-seed record (wall-clock stripped), for the
    byte-identity determinism check."""
    return {k: v for k, v in r.items() if k not in ("t_fit", "t_solve", "mean_t_solve")}


def deterministic_view(grid: dict) -> str:
    """Canonical JSON of the timing-free grid, for byte-identity comparison."""
    view = {
        "per_seed": grid["per_seed"],
        "results": {
            k: {
                arm: {kk: vv for kk, vv in cell.items() if kk != "mean_t_solve"}
                for arm, cell in arms.items()
            }
            for k, arms in grid["results"].items()
        },
    }
    return json.dumps(view, sort_keys=True, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# --------------------------------------------------------------------------
# crime-scene reproduction (exact 2026-07-17 config)
# --------------------------------------------------------------------------


def crime_scene_reproduction() -> dict:
    """Reproduce the exact 2026-07-17 d=20/n=800 config that produced the original
    deterministic false success: FD search (analytic_grad=False), 48 solver restarts,
    GP n_restarts=3, seed 0, +-0.8 box. Then run arm B (wrapped, default-on §13.2 gate)
    on the SAME (d, n, seed) and report whether the gate kills the miss."""
    cell = {"d": 20, "n": 800}
    raw = run_arm_raw(cell, 0, n_restarts=48, analytic_grad=False)
    wrapped = run_arm_wrapped(cell, 0, n_restarts=48, analytic_grad=False)
    return {
        "config": "d=20 n=800 seed=0 FD restarts=48 GP_restarts=3",
        "raw": _strip_timing(raw),
        "wrapped": _strip_timing(wrapped),
    }


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def smoke(n_restarts=48, analytic_grad=True) -> bool:
    """Tiny end-to-end grid run TWICE; assert the deterministic view is byte-identical.
    Returns True on PASS."""
    cells = ({"d": 2, "n": 24}, {"d": 8, "n": 96})
    g1 = run_grid(2, n_restarts=n_restarts, analytic_grad=analytic_grad, cells=cells, verbose=False)
    g2 = run_grid(2, n_restarts=n_restarts, analytic_grad=analytic_grad, cells=cells, verbose=False)
    v1, v2 = deterministic_view(g1), deterministic_view(g2)
    ok = v1 == v2
    print(f"[smoke] determinism double-run byte-identical: {ok}", flush=True)
    print(
        f"[smoke] d2 raw fsr/cand={g1['results']['d2_n24']['raw']['false_success_rate_per_candidate']}"
        f"  d8 wrapped abstention={g1['results']['d8_n96']['wrapped']['abstention_rate']}",
        flush=True,
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="run the full powered grid")
    ap.add_argument("--smoke", action="store_true", help="smoke + determinism check")
    ap.add_argument("--nseeds", type=int, default=20)
    ap.add_argument(
        "--n-restarts",
        type=int,
        default=48,
        help="solver multi-start budget (48 = house/crime-scene config)",
    )
    ap.add_argument(
        "--analytic-grad",
        action="store_true",
        default=True,
        help="analytic objective gradient (the established high-d speedup)",
    )
    ap.add_argument(
        "--fd",
        dest="analytic_grad",
        action="store_false",
        help="finite-difference search (the default solver path)",
    )
    ap.add_argument("--out-json", default=None)
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="write partial grid after each (cell, arm) so a late crash loses nothing",
    )
    args = ap.parse_args()

    if args.smoke:
        ok = smoke()
        sys.exit(0 if ok else 1)

    if not args.full:
        ap.print_help()
        return

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_json = args.out_json or os.path.join(repo, "docs", "false-success-study.json")
    t0 = time.time()
    print(
        f"[full] grid: {len(CELLS)} cells x 2 arms x {args.nseeds} seeds, "
        f"n_restarts={args.n_restarts} analytic_grad={args.analytic_grad}",
        flush=True,
    )
    grid = run_grid(
        args.nseeds,
        n_restarts=args.n_restarts,
        analytic_grad=args.analytic_grad,
        cells=CELLS,
        verbose=True,
        checkpoint_path=args.checkpoint,
    )
    print("[full] crime-scene reproduction (exact 2026-07-17 config)...", flush=True)
    crime = crime_scene_reproduction()
    wall = time.time() - t0
    payload = {
        "study": "false-success-rate-vs-dimension",
        "date": "2026-07-23",
        "design": {
            "cells": [dict(c) for c in CELLS],
            "nseeds": args.nseeds,
            "arms": {
                "raw": "unwrapped GP (gate inert, model-feasible)",
                "wrapped": "split-conformal wrapped GP (§13.2 gate default-on)",
            },
            "solver_policy": SOLVER_KW,
            "n_restarts": args.n_restarts,
            "analytic_grad": args.analytic_grad,
            "gp_fit_restarts": GP_FIT_RESTARTS,
            "cal_frac": CAL_FRAC,
            "spec_tol": TOL.tolist(),
            "ci": "Clopper-Pearson 95%",
        },
        "results": grid["results"],
        "crime_scene": crime,
        "wall_seconds": wall,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[full] wrote {out_json} ({wall / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
