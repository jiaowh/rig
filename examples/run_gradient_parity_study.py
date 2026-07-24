"""Analytic-gradient vs finite-difference PARITY study (2026-07-23).

``PessimisticInverseSolver(analytic_grad=True)`` (module docstring,
``src/rig/inverse/pessimistic.py``) replaces SciPy's finite-difference objective
gradient (``d+1`` model evaluations per L-BFGS-B step) with a closed-form one (ONE
evaluation per step). It is OPT-IN; the default stays finite differences (FD) because
``analytic_grad`` changes L-BFGS-B's SEARCH PATH, not the objective — the two paths are
equal in verdict and recipe to FD tolerance, NOT bitwise (see the ctor docstring). Making
it the default is owed but risky: it would silently move every recorded FD-path number.

This study does not flip the default. It measures whether flipping it WOULD be safe:
across dimension, seed, and target difficulty, does the analytic path ever reach a
DIFFERENT verdict than FD, and when it does, which one is right on GROUND TRUTH (never
the model's own opinion — the same circularity guard as
``run_false_success_study.py``)? The truth family, training design, and reachable-target
construction are reused BY IMPORT from ``run_false_success_study.build_cell`` (itself
built on ``run_dimensionality_study.make_truth``) — this study does not redefine the
process family, only adds a second, DELIBERATELY HARDER target class on the SAME
(truth, data) pair so target-difficulty is isolated from data variation.

Design
------
For each ``d in {2, 4, 8, 15, 20}`` and each of ``NSEEDS`` seeds:
  1. Build ONE cell (truth, Sobol training design, fitted GP) via ``build_cell`` +
     ``GPForwardModel`` — identical to the false-success study's arm A (unwrapped GP;
     the §13.2 conformal gate is orthogonal to this question and is not exercised here).
  2. Two target classes on the SAME cell (so only the spec box differs):
       * ``reachable`` — ``build_cell``'s own target: truth at a held-out on-support
         point, +-0.8 (generous; usually comfortably FEASIBLE).
       * ``hard`` — the SAME center, tolerance shrunk to +-``HARD_TOL`` (0.3): a
         boundary/marginal box, more likely to sit near the feasible/infeasible edge
         where a search-path difference between FD and analytic could plausibly flip
         the verdict.
  3. Solve TWICE per (cell, target): FD (``analytic_grad=False``, the default) and
     analytic (``analytic_grad=True``) — same model, same ``variables``, same spec, same
     ``n_restarts``/``max_iter`` (an explicit, FIXED budget for BOTH arms — see
     ``COMPUTE NOTE`` below), same binding solver policy
     (``kappa=z_epi=2.0, delta_frac=0.02``, imported from
     ``run_false_success_study.SOLVER_KW``), same ``seed``.
  4. Score the pair with :func:`score_pair` (the unit-tested core, see
     ``tests/test_gradient_parity.py``): verdict agreement; for agreeing FEASIBLE pairs,
     normalized recipe distance, margin difference (recomputed from the PUBLIC
     ``ForwardModel`` protocol — ``predict``/``jacobian``/``support_score`` — not solver
     internals), and whether EACH arm's own top candidate genuinely hits ground truth
     (near-equal margins with different recipes is fine if both genuinely hit); for
     disagreeing pairs, a drill-down naming which arm was right per ground truth.

COMPUTE NOTE (why the budget is smaller than production default)
------------------------------------------------------------------
Production ``n_restarts=None`` scales as ``24*dim`` (480 at d=20) — measured at ~150 s
FD per solve at d=20 (``docs/dimensionality-2026-07-17.md``). This study fixes
``n_restarts=16, max_iter=40`` for BOTH arms at every d (measured: FD ~23 s, analytic
~2.5 s at d=20, n=240) so the ~150-combo grid completes in the ~30-45 min compute budget.
The reduced budget is IDENTICAL across arms at every (d, seed, target) — the parity
question (do the two paths AGREE) does not depend on the absolute restart count, only on
it being shared. It does mean this study is not a re-measurement of the production
speedup ratio (that number is in the false-success study / dimensionality doc); the
speedup ratio reported here is directional evidence at a smaller, shared budget.

Deterministic (§13.4): every RNG draw derives from ``(d, seed)`` via ``build_cell`` and
the solvers' own ``seed=``. ``--smoke`` runs a tiny grid TWICE and asserts byte-identity.

Reproduce (full grid, ~30-45 min): python examples/run_gradient_parity_study.py --full
Smoke + determinism (seconds):     python examples/run_gradient_parity_study.py --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# reuse the house truth family + reachable-target construction; do NOT duplicate it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_false_success_study import (  # noqa: E402
    GP_FIT_RESTARTS,
    HI,
    LO,
    SOLVER_KW,
    build_cell,
    clopper_pearson,
    evaluate_on_truth,
)

from rig.forward import GPForwardModel  # noqa: E402
from rig.interfaces import Infeasible  # noqa: E402
from rig.inverse.pessimistic import PessimisticInverseSolver  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console guard

# --------------------------------------------------------------------------
# fixed experimental design
# --------------------------------------------------------------------------

D_VALUES = (2, 4, 8, 15, 20)
N_TRAIN = {2: 24, 4: 48, 8: 96, 15: 180, 20: 240}  # the house 12*d scaling
HARD_TOL = 0.3  # boundary/marginal spec box (vs build_cell's own +-0.8 "reachable")
N_RESTARTS = 16  # FIXED, shared by both arms at every d (see COMPUTE NOTE)
MAX_ITER = 40  # FIXED, shared by both arms at every d
NSEEDS_DEFAULT = 15

OUT_IDX = np.array([0, 1])  # OUT_KEYS order ("y0", "y1"), both constrained everywhere


# --------------------------------------------------------------------------
# target classes on ONE cell: reachable (from build_cell) + hard (same center,
# tighter tolerance) — isolates target difficulty from data/truth variation.
# --------------------------------------------------------------------------


def make_hard_spec(lower: np.ndarray, upper: np.ndarray, hard_tol: float = HARD_TOL):
    """Shrink a symmetric spec box to +-``hard_tol`` around the SAME center.

    ``build_cell``'s box is exactly ``y_ref +- TOL`` with a fixed module-level ``TOL``,
    so the center recovers ``y_ref`` exactly regardless of ``TOL``'s value: reusing
    ``(lower+upper)/2`` here means the hard target is centered on the identical
    reachable-target point, never a re-drawn one.
    """
    center = (np.asarray(lower, float) + np.asarray(upper, float)) / 2.0
    return center - hard_tol, center + hard_tol


def spec_from_bounds(lower: np.ndarray, upper: np.ndarray) -> dict:
    return {
        "targets": {
            "y0": (float(lower[0]), float(upper[0])),
            "y1": (float(lower[1]), float(upper[1])),
        },
        "max_candidates": 3,
    }


# --------------------------------------------------------------------------
# margin, recomputed from the PUBLIC ForwardModel protocol only (predict /
# jacobian / support_score) — never the solver's private _margins/_Restart.
# Mirrors the §8.1/§8.4/§8.5 formula the solver implements internally, so this
# script's parity check does not depend on (or duplicate) the solver's private
# state, only on the same public contract every ForwardModel backend honors.
# --------------------------------------------------------------------------


def public_margin(
    model,
    recipe: dict,
    dim: int,
    out_idx: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    kappa: float,
    z_epi: float,
    delta_frac: float,
    delta_raw: np.ndarray,
) -> float:
    """The §8 pessimistic margin at ``recipe``, from public predict/jacobian/
    support_score only. ``support_score`` itself is not part of the margin formula
    (it is a separate hard/soft manifold screen) so it is not read here."""
    x = np.array([recipe[f"x{i}"] for i in range(dim)], dtype=float)
    dist = model.predict(x)
    mu = np.atleast_1d(np.asarray(dist.mean, dtype=float))
    sig_ale = np.atleast_1d(np.asarray(dist.aleatoric_sigma, dtype=float))
    sig_epi = np.atleast_1d(np.asarray(dist.epistemic_sigma, dtype=float))
    s = z_epi * sig_epi
    if delta_frac > 0.0:
        jac = np.atleast_2d(np.asarray(model.jacobian(x), dtype=float))
        s = s + np.abs(jac) @ delta_raw
    scale = np.maximum(sig_ale, 1e-9 * (np.abs(mu) + 1.0))
    mu_s, s_s, sc = mu[out_idx], s[out_idx], scale[out_idx]
    u_hi = (upper - mu_s - s_s) / sc
    u_lo = (mu_s - lower - s_s) / sc
    return float(min(np.min(u_hi), np.min(u_lo)))


# --------------------------------------------------------------------------
# the scorer — pure, unit-tested (tests/test_gradient_parity.py)
# --------------------------------------------------------------------------


def score_pair(
    fd_res,
    an_res,
    *,
    truth,
    dim: int,
    lower: np.ndarray,
    upper: np.ndarray,
    model,
    kappa: float,
    z_epi: float,
    delta_frac: float,
    delta_raw: np.ndarray,
    flat_scale: np.ndarray,
) -> dict:
    """Compare one FD/analytic solve pair on the SAME (model, spec, seed).

    Ground truth (never the model's own opinion) is the authority for "which arm was
    right": a FEASIBLE verdict whose top candidate misses ground truth is scored as a
    false success, not a win, even though the solver certified it.
    """
    v_fd = "INFEASIBLE" if isinstance(fd_res, Infeasible) else "FEASIBLE"
    v_an = "INFEASIBLE" if isinstance(an_res, Infeasible) else "FEASIBLE"
    rec: dict = {"verdict_fd": v_fd, "verdict_an": v_an, "verdict_agree": v_fd == v_an}

    if v_fd == "FEASIBLE":
        top = fd_res[0]
        hit, exc, _y = evaluate_on_truth(top.recipe, truth, dim, lower, upper)
        rec["hit_fd"] = hit
        rec["excursion_fd"] = exc
        rec["confidence_fd"] = float(top.confidence)
        rec["margin_fd"] = public_margin(
            model,
            top.recipe,
            dim,
            OUT_IDX,
            lower,
            upper,
            kappa=kappa,
            z_epi=z_epi,
            delta_frac=delta_frac,
            delta_raw=delta_raw,
        )
    if v_an == "FEASIBLE":
        top = an_res[0]
        hit, exc, _y = evaluate_on_truth(top.recipe, truth, dim, lower, upper)
        rec["hit_an"] = hit
        rec["excursion_an"] = exc
        rec["confidence_an"] = float(top.confidence)
        rec["margin_an"] = public_margin(
            model,
            top.recipe,
            dim,
            OUT_IDX,
            lower,
            upper,
            kappa=kappa,
            z_epi=z_epi,
            delta_frac=delta_frac,
            delta_raw=delta_raw,
        )

    if rec["verdict_agree"] and v_fd == "FEASIBLE":
        x_fd = np.array([fd_res[0].recipe[f"x{i}"] for i in range(dim)], dtype=float)
        x_an = np.array([an_res[0].recipe[f"x{i}"] for i in range(dim)], dtype=float)
        rec["recipe_distance_normalized"] = float(
            np.linalg.norm((x_fd - x_an) / flat_scale) / np.sqrt(dim)
        )
        rec["margin_diff"] = abs(rec["margin_fd"] - rec["margin_an"])
        rec["gt_both_hit"] = bool(rec["hit_fd"] and rec["hit_an"])
        rec["gt_both_miss"] = bool((not rec["hit_fd"]) and (not rec["hit_an"]))
        rec["gt_split"] = bool(rec["hit_fd"] != rec["hit_an"])

    if not rec["verdict_agree"]:
        if v_fd == "FEASIBLE":
            # FD certified something, analytic abstained.
            rec["disagreement_type"] = "fd_feasible_an_infeasible"
            rec["verdict_favors"] = (
                # FD's certified recipe genuinely hits truth: analytic's abstention was
                # NEEDLESS (evidence AGAINST flipping — flipping would have lost this hit).
                "an_false_abstention"
                if rec["hit_fd"]
                # FD's certified recipe MISSES truth (a false success): analytic's
                # abstention, whatever its cause, did not ship one (evidence FOR flipping).
                else "fd_false_success"
            )
        else:
            # analytic certified something, FD abstained.
            rec["disagreement_type"] = "an_feasible_fd_infeasible"
            rec["verdict_favors"] = (
                # analytic's certified recipe genuinely hits truth: FD's abstention was
                # NEEDLESS (evidence FOR flipping — analytic found a hit FD missed).
                "fd_false_abstention"
                if rec["hit_an"]
                # analytic's certified recipe MISSES truth (a false success): evidence
                # AGAINST flipping.
                else "an_false_success"
            )
    return rec


# --------------------------------------------------------------------------
# one (d, seed) cell: shared GP, both target classes, both arms
# --------------------------------------------------------------------------


def make_solver(model, variables, X_train, *, analytic_grad: bool, seed: int):
    return PessimisticInverseSolver(
        model,
        variables=variables,
        output_keys=["y0", "y1"],
        X_train=X_train,
        n_restarts=N_RESTARTS,
        max_iter=MAX_ITER,
        analytic_grad=analytic_grad,
        seed=seed,
        **SOLVER_KW,
    )


def run_cell(d: int, n: int, seed: int) -> list[dict]:
    """One (d, seed): ONE fitted GP, TWO target classes, TWO arms each -> 2 score
    records ("reachable", "hard"). Returns a list of per-target-class result dicts."""
    truth, variables, X, Y, spec_reach, lower_reach, upper_reach = build_cell(d, n, seed)
    t0 = time.time()
    gp = GPForwardModel(n_restarts=GP_FIT_RESTARTS, seed=seed).fit(X, Y)
    t_fit = time.time() - t0

    lower_hard, upper_hard = make_hard_spec(lower_reach, upper_reach)
    spec_hard = spec_from_bounds(lower_hard, upper_hard)

    delta_raw = np.full(d, SOLVER_KW["delta_frac"] * (HI - LO))
    flat_scale = np.full(d, HI - LO)

    out = []
    for tag, spec, lower, upper in (
        ("reachable", spec_reach, lower_reach, upper_reach),
        ("hard", spec_hard, lower_hard, upper_hard),
    ):
        fd_solver = make_solver(gp, variables, X, analytic_grad=False, seed=seed)
        t0 = time.time()
        fd_res = fd_solver.solve(spec)
        t_fd = time.time() - t0

        an_solver = make_solver(gp, variables, X, analytic_grad=True, seed=seed)
        t0 = time.time()
        an_res = an_solver.solve(spec)
        t_an = time.time() - t0

        rec = score_pair(
            fd_res,
            an_res,
            truth=truth,
            dim=d,
            lower=lower,
            upper=upper,
            model=gp,
            kappa=SOLVER_KW["kappa"],
            z_epi=SOLVER_KW["z_epi"],
            delta_frac=SOLVER_KW["delta_frac"],
            delta_raw=delta_raw,
            flat_scale=flat_scale,
        )
        rec.update(
            d=d,
            n=n,
            seed=seed,
            target_class=tag,
            t_fit=t_fit,
            t_fd=t_fd,
            t_an=t_an,
        )
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# grid + aggregation
# --------------------------------------------------------------------------


def _strip_timing(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("t_fit", "t_fd", "t_an")}


def run_grid(nseeds: int, *, base_seed: int = 0, d_values=D_VALUES, verbose: bool = True) -> dict:
    """Deterministic given (base_seed, nseeds): every sub-seed derives from
    base_seed + seed_index, and every RNG draw inside a cell derives from (d, seed)."""
    per_seed: list[dict] = []
    for d in d_values:
        n = N_TRAIN[d]
        for si in range(nseeds):
            seed = base_seed + si
            recs = run_cell(d, n, seed)
            per_seed.extend(recs)
            if verbose:
                for r in recs:
                    print(
                        f"  d={d:2d} n={n:4d} seed={seed:2d} {r['target_class']:9s} "
                        f"fd={r['verdict_fd']:>11} an={r['verdict_an']:>11} "
                        f"agree={r['verdict_agree']!s:5} "
                        f"t_fd={r['t_fd']:.1f}s t_an={r['t_an']:.1f}s",
                        flush=True,
                    )
    return {"per_seed": per_seed}


def deterministic_view(grid: dict) -> str:
    view = [_strip_timing(r) for r in grid["per_seed"]]
    return json.dumps(view, sort_keys=True, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def aggregate(per_seed: list[dict]) -> dict:
    """Roll up per-(d, target_class) endpoints: verdict agreement (exact binomial CI),
    ground-truth outcome agreement among agreeing-FEASIBLE pairs, wall-time ratio, and
    the disagreement drill-down list."""
    by_cell: dict[str, list[dict]] = {}
    for r in per_seed:
        key = f"d{r['d']}_{r['target_class']}"
        by_cell.setdefault(key, []).append(r)

    cells = {}
    disagreements = []
    for key, recs in sorted(by_cell.items()):
        n = len(recs)
        n_agree = sum(1 for r in recs if r["verdict_agree"])
        agree_lo, agree_hi = clopper_pearson(n_agree, n)

        agree_feasible = [r for r in recs if r["verdict_agree"] and r["verdict_fd"] == "FEASIBLE"]
        n_af = len(agree_feasible)
        n_both_hit = sum(1 for r in agree_feasible if r.get("gt_both_hit"))
        n_both_miss = sum(1 for r in agree_feasible if r.get("gt_both_miss"))
        n_split = sum(1 for r in agree_feasible if r.get("gt_split"))
        mean_dist = (
            float(np.mean([r["recipe_distance_normalized"] for r in agree_feasible]))
            if agree_feasible
            else float("nan")
        )
        mean_margin_diff = (
            float(np.mean([r["margin_diff"] for r in agree_feasible]))
            if agree_feasible
            else float("nan")
        )
        max_margin_diff = (
            float(np.max([r["margin_diff"] for r in agree_feasible]))
            if agree_feasible
            else float("nan")
        )

        cell_disagree = [r for r in recs if not r["verdict_agree"]]
        for r in cell_disagree:
            disagreements.append(
                {
                    "d": r["d"],
                    "seed": r["seed"],
                    "target_class": r["target_class"],
                    "verdict_fd": r["verdict_fd"],
                    "verdict_an": r["verdict_an"],
                    "disagreement_type": r["disagreement_type"],
                    "verdict_favors": r["verdict_favors"],
                }
            )

        cells[key] = {
            "d": recs[0]["d"],
            "target_class": recs[0]["target_class"],
            "n_pairs": n,
            "n_verdict_agree": n_agree,
            "verdict_agreement_rate": n_agree / n if n else float("nan"),
            "verdict_agreement_ci95": [agree_lo, agree_hi],
            "n_agree_feasible": n_af,
            "gt_both_hit": n_both_hit,
            "gt_both_miss": n_both_miss,
            "gt_split": n_split,
            "gt_agreement_rate_among_agree_feasible": (
                (n_both_hit + n_both_miss) / n_af if n_af else float("nan")
            ),
            "mean_recipe_distance_normalized": mean_dist,
            "mean_margin_diff": mean_margin_diff,
            "max_margin_diff": max_margin_diff,
            "n_disagree": len(cell_disagree),
            "mean_t_fd": float(np.mean([r["t_fd"] for r in recs])),
            "mean_t_an": float(np.mean([r["t_an"] for r in recs])),
            "speedup_fd_over_an": (
                float(np.mean([r["t_fd"] for r in recs]) / np.mean([r["t_an"] for r in recs]))
            ),
        }

    by_d = {}
    for d in sorted({r["d"] for r in per_seed}):
        recs_d = [r for r in per_seed if r["d"] == d]
        by_d[str(d)] = {
            "n_pairs": len(recs_d),
            "n_verdict_agree": sum(1 for r in recs_d if r["verdict_agree"]),
            "mean_t_fd": float(np.mean([r["t_fd"] for r in recs_d])),
            "mean_t_an": float(np.mean([r["t_an"] for r in recs_d])),
            "speedup_fd_over_an": float(
                np.mean([r["t_fd"] for r in recs_d]) / np.mean([r["t_an"] for r in recs_d])
            ),
        }

    n_total = len(per_seed)
    n_agree_total = sum(1 for r in per_seed if r["verdict_agree"])
    lo, hi = clopper_pearson(n_agree_total, n_total)
    # FOR the flip: analytic found a genuine hit FD missed (false abstention), or
    # analytic's abstention avoided a false success FD would have shipped.
    n_for_flip = sum(
        1
        for d in disagreements
        if d["verdict_favors"] in ("fd_false_abstention", "fd_false_success")
    )
    # AGAINST the flip: FD found a genuine hit analytic missed (false abstention), or
    # FD's abstention avoided a false success analytic would have shipped.
    n_against_flip = sum(
        1
        for d in disagreements
        if d["verdict_favors"] in ("an_false_abstention", "an_false_success")
    )
    favors_counts: dict[str, int] = {}
    for d in disagreements:
        favors_counts[d["verdict_favors"]] = favors_counts.get(d["verdict_favors"], 0) + 1

    return {
        "cells": cells,
        "by_d": by_d,
        "disagreements": disagreements,
        "overall": {
            "n_pairs": n_total,
            "n_verdict_agree": n_agree_total,
            "verdict_agreement_rate": n_agree_total / n_total if n_total else float("nan"),
            "verdict_agreement_ci95": [lo, hi],
            "n_disagreements": len(disagreements),
            "disagreement_breakdown": favors_counts,
            # verdict_favors semantics (see score_pair): "fd_false_abstention" /
            # "fd_false_success" are evidence FOR flipping the default to analytic;
            # "an_false_abstention" / "an_false_success" are evidence AGAINST it.
            "n_disagreements_favoring_flip": n_for_flip,
            "n_disagreements_favoring_keep_fd": n_against_flip,
        },
    }


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def smoke() -> bool:
    """Tiny grid (d=2, 2 seeds) run TWICE; assert byte-identical. Returns True on PASS."""
    g1 = run_grid(2, d_values=(2,), verbose=False)
    g2 = run_grid(2, d_values=(2,), verbose=False)
    v1, v2 = deterministic_view(g1), deterministic_view(g2)
    ok = v1 == v2
    print(f"[smoke] determinism double-run byte-identical: {ok}", flush=True)
    agg = aggregate(g1["per_seed"])
    print(
        f"[smoke] d2 verdict agreement: {agg['overall']['verdict_agreement_rate']:.2f} "
        f"({agg['overall']['n_verdict_agree']}/{agg['overall']['n_pairs']})",
        flush=True,
    )
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="run the full powered grid")
    ap.add_argument("--smoke", action="store_true", help="smoke + determinism check")
    ap.add_argument("--nseeds", type=int, default=NSEEDS_DEFAULT)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.smoke:
        ok = smoke()
        sys.exit(0 if ok else 1)

    if not args.full:
        ap.print_help()
        return

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_json = args.out_json or os.path.join(repo, "docs", "gradient-parity.json")
    t0 = time.time()
    print(
        f"[full] grid: {len(D_VALUES)} d-values x 2 target-classes x {args.nseeds} seeds x 2 arms, "
        f"n_restarts={N_RESTARTS} max_iter={MAX_ITER}",
        flush=True,
    )
    grid = run_grid(args.nseeds, verbose=True)
    agg = aggregate(grid["per_seed"])
    wall = time.time() - t0
    payload = {
        "study": "analytic-gradient-vs-fd-parity",
        "date": "2026-07-23",
        "design": {
            "d_values": list(D_VALUES),
            "n_train": N_TRAIN,
            "nseeds": args.nseeds,
            "target_classes": {
                "reachable": "build_cell's own +-0.8 box around truth at a held-out point",
                "hard": f"same center, +-{HARD_TOL} box (boundary/marginal)",
            },
            "n_restarts": N_RESTARTS,
            "max_iter": MAX_ITER,
            "solver_policy": SOLVER_KW,
            "gp_fit_restarts": GP_FIT_RESTARTS,
            "ci": "Clopper-Pearson 95%",
        },
        "aggregate": agg,
        "per_seed": [_strip_timing(r) for r in grid["per_seed"]],
        "wall_seconds": wall,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    print(f"[full] wrote {out_json} ({wall / 60:.1f} min)", flush=True)
    print(
        f"[full] overall verdict agreement: "
        f"{agg['overall']['verdict_agreement_rate']:.3f} "
        f"({agg['overall']['n_verdict_agree']}/{agg['overall']['n_pairs']}), "
        f"disagreements: {agg['overall']['n_disagreements']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
