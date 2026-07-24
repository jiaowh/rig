"""M3 acceptance **v2** — the HONEST re-run (implementation-plan §15.4 gate: *"amortized
proposal matches per-query quality after refinement"*).

Why a v2 at all — the four audit critiques of v1
=================================================
The v1 harness (``examples/run_m3_acceptance.py`` + ``docs/M3-acceptance-2026-07-17.md``,
verdict PASS 5/5) was audited as **near-tautological**. This script is a from-scratch
re-run that fixes each critique; v1 is left byte-for-byte untouched (it still reproduces
the recorded toy-tanh result it was).

1. **Saturating pass rule.** v1's rule was ``d2_light_conf >= cold_heavy_conf - 0.02``
   while every confidence saturated at 0.9997-1.0 — almost no room to fail. **v2 scores
   arms by GROUND-TRUTH hit COUNTS only**: solve, then run the machine at the returned
   recipe and check the outcome lands in the spec box. Confidences never enter the
   verdict (:func:`m3_verdict` reads only the ``*_top_hit`` booleans). See the
   ``tests/test_m3_acceptance_v2.py`` non-saturation unit test.
2. **No gap for amortization to fill.** v1's ``cold_light`` already succeeded on 4/5
   targets, so the "amortization fills the gap" claim rested on n=1. **v2 pre-registers
   its targets via a cheap pre-probe** (:func:`select_targets`): it deliberately
   includes targets where the light cold solver FAILS (boundary pre-images that a single
   box-centre start cannot reach), plus HIT controls — so a gap provably exists on the
   shared set, and the comparison has room to show one. Selection uses ONLY ``cold_light``
   (never ``cold_heavy``/``d2``), so it cannot manufacture a d2 win.
3. **Toy tanh, not the machine.** v1 ran on a hand-built ``tanh`` function. **v2 runs on
   the WP-B :class:`~rig_adapters.mbe.machine.InSilicoMachine`** (calibrated MBE physics,
   ``metrology_noise`` ON like M2). The coupled spec channel is
   ``T_center x bow_cooldown_um`` (the same non-identity pair M2 uses: ``T_center`` pins
   ``T_heater``; ``bow_cooldown_um`` pins ``film_thickness`` given ``T_heater``).
4. **Scored against the surrogate.** v1 scored feasibility against the same GP that
   trained the generator. **v2 scores every arm against GROUND TRUTH** — the noise-free
   machine physics evaluated at the returned recipe (the binding BUILD_STATE rule,
   2026-07-17). The GP is only the object the §8 solver refines against; the verdict never
   reads the surrogate's own opinion of its recipe.

The experiment (unchanged QUESTION from v1)
===========================================
Does an amortized generator + one D2 light refinement recover what a cold HEAVY §8 solve
achieves, at a fraction of the budget — *where a gap actually exists*? Three arms solve
the SAME spec against the SAME GP model, differing only in search budget / warm-start:

  * ``cold_heavy`` — §8 solver, full cold Sobol multi-start (the per-query gold standard);
  * ``cold_light`` — same solver, 1 start (box centre) = the cut budget;
  * ``d2_light``   — D2: the amortized generator proposes ``n_proposals`` recipes that
    warm-start the SAME 1-start solver (``rig.inverse.AmortizedRefiner``).

**§14.6 gate is blocking.** The amortized generator's SBC/TARP :meth:`validate` gate must
PASS before any arm runs; a failed gate aborts with the gate's diagnosis (full mode).
Calibration attaches to the amortized proposal via that gate — never to the refined output.

**Pass rule (non-saturating, ground-truth).** The M3 claim holds iff, on the shared
target set, ``d2_light``'s ground-truth top-1 hit count is ``>=`` ``cold_heavy``'s AND
strictly ``>`` ``cold_light``'s. If d2_light does not beat cold_light, M3's amortization
value is NOT demonstrated on this machine — and the doc says so.

Run: ``python examples/run_m3_acceptance_v2.py`` (writes ``docs/m3-acceptance-v2.json``).
``--smoke`` runs a tiny, fast, deterministic config (gate recorded but not enforced, so
the whole arm/scoring/verdict path is exercised). Deterministic (seeded); the JSON's
``timing`` block is the only non-deterministic content and is excluded from the
determinism digest printed at the end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rig.forward import GPForwardModel, records_to_arrays  # noqa: E402
from rig.interfaces import Infeasible  # noqa: E402
from rig.inverse import (  # noqa: E402
    AmortizedInverseGenerator,
    AmortizedRefiner,
    PessimisticInverseSolver,
)

RESULT_JSON = Path(__file__).resolve().parents[1] / "docs" / "m3-acceptance-v2.json"

# The coupled, non-identity spec channel (M2's blessed pair on this machine):
# T_center pins T_heater; bow_cooldown_um pins film_thickness given T_heater.
SPEC_OUTPUTS = ("T_center", "bow_cooldown_um")


# ---------------------------------------------------------------------------
# configuration (all seeds + budgets in one place; --smoke swaps the sizes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    smoke: bool
    # model outputs the GP + generator learn (superset of SPEC_OUTPUTS). Chosen as the
    # non-near-deterministic channels: thickness_grown (SNR ~220, a near-noiseless
    # film_thickness identity) POISONS SBC calibration (teaches an overconfident
    # film_thickness posterior), so it is excluded; the SBC/TARP gate would not pass
    # with it in. T_center + bow_cooldown_um both carry a genuine, learnable posterior
    # width and jointly pin (T_heater, film_thickness).
    okeys: tuple[str, ...]
    n_train: int
    # generator (zuko NSF deep ensemble)
    n_members: int
    transforms: int
    hidden: tuple[int, ...]
    max_epochs: int
    region_hw: tuple[float, float]
    # §14.6 gate
    gate_n_sim: int
    gate_n_posterior: int
    gate_enforced: bool  # full: abort on fail; smoke: record but continue
    # §8 solver budgets (all arms share the binding §8 policy kappa=z_epi=2, delta=0.02)
    heavy_restarts: int
    light_restarts: int
    n_proposals: int
    max_candidates: int
    # target pre-probe / selection
    n_sobol_refs: int
    tol_std_frac: float  # box half-width = tol_std_frac * std(Y_train) per output
    k_miss: int  # # of cold_light-MISS targets to select (the gap)
    k_hit: int  # # of cold_light-HIT targets to select (controls)
    # seeds (each stream independent, §13.4)
    seed_train_design: int = 0
    seed_data_machine: int = 1000
    seed_gen: int = 0
    seed_gate: int = 0
    seed_gate_sim: int = 7
    seed_solver: int = 0
    seed_refs: int = 20
    seed_sigma: int = 4242


def full_config() -> Config:
    return Config(
        smoke=False,
        okeys=SPEC_OUTPUTS,
        # N=1024: v1 used 220 on a 2->2 TOY and never ran the §14.6 gate; v2 must PASS it on
        # the real 2->2 MBE map. 1024 is the smallest power-of-two in the config sweep that
        # clears SBC/TARP with margin (sbc_p=[0.90, 0.39], tarp_err=0.025); 512 was marginal
        # (SBC dim-2 p=0.022). Machine cost stays trivial (~1024 fast-path evals). This is a
        # gate-PREREQUISITE choice, not tuning-to-pass the arm verdict.
        n_train=1024,
        n_members=5,
        transforms=4,
        hidden=(128, 128),
        max_epochs=400,
        region_hw=(0.25, 2.5),
        gate_n_sim=200,
        gate_n_posterior=100,
        gate_enforced=True,
        heavy_restarts=48,  # = library default 24*dim at dim=2 (the full budget)
        light_restarts=1,  # box centre only (the cut budget)
        n_proposals=8,
        max_candidates=4,
        n_sobol_refs=20,
        tol_std_frac=0.5,
        k_miss=3,
        k_hit=3,
    )


def smoke_config() -> Config:
    return Config(
        smoke=True,
        okeys=SPEC_OUTPUTS,
        n_train=64,
        n_members=1,
        transforms=2,
        hidden=(32, 32),
        max_epochs=25,
        region_hw=(0.25, 2.0),
        gate_n_sim=40,
        gate_n_posterior=20,
        gate_enforced=False,  # tiny flow won't calibrate; exercise the path, don't abort
        heavy_restarts=6,
        light_restarts=1,
        n_proposals=3,
        max_candidates=3,
        n_sobol_refs=4,
        tol_std_frac=0.5,
        k_miss=1,
        k_hit=1,
    )


# ---------------------------------------------------------------------------
# machine helpers
# ---------------------------------------------------------------------------


def _mbe_bits():
    """Import the WP-B MBE adapter/machine lazily (keeps import-time torch/sim-free)."""
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    return RECIPE_VARIABLES, make_adapter, InSilicoMachine, PathologyConfig


def make_oracle_eval(clean_machine, ikeys, okeys):
    """Ground-truth evaluator: the NOISE-FREE machine physics at a recipe, returned in
    the ``okeys`` SI output order. Deterministic and call-order-independent (a clean
    machine has no noise / drift / hidden state), so it is the honest "true function"
    the returned recipes are scored against — never the surrogate."""

    def oracle(recipe):
        rec = clean_machine.run(recipe)
        _, y = records_to_arrays([rec], ikeys, okeys)
        return y[0]

    return oracle


def measure_metrology_sigma(make_noisy, ikeys, okeys, recipe, n_reps, seed):
    """Empirical per-output metrology sigma in the sweep's SI output space (as M2)."""
    m = make_noisy(seed)
    reps = np.array([records_to_arrays([m.run(recipe)], ikeys, okeys)[1][0] for _ in range(n_reps)])
    return reps.std(axis=0, ddof=1)


# ---------------------------------------------------------------------------
# scoring against ground truth (never the surrogate)
# ---------------------------------------------------------------------------


def score_result(res, oracle, out_idx, box_lo, box_hi):
    """Score one arm's :class:`InverseResult` against GROUND TRUTH.

    ``out_idx`` maps each spec output to its column in the oracle's output vector;
    ``box_lo``/``box_hi`` are the spec box on those outputs. Runs the oracle (the
    noise-free machine) at each returned recipe and records whether its TRUE outcome
    lands in the box. ``top_hit`` — the ranked-best candidate's ground-truth hit — is
    the ship-the-best success used by the verdict; ``any_hit`` is reported alongside.
    An :class:`Infeasible` verdict is a (recorded) miss.
    """
    if isinstance(res, Infeasible):
        return {
            "status": "INFEASIBLE",
            "feasible": False,
            "n_candidates": 0,
            "top_hit": False,
            "any_hit": False,
            "top_conf": None,
            "distance_to_feasible": float(res.distance_to_feasible),
            "reason": str(res.reason)[:160],
            "top_recipe": {k: float(v) for k, v in dict(res.nearest_achievable).items()},
            "top_outcome": None,
        }
    hits: list[bool] = []
    outcomes: list[list[float]] = []
    for c in res:
        y = oracle(c.recipe)
        yb = y[out_idx]
        inbox = bool(np.all(yb >= box_lo) and np.all(yb <= box_hi))
        hits.append(inbox)
        outcomes.append([float(v) for v in yb])
    return {
        "status": "FEASIBLE",
        "feasible": True,
        "n_candidates": len(res),
        "top_hit": bool(hits[0]),
        "any_hit": bool(any(hits)),
        "top_conf": float(res[0].confidence),
        "top_recipe": {k: float(v) for k, v in dict(res[0].recipe).items()},
        "top_outcome": outcomes[0],
        "all_hits": hits,
    }


def m3_verdict(rows):
    """The NON-SATURATING M3 pass rule. Reads ONLY the per-target ground-truth top-1
    hit booleans (``cold_heavy_top_hit`` / ``cold_light_top_hit`` / ``d2_light_top_hit``)
    — confidences are deliberately NOT arguments, so no saturated confidence can move the
    verdict (the v1 tautology). PASS iff d2_light's ground-truth hit count is >= cold_heavy's
    AND strictly > cold_light's (the gap amortization is supposed to fill, measured where
    a gap exists)."""
    gt_heavy = sum(1 for r in rows if r["cold_heavy_top_hit"])
    gt_light = sum(1 for r in rows if r["cold_light_top_hit"])
    gt_d2 = sum(1 for r in rows if r["d2_light_top_hit"])
    d2_ge_heavy = gt_d2 >= gt_heavy
    d2_gt_light = gt_d2 > gt_light
    return {
        "gate_pass": bool(d2_ge_heavy and d2_gt_light),
        "n_targets": len(rows),
        "gt_hits_cold_heavy": gt_heavy,
        "gt_hits_cold_light": gt_light,
        "gt_hits_d2_light": gt_d2,
        "d2_ge_heavy": bool(d2_ge_heavy),
        "d2_gt_light": bool(d2_gt_light),
    }


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


@dataclass
class _Built:
    """Everything the arms need, assembled once (deterministic)."""

    cfg: Config
    variables: list
    ikeys: list[str]
    okeys: list[str]
    out_idx: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    y_std: np.ndarray
    forward: GPForwardModel
    gen: AmortizedInverseGenerator
    oracle: object
    gate: dict
    sigma: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _make_solver(built: _Built, n_restarts: int):
    """§8 pessimistic solver, BINDING policy (kappa=z_epi=2.0, delta_frac=0.02). All three
    arms share this — they differ ONLY in ``n_restarts`` (and, for d2, warm starts)."""
    return PessimisticInverseSolver(
        built.forward,
        built.variables,
        built.okeys,
        X_train=built.X,
        n_restarts=n_restarts,
        seed=built.cfg.seed_solver,
    )


def build(cfg: Config) -> _Built:
    RECIPE_VARIABLES, make_adapter, InSilicoMachine, PathologyConfig = _mbe_bits()
    adapter = make_adapter()
    variables = list(RECIPE_VARIABLES)
    ikeys = [v.name for v in RECIPE_VARIABLES]
    okeys = list(cfg.okeys)
    out_idx = np.array([okeys.index(o) for o in SPEC_OUTPUTS], dtype=int)

    def make_noisy(seed):
        return InSilicoMachine(
            config=PathologyConfig(metrology_noise=True), seed=seed, adapter=adapter
        )

    # 1) training design on the NOISY machine (surrogate + generator must cope w/ noise)
    data_machine = make_noisy(cfg.seed_data_machine)
    recipes = adapter.seed_design(cfg.n_train, cfg.seed_train_design)
    runs = [data_machine.run(r) for r in recipes]
    X, Y = records_to_arrays(runs, ikeys, okeys)
    y_std = Y.std(axis=0)

    print(f"fitting GP forward tier on {cfg.n_train} noisy in-silico runs ...", flush=True)
    forward = GPForwardModel(seed=cfg.seed_gen).fit(X, Y)
    print(
        f"training amortized generator (zuko NSF x{cfg.n_members}, {cfg.max_epochs} epochs) ...",
        flush=True,
    )
    gen = AmortizedInverseGenerator(
        variables,
        okeys,
        n_members=cfg.n_members,
        transforms=cfg.transforms,
        hidden=cfg.hidden,
        max_epochs=cfg.max_epochs,
        region_hw=cfg.region_hw,
        seed=cfg.seed_gen,
    ).fit(X, Y)

    # 2) §14.6 SBC/TARP blocking gate — certify the amortized proposal relative to the
    # (noisy) generative model it was trained under; default prior bootstraps the
    # empirical training-u rows. A fresh seeded noisy machine keeps validate deterministic.
    gate_machine = make_noisy(cfg.seed_gate_sim)

    def gate_sim(recipe):
        return records_to_arrays([gate_machine.run(recipe)], ikeys, okeys)[1][0]

    print("running §14.6 SBC/TARP blocking gate on the generator ...", flush=True)
    gate_res = gen.validate(
        gate_sim, n_sim=cfg.gate_n_sim, n_posterior=cfg.gate_n_posterior, seed=cfg.seed_gate
    )
    gate = {
        "passed": bool(gate_res.passed),
        "sbc_passed": bool(gate_res.sbc_passed),
        "tarp_passed": bool(gate_res.tarp_passed),
        "sbc_p_values": [float(p) for p in gate_res.sbc_p_values],
        "tarp_max_calibration_error": float(gate_res.tarp_max_calibration_error),
        "n_sim": cfg.gate_n_sim,
        "n_posterior": cfg.gate_n_posterior,
        "enforced": cfg.gate_enforced,
    }

    oracle = make_oracle_eval(
        InSilicoMachine(config=PathologyConfig(), seed=0, adapter=adapter), ikeys, okeys
    )
    sigma = measure_metrology_sigma(
        make_noisy,
        ikeys,
        okeys,
        recipes[len(recipes) // 2],
        200 if not cfg.smoke else 40,
        cfg.seed_sigma,
    )
    return _Built(
        cfg, variables, ikeys, okeys, out_idx, X, Y, y_std, forward, gen, oracle, gate, sigma
    )


def _reference_recipes(built: _Built):
    """Pre-registered candidate reference recipes (SI): 4 near-corners FIRST (their
    coupled boxes have boundary pre-images that a single centre start struggles to
    reach — the cold_light-hard generators), then a Sobol interior spread. Order is
    fixed, so selection is mechanical."""
    RECIPE_VARIABLES, make_adapter, _, _ = _mbe_bits()
    lo = np.array([v.lower for v in RECIPE_VARIABLES], dtype=float)
    hi = np.array([v.upper for v in RECIPE_VARIABLES], dtype=float)
    names = [v.name for v in RECIPE_VARIABLES]
    refs = []
    for a in (0.04, 0.96):
        for b in (0.04, 0.96):
            refs.append(
                {
                    names[0]: float(lo[0] + a * (hi[0] - lo[0])),
                    names[1]: float(lo[1] + b * (hi[1] - lo[1])),
                }
            )
    refs.extend(make_adapter().seed_design(built.cfg.n_sobol_refs, built.cfg.seed_refs))
    return refs


def _box_for(built: _Built, ref_recipe):
    """Coupled (T_center, bow) box around the ref's ORACLE outcome, half-width
    ``tol_std_frac * std(Y_train)`` per output. Reachable by construction (the ref
    recipe hits its own box centre on the clean machine). ``tol_std_frac`` in standardized
    units = the box half-width the generator's region-augmentation was trained on."""
    y = built.oracle(ref_recipe)
    tol = built.cfg.tol_std_frac * built.y_std[built.out_idx]
    center = y[built.out_idx]
    box_lo = center - tol
    box_hi = center + tol
    spec_targets = {
        SPEC_OUTPUTS[j]: (float(box_lo[j]), float(box_hi[j])) for j in range(len(SPEC_OUTPUTS))
    }
    return spec_targets, box_lo, box_hi


def select_targets(built: _Built):
    """PRE-REGISTERED target selection via a cheap ``cold_light`` pre-probe.

    Rule (mechanical, derived ONLY from cold_light — never cold_heavy/d2, so it cannot
    manufacture a d2 win): for each candidate reference (corners first, then Sobol), build
    the coupled box, run the 1-start ``cold_light`` solver, and ground-truth-score its
    top-1. Classify HIT / MISS. Select the first ``k_miss`` MISS references and the first
    ``k_hit`` HIT references in pool order. Abort if either class is short — the comparison
    needs the split.
    """
    light = _make_solver(built, built.cfg.light_restarts)
    refs = _reference_recipes(built)
    classified = []
    for i, ref in enumerate(refs):
        spec_targets, box_lo, box_hi = _box_for(built, ref)
        spec = {"targets": spec_targets, "max_candidates": built.cfg.max_candidates}
        res = light.solve(spec)
        sc = score_result(res, built.oracle, built.out_idx, box_lo, box_hi)
        cls = "HIT" if sc["top_hit"] else "MISS"
        classified.append(
            {
                "idx": i,
                "reference_recipe": {k: float(v) for k, v in ref.items()},
                "box": spec_targets,
                "preprobe_class": cls,
                "preprobe_status": sc["status"],
            }
        )
        print(
            f"  pre-probe ref[{i}] {cls} ({sc['status']}, top_conf="
            f"{'-' if sc['top_conf'] is None else round(sc['top_conf'], 3)})",
            flush=True,
        )
    miss = [c for c in classified if c["preprobe_class"] == "MISS"]
    hit = [c for c in classified if c["preprobe_class"] == "HIT"]
    if len(miss) < built.cfg.k_miss or len(hit) < built.cfg.k_hit:
        if built.cfg.smoke:
            # SMOKE ONLY: the tiny undertrained GP's classification is environment-
            # sensitive; pad from the pool (by index) so the path still exercises arms +
            # scoring + verdict deterministically. The FULL run keeps the strict guard
            # below (it has a wide margin: ~10 MISS / ~14 HIT in the 24-candidate pool).
            need = built.cfg.k_miss + built.cfg.k_hit
            selected = miss + hit + classified
            seen: set[int] = set()
            uniq = []
            for c in selected:
                if c["idx"] not in seen:
                    seen.add(c["idx"])
                    uniq.append(c)
            return classified, uniq[:need]
        raise SystemExit(
            f"pre-probe could not form the target split: need {built.cfg.k_miss} MISS + "
            f"{built.cfg.k_hit} HIT, found {len(miss)} MISS + {len(hit)} HIT. "
            "cold_light is either too weak or too strong at this tol; adjust tol_std_frac "
            "or the reference pool (reportable outcome, not a silent pass)."
        )
    selected = miss[: built.cfg.k_miss] + hit[: built.cfg.k_hit]
    return classified, selected


def run_arms(built: _Built, selected):
    """Run the three arms on each SELECTED (frozen, pre-registered) target and score every
    one against ground truth. Returns (rows, per-arm wall times)."""
    cfg = built.cfg
    heavy = _make_solver(built, cfg.heavy_restarts)
    light = _make_solver(built, cfg.light_restarts)
    d2 = AmortizedRefiner(
        built.gen, _make_solver(built, cfg.light_restarts), n_proposals=cfg.n_proposals
    )
    times = {"cold_heavy": 0.0, "cold_light": 0.0, "d2_light": 0.0}
    rows = []
    for t in selected:
        spec = {"targets": t["box"], "max_candidates": cfg.max_candidates}
        box_lo = np.array([t["box"][o][0] for o in SPEC_OUTPUTS])
        box_hi = np.array([t["box"][o][1] for o in SPEC_OUTPUTS])
        r_heavy, d_heavy = _timed_solve(heavy, spec)
        r_light, d_light = _timed_solve(light, spec)
        r_d2, d_d2 = _timed_solve(d2, spec)
        times["cold_heavy"] += d_heavy
        times["cold_light"] += d_light
        times["d2_light"] += d_d2
        s_heavy = score_result(r_heavy, built.oracle, built.out_idx, box_lo, box_hi)
        s_light = score_result(r_light, built.oracle, built.out_idx, box_lo, box_hi)
        s_d2 = score_result(r_d2, built.oracle, built.out_idx, box_lo, box_hi)
        row = {
            "target_idx": t["idx"],
            "reference_recipe": t["reference_recipe"],
            "box": t["box"],
            "preprobe_class": t["preprobe_class"],
            "cold_heavy": s_heavy,
            "cold_light": s_light,
            "d2_light": s_d2,
            "cold_heavy_top_hit": s_heavy["top_hit"],
            "cold_light_top_hit": s_light["top_hit"],
            "d2_light_top_hit": s_d2["top_hit"],
        }
        rows.append(row)
        print(
            f"  target[{t['idx']}] ({t['preprobe_class']}): "
            f"heavy_hit={s_heavy['top_hit']} light_hit={s_light['top_hit']} "
            f"d2_hit={s_d2['top_hit']}  "
            f"(conf heavy={_rc(s_heavy)} light={_rc(s_light)} d2={_rc(s_d2)})",
            flush=True,
        )
    return rows, times


def _timed_solve(solver, spec):
    """Solve and return (result, wall_seconds). Module-level so no closure binds the loop
    variable (keeps the timing honest and the linter happy)."""
    t0 = time.perf_counter()
    res = solver.solve(spec)
    return res, time.perf_counter() - t0


def _rc(sc):
    return "-" if sc["top_conf"] is None else round(sc["top_conf"], 4)


def _determinism_digest(result: dict) -> str:
    """sha256 of the result with the non-deterministic ``timing`` block removed — the
    thing the smoke double-run must reproduce byte-identically."""
    clean = {k: v for k, v in result.items() if k != "timing"}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode("utf-8")).hexdigest()


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console guard (§/σ)
    ap = argparse.ArgumentParser(description="M3 acceptance v2 (honest, ground-truth).")
    ap.add_argument("--smoke", action="store_true", help="tiny fast deterministic config")
    ap.add_argument("--out", type=str, default=str(RESULT_JSON))
    args = ap.parse_args(argv)
    cfg = smoke_config() if args.smoke else full_config()

    from rig_adapters.mbe import simlink

    if not simlink.sim_available():
        print(
            f"MBE sim not found (set {simlink.MBE_SIM_ENV}); M3 v2 runs ONLY on the "
            "InSilicoMachine (no synthetic fallback — that would reintroduce the v1 "
            "toy-venue critique).",
            file=sys.stderr,
        )
        return 2

    t_start = time.perf_counter()
    built = build(cfg)

    gate = built.gate
    print(
        f"§14.6 gate: passed={gate['passed']} (sbc={gate['sbc_passed']} "
        f"tarp={gate['tarp_passed']} sbc_p={[round(p, 4) for p in gate['sbc_p_values']]} "
        f"tarp_err={round(gate['tarp_max_calibration_error'], 4)})",
        flush=True,
    )
    if not gate["passed"] and cfg.gate_enforced:
        result = {
            "meta": _meta(cfg, built),
            "gate": gate,
            "m3_verdict": {
                "gate_pass": False,
                "aborted": "GATE_FAILED",
                "note": "§14.6 SBC/TARP gate failed; per D2/§14.6 no posterior ships and "
                "no arm runs. This is the blocking semantics, not an arm result.",
            },
        }
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(
            "\nABORT: §14.6 gate FAILED — no arm runs (D2 blocking semantics). "
            f"Diagnosis: sbc_passed={gate['sbc_passed']}, tarp_passed={gate['tarp_passed']}, "
            f"sbc_p={[round(p, 4) for p in gate['sbc_p_values']]}, "
            f"tarp_err={round(gate['tarp_max_calibration_error'], 4)}.",
            flush=True,
        )
        print(f"wrote {Path(args.out).name}")
        print(f"DETERMINISM_DIGEST={_determinism_digest(result)}")
        return 1

    print("pre-probe + pre-registering targets (cold_light selection only) ...", flush=True)
    classified, selected = select_targets(built)
    print(
        f"selected {len(selected)} targets: "
        f"{sum(1 for s in selected if s['preprobe_class'] == 'MISS')} MISS + "
        f"{sum(1 for s in selected if s['preprobe_class'] == 'HIT')} HIT",
        flush=True,
    )

    print("running arms (cold_heavy / cold_light / d2_light), ground-truth scoring ...", flush=True)
    rows, times = run_arms(built, selected)
    verdict = m3_verdict(rows)

    heavy_starts = cfg.heavy_restarts
    d2_starts = cfg.light_restarts + cfg.n_proposals
    result = {
        "meta": _meta(cfg, built),
        "gate": gate,
        "budget": {
            "cold_heavy_starts": heavy_starts,
            "cold_light_starts": cfg.light_restarts,
            "d2_light_starts": d2_starts,
            "cost_ratio_starts_d2_over_heavy": round(d2_starts / heavy_starts, 4),
        },
        "selection": {
            "rule": "cold_light pre-probe; first k_miss MISS + first k_hit HIT in pool "
            "order (corners first, then Sobol). Selection uses cold_light ONLY.",
            "tol_std_frac": cfg.tol_std_frac,
            "pool_size": len(classified),
            "n_miss_in_pool": sum(1 for c in classified if c["preprobe_class"] == "MISS"),
            "n_hit_in_pool": sum(1 for c in classified if c["preprobe_class"] == "HIT"),
            "classification": classified,
        },
        "metrology_sigma": {
            SPEC_OUTPUTS[j]: float(built.sigma[built.out_idx][j]) for j in range(len(SPEC_OUTPUTS))
        },
        "targets": rows,
        "m3_verdict": verdict,
        "timing": {
            "seconds_per_arm": {k: round(v, 3) for k, v in times.items()},
            "cost_ratio_walltime_d2_over_heavy": round(
                times["d2_light"] / max(times["cold_heavy"], 1e-9), 4
            ),
            "total_seconds": round(time.perf_counter() - t_start, 3),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    v = verdict
    print(
        f"\nM3 v2 VERDICT: {'PASS' if v['gate_pass'] else 'FAIL'}  "
        f"(ground-truth top-1 hits — cold_heavy={v['gt_hits_cold_heavy']} "
        f"cold_light={v['gt_hits_cold_light']} d2_light={v['gt_hits_d2_light']} "
        f"of {v['n_targets']}; d2>=heavy={v['d2_ge_heavy']}, d2>light={v['d2_gt_light']})",
        flush=True,
    )
    print(
        f"budget: d2_light = {d2_starts}/{heavy_starts} = "
        f"{round(d2_starts / heavy_starts, 4)}x cold_heavy starts; wall-time ratio "
        f"{result['timing']['cost_ratio_walltime_d2_over_heavy']}x.",
        flush=True,
    )
    if not v["gate_pass"]:
        print(
            "HONEST NOTE: d2_light did NOT clear the bar — M3's amortization value is NOT "
            "demonstrated on this machine at this budget. See the doc.",
            flush=True,
        )
    print(f"wrote {Path(args.out).name}")
    print(f"DETERMINISM_DIGEST={_determinism_digest(result)}")
    return 0


def _meta(cfg: Config, built: _Built) -> dict:
    return {
        "machine": "InSilicoMachine(MBE)",
        "pathology": "metrology_noise=True (as M2); no drift/hidden-state (out of scope)",
        "ground_truth": "noise-free machine physics (clean InSilicoMachine) at the "
        "returned recipe — never the surrogate",
        "ikeys": built.ikeys,
        "okeys": built.okeys,
        "spec_outputs": list(SPEC_OUTPUTS),
        "n_train": cfg.n_train,
        "generator": {
            "n_members": cfg.n_members,
            "transforms": cfg.transforms,
            "hidden": list(cfg.hidden),
            "max_epochs": cfg.max_epochs,
            "region_hw": list(cfg.region_hw),
        },
        "solver_policy": "binding §8: kappa=z_epi=2.0, delta_frac=0.02",
        "smoke": cfg.smoke,
        "seeds": {
            "train_design": cfg.seed_train_design,
            "data_machine": cfg.seed_data_machine,
            "gen": cfg.seed_gen,
            "gate": cfg.seed_gate,
            "gate_sim": cfg.seed_gate_sim,
            "solver": cfg.seed_solver,
            "refs": cfg.seed_refs,
        },
    }


if __name__ == "__main__":
    sys.exit(main())
