"""IN-SILICO MULTI-TOOL M4 DRESS REHEARSAL (implementation-plan §10.4 / §5.8 / §8 / §11.4).

*** REHEARSAL — provenance = physics_sim — NOT headline evidence. ***
Real multi-tool data does not exist yet.  This script exercises the ENTIRE M4
story end-to-end on the WP-B in-silico machine, so integration bugs surface now
instead of on real wafers.  Every number here is produced by ``InSilicoMachine``
(the fast Arrhenius sim path) and is a MACHINERY PROOF only; the scientific claim
stays gated on M0 real data.  It also closes the "runner-level qualification
auto-invocation" crumb: Phase 4 demonstrates ``solve() -> ConfirmationCampaign``
automatically (both a direct campaign on the solver's output AND the
``ActiveLearningLoop(qualification=...)`` in-loop hook).

The fleet (3 synthetic "tools" = one MBE process, perturbed to emulate as-built
chamber-to-chamber variation the learning stack must DETECT, never being told):

  * Hidden per-tool physics via ``PathologyConfig(tool_perturbation=True)`` (the
    sim's E3 pathology), keyed by ``tool_id``: a fixed +-3% multiplicative offset
    on (substrate emissivity, source cosine_n, effective flux).  These stand in
    for run-to-run-INVISIBLE chamber differences (flux-cell calibration matched
    only to a few %, wall-coating emissivity history, beam-profile spread) — the
    §10.2 hidden-state the RunRecords never carry.  +-3% is a plausible
    chamber-matching delta, not tuned (it is the sim's own default scale).
  * Build geometry via ``machine_config`` (the split-plot HARD_TO_CHANGE
    whole-plot factors ``gap`` / ``source_height`` / ``heater_radius``): +-5-8%
    mechanical build tolerances, symmetric round offsets chosen BEFORE any
    outcome was measured and all well inside the sim's own MACHINE_CONFIG_BOUNDS.

  toolA = nominal geometry (the "golden" reference chamber).
  toolB = gap +8%, source_height -5%, heater_radius +5%.
  toolC = gap -8%, source_height +5%, heater_radius -5%   <- the HELD-OUT "new chamber".

Two modelled outputs make BOTH perturbation axes visible: ``thickness_grown``
(= film_thickness x flux_eff — the flux-sensitive channel, the canonical WP-B
handoff KPI; tool signal ~20x metrology noise) and ``T_center`` (radiative
balance — sensitive to emissivity AND geometry; tool signal ~6x noise).

Phases (each: seeded, deterministic, scored against MACHINE GROUND TRUTH where
accuracy is claimed, verdict printed, recorded to docs/multitool-rehearsal.json):

  1. FLEET            — build the 3 tools, seeded Sobol campaigns, report the
                        (ground-truth, never-modelled) hidden factors + the
                        between-tool signal vs the metrology-noise floor.
  2. POOL + LOTO      — fit the ICM multi-task GP (tool as task); leave-one-tool-out
                        zero-shot MUST show the §5.8 unknown-tool epistemic
                        domination (all 3 folds); few-shot (K=10/20) pooled vs a
                        from-scratch single-tool fit at equal n — honest verdict
                        whether pooling HELPS / HURTS / is a WASH.
  3. ONBOARDING       — §10.4 chamber onboarding: EPIG-driven active loop warm-started
                        from the pooled {A,B} model vs a cold single-tool loop, same
                        C-machine budget; runs-to-target-model-quality; ASSERT
                        EPIG > 0 nats on the unknown-tool path (the EPIG-collapse guard).
  4. SOLVE + QUALIFY  — bind the onboarded tool, pose a REACHABLE spec, §8 solve
                        (binding 2.0/2.0/0.02), auto-run a ConfirmationCampaign of
                        REAL confirmation runs on tool C; also pose an UNREACHABLE spec
                        and show NothingToQualify with ZERO machine calls; and drive the
                        ActiveLearningLoop qualification= hook (budget-charged).

Run (Windows cp1252 console -> force UTF-8):

    PYTHONIOENCODING=utf-8 python examples/run_multitool_rehearsal.py [--smoke] [--full] [--out P]

Determinism: everything is seeded; a re-run is byte-identical modulo ``timings``
(wall-clock, deliberately excluded from the determinism comparison — see
:func:`strip_volatile`).  ``--smoke`` is a tiny fast shape check; ``--full`` (the
default) is a modest, NOT powered study (this is a rehearsal; total wall < ~20 min).
No torch is imported anywhere on this path (all numpy/scipy GP tier).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rig.active import (
    ActiveLearningLoop,
    CampaignResult,
    ConfirmationCampaign,
    NothingToQualify,
)
from rig.active.acquisition import cost_cooled_acquisition, epig
from rig.active.batch import select_batch
from rig.calibration import ConformalForwardModel, SplitConformalCalibrator
from rig.forward import (
    GPForwardModel,
    MultiToolGPForwardModel,
    records_to_arrays,
)
from rig.interfaces import Infeasible
from rig.inverse.pessimistic import PessimisticInverseSolver
from rig.metrics import uq
from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # §/σ printing on a cp1252 console

# ---------------------------------------------------------------------------
# constants + fleet definition
# ---------------------------------------------------------------------------

SEED = 0
INPUT_KEYS = ["T_heater", "film_thickness"]
OUTPUT_KEYS = ["thickness_grown", "T_center"]  # (flux-sensitive, temperature)
THK, TC = 0, 1  # output column indices

# The 3 tools: (tool_id, machine_config geometry offsets).  Hidden (emissivity,
# cosine_n, flux) offsets are applied by tool_perturbation, keyed on tool_id, and
# are NOT set here (the machine draws them deterministically from the tool_id hash).
# Nominal geometry defaults: gap=0.012, source_height=0.20, heater_radius=0.018 m.
FLEET: tuple[tuple[str, dict[str, float]], ...] = (
    ("toolA", {}),  # golden reference chamber
    ("toolB", {"gap": 0.012 * 1.08, "source_height": 0.20 * 0.95, "heater_radius": 0.018 * 1.05}),
    ("toolC", {"gap": 0.012 * 0.92, "source_height": 0.20 * 1.05, "heater_radius": 0.018 * 0.95}),
)
TOOL_IDS = [tid for tid, _ in FLEET]
HELD_OUT = "toolC"  # the "new chamber" for LOTO / onboarding / solve+qualify
KNOWN = [t for t in TOOL_IDS if t != HELD_OUT]  # the fitted fleet {toolA, toolB}

# §8 BINDING feasibility policy (F3): kappa = z_epi = 2.0, delta_frac = 0.02.
KAPPA, Z_EPI, DELTA_FRAC = 2.0, 2.0, 0.02
EPIG_MIN_NATS = 1e-3  # the unknown-tool EPIG-collapse regression guard threshold

# Phase 4b — conformal wrap of the onboarded tool (§5.6 D4 / §13.2).
CONFORMAL_ALPHA = 0.1  # matches rig.calibration.conformal.DEFAULT_ALPHA
CAL_FRACTION = 1.0 / 3.0  # trailing fraction of the onboarded tool's runs held out
# a seed namespace strictly separate from every seed used by phases 1-4 (SEED+0..2,
# +100..124, +300..324, +500, +700, +900, +950, +12345) so phase 4b's own seeded
# Generator/design calls can never collide with -- and never perturb -- the RNG
# consumption of the recorded phases.
PHASE4B_SEED_BASE = SEED + 8000


@dataclass(frozen=True)
class Config:
    """Smoke vs full knobs (compute budget only — never changes the story)."""

    label: str
    n_train: int  # Sobol campaign size per tool
    n_test: int  # held-out ground-truth-scoring recipes
    n_restarts: int  # GP / ICM hyperparameter multi-start
    max_iter: int
    onboard_budget: int  # C-machine runs per onboarding arm
    q: int  # batch size
    pool: int  # acquisition candidate pool size
    fewshot_K: tuple[int, ...]
    gate_n_runs: int  # direct-flow confirmation batch size
    gate_min_rate: float
    gate_conf: float
    loop_n_runs: int  # in-loop qualification-hook confirmation batch size


SMOKE = Config(
    label="smoke",
    n_train=8,
    n_test=6,
    n_restarts=1,
    max_iter=40,
    onboard_budget=12,
    q=4,
    pool=32,
    fewshot_K=(4,),
    gate_n_runs=5,
    gate_min_rate=0.5,
    gate_conf=0.8,
    loop_n_runs=3,
)
FULL = Config(
    label="full",
    n_train=24,
    n_test=16,
    n_restarts=2,
    max_iter=100,
    onboard_budget=24,
    q=4,
    pool=64,
    fewshot_K=(10, 20),
    gate_n_runs=29,  # a flawless 29-run batch certifies p>=0.90 @ 95% and no fewer
    gate_min_rate=0.90,
    gate_conf=0.95,
    loop_n_runs=5,
)

_ADAPTER = make_adapter()


# ---------------------------------------------------------------------------
# machine plumbing — every consumer builds FRESH machine instances so run_index
# streams are local and the whole script re-runs byte-identically.
# ---------------------------------------------------------------------------


def _clean_machine(geom: dict[str, float]) -> InSilicoMachine:
    """Ground-truth machine: tool_perturbation only (NO metrology noise, NO
    seasoning) -> a deterministic, order-independent, noise-free truth oracle."""
    return InSilicoMachine(
        config=PathologyConfig(tool_perturbation=True),
        seed=SEED,
        adapter=_ADAPTER,
        machine_config=geom,
    )


def _noisy_machine(geom: dict[str, float]) -> InSilicoMachine:
    """Observation machine: adds heteroscedastic metrology noise (a realistic tool)."""
    return InSilicoMachine(
        config=PathologyConfig(tool_perturbation=True, metrology_noise=True),
        seed=SEED,
        adapter=_ADAPTER,
        machine_config=geom,
    )


def _geom_of(tool_id: str) -> dict[str, float]:
    return dict(next(g for tid, g in FLEET if tid == tool_id))


def _run_outputs(
    machine: InSilicoMachine, tool_id: str, recipes: list[dict[str, float]]
) -> np.ndarray:
    """(n, 2) output matrix (SI: m, K) from running ``recipes`` on ``machine``."""
    recs = [machine.run(r, tool_id=tool_id) for r in recipes]
    _, Y = records_to_arrays(recs, INPUT_KEYS, OUTPUT_KEYS)
    return Y


def truth_outputs(tool_id: str, recipes: list[dict[str, float]]) -> np.ndarray:
    """Noise-free MACHINE GROUND TRUTH for ``tool_id`` — the scoring oracle
    (fresh clean machine; order-independent because no noise/seasoning/first-wafer)."""
    return _run_outputs(_clean_machine(_geom_of(tool_id)), tool_id, recipes)


def noisy_campaign(
    tool_id: str, n: int, design_seed: int
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """A tool's seeded Sobol campaign as it would be OBSERVED (with metrology noise).
    Returns (X recipes-as-array, Y noisy outputs, recipe dicts). Deterministic."""
    recipes = _ADAPTER.seed_design(n, design_seed)
    Y = _run_outputs(_noisy_machine(_geom_of(tool_id)), tool_id, recipes)
    X = np.array([[r[k] for k in INPUT_KEYS] for r in recipes], dtype=float)
    return X, Y, recipes


def recipes_to_X(recipes: list[dict[str, float]]) -> np.ndarray:
    return np.array([[r[k] for k in INPUT_KEYS] for r in recipes], dtype=float)


def make_verifier(tool_id: str):
    """A ConfirmationBatchGate verifier: recipe -> {output: SI value} on a NOISY
    tool machine (real confirmation runs). Fresh instance -> deterministic sequence."""
    machine = _noisy_machine(_geom_of(tool_id))

    def verify(recipe: dict[str, float]) -> dict[str, float]:
        rec = machine.run(dict(recipe), tool_id=tool_id)
        return {o.name: float(o.value.magnitude) for o in rec.outcomes if o.name in OUTPUT_KEYS}

    return verify


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def banner(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def rmse_per_output(mu: np.ndarray, y: np.ndarray) -> list[float]:
    return [float(v) for v in uq.rmse(np.asarray(mu), np.asarray(y))]


def strip_volatile(payload: dict[str, Any]) -> dict[str, Any]:
    """A copy with wall-clock timings removed — the determinism comparison unit
    (everything else is a pure function of SEED + config)."""
    out = deepcopy(payload)
    out.get("meta", {}).pop("timings", None)
    return out


# ---------------------------------------------------------------------------
# PHASE 1 — FLEET
# ---------------------------------------------------------------------------


def phase1_fleet(cfg: Config) -> dict[str, Any]:
    banner("PHASE 1 -- FLEET (3 synthetic tools = one MBE process, as-built variation)")
    print("REHEARSAL: provenance=physics_sim, machinery proof only (M0 real data still owed).")

    # ground-truth hidden factors (never fed to any model) + a fixed-recipe signal probe
    ref = {"T_heater": 1325.0, "film_thickness": 2.5e-6}
    tools = {}
    print(f"\n{'tool':<7}{'geometry offsets':<44}{'hidden (eps,cos,flux) factors':<32}")
    print("-" * 83)
    for tid, geom in FLEET:
        factors = _clean_machine(geom)._tool_factors(tid)  # ground truth only
        gtxt = ", ".join(f"{k}x{v / _ADAPTER_default(k):.3f}" for k, v in geom.items()) or "nominal"
        tools[tid] = {
            "geometry": {k: float(v) for k, v in geom.items()},
            "hidden_factors_emissivity_cosine_flux": [float(x) for x in factors],
        }
        print(f"{tid:<7}{gtxt:<44}{'(' + ', '.join(f'{x:.4f}' for x in factors) + ')':<32}")

    # between-tool signal at a fixed recipe vs the metrology-noise floor
    print("\nfixed-recipe tool signal vs metrology-noise floor (T_heater=1325 K, film=2.5 um):")
    clean_ref = {tid: truth_outputs(tid, [ref])[0] for tid in TOOL_IDS}
    noise_std = {}
    for tid, geom in FLEET:
        nm = _noisy_machine(geom)
        reps = np.array([_run_outputs(nm, tid, [ref])[0] for _ in range(20)])
        noise_std[tid] = reps.std(axis=0)
    b, c = clean_ref["toolB"], clean_ref["toolC"]
    thk_sig, tc_sig = abs(b[THK] - c[THK]), abs(b[TC] - c[TC])
    thk_noise = float(np.mean([noise_std[t][THK] for t in TOOL_IDS]))
    tc_noise = float(np.mean([noise_std[t][TC] for t in TOOL_IDS]))
    print(
        f"  thickness_grown: B-C tool gap {thk_sig:.3e} m  vs noise std {thk_noise:.3e} m  "
        f"-> {thk_sig / thk_noise:.1f}x"
    )
    print(
        f"  T_center       : B-C tool gap {tc_sig:.3f} K  vs noise std {tc_noise:.3f} K  -> {tc_sig / tc_noise:.1f}x"
    )

    # seeded per-tool campaigns (the OBSERVED, noisy data the fleet is fit on)
    campaigns = {}
    for i, tid in enumerate(TOOL_IDS):
        X, Y, recipes = noisy_campaign(tid, cfg.n_train, design_seed=SEED + i)
        campaigns[tid] = dict(X=X, Y=Y, recipes=recipes)
        print(
            f"  {tid}: {cfg.n_train} seeded Sobol runs "
            f"thk[{Y[:, THK].min():.3e},{Y[:, THK].max():.3e}]m  "
            f"Tc[{Y[:, TC].min():.1f},{Y[:, TC].max():.1f}]K"
        )

    result = {
        "provenance": "physics_sim",
        "caveat": "REHEARSAL / machinery proof; not headline evidence (M0 real data owed)",
        "held_out_tool": HELD_OUT,
        "known_fleet": KNOWN,
        "n_train_per_tool": cfg.n_train,
        "tools": tools,
        "fixed_recipe_signal": {
            "reference_recipe": ref,
            "clean_outputs_per_tool": {t: [float(v) for v in clean_ref[t]] for t in TOOL_IDS},
            "metrology_noise_std_mean": {"thickness_grown": thk_noise, "T_center": tc_noise},
            "B_minus_C_gap": {"thickness_grown": float(thk_sig), "T_center": float(tc_sig)},
            "signal_to_noise": {
                "thickness_grown": float(thk_sig / thk_noise),
                "T_center": float(tc_sig / tc_noise),
            },
        },
        "verdict": (
            "3 tools built; the tool-to-tool signal is well above the metrology-noise floor "
            "on both outputs -> the perturbations are learnable (and NOT so large as to be a "
            "different process). Not tuned: symmetric round offsets inside the sim's own bounds."
        ),
    }
    return result, campaigns


def _ADAPTER_default(name: str) -> float:
    from rig_adapters.mbe.adapter import MACHINE_CONFIG_DEFAULTS

    return MACHINE_CONFIG_DEFAULTS[name]


# ---------------------------------------------------------------------------
# PHASE 2 — POOLED FIT + LEAVE-ONE-TOOL-OUT
# ---------------------------------------------------------------------------


def _fit_pooled(
    campaigns: dict[str, Any], tools: list[str], cfg: Config
) -> MultiToolGPForwardModel:
    X = np.vstack([campaigns[t]["X"] for t in tools])
    Y = np.vstack([campaigns[t]["Y"] for t in tools])
    tool_labels: list[str] = []
    for t in tools:
        tool_labels += [t] * campaigns[t]["X"].shape[0]
    return MultiToolGPForwardModel(
        rank=1, n_restarts=cfg.n_restarts, seed=SEED, max_iter=cfg.max_iter
    ).fit(X, Y, tool_labels)


def phase2_pooling(cfg: Config, campaigns: dict[str, Any]) -> dict[str, Any]:
    banner("PHASE 2 -- POOLED ICM FIT + LEAVE-ONE-TOOL-OUT (§10.4 / §5.8)")

    # held-out ground-truth scoring recipes (distinct seed from any training design)
    test_recipes = _ADAPTER.seed_design(cfg.n_test, seed=SEED + 900)
    X_test = recipes_to_X(test_recipes)

    # (a) §5.8 leave-one-tool-out zero-shot epistemic domination — all 3 folds.
    print("\n(a) §5.8 leave-one-tool-out zero-shot epistemic domination (unknown tool must")
    print("    carry MORE epistemic than every fitted tool, elementwise on both outputs):")
    loto = {}
    all_dominate = True
    for held in TOOL_IDS:
        known = [t for t in TOOL_IDS if t != held]
        model = _fit_pooled(campaigns, known, cfg)
        Xh = recipes_to_X(_ADAPTER.seed_design(cfg.n_test, seed=SEED + 950))
        epi_unknown = np.asarray(model.predict(Xh, tool_id=held).epistemic_sigma)  # held is UNKNOWN
        dom = True
        per_known = {}
        for kt in known:
            epi_k = np.asarray(model.predict(Xh, tool_id=kt).epistemic_sigma)
            ok = bool(np.all(epi_unknown >= epi_k - 1e-12))
            dom = dom and ok
            per_known[kt] = {
                "mean_epi_known": [float(v) for v in epi_k.mean(axis=0)],
                "unknown_dominates_elementwise": ok,
            }
        all_dominate = all_dominate and dom
        loto[held] = {
            "known_fleet": known,
            "mean_epi_unknown": [float(v) for v in epi_unknown.mean(axis=0)],
            "per_known_tool": per_known,
            "dominates_all": dom,
        }
        print(f"    held-out {held} (fit {known}): unknown dominates all fitted -> {dom}")

    # (b) few-shot pooled vs from-scratch single-tool, EQUAL n — the honest verdict.
    print("\n(b) onboarding the held-out tool with K few-shot runs: POOLED (warm) vs SCRATCH")
    print(
        f"    (a from-scratch single-tool GP on the SAME K rows). RMSE vs {HELD_OUT} ground truth."
    )
    y_truth = truth_outputs(HELD_OUT, test_recipes)
    Xc, Yc = campaigns[HELD_OUT]["X"], campaigns[HELD_OUT]["Y"]

    # zero-shot pooled (K=0): the {A,B} population fallback for the unknown tool
    pooled_known = _fit_pooled(campaigns, KNOWN, cfg)
    rmse_zero = rmse_per_output(pooled_known.predict(X_test, tool_id=HELD_OUT).mean, y_truth)
    print(
        f"    K=0  pooled zero-shot (unknown-tool fallback): thk-RMSE {rmse_zero[THK]:.3e}  Tc-RMSE {rmse_zero[TC]:.3f}"
    )

    fewshot = {
        "zero_shot_pooled_rmse": {"thickness_grown": rmse_zero[THK], "T_center": rmse_zero[TC]}
    }
    helps_thk = []
    for K in cfg.fewshot_K:
        Kc = min(K, Xc.shape[0])
        # POOLED warm: fit {A,B}, then fold in K held-out rows (adapt_to_tool = full refit)
        warm = _fit_pooled(campaigns, KNOWN, cfg)
        warm.adapt_to_tool(HELD_OUT, Xc[:Kc], Yc[:Kc])
        rmse_pool = rmse_per_output(warm.predict(X_test, tool_id=HELD_OUT).mean, y_truth)
        # SCRATCH single-tool at equal n
        scratch = GPForwardModel(n_restarts=cfg.n_restarts, seed=SEED).fit(Xc[:Kc], Yc[:Kc])
        rmse_scr = rmse_per_output(scratch.predict(X_test).mean, y_truth)
        d_thk = rmse_scr[THK] - rmse_pool[THK]  # >0 => pooling helps on thickness
        helps_thk.append(d_thk)
        fewshot[f"K={Kc}"] = {
            "pooled_rmse": {"thickness_grown": rmse_pool[THK], "T_center": rmse_pool[TC]},
            "scratch_rmse": {"thickness_grown": rmse_scr[THK], "T_center": rmse_scr[TC]},
            "pooled_minus_scratch_thk": float(rmse_pool[THK] - rmse_scr[THK]),
            "pooling_helps_thk": bool(d_thk > 0),
        }
        verdict = "POOLING HELPS" if d_thk > 0 else "SCRATCH WINS"
        print(
            f"    K={Kc:<3d} pooled thk-RMSE {rmse_pool[THK]:.3e} | scratch thk-RMSE {rmse_scr[THK]:.3e}"
            f"  -> {verdict} (Δthk {d_thk:+.3e})"
        )

    mean_help = float(np.mean(helps_thk))
    if mean_help > 0:
        pooling_verdict = "HELPS"
    elif mean_help < 0:
        pooling_verdict = "HURTS"
    else:
        pooling_verdict = "WASH"
    print(
        f"\n    HONEST POOLING VERDICT (on this 3-tool fleet, thickness KPI): {pooling_verdict} "
        f"(mean scratch-minus-pooled RMSE {mean_help:+.3e} m)"
    )

    return (
        {
            "protocol": (
                "pooled ICM multi-task GP (tool as task); (a) 3-fold leave-one-tool-out zero-shot "
                "§5.8 domination; (b) few-shot pooled vs from-scratch single-tool at equal n, RMSE "
                "vs the held-out tool's noise-free machine ground truth"
            ),
            "n_test": cfg.n_test,
            "loto_zero_shot_domination": {"all_folds_dominate": all_dominate, "folds": loto},
            "few_shot": fewshot,
            "pooling_verdict": pooling_verdict,
            "pooling_mean_scratch_minus_pooled_thk": mean_help,
            "verdict": (
                f"§5.8 domination holds on all folds: {all_dominate}. Pooling on this fleet {pooling_verdict} "
                "on the thickness KPI (measured, not assumed) — any answer is a legitimate finding."
            ),
        },
        X_test,
        test_recipes,
        y_truth,
    )


# ---------------------------------------------------------------------------
# PHASE 3 — NEW-CHAMBER ONBOARDING (§10.4)
# ---------------------------------------------------------------------------


def _held_out_rmse_thk(
    model, X_test: np.ndarray, y_truth: np.ndarray, tool_id: str | None
) -> float:
    pred = model.predict(X_test, tool_id=tool_id) if tool_id is not None else model.predict(X_test)
    return rmse_per_output(pred.mean, y_truth)[THK]


def phase3_onboarding(
    cfg: Config,
    campaigns: dict[str, Any],
    X_test: np.ndarray,
    y_truth: np.ndarray,
) -> dict[str, Any]:
    banner("PHASE 3 -- NEW-CHAMBER ONBOARDING (§10.4 EPIG-driven; warm pooled vs cold scratch)")

    # A fresh reference set of "operating recipes" is X_star for EPIG (distinct seed
    # from the RMSE test set -> no teaching-to-the-test). Onboard to be accurate where
    # we'll operate; score RMSE on the separate held-out set.
    xstar_recipes = _ADAPTER.seed_design(max(8, cfg.q * 3), seed=SEED + 700)
    X_star = recipes_to_X(xstar_recipes)

    # quality threshold: 1.5x the cold-arm full-data ceiling (single-tool GP on ALL of C).
    Xc, Yc = campaigns[HELD_OUT]["X"], campaigns[HELD_OUT]["Y"]
    ceiling_model = GPForwardModel(n_restarts=cfg.n_restarts, seed=SEED).fit(Xc, Yc)
    ceiling = _held_out_rmse_thk(ceiling_model, X_test, y_truth, None)
    threshold = 1.5 * ceiling
    print(
        f"full-data ceiling thk-RMSE {ceiling:.3e} m  ->  onboarding target threshold {threshold:.3e} m"
    )

    noisy_C = _noisy_machine(_geom_of(HELD_OUT))  # ONE instance -> local deterministic run_index
    q = cfg.q

    # ---- WARM arm: start from pooled {A,B}, C UNKNOWN; EPIG-select, run, adapt ----
    warm = _fit_pooled(campaigns, KNOWN, cfg)
    warm_traj: list[dict[str, Any]] = []
    epig_unknown_max = None
    C_X = np.empty((0, len(INPUT_KEYS)))
    C_Y = np.empty((0, len(OUTPUT_KEYS)))
    n_runs = 0
    rmse0 = _held_out_rmse_thk(warm, X_test, y_truth, HELD_OUT)
    warm_traj.append({"n_C_runs": 0, "thk_rmse": rmse0})
    while n_runs < cfg.onboard_budget:
        view = warm.for_tool(HELD_OUT)  # unknown-tool fallback until first adapt
        pool_recipes = _ADAPTER.seed_design(cfg.pool, seed=SEED + 100 + n_runs)
        pool_X = recipes_to_X(pool_recipes)
        acq = cost_cooled_acquisition(view, pool_X, X_star, lam=0.7, beta=0.0)  # EPIG-heavy
        if n_runs == 0:
            # THE EPIG-collapse regression guard: on the UNKNOWN-tool path EPIG must be > 0 nats.
            e = epig(view, pool_X, X_star)
            epig_unknown_max = float(np.max(e))
            print(
                f"  [guard] unknown-tool EPIG max = {epig_unknown_max:.4f} nats "
                f"({'OK > 0' if epig_unknown_max > EPIG_MIN_NATS else 'COLLAPSED!'})"
            )
        take = min(q, cfg.onboard_budget - n_runs)
        idx = select_batch(acq, pool_X, take, model=view)
        picked = [pool_recipes[i] for i in idx]
        Yb = _run_outputs(noisy_C, HELD_OUT, picked)
        Xb = recipes_to_X(picked)
        C_X, C_Y = np.vstack([C_X, Xb]), np.vstack([C_Y, Yb])
        n_runs += len(picked)
        # refit the pooled model with ALL accumulated C data folded in (C now KNOWN)
        warm = _fit_pooled(campaigns, KNOWN, cfg)
        warm.adapt_to_tool(HELD_OUT, C_X, C_Y)
        r = _held_out_rmse_thk(warm, X_test, y_truth, HELD_OUT)
        warm_traj.append({"n_C_runs": n_runs, "thk_rmse": r})

    # ---- COLD arm: single-tool GP on C only; Sobol seed then EPIG/BALD-select ----
    cold_X, cold_Y = np.empty((0, len(INPUT_KEYS))), np.empty((0, len(OUTPUT_KEYS)))
    cold_traj: list[dict[str, Any]] = []
    n_runs = 0
    # seed DoE (q runs) — a single-tool GP cannot fit on zero rows
    seed_recipes = _ADAPTER.seed_design(q, seed=SEED + 500)
    Yb = _run_outputs(noisy_C, HELD_OUT, seed_recipes)
    cold_X, cold_Y = recipes_to_X(seed_recipes), Yb
    n_runs = q
    cold = GPForwardModel(n_restarts=cfg.n_restarts, seed=SEED).fit(cold_X, cold_Y)
    cold_traj.append(
        {"n_C_runs": n_runs, "thk_rmse": _held_out_rmse_thk(cold, X_test, y_truth, None)}
    )
    while n_runs < cfg.onboard_budget:
        pool_recipes = _ADAPTER.seed_design(cfg.pool, seed=SEED + 300 + n_runs)
        pool_X = recipes_to_X(pool_recipes)
        acq = cost_cooled_acquisition(cold, pool_X, X_star, lam=0.7, beta=0.0)
        take = min(q, cfg.onboard_budget - n_runs)
        idx = select_batch(acq, pool_X, take, model=cold)
        picked = [pool_recipes[i] for i in idx]
        Yb = _run_outputs(noisy_C, HELD_OUT, picked)
        cold_X, cold_Y = np.vstack([cold_X, recipes_to_X(picked)]), np.vstack([cold_Y, Yb])
        n_runs += len(picked)
        cold = GPForwardModel(n_restarts=cfg.n_restarts, seed=SEED).fit(cold_X, cold_Y)
        cold_traj.append(
            {"n_C_runs": n_runs, "thk_rmse": _held_out_rmse_thk(cold, X_test, y_truth, None)}
        )

    def runs_to_threshold(traj: list[dict[str, Any]]) -> int | None:
        for p in traj:
            if p["thk_rmse"] <= threshold:
                return int(p["n_C_runs"])
        return None

    warm_rt, cold_rt = runs_to_threshold(warm_traj), runs_to_threshold(cold_traj)
    print(
        "\n  warm (pooled) RMSE trajectory:",
        [f"{p['n_C_runs']}:{p['thk_rmse']:.2e}" for p in warm_traj],
    )
    print(
        "  cold (scratch) RMSE trajectory:",
        [f"{p['n_C_runs']}:{p['thk_rmse']:.2e}" for p in cold_traj],
    )
    print(
        f"  runs-to-threshold: warm={warm_rt}  cold={cold_rt}  (None = not reached within budget)"
    )

    if warm_rt is not None and (cold_rt is None or warm_rt < cold_rt):
        onboard_verdict = "WARM START WINS (pooled onboarded in fewer C-runs)"
    elif cold_rt is not None and (warm_rt is None or cold_rt < warm_rt):
        onboard_verdict = "COLD WINS (warm start did not pay off on this fleet)"
    else:
        onboard_verdict = "TIE (both reached threshold at the same budget, or neither did)"
    print(f"  ONBOARDING VERDICT: {onboard_verdict}")

    epig_ok = epig_unknown_max is not None and epig_unknown_max > EPIG_MIN_NATS
    return (
        {
            "protocol": (
                "§10.4 chamber onboarding: EPIG-driven active selection (cost_cooled_acquisition, "
                "lam=0.7 EPIG-heavy) warm-started from the pooled {A,B} model vs a cold single-tool "
                "GP, same C-machine budget; held-out thickness RMSE vs ground truth after each batch"
            ),
            "onboard_budget_C_runs": cfg.onboard_budget,
            "batch_q": q,
            "full_data_ceiling_thk_rmse": float(ceiling),
            "target_threshold_thk_rmse": float(threshold),
            "epig_unknown_tool_max_nats": epig_unknown_max,
            "epig_positive_on_unknown_tool": bool(epig_ok),
            "warm_trajectory": warm_traj,
            "cold_trajectory": cold_traj,
            "warm_runs_to_threshold": warm_rt,
            "cold_runs_to_threshold": cold_rt,
            "onboarding_verdict": onboard_verdict,
            "warm_final_model_C_runs": int(C_X.shape[0]),
            "verdict": (
                f"EPIG > 0 on the unknown-tool path: {epig_ok} (max {epig_unknown_max}). "
                f"{onboard_verdict}."
            ),
        },
        warm,
        C_X,
        C_Y,
    )


# ---------------------------------------------------------------------------
# PHASE 4 — SOLVE + AUTO-QUALIFICATION
# ---------------------------------------------------------------------------


def _summarize_campaign(res: CampaignResult) -> dict[str, Any]:
    per_cand = []
    for c in list(res.certified) + list(res.rejected):
        ev = c.evidence
        per_cand.append(
            {
                "passed": bool(c.passed),
                "n_in_spec": int(ev["n_in_spec"]),
                "n_runs": int(ev["n_runs"]),
                "binomial_lower_bound": float(ev["binomial_lower_bound"]),
                "min_in_spec_rate": float(ev["min_in_spec_rate"]),
            }
        )
    return {
        "n_candidates": res.n_candidates,
        "n_certified": res.n_certified,
        "n_rejected": res.n_rejected,
        "n_machine_calls": res.n_machine_calls,
        "provenance_source": res.provenance_source,
        "headline_eligible": res.headline_eligible,
        "confidence_per_candidate": float(res.confidence_per_candidate),
        "per_candidate": per_cand,
        "caveats": list(res.caveats),
    }


def phase4_solve_qualify(
    cfg: Config,
    onboarded_model: MultiToolGPForwardModel,
    C_X: np.ndarray,
    campaigns: dict[str, Any],
) -> dict[str, Any]:
    banner("PHASE 4 -- SOLVE + AUTO-QUALIFICATION on the onboarded tool (§8 + §11.4)")
    print(
        "Auto-qualification via a DIRECT ConfirmationCampaign on solve()'s output — the flow that\n"
        "closes the 'runner-level qualification auto-invocation' crumb AND is the only one that\n"
        "yields NothingToQualify (zero machine calls) for an Infeasible solve. Plus the\n"
        "ActiveLearningLoop qualification= hook to show in-loop auto-invocation + budget charging."
    )
    view = onboarded_model.for_tool(HELD_OUT)
    variables = list(RECIPE_VARIABLES)

    # A reachable spec, anchored on an on-support reference recipe evaluated at C ground
    # truth (so a witness recipe provably exists), tol = 8% of the campaign output range.
    Yc = campaigns[HELD_OUT]["Y"]
    ref_recipe = _ADAPTER.seed_design(1, seed=SEED + 12345)[0]
    y_ref = truth_outputs(HELD_OUT, [ref_recipe])[0]
    thk_rng = float(Yc[:, THK].max() - Yc[:, THK].min())
    tc_rng = float(Yc[:, TC].max() - Yc[:, TC].min())
    tol_thk, tol_tc = 0.08 * thk_rng, 0.08 * tc_rng
    reachable = {
        "targets": {
            "thickness_grown": {"target": float(y_ref[THK]), "tol": tol_thk},
            "T_center": {"target": float(y_ref[TC]), "tol": tol_tc},
        },
        "max_candidates": 4,
    }
    print(
        f"\nreachable spec (anchored on-support): thickness_grown={y_ref[THK]:.3e}+-{tol_thk:.2e} m, "
        f"T_center={y_ref[TC]:.1f}+-{tol_tc:.1f} K"
    )

    solver = PessimisticInverseSolver(
        view,
        variables=variables,
        output_keys=OUTPUT_KEYS,
        X_train=C_X,
        kappa=KAPPA,
        z_epi=Z_EPI,
        delta_frac=DELTA_FRAC,
        n_restarts=cfg.n_restarts * 24,  # modest multi-start (default is 24*dim = 48 at d=2)
        seed=SEED,
    )
    res = solver.solve(reachable)

    # ---- (i) DIRECT auto-qualification on the reachable solve -------------------
    reach_block: dict[str, Any] = {}
    if isinstance(res, Infeasible):
        print("  reachable solve unexpectedly INFEASIBLE:", res.reason[:90])
        reach_block = {"solve": "INFEASIBLE", "reason": res.reason}
    else:
        print(
            f"  §8 solve -> {len(res)} FEASIBLE candidate(s); auto-running a ConfirmationCampaign..."
        )
        campaign = ConfirmationCampaign(
            process_id="mbe",
            tool_id=HELD_OUT,
            seed=SEED,
            machine=make_verifier(HELD_OUT),
            gate_params=dict(
                targets=reachable["targets"],
                n_runs=cfg.gate_n_runs,
                min_in_spec_rate=cfg.gate_min_rate,
                confidence=cfg.gate_conf,
                provenance_source="physics_sim",  # <- REHEARSAL: not headline-eligible
                output_keys=OUTPUT_KEYS,
            ),
        )
        outcome = campaign.run(res)
        assert isinstance(outcome, CampaignResult)
        summ = _summarize_campaign(outcome)
        print(
            f"  campaign: {summ['n_certified']}/{summ['n_candidates']} certified in "
            f"{summ['n_machine_calls']} confirmation runs (provenance={summ['provenance_source']}, "
            f"headline_eligible={summ['headline_eligible']})"
        )
        reach_block = {
            "solve": "FEASIBLE",
            "n_candidates": len(res),
            "first_candidate_calibration_status": res[0].calibration_status,
            "campaign": summ,
        }

    # ---- (ii) UNREACHABLE spec -> Infeasible -> NothingToQualify, ZERO calls -----
    unreachable = {
        "targets": {
            "thickness_grown": (1.5 * float(Yc[:, THK].max()), 2.0 * float(Yc[:, THK].max()))
        },
        "max_candidates": 4,
    }
    print(
        f"\nunreachable spec: thickness_grown in [{unreachable['targets']['thickness_grown'][0]:.3e},"
        f"{unreachable['targets']['thickness_grown'][1]:.3e}] m (above achievable)"
    )
    res_u = solver.solve(unreachable)
    calls_counter = {"n": 0}

    def counting_verifier(recipe: dict[str, float]) -> dict[str, float]:
        calls_counter["n"] += 1  # must stay 0 for an Infeasible input
        return make_verifier(HELD_OUT)(recipe)

    campaign_u = ConfirmationCampaign(
        process_id="mbe",
        tool_id=HELD_OUT,
        seed=SEED,
        machine=counting_verifier,
        gate_params=dict(
            targets=unreachable["targets"],
            n_runs=cfg.gate_n_runs,
            min_in_spec_rate=cfg.gate_min_rate,
            confidence=cfg.gate_conf,
            provenance_source="physics_sim",
            output_keys=OUTPUT_KEYS,
        ),
    )
    outcome_u = campaign_u.run(res_u)
    is_ntq = isinstance(outcome_u, NothingToQualify)
    print(
        f"  §8 solve -> {'INFEASIBLE' if isinstance(res_u, Infeasible) else 'FEASIBLE'}; "
        f"campaign -> {'NothingToQualify' if is_ntq else type(outcome_u).__name__}; "
        f"machine calls fired = {calls_counter['n']} (must be 0)"
    )
    unreach_block = {
        "solve": "INFEASIBLE" if isinstance(res_u, Infeasible) else "FEASIBLE",
        "reason": res_u.reason if isinstance(res_u, Infeasible) else None,
        "nothing_to_qualify": bool(is_ntq),
        "machine_calls_fired": calls_counter["n"],
    }

    # ---- (iii) ActiveLearningLoop qualification= hook: in-loop auto-invocation ---
    print(
        "\nActiveLearningLoop(qualification=...) hook on tool C (in-loop auto-invocation + budget):"
    )
    loop_geom = _geom_of(HELD_OUT)
    loop_machine_inst = _noisy_machine(loop_geom)

    def loop_machine(recipe: dict[str, float]) -> np.ndarray:
        rec = loop_machine_inst.run(dict(recipe), tool_id=HELD_OUT)
        _, y = records_to_arrays([rec], INPUT_KEYS, OUTPUT_KEYS)
        return y[0]

    lo_thk, hi_thk = float(y_ref[THK] - tol_thk), float(y_ref[THK] + tol_thk)
    lo_tc, hi_tc = float(y_ref[TC] - tol_tc), float(y_ref[TC] + tol_tc)

    def loop_in_spec(y: np.ndarray) -> bool:
        return bool(lo_thk <= float(y[THK]) <= hi_thk and lo_tc <= float(y[TC]) <= hi_tc)

    loop_targets = {
        "thickness_grown": (lo_thk, hi_thk),
        "T_center": (lo_tc, hi_tc),
    }
    loop_campaign = ConfirmationCampaign(
        process_id="mbe",
        tool_id=HELD_OUT,
        seed=SEED,
        machine=make_verifier(HELD_OUT),
        gate_params=dict(
            targets=loop_targets,
            n_runs=cfg.loop_n_runs,
            min_in_spec_rate=0.5,
            confidence=0.8,
            provenance_source="physics_sim",
            output_keys=OUTPUT_KEYS,
        ),
    )
    n_seed = 8
    # budget must cover the seed DoE + a worst-case confirmation of every hitting seed run
    budget = n_seed + n_seed * cfg.loop_n_runs + 8
    loop = ActiveLearningLoop(
        machine=loop_machine,
        in_spec=loop_in_spec,
        variables=variables,
        input_keys=INPUT_KEYS,
        output_keys=OUTPUT_KEYS,
        spec={"targets": loop_targets, "max_candidates": 4},
        cost_recipe=lambda r: 1000.0,
        c_batch=1000.0,
        budget=budget,
        q=4,
        n_seed=n_seed,
        n_pool=cfg.pool,
        kappa=KAPPA,
        z_epi=Z_EPI,
        delta_frac=DELTA_FRAC,
        seed=SEED,
        qualification=loop_campaign,
    )
    traj = loop.run()
    fired = traj.qualification_outcome is not None or len(traj.qualification_rejections) > 0
    n_conf = 0
    if traj.qualification_outcome is not None:
        n_conf += traj.qualification_outcome.n_machine_calls
    for rej in traj.qualification_rejections:
        n_conf += rej.n_machine_calls
    print(
        f"  loop: hit={traj.hit} stop_reason={traj.stop_reason!r}; qualification fired={fired}; "
        f"n_queries={traj.n_queries} (incl. {n_conf} confirmation runs charged to budget)"
    )
    loop_block = {
        "hit": bool(traj.hit),
        "stop_reason": traj.stop_reason,
        "qualification_fired": bool(fired),
        "n_queries_total": int(traj.n_queries),
        "confirmation_runs_charged": int(n_conf),
        "n_rejections": len(traj.qualification_rejections),
        "qualification_outcome": (
            _summarize_campaign(traj.qualification_outcome)
            if traj.qualification_outcome is not None
            else None
        ),
    }

    qualification_ok = (
        reach_block.get("solve") == "FEASIBLE"
        and reach_block.get("campaign", {}).get("n_certified", 0) >= 1
        and is_ntq
        and calls_counter["n"] == 0
        and fired
    )
    print(f"\n  PHASE 4 VERDICT: auto-qualification wired end-to-end: {qualification_ok}")
    return {
        "protocol": (
            "bind onboarded tool; §8 solve (binding 2.0/2.0/0.02); (i) DIRECT ConfirmationCampaign "
            "on the reachable solve output (real confirmation runs on tool C); (ii) unreachable "
            "solve -> Infeasible -> NothingToQualify, zero machine calls; (iii) ActiveLearningLoop "
            "qualification= hook (in-loop auto-invocation, confirmation runs charged to budget)"
        ),
        "flow_choice_justification": (
            "DIRECT ConfirmationCampaign.run(solver.solve(spec)) is the faithful 'solve -> qualify' "
            "runner flow AND the only one producing NothingToQualify (zero calls) on Infeasible; the "
            "ActiveLearningLoop hook is ALSO exercised to demonstrate in-loop auto-invocation + budget "
            "charging."
        ),
        "reachable_spec": reachable["targets"],
        "reachable": reach_block,
        "unreachable_spec": {"thickness_grown": list(unreachable["targets"]["thickness_grown"])},
        "unreachable": unreach_block,
        "loop_qualification_hook": loop_block,
        "rehearsal_caveat": (
            "provenance_source=physics_sim on every confirmation run -> headline_eligible=False; this "
            "is a §11.4 pre-filter rung (in-silico independent-solver + confirmation), NOT tool "
            "qualification. Production promotion requires a real_tool campaign. The Clopper-Pearson "
            "bound also assumes i.i.d. Bernoulli runs (optimistic under the first-wafer/seasoning "
            "serial correlation this very sim can model), and no Cpk/process-window is computed."
        ),
        "qualification_ok": bool(qualification_ok),
        "verdict": f"solve -> ConfirmationCampaign auto-invocation demonstrated end-to-end: {qualification_ok}",
    }


# ---------------------------------------------------------------------------
# PHASE 4b — CONFORMAL WRAP OF THE ONBOARDED TOOL (§5.6 D4 / §13.2 upgrade)
# ---------------------------------------------------------------------------


def _min_n_cal(alpha: float) -> int:
    """Smallest calibration-set size ``n`` admitting a FINITE split-conformal
    quantile at this alpha: the order-statistic index ``k =
    ceil((1-alpha)(n+1))`` (:func:`rig.calibration.conformal.conformal_quantile`)
    must satisfy ``k <= n``, else the honest result is a +inf band."""
    n = 1
    while int(np.ceil((1.0 - alpha) * (n + 1))) > n:
        n += 1
    return n


def _reachable_spec(cfg: Config, campaigns: dict[str, Any]) -> dict[str, Any]:
    """The SAME reachable-spec construction as ``phase4_solve_qualify`` --
    deliberately DUPLICATED rather than factored into a shared call, so a future
    edit to Phase 4b can never perturb Phase 4's recorded numbers. Deterministic
    given ``cfg`` + ``campaigns`` (same seeds, same formula)."""
    Yc = campaigns[HELD_OUT]["Y"]
    ref_recipe = _ADAPTER.seed_design(1, seed=SEED + 12345)[0]
    y_ref = truth_outputs(HELD_OUT, [ref_recipe])[0]
    thk_rng = float(Yc[:, THK].max() - Yc[:, THK].min())
    tc_rng = float(Yc[:, TC].max() - Yc[:, TC].min())
    tol_thk, tol_tc = 0.08 * thk_rng, 0.08 * tc_rng
    return {
        "targets": {
            "thickness_grown": {"target": float(y_ref[THK]), "tol": tol_thk},
            "T_center": {"target": float(y_ref[TC]), "tol": tol_tc},
        },
        "max_candidates": 4,
    }


def phase4b_conformal_wrap(
    cfg: Config, campaigns: dict[str, Any], C_X: np.ndarray, C_Y: np.ndarray
) -> dict[str, Any]:
    banner("PHASE 4b -- CONFORMAL WRAP OF THE ONBOARDED TOOL (§5.6 D4 / §13.2 upgrade)")
    print(
        "Extends Phase 4: carve a HELD-OUT calibration split from the onboarded tool's own\n"
        "runs, wrap its tool view in the real ConformalForwardModel, and re-solve the SAME\n"
        "reachable spec -- so this rehearsal exercises the FULL certified §13.2 path a real\n"
        "M4 tool onboarding would use, not just the raw-sigma kappa margins."
    )

    variables = list(RECIPE_VARIABLES)
    reachable = _reachable_spec(cfg, campaigns)
    tgt = reachable["targets"]
    print(
        f"reachable spec (identical to Phase 4): thickness_grown={tgt['thickness_grown']['target']:.3e}"
        f"+-{tgt['thickness_grown']['tol']:.2e} m, "
        f"T_center={tgt['T_center']['target']:.1f}+-{tgt['T_center']['tol']:.1f} K"
    )

    # ---- (a) split rule --------------------------------------------------------
    n_total = int(C_X.shape[0])
    n_cal_natural = int(round(n_total * CAL_FRACTION))
    n_fit = n_total - n_cal_natural
    n_cal_min = _min_n_cal(CONFORMAL_ALPHA)
    print(
        f"\nsplit rule: trailing {CAL_FRACTION:.0%} of the onboarded tool's {n_total} runs held "
        f"out as calibration (chronological order -- the most-recently-acquired onboarding "
        f"batches), the rest used to fit the tool view -> n_fit={n_fit}  n_cal={n_cal_natural}. "
        f"A finite split-conformal quantile at alpha={CONFORMAL_ALPHA} needs n_cal >= {n_cal_min}."
    )

    # A model fit ONLY on the fit split (never the calibration rows) -- the same
    # pooled-{A,B}+adapt_to_tool recipe as the Phase-3 warm arm, but on a SUBSET of
    # C's runs. This is deliberately NOT Phase 4's fully-onboarded `view`: that one
    # already consumed every C run at fit time, so it could not honestly supply a
    # held-out calibration set (the §5.6 leakage guard SplitConformalCalibrator's
    # own docstring calls out as the caller's responsibility).
    fit_model = _fit_pooled(campaigns, KNOWN, cfg)
    fit_model.adapt_to_tool(HELD_OUT, C_X[:n_fit], C_Y[:n_fit])
    fit_view = fit_model.for_tool(HELD_OUT)

    def _solve(model: Any, X_train: np.ndarray):
        return PessimisticInverseSolver(
            model,
            variables=variables,
            output_keys=OUTPUT_KEYS,
            X_train=X_train,
            kappa=KAPPA,
            z_epi=Z_EPI,
            delta_frac=DELTA_FRAC,
            n_restarts=cfg.n_restarts * 24,
            seed=SEED,
        ).solve(reachable)

    # ---- (b) raw (unwrapped, reduced-data) baseline: what the kappa margins alone admit
    raw_res = _solve(fit_view, C_X[:n_fit])
    raw_candidates = [] if isinstance(raw_res, Infeasible) else list(raw_res)
    if isinstance(raw_res, Infeasible):
        print(f"  raw (fit-split-only) solve unexpectedly INFEASIBLE: {raw_res.reason[:90]}")
    print(
        f"\nraw (unwrapped, fit-split-only model) solve: {len(raw_candidates)} FEASIBLE "
        f"candidate(s), calibration_status="
        f"{raw_candidates[0].calibration_status if raw_candidates else None!r}"
    )

    def _band_widths(model: Any, recipe: dict[str, float]) -> dict[str, float]:
        x = recipes_to_X([recipe])[0]
        cs = np.atleast_2d(model.predict(x).conformal_set)
        return {name: float(cs[j, 1] - cs[j, 0]) for j, name in enumerate(OUTPUT_KEYS)}

    def _raw_widths(cand: Any) -> dict[str, float]:
        return {n: float(hi - lo) for n, (lo, hi) in cand.predicted_outcome_interval.items()}

    def _in_box(model: Any, recipe: dict[str, float]) -> bool:
        x = recipes_to_X([recipe])[0]
        cs = np.atleast_2d(model.predict(x).conformal_set)
        ok = True
        for j, name in enumerate(OUTPUT_KEYS):
            t = tgt.get(name)
            if t is None:
                continue
            lo, hi = t["target"] - t["tol"], t["target"] + t["tol"]
            ok = ok and bool(cs[j, 0] >= lo - 1e-9) and bool(cs[j, 1] <= hi + 1e-9)
        return ok

    def _run_gate(cand_result: Any, tag: str) -> dict[str, Any]:
        campaign = ConfirmationCampaign(
            process_id="mbe",
            tool_id=HELD_OUT,
            seed=SEED,
            machine=make_verifier(HELD_OUT),
            gate_params=dict(
                targets=tgt,
                n_runs=cfg.gate_n_runs,
                min_in_spec_rate=cfg.gate_min_rate,
                confidence=cfg.gate_conf,
                provenance_source="physics_sim",
                output_keys=OUTPUT_KEYS,
            ),
        )
        outcome = campaign.run(cand_result)
        if isinstance(outcome, NothingToQualify):
            print(f"  [{tag}] campaign -> NothingToQualify (0 machine calls)")
            return {"nothing_to_qualify": True, "n_machine_calls": 0}
        summ = _summarize_campaign(outcome)
        print(
            f"  [{tag}] campaign: {summ['n_certified']}/{summ['n_candidates']} certified in "
            f"{summ['n_machine_calls']} confirmation runs"
        )
        return {"nothing_to_qualify": False, **summ}

    # ---- (c) HONEST branch: calibrate on the natural (possibly tiny) split -----
    print(f"\n(c) HONEST branch -- calibrate on the natural n_cal={n_cal_natural} split:")
    calibrator_natural = SplitConformalCalibrator(alpha=CONFORMAL_ALPHA)
    calibrator_natural.fit(fit_view, C_X[n_fit:], C_Y[n_fit:])
    kappa_natural = calibrator_natural.kappa()
    finite_natural = bool(np.all(np.isfinite(kappa_natural)))
    print(
        f"  n_cal={n_cal_natural} vs n_cal_min={n_cal_min}: "
        f"{'finite band -- coverage claim possible' if finite_natural else 'TOO SMALL -> +inf band (honest, no coverage claim)'}"
        f"; kappa={[float(k) for k in kappa_natural]}"
    )
    wrapped_natural = ConformalForwardModel(fit_view, calibrator_natural)
    res_natural = _solve(wrapped_natural, C_X[:n_fit])
    gate_rejected_raw_natural = sum(
        1 for c in raw_candidates if not _in_box(wrapped_natural, c.recipe)
    )
    if isinstance(res_natural, Infeasible):
        print(f"  wrapped(natural) solve -> INFEASIBLE: {res_natural.reason[:110]}")
        natural_block = {
            "n_cal": n_cal_natural,
            "kappa_per_output": [float(k) for k in kappa_natural],
            "finite_band": finite_natural,
            "solve": "INFEASIBLE",
            "reason": res_natural.reason,
        }
    else:
        cands = list(res_natural)
        print(
            f"  wrapped(natural) solve -> {len(cands)} candidate(s), "
            f"calibration_status={cands[0].calibration_status!r}"
        )
        natural_block = {
            "n_cal": n_cal_natural,
            "kappa_per_output": [float(k) for k in kappa_natural],
            "finite_band": finite_natural,
            "solve": "FEASIBLE",
            "n_candidates": len(cands),
            "first_candidate_calibration_status": cands[0].calibration_status,
        }
    natural_block["n_raw_admitted_checked"] = len(raw_candidates)
    natural_block["n_raw_admitted_rejected_by_gate"] = gate_rejected_raw_natural
    natural_block["campaign"] = _run_gate(res_natural, "natural")

    # ---- (d) extra-collection variant: charge the runs for a MINIMAL viable n_cal
    extra_needed = max(0, n_cal_min - n_cal_natural)
    print(
        f"\n(d) EXTRA-COLLECTION variant -- charge {extra_needed} additional tool-C run(s) to "
        f"reach the minimal viable n_cal={n_cal_min} for a finite alpha={CONFORMAL_ALPHA} quantile:"
    )
    if extra_needed > 0:
        extra_recipes = _ADAPTER.seed_design(extra_needed, seed=PHASE4B_SEED_BASE)
        # a PLAIN seeded design (not EPIG-selected): calibration data should stay
        # exchangeable with the operating distribution, not actively chosen, so a
        # fresh Sobol/QMC draw is the honest choice here (rather than reusing the
        # onboarding loop's acquisition, which would bias the calibration sample).
        extra_machine = _noisy_machine(_geom_of(HELD_OUT))  # fresh instance -> local run_index
        Y_extra = _run_outputs(extra_machine, HELD_OUT, extra_recipes)
        X_extra = recipes_to_X(extra_recipes)
        cal_X_ext = np.vstack([C_X[n_fit:], X_extra])
        cal_Y_ext = np.vstack([C_Y[n_fit:], Y_extra])
    else:
        cal_X_ext, cal_Y_ext = C_X[n_fit:], C_Y[n_fit:]
    n_cal_extended = int(cal_X_ext.shape[0])
    calibrator_extended = SplitConformalCalibrator(alpha=CONFORMAL_ALPHA)
    calibrator_extended.fit(fit_view, cal_X_ext, cal_Y_ext)
    kappa_extended = calibrator_extended.kappa()
    finite_extended = bool(np.all(np.isfinite(kappa_extended)))
    print(
        f"  n_cal={n_cal_extended} (natural {n_cal_natural} + {extra_needed} extra): "
        f"{'FINITE band' if finite_extended else 'STILL INFINITE'}; "
        f"kappa={[float(k) for k in kappa_extended]}"
    )
    wrapped_extended = ConformalForwardModel(fit_view, calibrator_extended)
    res_extended = _solve(wrapped_extended, C_X[:n_fit])
    gate_rejected_raw_extended = sum(
        1 for c in raw_candidates if not _in_box(wrapped_extended, c.recipe)
    )

    width_comparison = None
    if raw_candidates:
        probe_recipe = raw_candidates[0].recipe
        width_comparison = {
            "probe_recipe": dict(probe_recipe),
            "raw_pessimistic_interval_width": _raw_widths(raw_candidates[0]),
            "conformal_width_natural": _band_widths(wrapped_natural, probe_recipe),
            "conformal_width_extended": _band_widths(wrapped_extended, probe_recipe),
        }

    upgraded_to_conformal_checked = False
    if isinstance(res_extended, Infeasible):
        print(f"  wrapped(extended) solve -> INFEASIBLE: {res_extended.reason[:110]}")
        extended_block = {
            "n_cal": n_cal_extended,
            "extra_runs_collected": extra_needed,
            "kappa_per_output": [float(k) for k in kappa_extended],
            "finite_band": finite_extended,
            "solve": "INFEASIBLE",
            "reason": res_extended.reason,
        }
    else:
        cands = list(res_extended)
        upgraded_to_conformal_checked = bool(
            cands and cands[0].calibration_status == "conformal-checked"
        )
        print(
            f"  wrapped(extended) solve -> {len(cands)} candidate(s), "
            f"calibration_status={cands[0].calibration_status!r} "
            f"(upgraded to conformal-checked: {upgraded_to_conformal_checked})"
        )
        extended_block = {
            "n_cal": n_cal_extended,
            "extra_runs_collected": extra_needed,
            "kappa_per_output": [float(k) for k in kappa_extended],
            "finite_band": finite_extended,
            "solve": "FEASIBLE",
            "n_candidates": len(cands),
            "first_candidate_calibration_status": cands[0].calibration_status,
        }
    extended_block["n_raw_admitted_checked"] = len(raw_candidates)
    extended_block["n_raw_admitted_rejected_by_gate"] = gate_rejected_raw_extended
    extended_block["campaign"] = _run_gate(res_extended, "extended")

    budget = {
        "onboard_budget_runs_phase3": cfg.onboard_budget,
        "n_total_C_runs_available": n_total,
        "n_fit": n_fit,
        "n_cal_natural": n_cal_natural,
        "extra_collection_runs_charged": extra_needed,
        "total_new_machine_calls_this_phase": extra_needed,
    }
    print(
        f"\nbudget: {n_total} onboarding run(s) already charged in Phase 3 + "
        f"{extra_needed} extra calibration run(s) charged HERE (Phase 4b)."
    )

    verdict = (
        f"n_cal_natural={n_cal_natural} < n_cal_min={n_cal_min}: honest branch band is "
        f"{'finite' if finite_natural else 'INFINITE -> the §13.2 gate rejects every candidate (0 conformal-checked)'}. "
        f"Extra-collection branch (n_cal={n_cal_extended}) band is "
        f"{'finite' if finite_extended else 'still infinite'}; candidates upgrade to "
        f"calibration_status='conformal-checked': {upgraded_to_conformal_checked}."
    )
    print(f"\nPHASE 4b VERDICT: {verdict}")

    return {
        "protocol": (
            "carve a held-out calibration split from the onboarded tool's own accumulated "
            "runs (never the model's own fit rows -- the §5.6 leakage guard); fit a "
            "SplitConformalCalibrator on it; wrap the tool view in ConformalForwardModel; "
            "re-solve the SAME reachable spec used by Phase 4. Reports the honest natural-"
            "split outcome AND a labelled extra-collection variant that charges the runs "
            "needed for a minimal viable n_cal at alpha=0.1."
        ),
        "reachable_spec": tgt,
        "split_rule": (
            f"trailing {CAL_FRACTION:.3f} fraction of the {n_total} onboarded-tool runs "
            "(chronological -- most-recently-acquired onboarding batches held out)"
        ),
        "alpha": CONFORMAL_ALPHA,
        "n_cal_min_for_finite_quantile": n_cal_min,
        "raw_unwrapped_reduced_data_model": {
            "n_fit": n_fit,
            "n_candidates": len(raw_candidates),
            "calibration_status": (
                raw_candidates[0].calibration_status if raw_candidates else None
            ),
        },
        "honest_natural_split": natural_block,
        "extra_collection_variant": extended_block,
        "band_width_comparison": width_comparison,
        "budget": budget,
        "upgraded_to_conformal_checked": upgraded_to_conformal_checked,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def run_rehearsal(cfg: Config) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t_all = time.perf_counter()

    t0 = time.perf_counter()
    phase1, campaigns = phase1_fleet(cfg)
    timings["phase1_fleet"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    phase2, X_test, test_recipes, y_truth = phase2_pooling(cfg, campaigns)
    timings["phase2_pooling"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    phase3, warm_model, C_X, C_Y = phase3_onboarding(cfg, campaigns, X_test, y_truth)
    timings["phase3_onboarding"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    phase4 = phase4_solve_qualify(cfg, warm_model, C_X, campaigns)
    timings["phase4_solve_qualify"] = round(time.perf_counter() - t0, 2)

    # Phase 4b is added STRICTLY AFTER phases 1-4 and uses ONLY its own seed
    # namespace (PHASE4B_SEED_BASE) plus deterministic re-derivations of C_X/C_Y
    # already returned by phase 3 -- it consumes no shared global RNG state, so
    # phases 1-4's recorded numbers are unperturbed by its presence.
    t0 = time.perf_counter()
    phase4b = phase4b_conformal_wrap(cfg, campaigns, C_X, C_Y)
    timings["phase4b_conformal_wrap"] = round(time.perf_counter() - t0, 2)

    timings["total"] = round(time.perf_counter() - t_all, 2)

    return {
        "meta": {
            "title": "in-silico multi-tool M4 dress rehearsal",
            "provenance": "physics_sim",
            "REHEARSAL": True,
            "headline_eligible": False,
            "caveat": (
                "Every number is produced by the in-silico MBE machine (physics_sim) and is a "
                "MACHINERY PROOF only. NOT headline evidence; the scientific claim stays gated on "
                "M0 real data."
            ),
            "seed": SEED,
            "mode": cfg.label,
            "config": {
                "n_train_per_tool": cfg.n_train,
                "n_test": cfg.n_test,
                "n_restarts": cfg.n_restarts,
                "max_iter": cfg.max_iter,
                "onboard_budget": cfg.onboard_budget,
                "q": cfg.q,
                "pool": cfg.pool,
                "fewshot_K": list(cfg.fewshot_K),
                "gate_n_runs": cfg.gate_n_runs,
                "gate_min_in_spec_rate": cfg.gate_min_rate,
                "gate_confidence": cfg.gate_conf,
                "loop_n_runs": cfg.loop_n_runs,
            },
            "feasibility_policy": {"kappa": KAPPA, "z_epi": Z_EPI, "delta_frac": DELTA_FRAC},
            "fleet": {tid: FLEET[i][1] for i, tid in enumerate(TOOL_IDS)},
            "held_out_tool": HELD_OUT,
            "timings": timings,
        },
        "phase1_fleet": phase1,
        "phase2_pooling": phase2,
        "phase3_onboarding": phase3,
        "phase4_solve_qualify": phase4,
        "phase4b_conformal_wrap": phase4b,
        "verdicts": {
            "loto_domination_all_folds": phase2["loto_zero_shot_domination"]["all_folds_dominate"],
            "pooling_verdict": phase2["pooling_verdict"],
            "epig_positive_on_unknown_tool": phase3["epig_positive_on_unknown_tool"],
            "onboarding_verdict": phase3["onboarding_verdict"],
            "auto_qualification_ok": phase4["qualification_ok"],
            "conformal_wrap_upgraded_to_conformal_checked": phase4b[
                "upgraded_to_conformal_checked"
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="tiny fast shape check")
    parser.add_argument("--full", action="store_true", help="modest full run (default)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    cfg = SMOKE if (args.smoke and not args.full) else FULL

    # full -> the committed artifact docs/multitool-rehearsal.json; smoke -> a throwaway
    # in the OS temp dir (a --smoke run must never overwrite or shadow the full artifact).
    if cfg.label == "smoke":
        default_out = Path(tempfile.gettempdir()) / "multitool-rehearsal.smoke.json"
    else:
        default_out = Path(__file__).resolve().parents[1] / "docs" / "multitool-rehearsal.json"
    out_path = args.out or default_out

    banner(
        f"IN-SILICO MULTI-TOOL M4 DRESS REHEARSAL ({cfg.label.upper()}) -- provenance=physics_sim"
    )
    print(
        "*** REHEARSAL: machinery proof only, NOT headline evidence (M0 real data still owed). ***"
    )

    payload = run_rehearsal(cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")

    banner("SUMMARY")
    v = payload["verdicts"]
    print(f"  §5.8 LOTO domination (all folds) : {v['loto_domination_all_folds']}")
    print(f"  pooling verdict                  : {v['pooling_verdict']}")
    print(f"  EPIG > 0 on unknown-tool path    : {v['epig_positive_on_unknown_tool']}")
    print(f"  onboarding                       : {v['onboarding_verdict']}")
    print(f"  solve -> auto-qualification wired : {v['auto_qualification_ok']}")
    print(
        f"  conformal wrap -> conformal-checked: {v['conformal_wrap_upgraded_to_conformal_checked']}"
    )
    print(f"\nresults -> {out_path}")
    print(f"total wall time: {payload['meta']['timings']['total']} s")
    banner("DONE (REHEARSAL -- physics_sim -- not headline evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
