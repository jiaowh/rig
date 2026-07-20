"""Pre-register the bake-off target set → frozen ``targets.json`` (prereg §4.1).

Run this ONCE, BEFORE any arm. It:
  1. dense-searches the MBE machine's NOISE-FREE path (``evaluate_physics`` with nominal
     params — see ``run_bakeoff.MachineHarness``) over the 2-D recipe box: Sobol ≥4096
     points → the reachable output cloud;
  2. generates ≥20 targets — ≥10 clearly-feasible, ≥5 feasible-but-hard (tiny boxes on the
     reachable-set boundary), ≥5 ground-truth-INFEASIBLE (boxes off the reachable manifold);
  3. LABELS each target's ``feasible_truth`` by dense search + local refine — never by any
     method under test (prereg §4.1);
  4. writes ``targets.json`` with a per-target spec box (SI units), the feasibility verdict,
     a ground-truth witness recipe/outcome when feasible, and a sha256 FREEZE hash the
     runner refuses to run against if it changes.

Ground truth = the machine's deterministic noise-free path; a genuine deterministic
mechanism exists (see the harness docstring), so the seeded-replicate-average fallback is
not used.

    PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/pre_register_targets.py
"""

from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from rig_adapters.mbe.adapter import (
    MACHINE_CONFIG_DEFAULTS,
    RECIPE_VARIABLES,
    evaluate_physics,
)
from rig_adapters.mbe.outcomes import metrics_to_outcomes

from run_bakeoff import CONTROLLED_OUTPUTS, RECIPE_KEYS, canonical_hash, in_spec

_OUT_PATH = Path(__file__).resolve().parent / "targets.json"
_N_DENSE = 4096
_SEED = 20260717
# One-sided spec: slip_max_ratio ≤ _SLIP_UPPER. Set to the 85th percentile of the
# REACHABLE slip range (filled in build_targets), NOT the nominal ≤1 — slip only reaches
# ~0.3 here, so ≤1 is trivially satisfied AND makes MFL's one-sided point rule
# (bound − σ ≈ 0.95) chase an unreachable value and wreck the other outputs. A
# reachable-relative bound is a genuine, satisfiable-with-effort one-sided constraint.
_SLIP_UPPER = 1.0  # placeholder; set per reachable range in build_targets

_X_LO = np.array([v.lower for v in RECIPE_VARIABLES], dtype=float)
_X_HI = np.array([v.upper for v in RECIPE_VARIABLES], dtype=float)


def _outcome_si(recipe_vec: np.ndarray) -> dict[str, float]:
    recipe = {k: float(v) for k, v in zip(RECIPE_KEYS, recipe_vec, strict=True)}
    metrics = evaluate_physics(recipe, MACHINE_CONFIG_DEFAULTS)
    return {o.name: float(o.value.magnitude) for o in metrics_to_outcomes(metrics)}


def _controlled(outcome: dict[str, float]) -> np.ndarray:
    return np.array([outcome[n] for n in CONTROLLED_OUTPUTS], dtype=float)


def dense_cloud() -> tuple[np.ndarray, np.ndarray]:
    """(recipes (N,2), controlled outputs (N,3)) over a Sobol design of the recipe box."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=_SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = sampler.random(_N_DENSE)
    X = qmc.scale(u, _X_LO, _X_HI)
    Z = np.array([_controlled(_outcome_si(x)) for x in X])
    return X, Z


def _box_distance(z: np.ndarray, lo: np.ndarray, hi: np.ndarray, scale: np.ndarray) -> float:
    """Normalized L2 excess of a controlled-output point beyond a spec box (0 iff inside)."""
    excess = np.maximum(np.maximum(lo - z, z - hi), 0.0) / scale
    return float(np.linalg.norm(excess))


def label_feasibility(
    lo: np.ndarray, hi: np.ndarray, scale: np.ndarray, X: np.ndarray, Z: np.ndarray
) -> tuple[bool, dict | None, dict | None]:
    """Dense search + local refine on the NOISE-FREE path against the EFFECTIVE (stored)
    spec bounds — ``lo``/``hi`` must already carry the one-sided-slip override
    (:func:`_effective_bounds`), so ``feasible_truth`` and the witness are labelled against
    exactly the spec the runner scores. Returns
    (feasible_truth, witness_recipe, witness_outcome)."""
    d = np.array([_box_distance(z, lo, hi, scale) for z in Z])
    k = int(np.argmin(d))
    if d[k] <= 1e-9:
        witness = X[k]
    else:
        # local refine from the best dense point (Sobol ≥4096 + local refine, prereg §4.1).
        res = minimize(
            lambda x: _box_distance(_controlled(_outcome_si(x)), lo, hi, scale),
            X[k],
            method="L-BFGS-B",
            bounds=list(zip(_X_LO, _X_HI, strict=True)),
            options={"maxiter": 200},
        )
        if res.fun > 1e-6:
            return False, None, None
        witness = res.x
    outcome = _outcome_si(witness)
    recipe = {k2: float(v) for k2, v in zip(RECIPE_KEYS, witness, strict=True)}
    return True, recipe, outcome


def _effective_bounds(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The spec box that is ACTUALLY stored and scored: ``slip_max_ratio`` is one-sided
    upper (lower = -inf, upper = ``_SLIP_UPPER``); the other controlled outputs keep their
    two-sided box.

    This is the SINGLE definition of the spec used for BOTH feasibility labelling and
    storage — the fix for the target-label/scored-spec mismatch. Previously the label was
    computed against the two-sided slip box while the stored/scored spec overrode slip to
    one-sided, so a witness that satisfied the two-sided slip band could VIOLATE the
    one-sided bound actually scored (proven: 6 stored witnesses did, and hard_01/03/04 had
    zero feasible recipes under the stored spec yet were labelled feasible)."""
    eff_lo = np.asarray(lo, dtype=float).copy()
    eff_hi = np.asarray(hi, dtype=float).copy()
    idx = CONTROLLED_OUTPUTS.index("slip_max_ratio")
    eff_lo[idx] = -np.inf
    eff_hi[idx] = float(_SLIP_UPPER)
    return eff_lo, eff_hi


def _spec_json(lo: np.ndarray, hi: np.ndarray) -> dict[str, list[float | None]]:
    """spec as {output: [lower, upper]} in SI, null on an open side. Derived from the SAME
    effective bounds used to label feasibility (:func:`_effective_bounds`), so the stored
    spec and the labelled spec can never diverge again."""
    eff_lo, eff_hi = _effective_bounds(lo, hi)
    spec: dict[str, list[float | None]] = {}
    for i, name in enumerate(CONTROLLED_OUTPUTS):
        lo_i = None if not np.isfinite(eff_lo[i]) else float(eff_lo[i])
        hi_i = None if not np.isfinite(eff_hi[i]) else float(eff_hi[i])
        spec[name] = [lo_i, hi_i]
    return spec


def build_targets(X: np.ndarray, Z: np.ndarray) -> list[dict]:
    global _SLIP_UPPER
    rng = np.random.default_rng(_SEED)
    zmin, zmax = Z.min(axis=0), Z.max(axis=0)
    zrange = np.maximum(zmax - zmin, 1e-9)
    scale = zrange  # normalize box distances by reachable range per output
    idx_slip = CONTROLLED_OUTPUTS.index("slip_max_ratio")
    # a genuine, satisfiable-with-effort one-sided bound: 85th pct of reachable slip.
    _SLIP_UPPER = float(np.percentile(Z[:, idx_slip], 85.0))

    def make(spec_lo, spec_hi, cls) -> dict:
        # label AND store against the SAME effective (one-sided-slip) bounds.
        eff_lo, eff_hi = _effective_bounds(spec_lo, spec_hi)
        feas, wr, wo = label_feasibility(eff_lo, eff_hi, scale, X, Z)
        spec = _spec_json(spec_lo, spec_hi)
        if feas:
            # every stored witness MUST satisfy the stored (scored) spec — asserted here so
            # a label/spec mismatch can never survive generation again (the blocker fix).
            assert wo is not None and in_spec(wo, spec, tol=0.0), (
                f"{cls}: labelled feasible but the witness violates the stored spec {spec}"
            )
        return {
            "class": cls,
            "spec": spec,
            "feasible_truth": bool(feas),
            "witness_recipe": wr,
            "witness_outcome": wo,
        }

    def take(pool, want: bool, n: int) -> list[dict]:
        """First ``n`` labelled candidates whose feasibility == ``want`` (dense-search
        verified against the STORED spec — the class is what it CLAIMS, never assumed)."""
        picked = [c for c in pool if c["feasible_truth"] is want][:n]
        return picked

    # Dense points that meet the one-sided slip bound = the reachable set UNDER THE STORED
    # SPEC. Feasible + feasible-but-hard targets are anchored here, so each witness (its
    # anchor) genuinely satisfies the scored spec, slip included — not the two-sided box.
    slip_ok = np.where(Z[:, idx_slip] <= _SLIP_UPPER)[0]
    Zf = Z[slip_ok]

    # -- clearly-feasible: moderate boxes around random slip-feasible reachable anchors --
    half_easy = 0.12 * zrange
    feas_anchors = rng.choice(slip_ok, size=min(24, len(slip_ok)), replace=False)
    feas_pool = [make(Z[a] - half_easy, Z[a] + half_easy, "feasible") for a in feas_anchors]
    feasible = take(feas_pool, True, 10)

    # -- feasible-but-hard: tiny boxes on the reachable BOUNDARY *of the slip-feasible
    #    sub-cloud* (extremes + outer shell in the two two-sided outputs) --------------
    half_hard = 0.02 * zrange
    edge_local = [
        int(np.argmin(Zf[:, 0])),
        int(np.argmax(Zf[:, 0])),
        int(np.argmin(Zf[:, 1])),
        int(np.argmax(Zf[:, 1])),
    ]
    # plus outer-shell points (far from the sub-cloud centroid) for boundary variety.
    c2 = Zf[:, :2].mean(axis=0)
    shell_local = np.argsort(-np.linalg.norm((Zf[:, :2] - c2) / zrange[:2], axis=1))[:20]
    hard_local = list(dict.fromkeys([*edge_local, *(int(i) for i in shell_local)]))
    hard_pool = [make(Zf[a] - half_hard, Zf[a] + half_hard, "hard") for a in hard_local]
    hard = take(hard_pool, True, 5)

    # -- ground-truth-infeasible: boxes off the reachable manifold ---------------------
    infeas_pool: list[dict] = []
    # (a) nonuniformity BELOW the global reachable minimum (varied T_center bands).
    for tc in np.linspace(zmin[1] + 0.2 * zrange[1], zmax[1] - 0.2 * zrange[1], 4):
        lo = np.array([0.0, tc - 0.05 * zrange[1], 0.0])
        hi = np.array([0.4 * zmin[0], tc + 0.05 * zrange[1], _SLIP_UPPER])
        infeas_pool.append(make(lo, hi, "infeasible"))
    # (b) T_center OUTSIDE the reachable range (above max / below min).
    mid0 = 0.5 * (zmin[0] + zmax[0])
    for tc_lo, tc_hi in (
        (zmax[1] + 0.3 * zrange[1], zmax[1] + 0.6 * zrange[1]),
        (zmin[1] - 0.6 * zrange[1], zmin[1] - 0.3 * zrange[1]),
    ):
        lo = np.array([mid0 - 0.1 * zrange[0], tc_lo, 0.0])
        hi = np.array([mid0 + 0.1 * zrange[0], tc_hi, _SLIP_UPPER])
        infeas_pool.append(make(lo, hi, "infeasible"))
    infeasible = take(infeas_pool, False, 5)

    targets: list[dict] = []
    for cls_list in (feasible, hard, infeasible):
        for c in cls_list:
            cls = c["class"]
            c["id"] = f"{cls}_{sum(1 for t in targets if t['class'] == cls):02d}"
            targets.append(c)
    return targets


def main() -> None:
    print(f"dense search: {_N_DENSE} noise-free evals over the recipe box ...")
    X, Z = dense_cloud()
    print(
        "reachable ranges (SI):",
        {
            n: (round(float(Z[:, i].min()), 6), round(float(Z[:, i].max()), 6))
            for i, n in enumerate(CONTROLLED_OUTPUTS)
        },
    )
    targets = build_targets(X, Z)

    counts: dict[str, int] = {}
    feas: dict[str, int] = {}
    for t in targets:
        counts[t["class"]] = counts.get(t["class"], 0) + 1
        feas[t["class"]] = feas.get(t["class"], 0) + int(t["feasible_truth"])
    print("targets by class (feasible_truth count):")
    for cls in ("feasible", "hard", "infeasible"):
        print(f"  {cls:<11} n={counts.get(cls, 0)}  feasible_truth={feas.get(cls, 0)}")

    # sanity: the classes must actually be what they claim (else the target set is broken).
    assert counts.get("feasible", 0) >= 10 and feas.get("feasible", 0) == counts["feasible"]
    assert counts.get("hard", 0) >= 5 and feas.get("hard", 0) == counts["hard"]
    assert counts.get("infeasible", 0) >= 5 and feas.get("infeasible", 0) == 0, (
        "an 'infeasible' target was reachable by dense search — tighten the generator"
    )
    # every stored witness satisfies the stored (scored) spec; infeasibles carry none.
    for t in targets:
        if t["feasible_truth"]:
            assert t["witness_outcome"] is not None and in_spec(
                t["witness_outcome"], t["spec"], tol=0.0
            ), f"{t['id']}: stored witness violates the stored spec {t['spec']}"
        else:
            assert t["witness_recipe"] is None and t["witness_outcome"] is None

    payload = {
        "meta": {
            "created": datetime.now(UTC).isoformat(),
            "comparator": "Gu et al. 2025, arXiv:2505.16060 (MFL)",
            "prereg": "docs/prereg-mfl-bakeoff-2026-07-17.md",
            "controlled_outputs": list(CONTROLLED_OUTPUTS),
            "units": "SI (nonuniformity fraction, T_center K, slip dimensionless)",
            "slip_upper_spec": _SLIP_UPPER,
            "n_dense": _N_DENSE,
            "dense_seed": _SEED,
            "machine_config": dict(MACHINE_CONFIG_DEFAULTS),
            "ground_truth": "evaluate_physics nominal (noise-free fast-Arrhenius path)",
        },
        "targets": targets,
    }
    payload["hash"] = canonical_hash(payload)
    _OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {_OUT_PATH} ({len(targets)} targets)  hash={payload['hash'][:16]}")


if __name__ == "__main__":
    main()
