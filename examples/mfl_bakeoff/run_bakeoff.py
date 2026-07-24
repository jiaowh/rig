"""RIG vs MFL bake-off runner (pre-registered: docs/prereg-mfl-bakeoff-2026-07-17.md).

Four arms on the MBE :class:`~rig_adapters.mbe.machine.InSilicoMachine`, scored against
GROUND TRUTH (the machine's noise-free path — see ``ground_truth_outcome``), never
against any method's own surrogate (prereg §4.3 / audit F2):

  * ``rig``            — GPForwardModel on the seed runs → PessimisticInverseSolver (§8).
  * ``rig-reval``      — + a ConformalForwardModel re-validation gate (§13.2), fit on a
                         held-out slice of the SAME seed runs (the owed "does conformal
                         re-validation change the miss rate" experiment).
  * ``mfl-charitable`` — MFL (Gu et al. 2025), FD Loop-B probes NOT counted (their setting:
                         a differentiable deployed M).
  * ``mfl-deployable`` — the SAME trained MFL, every machine touch counted (a real tool).

The two MFL arms share ONE trained reverse model R and differ ONLY in the machine-query
LEDGER (prereg §3 resolution). Metrics use the prereg §0 names VERBATIM. Both venues are
in-silico — say so in every write-up.

The metric + scorer helpers at module top depend only on numpy/stdlib and are imported by
``tests/test_mfl_baseline.py`` (test 6); every heavy import (torch, the GP, the MBE sim)
is deferred into the run functions so the scorer stays importable without them.

Usage
-----
    PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/run_bakeoff.py --smoke
    PYTHONIOENCODING=utf-8 python examples/mfl_bakeoff/run_bakeoff.py --full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# controlled outputs every target constrains (a FIXED set so one R conditions on a
# fixed-dim z'): two two-sided outputs + one one-sided (slip ≤ 1.0) to exercise the
# one-sided code paths in the scorer and MFL's point-target rule. SI units throughout
# (nonuniformity_pct → fraction, T_center → K, slip_max_ratio → dimensionless).
CONTROLLED_OUTPUTS: tuple[str, ...] = ("nonuniformity_pct", "T_center", "slip_max_ratio")
RECIPE_KEYS: tuple[str, ...] = ("T_heater", "film_thickness")

_TARGETS_PATH = Path(__file__).resolve().parent / "targets.json"

# ==========================================================================
# metric + scorer helpers (numpy/stdlib only — imported by test 6)
# ==========================================================================


@dataclass
class ArmTargetResult:
    """One arm's outcome for one target (the scorer's unit of account)."""

    target_id: str
    presented: bool  # did the method present a recipe? (RIG: FEASIBLE only; MFL: always)
    recipe: dict[str, float] | None
    ground_truth: dict[str, float] | None  # noise-free outcome at ``recipe`` (None if abstained)
    spec: dict[str, tuple[float | None, float | None]]
    feasible_truth: bool
    yield_under_noise: float | None = None
    normalized_margin: float | None = None


def _bounds(entry: Sequence[float | None]) -> tuple[float, float]:
    lo, hi = entry
    return (-math.inf if lo is None else float(lo), math.inf if hi is None else float(hi))


def in_spec(
    outcome: Mapping[str, float],
    spec: Mapping[str, Sequence[float | None]],
    tol: float = 0.0,
) -> bool:
    """True iff every constrained output is inside its (possibly one-sided) spec edge."""
    for name, entry in spec.items():
        lo, hi = _bounds(entry)
        y = float(outcome[name])
        if not (lo - tol <= y <= hi + tol):
            return False
    return True


def certified_miss_rate(results: Sequence[ArmTargetResult]) -> float:
    """prereg §0 headline: among recipes the method PRESENTS as meeting spec, the
    fraction out-of-spec on ground truth. RIG abstentions are ``presented=False`` and are
    therefore NEITHER a miss NOR in the denominator. ``nan`` when nothing was presented."""
    presented = [r for r in results if r.presented]
    if not presented:
        return float("nan")
    misses = sum(1 for r in presented if not in_spec(r.ground_truth, r.spec))
    return misses / len(presented)


def false_abstention_rate(results: Sequence[ArmTargetResult]) -> float:
    """prereg §0: among ground-truth-FEASIBLE targets, the fraction the method refused.
    MFL is structurally 0 (no abstention branch — prereg P4). ``nan`` if no feasible
    targets."""
    feasible = [r for r in results if r.feasible_truth]
    if not feasible:
        return float("nan")
    return sum(1 for r in feasible if not r.presented) / len(feasible)


def normalized_margin_of(
    outcome: Mapping[str, float],
    spec: Mapping[str, Sequence[float | None]],
    sigma: Mapping[str, float],
) -> float:
    """prereg §0: ``min_j`` distance-to-nearest-spec-edge / aleatoric σ_j at x. One-sided
    constraints use the single finite edge. Negative when out of spec."""
    per_output = []
    for name, entry in spec.items():
        lo, hi = _bounds(entry)
        y = float(outcome[name])
        s = max(float(sigma[name]), 1e-12)
        edges = []
        if math.isfinite(lo):
            edges.append((y - lo) / s)
        if math.isfinite(hi):
            edges.append((hi - y) / s)
        per_output.append(min(edges))
    return min(per_output)


def yield_from_replicates(
    replicates: Sequence[Mapping[str, float]],
    spec: Mapping[str, Sequence[float | None]],
) -> float:
    """prereg §0 ``yield_under_noise``: fraction of noisy replicates in spec at fixed x."""
    if not replicates:
        return float("nan")
    return sum(1 for r in replicates if in_spec(r, spec)) / len(replicates)


def _median(values: Sequence[float]) -> float:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.median(vals)) if vals else float("nan")


# ==========================================================================
# targets.json freeze check (prereg §4.1 — frozen before any arm runs)
# ==========================================================================


def canonical_hash(payload: Mapping) -> str:
    """sha256 over the canonical (sorted-key) JSON of ``{meta, targets}`` — the frozen
    identity of the target set. The runner refuses if the on-disk hash disagrees."""
    body = {"meta": payload["meta"], "targets": payload["targets"]}
    blob = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_frozen_targets(path: Path = _TARGETS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run pre_register_targets.py FIRST (targets are frozen "
            "before any arm runs, prereg §4.1)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recomputed = canonical_hash(payload)
    if recomputed != payload.get("hash"):
        raise RuntimeError(
            "targets.json HASH MISMATCH — the frozen target set was edited after "
            f"registration (stored {payload.get('hash')}, recomputed {recomputed}). "
            "Refusing to run: re-register the targets or restore the file (prereg §4.1)."
        )
    return payload


def _spec_from_json(
    spec_json: Mapping[str, Sequence],
) -> dict[str, tuple[float | None, float | None]]:
    return {name: (entry[0], entry[1]) for name, entry in spec_json.items()}


# ==========================================================================
# machine harness — ground truth (noise-free) + noisy queries (heavy imports)
# ==========================================================================


class MachineHarness:
    """The MBE machine, split into the two paths the prereg needs.

    GROUND-TRUTH / NOISE-FREE — ``ground_truth_outcome``. The deterministic mechanism
    found in :mod:`rig_adapters.mbe.machine`: ``evaluate_physics(recipe, machine_config)``
    with its nominal keyword params (``emissivity=NOMINAL_EMISSIVITY``,
    ``cosine_n=NOMINAL_COSINE_N``, ``flux_eff=1.0``) is the fast-Arrhenius path with NO
    metrology noise, NO first-wafer offset, NO seasoning, NO tool perturbation — i.e.
    exactly an ``InSilicoMachine(PathologyConfig())`` (all pathologies OFF) run, which the
    machine's own determinism contract makes bit-identical. We take the outcome through
    ``metrics_to_outcomes`` so its SI canonicalization matches the noisy path byte-for-byte.
    (A genuine deterministic path exists, so the seeded-replicate-average fallback is not
    needed.)

    NOISY QUERIES — ``noisy_outcome`` / ``noisy_replicates``: an
    ``InSilicoMachine(PathologyConfig(metrology_noise=True))`` whose heteroscedastic
    metrology noise is ON. Seeded and deterministic.
    """

    def __init__(self, machine_config: Mapping[str, float] | None, base_seed: int) -> None:
        from rig_adapters.mbe.adapter import MACHINE_CONFIG_DEFAULTS

        self.machine_config = dict(MACHINE_CONFIG_DEFAULTS, **(machine_config or {}))
        self.base_seed = int(base_seed)

    def _recipe_si(self, recipe: Mapping[str, float]) -> dict[str, float]:
        return {k: float(recipe[k]) for k in RECIPE_KEYS}

    def ground_truth_outcome(self, recipe: Mapping[str, float]) -> dict[str, float]:
        """Noise-free SI outcome (all declared outputs)."""
        from rig_adapters.mbe.adapter import evaluate_physics
        from rig_adapters.mbe.outcomes import metrics_to_outcomes

        metrics = evaluate_physics(self._recipe_si(recipe), self.machine_config)
        return {o.name: float(o.value.magnitude) for o in metrics_to_outcomes(metrics)}

    def _noisy_machine(self, seed: int):
        from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

        return InSilicoMachine(
            PathologyConfig(metrology_noise=True),
            seed=seed,
            machine_config=self.machine_config,
        )

    def noisy_replicates(
        self, recipe: Mapping[str, float], n: int, seed: int
    ) -> list[dict[str, float]]:
        """``n`` independent noisy SI outcomes at a fixed recipe (seeded)."""
        machine = self._noisy_machine(seed)
        recipe_si = self._recipe_si(recipe)
        out = []
        for _ in range(n):
            rec = machine.run(recipe_si)
            out.append({o.name: float(o.value.magnitude) for o in rec.outcomes})
        return out


# ==========================================================================
# seed data (shared, identical budget for both methods — prereg §4.4)
# ==========================================================================


def generate_seed_runs(harness: MachineHarness, n_seed: int, seed: int):
    """The SAME seed RunRecords both RIG's GP and MFL's emulator train on (prereg §4.4).
    Returns ``(records, X, Y_all, Z_controlled)`` — SI arrays; ``Y_all`` over every
    declared output (for the GP), ``Z_controlled`` over the controlled subset (for E)."""
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig
    from rig_adapters.mbe.outcomes import OUTPUT_ORDER

    ranges = {v.name: (v.lower, v.upper) for v in RECIPE_VARIABLES}
    from rig.interfaces import sobol_seed_design

    design = sobol_seed_design(ranges, n_seed, seed)
    machine = InSilicoMachine(
        PathologyConfig(metrology_noise=True), seed=seed, machine_config=harness.machine_config
    )
    records = [machine.run({k: d[k] for k in RECIPE_KEYS}) for d in design]

    from rig.forward.data import records_to_arrays

    X, Y_all = records_to_arrays(records, RECIPE_KEYS, OUTPUT_ORDER)
    ci = [OUTPUT_ORDER.index(n) for n in CONTROLLED_OUTPUTS]
    return records, X, Y_all, Y_all[:, ci]


# ==========================================================================
# RIG arms
# ==========================================================================


def _spec_for_solver(spec: Mapping[str, tuple]) -> dict[str, tuple[float | None, float | None]]:
    """parse_targets accepts (lower, upper) with None for an open side — reuse directly."""
    return {name: (entry[0], entry[1]) for name, entry in spec.items()}


def run_rig_arm(harness, targets, X_seed, Y_all, *, use_reval: bool, seed: int):
    """Fit the GP on the seed runs and solve each target with the §8 pessimistic inverse.
    ``use_reval`` adds the §13.2 ConformalForwardModel re-validation gate on a held-out
    slice of the SAME seed runs."""
    from rig.calibration import ConformalForwardModel, SplitConformalCalibrator
    from rig.forward.gp import GPForwardModel
    from rig.interfaces import Infeasible
    from rig.inverse.pessimistic import PessimisticInverseSolver
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES
    from rig_adapters.mbe.outcomes import OUTPUT_ORDER

    reval_model = None
    if use_reval:
        n = X_seed.shape[0]
        n_fit = max(4, int(round(0.7 * n)))
        gp = GPForwardModel(list(RECIPE_KEYS), list(OUTPUT_ORDER), seed=seed).fit(
            X_seed[:n_fit], Y_all[:n_fit]
        )
        cal = SplitConformalCalibrator(alpha=0.1)
        cal.fit(gp, X_seed[n_fit:], Y_all[n_fit:])
        reval_model = ConformalForwardModel(gp, cal)
        X_train = X_seed[:n_fit]
    else:
        gp = GPForwardModel(list(RECIPE_KEYS), list(OUTPUT_ORDER), seed=seed).fit(X_seed, Y_all)
        X_train = X_seed

    solver = PessimisticInverseSolver(
        gp,
        list(RECIPE_VARIABLES),
        list(OUTPUT_ORDER),
        X_train=X_train,
        revalidation_model=reval_model,
    )

    results: list[ArmTargetResult] = []
    for tgt in targets:
        spec = _spec_from_json(tgt["spec"])
        out = solver.solve({"targets": _spec_for_solver(spec), "max_candidates": 4})
        if isinstance(out, Infeasible):
            results.append(
                ArmTargetResult(tgt["id"], False, None, None, spec, bool(tgt["feasible_truth"]))
            )
        else:
            recipe = {k: float(out[0].recipe[k]) for k in RECIPE_KEYS}
            gt = harness.ground_truth_outcome(recipe)
            results.append(
                ArmTargetResult(tgt["id"], True, recipe, gt, spec, bool(tgt["feasible_truth"]))
            )
    ledger = {
        "seed_runs": int(X_seed.shape[0]),
        "loopB_evals": 0,
        "fd_probe_evals": 0,
        "revalidation_evals": 0,
        "charitable_total": int(X_seed.shape[0]),
        "deployable_total": int(X_seed.shape[0]),
    }
    return results, ledger


# ==========================================================================
# MFL arm (one trained R; charitable + deployable differ only in the ledger)
# ==========================================================================


def _mfl_point_target(spec: Mapping[str, tuple], sigma: Mapping[str, float]) -> np.ndarray:
    """prereg §1 point-target rule for a box: box CENTER for a two-sided output;
    (finite edge ∓ 1·σ_seed, moved INSIDE the feasible side) for a one-sided output."""
    pt = []
    for name in CONTROLLED_OUTPUTS:
        lo, hi = _bounds(spec[name])
        s = float(sigma[name])
        if math.isfinite(lo) and math.isfinite(hi):
            pt.append(0.5 * (lo + hi))
        elif math.isfinite(hi):  # upper-only (slip ≤ U): one σ inside
            pt.append(hi - s)
        elif math.isfinite(lo):  # lower-only: one σ inside
            pt.append(lo + s)
        else:
            pt.append(0.0)
    return np.asarray(pt, dtype=float)


def run_mfl_arms(harness, targets, X_seed, Z_ctrl, *, mfl_kwargs, seed: int, n_seed: int):
    """Train ONE MFL reverse model and present a recipe per target. Returns the shared
    per-target results plus the charitable/deployable ledgers."""
    from rig.baselines.mfl import MFLLedger, ModelFeedbackLearning
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES

    x_lo = np.array([v.lower for v in RECIPE_VARIABLES], dtype=float)
    x_hi = np.array([v.upper for v in RECIPE_VARIABLES], dtype=float)
    sigma_seed = {name: float(Z_ctrl[:, i].std()) for i, name in enumerate(CONTROLLED_OUTPUTS)}

    mfl = ModelFeedbackLearning(
        x_dim=2, z_dim=len(CONTROLLED_OUTPUTS), x_lower=x_lo, x_upper=x_hi, seed=seed, **mfl_kwargs
    )
    mfl.fit_emulator(X_seed, Z_ctrl)

    specs = [_spec_from_json(t["spec"]) for t in targets]
    targets_z = np.stack([_mfl_point_target(s, sigma_seed) for s in specs])

    # Loop-B machine: raw recipe batch → raw controlled outputs, noisy (a real per-tool
    # tool). One shared noisy machine so run_index advances deterministically.
    loopb_machine = _LoopBMachine(harness, seed=seed + 777)
    ledger = MFLLedger(seed_runs=n_seed)
    mfl.train_reverse(targets_z, machine=loopb_machine, ledger=ledger)

    results: list[ArmTargetResult] = []
    for tgt, spec, z in zip(targets, specs, targets_z, strict=True):
        recipe = mfl.propose_recipe(z, RECIPE_KEYS)
        gt = harness.ground_truth_outcome(recipe)
        results.append(
            ArmTargetResult(tgt["id"], True, recipe, gt, spec, bool(tgt["feasible_truth"]))
        )
    return results, ledger


class _LoopBMachine:
    """Batch RAW-recipe → RAW-controlled-output noisy machine for MFL's Loop B."""

    def __init__(self, harness: MachineHarness, seed: int) -> None:
        self._machine = harness._noisy_machine(seed)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        rows = []
        for x in X:
            recipe = {k: float(v) for k, v in zip(RECIPE_KEYS, x, strict=True)}
            rec = self._machine.run(recipe)
            vals = {o.name: float(o.value.magnitude) for o in rec.outcomes}
            rows.append([vals[n] for n in CONTROLLED_OUTPUTS])
        return np.asarray(rows, dtype=float)


# ==========================================================================
# scoring pass (yield + margin under noise) — the noisy machine, shared σ
# ==========================================================================


def score_under_noise(harness, results, *, n_reps: int, seed: int) -> None:
    """Fill ``yield_under_noise`` and ``normalized_margin`` on every PRESENTED recipe,
    using ``n_reps`` noisy replicates at the fixed recipe (prereg §0)."""
    for k, r in enumerate(results):
        if not r.presented:
            continue
        reps = harness.noisy_replicates(r.recipe, n_reps, seed=seed + k)
        sigma = {
            name: float(np.std([rep[name] for rep in reps]) if reps else 0.0)
            for name in CONTROLLED_OUTPUTS
        }
        r.yield_under_noise = yield_from_replicates(reps, r.spec)
        r.normalized_margin = normalized_margin_of(r.ground_truth, r.spec, sigma)


def summarize_arm(name: str, results, ledger: dict) -> dict:
    presented = [r for r in results if r.presented]
    return {
        "arm": name,
        "n_targets": len(results),
        "n_presented": len(presented),
        "certified_miss_rate": certified_miss_rate(results),
        "false_abstention_rate": false_abstention_rate(results),
        "yield_under_noise_median": _median([r.yield_under_noise for r in presented]),
        "normalized_margin_median": _median([r.normalized_margin for r in presented]),
        "machine_queries": ledger,
        "per_target": [
            {
                "target_id": r.target_id,
                "presented": r.presented,
                "feasible_truth": r.feasible_truth,
                "in_spec_ground_truth": (
                    bool(in_spec(r.ground_truth, r.spec)) if r.presented else None
                ),
                "yield_under_noise": r.yield_under_noise,
                "normalized_margin": r.normalized_margin,
                "recipe": r.recipe,
            }
            for r in results
        ],
    }


# ==========================================================================
# driver
# ==========================================================================


@dataclass
class BakeoffConfig:
    n_seed: int
    n_reps: int
    mfl_kwargs: dict
    target_classes: tuple[str, ...] | None  # None → all
    label: str
    citable: bool
    seed: int = 20260718


def smoke_config() -> BakeoffConfig:
    # 4 targets (2 easy / 1 hard / 1 infeasible), N=30, reduced loops. NOT citable.
    return BakeoffConfig(
        n_seed=30,
        n_reps=50,
        mfl_kwargs=dict(
            emulator_epochs=150, T=200, T0=150, tau=25, tau0=15, delta=0.9, alpha1=0.02
        ),
        target_classes=("feasible", "feasible", "hard", "infeasible"),
        label="smoke",
        citable=False,
    )


def full_config() -> BakeoffConfig:
    # all frozen targets, N=60, Table-10 defaults (from the class), 200 replicates.
    return BakeoffConfig(
        n_seed=60,
        n_reps=200,
        mfl_kwargs={},  # Table-10 defaults on ModelFeedbackLearning
        target_classes=None,
        label="full",
        citable=True,
    )


def _select_targets(all_targets: list[dict], classes: tuple[str, ...] | None) -> list[dict]:
    if classes is None:
        return list(all_targets)
    by_class: dict[str, list[dict]] = {}
    for t in all_targets:
        by_class.setdefault(t["class"], []).append(t)
    picked: list[dict] = []
    used: dict[str, int] = {}
    for cls in classes:
        i = used.get(cls, 0)
        pool = by_class.get(cls, [])
        if i < len(pool):
            picked.append(pool[i])
            used[cls] = i + 1
    return picked


def run(cfg: BakeoffConfig) -> dict:
    payload = load_frozen_targets()
    all_targets = payload["targets"]
    targets = _select_targets(all_targets, cfg.target_classes)
    machine_config = payload["meta"].get("machine_config")
    harness = MachineHarness(machine_config, base_seed=cfg.seed)

    t0 = time.time()
    _records, X, Y_all, Z_ctrl = generate_seed_runs(harness, cfg.n_seed, cfg.seed)

    arms: list[dict] = []

    rig_res, rig_led = run_rig_arm(harness, targets, X, Y_all, use_reval=False, seed=cfg.seed)
    score_under_noise(harness, rig_res, n_reps=cfg.n_reps, seed=cfg.seed + 1)
    arms.append(summarize_arm("rig", rig_res, rig_led))

    reval_res, reval_led = run_rig_arm(harness, targets, X, Y_all, use_reval=True, seed=cfg.seed)
    score_under_noise(harness, reval_res, n_reps=cfg.n_reps, seed=cfg.seed + 2)
    arms.append(summarize_arm("rig-reval", reval_res, reval_led))

    mfl_res, mfl_led = run_mfl_arms(
        harness, targets, X, Z_ctrl, mfl_kwargs=cfg.mfl_kwargs, seed=cfg.seed, n_seed=cfg.n_seed
    )
    score_under_noise(harness, mfl_res, n_reps=cfg.n_reps, seed=cfg.seed + 3)
    # charitable + deployable share the recipes; only the ledger total differs.
    char_led = dict(mfl_led.as_dict())
    dep_led = dict(mfl_led.as_dict())
    arms.append(summarize_arm("mfl-charitable", mfl_res, char_led))
    arms.append(summarize_arm("mfl-deployable", mfl_res, dep_led))

    return {
        "meta": {
            "label": cfg.label,
            "citable": cfg.citable,
            "warning": None
            if cfg.citable
            else "SMOKE RUN — reduced budget, NOT for citation (prereg protocol).",
            "in_silico_only": "Both venues are simulators (prereg §5). Nothing here is "
            "about real hardware.",
            "n_seed": cfg.n_seed,
            "n_reps": cfg.n_reps,
            "n_targets": len(targets),
            "seed": cfg.seed,
            "targets_hash": payload["hash"],
            "wall_clock_s": round(time.time() - t0, 1),
            "created": datetime.now(UTC).isoformat(),
        },
        "arms": arms,
    }


def _fmt(v) -> str:
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return "nan" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def print_summary(result: dict) -> None:
    m = result["meta"]
    print(f"\n=== MFL bake-off [{m['label']}] {'' if m['citable'] else '(NON-CITABLE)'} ===")
    print(f"    {m['in_silico_only']}")
    print(
        f"    n_seed={m['n_seed']} n_targets={m['n_targets']} n_reps={m['n_reps']} "
        f"wall={m['wall_clock_s']}s  targets_hash={m['targets_hash'][:12]}"
    )
    header = (
        f"{'arm':<16}{'miss':>8}{'false_abst':>12}{'yield':>8}"
        f"{'margin':>9}{'q_charit':>10}{'q_deploy':>10}"
    )
    print(header)
    print("-" * len(header))
    for a in result["arms"]:
        led = a["machine_queries"]
        print(
            f"{a['arm']:<16}"
            f"{_fmt(a['certified_miss_rate']):>8}"
            f"{_fmt(a['false_abstention_rate']):>12}"
            f"{_fmt(a['yield_under_noise_median']):>8}"
            f"{_fmt(a['normalized_margin_median']):>9}"
            f"{led['charitable_total']:>10}"
            f"{led['deployable_total']:>10}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="RIG vs MFL bake-off (pre-registered).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="4 targets, reduced budget, NOT citable")
    g.add_argument("--full", action="store_true", help="all frozen targets, Table-10 defaults")
    args = ap.parse_args()

    cfg = smoke_config() if args.smoke else full_config()
    result = run(cfg)
    print_summary(result)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(__file__).resolve().parent / "results" / f"{cfg.label}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bakeoff_results.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote {out_dir / 'bakeoff_results.json'}")


if __name__ == "__main__":
    main()
