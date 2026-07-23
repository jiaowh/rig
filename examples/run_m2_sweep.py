"""Powered M2 result: RIG active-learning loop vs warm-started GP-EI BO on the
in-silico MBE machine, scored by difference-in-RMST (implementation-plan §12.2/§12.3).

This is the M2 milestone *measurement* (machinery proof, in-silico — the
real-data headline stays gated on M0). It runs the WP-B ``InSilicoMachine``
(calibrated MBE physics + injected pathologies) when the sibling sim is
available; otherwise it falls back to a CI-portable synthetic machine.

Honest-configuration notes (see docs/M2-result-2026-07-16.md and the adversarial
validation trail). An earlier v1 of this driver was refuted by a 6-lens skeptic
fleet: (TS1, fatal) its spec tolerance was a free ``tol_frac*span`` knob that sat
in the exact window manufacturing RIG's win — doubling it erased the effect; (A3,
major) its "joint" target was a *separable identity* (``thickness_grown`` is
literally ``film_thickness``), the easiest geometry for a direct inverse; and
(A1/A2/TS2) it ran the machine with ALL pathologies OFF, so it was deterministic,
seeds were not real machine realizations, and the §12.1 exploitation mode was
never exercised. This driver fixes all of that:

* **metrology noise ON** (``PathologyConfig(metrology_noise=True)``) — the machine
  is stochastic, so the 50 seeds are genuine independent realizations, the spec
  hit is a *noisy* acceptance test, and the surrogate must cope with noise. The
  config is serialized into the result JSON (A4).
* **coupled non-identity target** — the physics probe (scratchpad/probe_machine)
  shows the thermal KPIs collapse to ``T_heater`` and ``thickness_grown`` is a
  literal identity onto ``film_thickness``; only ``bow_cooldown_um`` genuinely
  depends on BOTH recipe vars. We target ``T_center`` × ``bow_cooldown_um``: a
  *triangular*-coupled 2→2 inverse (bow depends on the T_heater picked for
  T_center), not a diagonal identity. (This is the strongest coupling the 2-D
  recipe space offers; we say so plainly rather than overclaim a Pareto trade-off.)
* **metrology-anchored tolerance + sensitivity curve** — the spec half-width is
  ``tol_k × σ_metrology`` (a Gage-R&R-style spec, per-output σ measured empirically
  on the noisy machine), NOT a free fraction of an arbitrary probe span. The
  headline uses ``tol_k=6`` (a six-sigma window); a tol-sensitivity curve over
  several ``tol_k`` is reported so no single-knob artifact can hide (TS1 fix).
* **scale-fair BO** (BF-1) — the real baseline fix is in ``warm_bo.py``: its
  distance-to-box scalarization is now normalized per-output by the spec
  tolerance, so a multi-KPI spec whose outputs span orders of magnitude (here
  ``T_center`` ~1e3 vs ``bow_cooldown_um`` ~1e-4 in SI) is not numerically blind
  to the small-scale KPI. Without it, BO nails T_center and never sees the tight
  bow box (an artifact, not a method gap); with it, BO becomes a genuine
  comparator (hit-rate ~0.58 here). ``n_pool`` (fresh Sobol pool per batch,
  identical for both arms) is kept modest since the 2-D pool is dense enough and
  the validator's own check showed continuous-EI does not change the verdict.
* **richer reporting** — both-hit-restricted ΔRMST is reported alongside the
  pooled ΔRMST so the "cheaper" claim is separated from the "more reliable"
  (hit-rate) claim (SDM-2); a support-score honesty readout checks the winning
  inverse proposal is on-support (A5).
* **feasibility policy selectable via ``--policy`` (F3, audit 2026-07-21/22)** — the
  RIG arm's §8 conservatism is chosen at run time. ``--policy ablation`` (DEFAULT, the
  published M2 config) pins ``kappa=z_epi=1.0``, ``delta_frac=0.01`` — the MORE
  PERMISSIVE ablation — and writes ``docs/m2-result.json``. ``--policy binding`` pins
  the binding §8 ``2.0/2.0/0.02`` policy that ``PessimisticInverseSolver`` and
  ``ActiveLearningLoop`` default to, and writes ``docs/m2-result-binding.json``. The
  knobs are applied at BOTH RIG pin sites (the active-loop factory ``_make_factories``
  and the inverse readout ``_inverse_readout``); the warm-BO comparator is NEVER
  touched (the policy is RIG's feasibility conservatism, not BO's search — and
  ``WarmStartedBO`` does not even accept these knobs). The selected label is written
  into the result JSON (``feasibility_policy``) and printed at run start so no reader
  mistakes ablation numbers for binding-policy numbers.

Usage (from repo root, with the sim on MBE_SIM_PATH):
    python examples/run_m2_sweep.py --policy binding      # binding §8 2.0/2.0/0.02
    python examples/run_m2_sweep.py --seeds 50 --targets 4 --tol-k 6 \
        --out docs/m2-result.json                          # ablation (default)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rig.active import ActiveLearningLoop
from rig.baselines import WarmStartedBO
from rig.eval.diversity import vendi_score
from rig.eval.m2_sweep import Target, run_m2_sweep
from rig.eval.survival import rmst_difference_test
from rig.interfaces import ContinuousVariable, Infeasible
from rig.inverse import PessimisticInverseSolver

# metrology-noise multiplier for the headline spec half-width (six-sigma window).
DEFAULT_TOL_K = 6.0
# The coupled non-identity KPI pair (see module docstring / probe).
INSILICO_OKEYS = ["T_center", "bow_cooldown_um"]

# §8 feasibility policy under which THIS driver runs its RIG arm (F3, audit
# 2026-07-21/22). The `--policy` flag selects one of these knob sets and applies it at
# BOTH RIG pin sites (`_make_factories` active-loop factory + `_inverse_readout`); the
# warm-BO comparator is never touched. The `ablation` set (the MORE PERMISSIVE 1.0/1.0/
# 0.01) is what produced the published M2 numbers; `binding` is the §8 2.0/2.0/0.02
# policy that PessimisticInverseSolver and ActiveLearningLoop default to. The selected
# label is written into the result JSON (`feasibility_policy`) and printed at run start
# so no reader mistakes ablation numbers for binding-policy numbers.
FEASIBILITY_POLICIES: dict[str, dict[str, float]] = {
    "ablation": {"kappa": 1.0, "z_epi": 1.0, "delta_frac": 0.01},
    "binding": {"kappa": 2.0, "z_epi": 2.0, "delta_frac": 0.02},
}
FEASIBILITY_POLICY_LABELS: dict[str, str] = {
    "ablation": (
        "ablation-1.0/1.0/0.01 (kappa/z_epi/delta_frac) — the MORE PERMISSIVE "
        "ablation, NOT the binding §8 2.0/2.0/0.02 policy; see "
        "docs/m2-result-binding.json for the binding-policy re-run"
    ),
    "binding": (
        "binding-2.0/2.0/0.02 (kappa/z_epi/delta_frac) — the binding §8 policy that "
        "PessimisticInverseSolver and ActiveLearningLoop default to"
    ),
}


def _synthetic(n_targets: int, tol_k: float):
    """CI-portable non-monotone stochastic machine: y = 2 + 1.5 sin(5x) + noise."""
    sigma = 0.05  # fixed metrology sigma for the synthetic fallback

    def make_machine(seed: int) -> Callable[[dict], np.ndarray]:
        rng = np.random.default_rng(seed)

        def machine(recipe):
            return np.array([2.0 + 1.5 * np.sin(5.0 * recipe["x"]) + sigma * rng.standard_normal()])

        return machine

    lo, hi = np.array([0.5]), np.array([3.5])
    okeys = ["y"]
    targets = _build_targets(okeys, lo, hi, np.array([sigma]), n_targets, tol_k)
    variables = [ContinuousVariable("x", 0.0, 1.0)]
    meta = dict(
        sigma={okeys[0]: sigma},
        clean_lo=lo.tolist(),
        clean_hi=hi.tolist(),
        pathology={"metrology_noise": True, "note": "synthetic fallback"},
    )
    return (
        make_machine,
        targets,
        variables,
        ["x"],
        okeys,
        dict(cost_recipe=lambda r: 1.0, c_batch=0.0),
        "synthetic-sin",
        meta,
    )


def _measure_sigma(make_noisy_machine, ikeys, okeys, *, recipe, n_reps=400, seed=0):
    """Empirical per-output metrology sigma IN THE SWEEP'S OUTPUT SPACE (SI units
    as records_to_arrays returns them), so the anchored tolerance is unit-correct."""
    machine = make_noisy_machine(seed)
    reps = np.array([np.asarray(machine(recipe), dtype=float) for _ in range(n_reps)])
    return reps.std(axis=0, ddof=1)


def _insilico_common(probe_seed: int = 7):
    """Shared in-silico setup: adapter, clean output ranges, and metrology sigma.

    Returns everything the target-builder and machine-factory need. The machine
    runs with ``metrology_noise=True`` (the honest config); the CLEAN probe (used
    only to place target CENTERS) is deterministic so target definitions are
    reproducible and noise-independent, while the spec HALF-WIDTH is anchored to
    the measured metrology sigma."""
    from rig.forward import records_to_arrays
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    adapter = make_adapter()
    ikeys = [v.name for v in RECIPE_VARIABLES]
    okeys = INSILICO_OKEYS
    pathology = PathologyConfig(metrology_noise=True)

    # clean probe -> reachable output ranges (deterministic, for target centers)
    clean = InSilicoMachine(config=PathologyConfig(), seed=probe_seed, adapter=adapter)
    probe = [clean.run(p, tool_id="A") for p in adapter.seed_design(24, probe_seed)]
    _, yp = records_to_arrays(probe, ikeys, okeys)
    clean_lo, clean_hi = yp.min(axis=0), yp.max(axis=0)

    def make_machine(seed: int) -> Callable[[dict], np.ndarray]:
        sim = InSilicoMachine(config=pathology, seed=seed, adapter=adapter)

        def machine(recipe):
            rec = sim.run(recipe, tool_id="A")
            _, y = records_to_arrays([rec], ikeys, okeys)
            return y[0]

        return machine

    mid = {k: 0.5 * (v.lower + v.upper) for k, v in ((v.name, v) for v in RECIPE_VARIABLES)}
    sigma = _measure_sigma(make_machine, ikeys, okeys, recipe=mid)

    variables = list(RECIPE_VARIABLES)
    cost = dict(cost_recipe=lambda r: 1000.0, c_batch=1000.0)  # Kanarik cost model
    meta = dict(
        sigma={okeys[j]: float(sigma[j]) for j in range(len(okeys))},
        clean_lo=clean_lo.tolist(),
        clean_hi=clean_hi.tolist(),
        pathology=asdict(pathology),
        probe_seed=probe_seed,
    )
    return make_machine, variables, ikeys, okeys, clean_lo, clean_hi, sigma, cost, meta


def _build_targets(okeys, lo, hi, sigma, n_targets, tol_k):
    """Metrology-anchored coupled targets: center at fraction f of the CLEAN
    output span; half-width ``tol = tol_k × σ_metrology`` per output (a
    Gage-R&R-style spec, NOT a free fraction of the probe span). The hit test is
    evaluated on the NOISY observed output (a realistic acceptance test)."""
    lo, hi, sigma = np.asarray(lo), np.asarray(hi), np.asarray(sigma)
    span = hi - lo
    tol = tol_k * sigma
    targets = []
    fracs = np.linspace(0.3, 0.7, n_targets) if n_targets > 1 else np.array([0.5])
    for f in fracs:
        center = lo + f * span
        spec = {
            "targets": {
                k: {"target": float(center[j]), "tol": float(tol[j])} for j, k in enumerate(okeys)
            }
        }

        def in_spec(y, center=center, tol=tol):
            return bool(np.all(np.abs(np.asarray(y, dtype=float) - center) <= tol))

        targets.append(Target(f"joint_{f:.2f}", spec, in_spec))
    return targets


def _make_factories(variables, ikeys, okeys, cost, *, budget, q, n_seed, n_pool, policy_knobs):
    def rig(*, machine, in_spec, spec, seed):
        return ActiveLearningLoop(
            machine=machine,
            in_spec=in_spec,
            variables=variables,
            input_keys=ikeys,
            output_keys=okeys,
            spec=spec,
            budget=budget,
            q=q,
            n_seed=n_seed,
            n_pool=n_pool,
            # F3 (audit 2026-07-21/22): the RIG arm's §8 conservatism is the
            # `--policy`-selected knob set (see FEASIBILITY_POLICIES) — ablation
            # 1.0/1.0/0.01 (the published M2 config) or binding 2.0/2.0/0.02 — applied
            # identically here and in `_inverse_readout`. The `bo` factory below never
            # receives these: the policy is RIG's feasibility conservatism, not BO's
            # search (WarmStartedBO does not accept kappa/z_epi/delta_frac at all).
            kappa=policy_knobs["kappa"],
            z_epi=policy_knobs["z_epi"],
            delta_frac=policy_knobs["delta_frac"],
            seed=seed,
            **cost,
        )

    def bo(*, machine, in_spec, spec, seed):
        return WarmStartedBO(
            machine=machine,
            in_spec=in_spec,
            variables=variables,
            input_keys=ikeys,
            output_keys=okeys,
            spec=spec,
            budget=budget,
            q=q,
            n_seed=n_seed,
            n_pool=n_pool,
            seed=seed,
            **cost,
        )

    return {"rig": rig, "bo": bo}


def _both_hit_delta_rmst(report, horizon):
    """ΔRMST restricted to (target,seed) pairs where BOTH methods hit — separates
    the 'cheaper' claim from the 'more reliable' (hit-rate) claim (SDM-2)."""
    ref, other = report.reference, report.other
    by_key: dict[tuple[str, int], dict[str, object]] = {}
    for c in report.campaigns:
        by_key.setdefault((c.target, c.seed), {})[c.method] = c
    both = [
        (v[ref], v[other])
        for v in by_key.values()
        if ref in v and other in v and v[ref].event and v[other].event
    ]
    if len(both) < 2:
        return {
            "n_both_hit": len(both),
            "delta_rmst": float("nan"),
            "p_value": float("nan"),
            "median_saving": float("nan"),
        }
    rt = np.array([r.time for r, _ in both])
    ot = np.array([o.time for _, o in both])
    ev = np.ones(len(both), dtype=bool)
    d = rmst_difference_test(rt, ev, ot, ev, horizon)
    return {
        "n_both_hit": len(both),
        "delta_rmst": float(d.delta),
        "p_value": float(d.p_value),
        "median_saving": float(np.median(ot - rt)),
    }


def _inverse_readout(
    make_machine, target, variables, ikeys, okeys, *, n_design, seed, policy_knobs
):
    """RIG returns a DIVERSE, ON-SUPPORT pre-image (§8.7); BO returns one point.
    Fit a GP on a shared design, run the inverse solver, and report (a) Vendi
    diversity of the candidate set vs BO's singleton and (b) the winning
    candidate's support_score vs the solver's support_floor (A5 honesty check)."""
    import warnings

    from scipy.stats import qmc

    from rig.forward import GPForwardModel
    from rig.transforms import RecipeTransform

    machine = make_machine(seed)
    rt = RecipeTransform(variables)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = (2.0 * qmc.Sobol(d=rt.dim, scramble=True, seed=seed).random(n_design) - 1.0) * 5.0
    recipes = [rt.forward(ui) for ui in u]
    flat = [v.name for v in variables]
    X = np.array([[r[k] for k in flat] for r in recipes], dtype=float)
    Y = np.array([np.asarray(machine(r), dtype=float) for r in recipes], dtype=float)
    gp = GPForwardModel(n_restarts=2, seed=seed).fit(X, Y)
    # F3: the solver runs under the `--policy`-selected knobs (see FEASIBILITY_POLICIES),
    # IDENTICAL to the active-loop factory, so the readout reflects the same conservatism
    # the loop's exploit pick was solved under.
    solver = PessimisticInverseSolver(
        gp,
        variables,
        okeys,
        X_train=X,
        kappa=policy_knobs["kappa"],
        z_epi=policy_knobs["z_epi"],
        delta_frac=policy_knobs["delta_frac"],
        seed=seed,
    )
    res = solver.solve({**target.spec, "max_candidates": 6})
    if isinstance(res, list) and len(res) >= 1:
        cand = np.array([[c.recipe[k] for k in flat] for c in res], dtype=float)
        rig_vendi = float(vendi_score(cand)) if len(cand) >= 2 else 1.0
        supports = [float(c.support_score) for c in res]
        top_support = supports[0]
        on_support = bool(top_support >= solver.support_floor)
        return {
            "target": target.id,
            "verdict": "FEASIBLE_CERTIFIED",
            "rig_candidates": len(cand),
            "rig_vendi": rig_vendi,
            "bo_vendi": 1.0,
            "top_support_score": top_support,
            "support_floor": float(solver.support_floor),
            "top_on_support": on_support,
            "all_on_support": bool(all(s >= solver.support_floor for s in supports)),
        }
    # INFEASIBLE at this design → the loop proceeds via the margin-guided
    # nearest_achievable fallback (IF-1). Report the §8.8 attribution honestly so
    # the write-up never conflates "FEASIBLE-certified" with "fell back".
    if isinstance(res, Infeasible):
        return {
            "target": target.id,
            "verdict": "INFEASIBLE_FALLBACK",
            "rig_candidates": 0,
            "feasible": False,
            "distance_to_feasible": float(res.distance_to_feasible),
            "reason": str(res.reason),
        }
    return {
        "target": target.id,
        "verdict": "INFEASIBLE_FALLBACK",
        "rig_candidates": 0,
        "feasible": False,
    }


def _horizon(budget, n_seed, q, cost):
    cr, cb = cost["cost_recipe"]({}), cost["c_batch"]
    n_extra = math.ceil(max(budget - n_seed, 0) / q)
    return (n_seed * cr + cb) + n_extra * (q * cr + cb) + 1.0


def _run_curve(
    make_machine,
    variables,
    ikeys,
    okeys,
    clean_lo,
    clean_hi,
    sigma,
    cost,
    *,
    curve_ks,
    curve_seeds,
    curve_targets,
    budget,
    q,
    n_seed,
    n_pool,
    bootstrap,
    policy_knobs,
):
    """Tol-sensitivity curve (TS1): is RIG's advantage knob-dependent? Sweep the
    metrology multiplier tol_k at reduced seeds/targets and report the pooled
    ΔRMST + per-method hit-rate at each spec tightness. Runs the RIG arm under the
    same `--policy` knob set as the headline sweep (`policy_knobs`)."""
    horizon = _horizon(budget, n_seed, q, cost)
    methods = _make_factories(
        variables,
        ikeys,
        okeys,
        cost,
        budget=budget,
        q=q,
        n_seed=n_seed,
        n_pool=n_pool,
        policy_knobs=policy_knobs,
    )
    curve = []
    for k in curve_ks:
        targets = _build_targets(okeys, clean_lo, clean_hi, sigma, curve_targets, k)
        rep = run_m2_sweep(
            make_machine=make_machine,
            methods=methods,
            targets=targets,
            seeds=range(curve_seeds),
            horizon=horizon,
            n_bootstrap=bootstrap,
            bootstrap_seed=0,
        )
        curve.append(
            {
                "tol_k": float(k),
                "delta_rmst": float(rep.pooled_delta_rmst),
                "p_value": float(rep.pooled_p_value),
                "hit_rate": {m: float(rep.pooled_hit_rate[m]) for m in rep.methods},
                "win_rate": float(rep.pooled_win_rate),
                "tie_rate": float(rep.pooled_tie_rate),
            }
        )
        print(
            f"[m2:curve] tol_k={k:.1f}  dRMST={rep.pooled_delta_rmst:.4g}  "
            f"p={rep.pooled_p_value:.2g}  hit rig={rep.pooled_hit_rate['rig']:.2f} "
            f"bo={rep.pooled_hit_rate['bo']:.2f}  win={rep.pooled_win_rate:.2f}",
            flush=True,
        )
    return curve


def _scoped_verdict(report, meta, tol_k):
    noisy = meta.get("pathology", {}).get("metrology_noise", False)
    return (
        f"IN-SILICO MACHINERY PROOF (not real-tool evidence; headline gated on M0). "
        f"On the InSilicoMachine (metrology_noise={noisy}) with a coupled "
        f"{'×'.join(INSILICO_OKEYS)} target and a metrology-anchored spec "
        f"(tol={tol_k:g}σ): {report.verdict}"
    )


def main(argv=None) -> int:
    # cp1252 console guard (audit 2026-07-17) — the two sibling example scripts
    # (run_m1_sputtering, run_m3_acceptance) both do this and this one did not, so
    # printing the IF-1 readout (whose `reason` string contains 'σ') raised
    # UnicodeEncodeError on a Windows console — *before* the `out_path.write_text`
    # at the end. The documented reproduce command therefore crashed after doing
    # ALL the compute and never wrote docs/m2-result.json. That is why the shipped
    # artifact still carries the pre-IF-1 `inverse_readout` schema (no `verdict`
    # key) while the doc advertises the IF-1 attribution as wired in.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="Powered M2 RIG-vs-BO sweep (§12.3).")
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--targets", type=int, default=4)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--n-seed", type=int, default=8)
    ap.add_argument("--n-pool", type=int, default=128)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument(
        "--tol-k",
        type=float,
        default=DEFAULT_TOL_K,
        help="headline spec half-width as a multiple of metrology sigma",
    )
    ap.add_argument(
        "--tol-curve",
        type=str,
        default="2,3,4,6,8",
        help="comma-separated tol_k values for the sensitivity curve",
    )
    ap.add_argument("--curve-seeds", type=int, default=16)
    ap.add_argument("--curve-targets", type=int, default=2)
    ap.add_argument("--no-curve", action="store_true", help="skip the tol-sensitivity curve")
    ap.add_argument(
        "--policy",
        choices=("ablation", "binding"),
        default="ablation",
        help=(
            "RIG feasibility policy (kappa/z_epi/delta_frac), applied to the RIG arm's "
            "loop + inverse readout but NEVER the BO comparator: 'ablation' (default) = "
            "1.0/1.0/0.01 (published M2 config); 'binding' = the §8 2.0/2.0/0.02 default"
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help=(
            "output JSON path; default docs/m2-result.json for --policy ablation, "
            "docs/m2-result-binding.json for --policy binding"
        ),
    )
    ap.add_argument("--force-synthetic", action="store_true")
    args = ap.parse_args(argv)

    # F3 (audit 2026-07-22): resolve the RIG feasibility policy once, then thread the
    # SAME knob set through both RIG pin sites (loop factory + inverse readout). The
    # default --out follows the policy so a binding run never clobbers the published
    # ablation artifact. --out is resolved before vars(args) is serialized, so the
    # config block records the concrete path.
    policy_knobs = FEASIBILITY_POLICIES[args.policy]
    feasibility_policy = FEASIBILITY_POLICY_LABELS[args.policy]
    if args.out is None:
        args.out = (
            "docs/m2-result-binding.json" if args.policy == "binding" else "docs/m2-result.json"
        )

    from rig_adapters.mbe import simlink

    use_sim = simlink.sim_available() and not args.force_synthetic
    if use_sim:
        (make_machine, variables, ikeys, okeys, clean_lo, clean_hi, sigma, cost, meta) = (
            _insilico_common()
        )
        targets = _build_targets(okeys, clean_lo, clean_hi, sigma, args.targets, args.tol_k)
        machine_name = "InSilicoMachine(MBE)"
    else:
        make_machine, targets, variables, ikeys, okeys, cost, machine_name, meta = _synthetic(
            args.targets, args.tol_k
        )
        clean_lo, clean_hi = np.asarray(meta["clean_lo"]), np.asarray(meta["clean_hi"])
        sigma = np.asarray([meta["sigma"][k] for k in okeys])

    horizon = _horizon(args.budget, args.n_seed, args.q, cost)
    methods = _make_factories(
        variables,
        ikeys,
        okeys,
        cost,
        budget=args.budget,
        q=args.q,
        n_seed=args.n_seed,
        n_pool=args.n_pool,
        policy_knobs=policy_knobs,
    )

    n_campaigns = 2 * args.targets * args.seeds
    print(
        f"[m2] machine={machine_name}  okeys={okeys}  targets={args.targets}  "
        f"seeds={args.seeds}  tol_k={args.tol_k}  budget={args.budget}  "
        f"n_pool={args.n_pool}  horizon={horizon:.0f}  campaigns={n_campaigns}",
        flush=True,
    )
    print(f"[m2] pathology={meta['pathology']}  sigma={meta['sigma']}", flush=True)
    print(f"[m2] feasibility_policy [{args.policy}]: {feasibility_policy}", flush=True)
    if not use_sim:
        print(
            "[m2] WARNING: sim unavailable -> SYNTHETIC fallback (not MBE physics). "
            "Set MBE_SIM_PATH for the headline result.",
            flush=True,
        )

    done, t0 = [0], time.time()

    def progress(msg: str) -> None:
        done[0] += 1
        if done[0] % 10 == 0 or done[0] == args.targets * args.seeds:
            print(
                f"[m2] {done[0]}/{args.targets * args.seeds} target-seed pairs "
                f"({time.time() - t0:.0f}s) — {msg}",
                flush=True,
            )

    report = run_m2_sweep(
        make_machine=make_machine,
        methods=methods,
        targets=targets,
        seeds=range(args.seeds),
        horizon=horizon,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=0,
        progress=progress,
    )

    both_hit = _both_hit_delta_rmst(report, horizon)

    curve = []
    if not args.no_curve:
        curve_ks = [float(s) for s in args.tol_curve.split(",") if s.strip()]
        print(
            f"[m2] tol-sensitivity curve over tol_k={curve_ks} "
            f"({args.curve_seeds} seeds x {args.curve_targets} targets)",
            flush=True,
        )
        curve = _run_curve(
            make_machine,
            variables,
            ikeys,
            okeys,
            clean_lo,
            clean_hi,
            sigma,
            cost,
            curve_ks=curve_ks,
            curve_seeds=args.curve_seeds,
            curve_targets=args.curve_targets,
            budget=args.budget,
            q=args.q,
            n_seed=args.n_seed,
            n_pool=args.n_pool,
            bootstrap=args.bootstrap,
            policy_knobs=policy_knobs,
        )

    try:
        readout = _inverse_readout(
            make_machine,
            targets[0],
            variables,
            ikeys,
            okeys,
            n_design=max(24, 4 * len(variables) + 8),
            seed=0,
            policy_knobs=policy_knobs,
        )
    except Exception as e:  # noqa: BLE001 — a readout must never sink the run
        readout = {"error": repr(e)}

    out = report.to_dict()
    out["machine"] = machine_name
    out["used_sim"] = use_sim
    out["okeys"] = okeys
    out["ikeys"] = ikeys
    out["tol_k"] = args.tol_k
    out["metrology_sigma"] = meta["sigma"]
    out["clean_output_range"] = {
        okeys[j]: [float(clean_lo[j]), float(clean_hi[j])] for j in range(len(okeys))
    }
    out["pathology_config"] = meta["pathology"]
    out["both_hit_delta_rmst"] = both_hit
    out["tol_sensitivity_curve"] = curve
    out["inverse_readout"] = readout
    out["scoped_verdict"] = _scoped_verdict(report, meta, args.tol_k)
    out["config"] = vars(args)
    out["policy"] = args.policy  # F3: "ablation" | "binding"
    out["feasibility_policy"] = feasibility_policy  # F3: the --policy-selected label

    print("\n================= M2 RESULT (honest config) =================")
    print(f"machine: {machine_name}   (used_sim={use_sim})  pathology={meta['pathology']}")
    print(f"target: coupled {okeys}   spec: tol={args.tol_k:g}*sigma (metrology-anchored)")
    print(f"feasibility_policy [{args.policy}]: {feasibility_policy}")
    print(
        f"pooled RMST  rig={report.pooled_rmst['rig']:.4g}  bo={report.pooled_rmst['bo']:.4g}  "
        f"(smaller = cheaper)"
    )
    print(
        f"pooled hit-rate  rig={report.pooled_hit_rate['rig']:.2f}  bo={report.pooled_hit_rate['bo']:.2f}"
    )
    print(
        f"pooled win-rate (rig<bo per seed): {report.pooled_win_rate:.2f}  "
        f"(tie {report.pooled_tie_rate:.2f})"
    )
    print(
        f"dRMST (rig-bo) = {report.pooled_delta_rmst:.4g}  "
        f"95% CI [{report.bootstrap_ci95[0]:.4g}, {report.bootstrap_ci95[1]:.4g}]  "
        f"p={report.pooled_p_value:.3g}  P(rig better)={report.prob_reference_better:.2f}"
    )
    print(
        f"both-hit dRMST = {both_hit['delta_rmst']:.4g}  (n={both_hit['n_both_hit']}, "
        f"p={both_hit['p_value']:.3g}, median saving={both_hit['median_saving']:.4g})"
    )
    print(f"VERDICT: {report.verdict}")
    print(f"inverse readout: {readout}")
    print("per-target:")
    for tv in report.per_target:
        print(
            f"  {tv.target}: rig_hit={tv.hit_rate['rig']:.2f} bo_hit={tv.hit_rate['bo']:.2f} "
            f"dRMST={tv.delta_rmst:.4g} p={tv.p_value:.3g} win={tv.win_rate:.2f}"
        )
    if curve:
        print("tol-sensitivity curve:")
        for c in curve:
            print(
                f"  tol_k={c['tol_k']:.1f}: dRMST={c['delta_rmst']:.4g} p={c['p_value']:.2g} "
                f"hit rig={c['hit_rate']['rig']:.2f} bo={c['hit_rate']['bo']:.2f} "
                f"win={c['win_rate']:.2f}"
            )
    print("============================================\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[m2] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
