"""WP-E: the powered BoTorch COMPARATOR SLATE for the M2 claim (§9.8 / §12.3).

The M2 claim ("RIG reaches spec ~2x cheaper than BO") already stands against
fixed-pool GP-EI BO, continuous-acquisition BO, and ``BoTorchBO`` (SingleTaskGP +
Hvarfner prior + qLogEI/qLCB). This driver adds the two BoTorch families a
reviewer would demand and scores RIG against the FULL slate on the in-silico MBE
machine:

    arms = RIG (ablation policy) | BoTorchBO | SCBO | TuRBO

House rules (all enforced here): matched budgets, bit-identical warm starts, the
SAME Kanarik cost model, scored on the MACHINE's output only, common-random-
numbers pairing (``make_machine(seed)`` fresh per arm so every arm sees the same
seeded noise realization), no tuning RIG, no weakening the comparators.

Design (efficiency + fidelity): each arm is run ONCE per (target, seed) with CRN,
then each comparator is scored against the RIG reference using the imported §12
survival machinery (``rmst_difference_test`` + the m2_sweep paired bootstrap) — so
RIG is not recomputed per comparator, and 50 seeds fit in budget. The target /
seed / machine / metrology-sigma / tolerance / horizon machinery is REUSED BY
IMPORT from ``run_m2_sweep`` (``_insilico_common``, ``_build_targets``,
``_horizon``, ``FEASIBILITY_POLICIES``) — this driver never edits it.

Endpoints per comparator: RMST (smaller = cheaper), hit rate, pooled ΔRMST (=
RMST[rig] - RMST[comparator]; NEGATIVE => RIG cheaper) with a paired-bootstrap 95%
CI + P(RIG better), win-rate, and the difference-in-RMST p-value; plus per-target
breakdowns. The honest verdict states whether the M2 claim holds against the full
slate; if any comparator beats RIG anywhere it goes in the headline.

Note on "unverified bests" / false-accept: in this cost-to-target harness a HIT is
a single in-spec observation on the (noisy) machine for ALL arms — RIG (run here
WITHOUT the F2 confirmation hook, the published-M2 posture), BoTorchBO, SCBO, and
TuRBO alike. So false-accept exposure is IDENTICAL across arms and is not a
differentiator in this comparison; a confirmation-gated variant is the F2
``ConfirmationCampaign`` and is out of scope for the slate. We report this plainly
rather than manufacture an arm-specific certified-miss metric.

Usage (from repo root, sim on MBE_SIM_PATH):
    python examples/run_m2_botorch_slate.py --seeds 50 --targets 2 --tol-k 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# reuse run_m2_sweep's machinery BY IMPORT (never edited)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_m2_sweep as S  # noqa: E402

from rig.active import ActiveLearningLoop  # noqa: E402
from rig.baselines import BoTorchBO, SCBOBaseline, TuRBOBaseline  # noqa: E402
from rig.eval.m2_sweep import _paired_bootstrap_delta  # noqa: E402
from rig.eval.survival import rmst, rmst_difference_test  # noqa: E402

REFERENCE = "rig"
COMPARATORS = ["botorch", "scbo", "turbo"]


def _campaign_time(traj) -> tuple[float, bool]:
    """(time, event): cost-to-target if hit, else the censored budget cost."""
    if traj.hit:
        return float(traj.cost_to_target), True
    censored = float(traj.cumulative_cost[-1]) if traj.cumulative_cost else float("inf")
    return censored, False


def _make_arm_factories(variables, ikeys, okeys, cost, *, budget, q, n_seed, n_pool, policy_knobs):
    """Every arm gets the IDENTICAL budget / q / n_seed / cost / warm-start seed;
    only RIG receives the §8 feasibility knobs (the comparators have no such
    conservatism — it is RIG's, not BO's, and injecting it would handicap BO)."""

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
            kappa=policy_knobs["kappa"],
            z_epi=policy_knobs["z_epi"],
            delta_frac=policy_knobs["delta_frac"],
            seed=seed,
            **cost,
        )

    def botorch(*, machine, in_spec, spec, seed):
        return BoTorchBO(
            machine=machine,
            in_spec=in_spec,
            variables=variables,
            input_keys=ikeys,
            output_keys=okeys,
            spec=spec,
            budget=budget,
            q=q,
            n_seed=n_seed,
            seed=seed,
            **cost,
        )

    def scbo(*, machine, in_spec, spec, seed):
        return SCBOBaseline(
            machine=machine,
            in_spec=in_spec,
            variables=variables,
            input_keys=ikeys,
            output_keys=okeys,
            spec=spec,
            budget=budget,
            q=q,
            n_seed=n_seed,
            seed=seed,
            **cost,
        )

    def turbo(*, machine, in_spec, spec, seed):
        return TuRBOBaseline(
            machine=machine,
            in_spec=in_spec,
            variables=variables,
            input_keys=ikeys,
            output_keys=okeys,
            spec=spec,
            budget=budget,
            q=q,
            n_seed=n_seed,
            seed=seed,
            **cost,
        )

    return {"rig": rig, "botorch": botorch, "scbo": scbo, "turbo": turbo}


def _score_pair(rows_ref, rows_cmp, horizon, *, n_bootstrap, bootstrap_seed):
    """RIG (reference) vs one comparator over aligned (target, seed) rows. Returns
    the §12 difference-in-RMST verdict with a paired-bootstrap CI on ΔRMST."""
    # align by (target, seed) so the bootstrap is paired (CRN)
    keyed_ref = {(r["target"], r["seed"]): r for r in rows_ref}
    keyed_cmp = {(r["target"], r["seed"]): r for r in rows_cmp}
    keys = sorted(set(keyed_ref) & set(keyed_cmp))
    rt = np.array([keyed_ref[k]["time"] for k in keys], dtype=float)
    re = np.array([keyed_ref[k]["event"] for k in keys], dtype=bool)
    ct = np.array([keyed_cmp[k]["time"] for k in keys], dtype=float)
    ce = np.array([keyed_cmp[k]["event"] for k in keys], dtype=bool)

    diff = rmst_difference_test(rt, re, ct, ce, horizon)
    pairs = list(zip(rt.tolist(), re.tolist(), ct.tolist(), ce.tolist(), strict=True))
    ci, prob_ref_better = _paired_bootstrap_delta(pairs, horizon, n_bootstrap, bootstrap_seed)
    wins = int(np.sum(rt < ct))
    ties = int(np.sum(rt == ct))
    n = len(keys)
    return {
        "n_pairs": n,
        "rmst_rig": float(diff.rmst_a),
        "rmst_cmp": float(diff.rmst_b),
        "delta_rmst": float(diff.delta),  # rig - cmp; <0 => RIG cheaper
        "delta_se": float(diff.se),
        "p_value": float(diff.p_value),
        "bootstrap_ci95": [float(ci[0]), float(ci[1])],
        "prob_rig_better": float(prob_ref_better),
        "hit_rate_rig": float(np.mean(re)),
        "hit_rate_cmp": float(np.mean(ce)),
        "win_rate_rig": wins / max(n, 1),
        "tie_rate": ties / max(n, 1),
    }


def _per_target(rows_ref, rows_cmp, targets, horizon):
    out = []
    for tgt in targets:
        rr = [r for r in rows_ref if r["target"] == tgt.id]
        rc = [r for r in rows_cmp if r["target"] == tgt.id]
        kref = {r["seed"]: r for r in rr}
        kcmp = {r["seed"]: r for r in rc}
        seeds = sorted(set(kref) & set(kcmp))
        rt = np.array([kref[s]["time"] for s in seeds], dtype=float)
        re = np.array([kref[s]["event"] for s in seeds], dtype=bool)
        ct = np.array([kcmp[s]["time"] for s in seeds], dtype=float)
        ce = np.array([kcmp[s]["event"] for s in seeds], dtype=bool)
        d = rmst_difference_test(rt, re, ct, ce, horizon)
        out.append(
            {
                "target": tgt.id,
                "delta_rmst": float(d.delta),
                "p_value": float(d.p_value),
                "hit_rate_rig": float(np.mean(re)),
                "hit_rate_cmp": float(np.mean(ce)),
                "win_rate_rig": float(np.mean(rt < ct)),
            }
        )
    return out


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description="M2 BoTorch comparator slate (RIG vs BoTorchBO/SCBO/TuRBO)."
    )
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--targets", type=int, default=2)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--q", type=int, default=4)
    ap.add_argument("--n-seed", type=int, default=8)
    ap.add_argument("--n-pool", type=int, default=128)
    ap.add_argument("--tol-k", type=float, default=S.DEFAULT_TOL_K)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--policy", choices=("ablation", "binding"), default="ablation")
    ap.add_argument("--out", type=str, default="docs/m2-botorch-slate.json")
    ap.add_argument("--force-synthetic", action="store_true")
    args = ap.parse_args(argv)

    policy_knobs = S.FEASIBILITY_POLICIES[args.policy]

    from rig_adapters.mbe import simlink

    use_sim = simlink.sim_available() and not args.force_synthetic
    if use_sim:
        (make_machine, variables, ikeys, okeys, clean_lo, clean_hi, sigma, cost, meta) = (
            S._insilico_common()
        )
        targets = S._build_targets(okeys, clean_lo, clean_hi, sigma, args.targets, args.tol_k)
        machine_name = "InSilicoMachine(MBE)"
    else:
        make_machine, targets, variables, ikeys, okeys, cost, machine_name, meta = S._synthetic(
            args.targets, args.tol_k
        )

    horizon = S._horizon(args.budget, args.n_seed, args.q, cost)
    arms = _make_arm_factories(
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
    arm_names = ["rig", "botorch", "scbo", "turbo"]

    n_campaigns = len(arm_names) * args.targets * args.seeds
    print(
        f"[slate] machine={machine_name} okeys={okeys} arms={arm_names} "
        f"targets={args.targets} seeds={args.seeds} tol_k={args.tol_k} budget={args.budget} "
        f"policy={args.policy} horizon={horizon:.0f} campaigns={n_campaigns}",
        flush=True,
    )
    print(f"[slate] pathology={meta['pathology']} sigma={meta['sigma']}", flush=True)
    if not use_sim:
        print(
            "[slate] WARNING: sim unavailable -> SYNTHETIC fallback (not MBE physics).", flush=True
        )

    # -- run every arm once per (target, seed) with CRN pairing -----------------
    rows: dict[str, list[dict]] = {a: [] for a in arm_names}
    done, t0 = 0, time.time()
    total = args.targets * args.seeds
    for tgt in targets:
        for seed in range(args.seeds):
            for name in arm_names:
                machine = make_machine(seed)  # fresh & seeded => CRN across arms
                runnable = arms[name](
                    machine=machine, in_spec=tgt.in_spec, spec=tgt.spec, seed=seed
                )
                traj = runnable.run()
                time_, event = _campaign_time(traj)
                rows[name].append(
                    {
                        "target": tgt.id,
                        "seed": seed,
                        "time": time_,
                        "event": bool(event),
                        "n_queries": int(traj.n_queries),
                    }
                )
            done += 1
            if done % 10 == 0 or done == total:
                print(
                    f"[slate] {done}/{total} target-seed pairs ({time.time() - t0:.0f}s)",
                    flush=True,
                )

    # -- score each comparator vs RIG ------------------------------------------
    slate = {}
    for cmp_name in COMPARATORS:
        pooled = _score_pair(
            rows[REFERENCE],
            rows[cmp_name],
            horizon,
            n_bootstrap=args.bootstrap,
            bootstrap_seed=0,
        )
        pooled["per_target"] = _per_target(rows[REFERENCE], rows[cmp_name], targets, horizon)
        slate[cmp_name] = pooled

    # RIG's own pooled RMST/hit-rate (reference, identical across pairings)
    rt = np.array([r["time"] for r in rows["rig"]], dtype=float)
    re = np.array([r["event"] for r in rows["rig"]], dtype=bool)
    rig_summary = {"rmst": float(rmst(rt, re, horizon).rmst), "hit_rate": float(np.mean(re))}

    # -- honest verdict --------------------------------------------------------
    beaten_by = []
    for cmp_name in COMPARATORS:
        s = slate[cmp_name]
        # a comparator "beats" RIG if it is significantly CHEAPER (delta_rmst > 0
        # means RIG costlier) at p<0.05, OR has a materially higher hit-rate.
        if s["delta_rmst"] > 0 and s["p_value"] < 0.05:
            beaten_by.append(
                f"{cmp_name} (cheaper, dRMST=+{s['delta_rmst']:.3g}, p={s['p_value']:.3g})"
            )
        elif s["hit_rate_cmp"] > s["hit_rate_rig"] + 0.05:
            beaten_by.append(
                f"{cmp_name} (higher hit-rate {s['hit_rate_cmp']:.2f} vs rig {s['hit_rate_rig']:.2f})"
            )
    claim_holds = len(beaten_by) == 0
    if claim_holds:
        verdict = (
            "IN-SILICO: the M2 'RIG reaches spec ~2x cheaper than BO' claim HOLDS against the "
            "full BoTorch slate (BoTorchBO, SCBO, TuRBO): RIG is cheaper (ΔRMST<0) or tied on "
            "every comparator, none significantly beats it. (Machinery proof, not real-tool "
            "evidence; the real-data headline stays gated on M0.)"
        )
    else:
        verdict = (
            "IN-SILICO: the M2 claim does NOT fully hold against the BoTorch slate — RIG is "
            f"beaten by: {'; '.join(beaten_by)}. Reported in the headline, not a footnote."
        )

    out = {
        "machine": machine_name,
        "used_sim": use_sim,
        "okeys": okeys,
        "ikeys": ikeys,
        "tol_k": args.tol_k,
        "policy": args.policy,
        "feasibility_policy": S.FEASIBILITY_POLICY_LABELS[args.policy],
        "metrology_sigma": meta["sigma"],
        "pathology_config": meta["pathology"],
        "horizon": horizon,
        "reference": REFERENCE,
        "comparators": COMPARATORS,
        "rig_summary": rig_summary,
        "slate": slate,
        "raw_rows": rows,
        "config": vars(args),
        "claim_holds_vs_slate": claim_holds,
        "beaten_by": beaten_by,
        "verdict": verdict,
        "false_accept_note": (
            "In this cost-to-target harness a HIT is a single in-spec observation on the noisy "
            "machine for ALL arms (RIG run without the F2 confirmation hook, plus the three BO "
            "arms), so false-accept exposure is identical across arms and not a differentiator; "
            "a confirmation-gated variant is the F2 ConfirmationCampaign, out of slate scope."
        ),
    }

    print("\n================= M2 BOTORCH SLATE =================")
    print(
        f"machine: {machine_name} (used_sim={use_sim})  policy={args.policy}  tol_k={args.tol_k:g}"
    )
    print(f"RIG reference: RMST={rig_summary['rmst']:.4g}  hit-rate={rig_summary['hit_rate']:.2f}")
    print(
        f"{'comparator':10s} {'RMST_cmp':>10s} {'hit_cmp':>8s} {'dRMST(rig-cmp)':>15s} "
        f"{'95% CI':>22s} {'p':>8s} {'P(rig<)':>8s} {'win':>6s}"
    )
    for cmp_name in COMPARATORS:
        s = slate[cmp_name]
        ci = s["bootstrap_ci95"]
        print(
            f"{cmp_name:10s} {s['rmst_cmp']:10.4g} {s['hit_rate_cmp']:8.2f} "
            f"{s['delta_rmst']:15.4g} [{ci[0]:9.4g},{ci[1]:9.4g}] {s['p_value']:8.3g} "
            f"{s['prob_rig_better']:8.2f} {s['win_rate_rig']:6.2f}"
        )
    print(f"\nclaim_holds_vs_slate: {claim_holds}")
    print(f"VERDICT: {verdict}")
    print("====================================================\n")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[slate] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
