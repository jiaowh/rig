"""M3 acceptance run (implementation-plan §15.4 gate: *'amortized proposal matches
per-query quality after refinement'*).

End-to-end D2 on a self-contained in-silico process, exercising the FULL real stack:
a fitted GP forward tier (§5, WP-C) as the surrogate the §8 solver refines against, a
real zuko conditional-flow amortized generator (§14.3, M3), and the ONE per-query
pessimistic refinement wired through :class:`~rig.inverse.AmortizedRefiner` (D2). Over a
sweep of target boxes it compares, per query:

  * ``cold_heavy`` — the canonical §8 solver with a HEAVY cold Sobol multi-start (the
    per-query gold standard),
  * ``cold_light`` — the same solver with a LIGHT budget (1 start = the box centre),
  * ``d2_light``   — D2: the amortized generator proposes, then the LIGHT solver refines.

The M3 gate is met when ``d2_light`` matches ``cold_heavy``'s quality (feasibility +
pessimistic confidence) at ``cold_light``'s budget — i.e. amortization pays for the
per-query refinement. This is a synthetic in-silico stand-in; the WP-B MBE
``InSilicoMachine`` is the hardware-faithful drop-in for the same harness.

Run: ``python examples/run_m3_acceptance.py`` (writes docs/M3-acceptance-2026-07-17.md +
docs/m3-acceptance.json). Deterministic (seeded).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rig.forward import GPForwardModel  # noqa: E402
from rig.interfaces import ContinuousVariable, Infeasible  # noqa: E402
from rig.inverse import (  # noqa: E402
    AmortizedInverseGenerator,
    AmortizedRefiner,
    PessimisticInverseSolver,
)

RESULT_MD = Path(__file__).resolve().parents[1] / "docs" / "M3-acceptance-2026-07-17.md"
RESULT_JSON = Path(__file__).resolve().parents[1] / "docs" / "m3-acceptance.json"

VARIABLES = [ContinuousVariable("flux", 0.0, 4.0), ContinuousVariable("temp", 0.0, 4.0)]
OUTPUTS = ["rate", "uniformity"]


def in_silico(X: np.ndarray) -> np.ndarray:
    """Coupled 2-in → 2-out in-silico process. ``rate`` is PLATEAU-shaped in ``flux``
    (flat away from flux≈2) so a cold start from the box centre is trapped; both outputs
    are coupled, giving a genuine 2-D pre-image."""
    X = np.atleast_2d(np.asarray(X, float))
    flux, temp = X[:, 0], X[:, 1]
    rate = 5.0 * (1.0 + np.tanh(1.6 * (flux - 2.0))) + 0.4 * temp
    uniformity = 8.0 - (temp - 2.0) ** 2 + 0.5 * flux
    return np.column_stack([rate, uniformity])


def _best_conf(res) -> float:
    return 0.0 if isinstance(res, Infeasible) else max(c.confidence for c in res)


def _feasible(res) -> bool:
    return not isinstance(res, Infeasible) and len(res) > 0


def _raw_proposal_hit_rate(gen, forward, spec, n=64) -> float:
    """Fraction of the generator's RAW (unrefined) proposals whose GP-mean already lands
    in the spec box — the amortized proposal quality before any §8 refinement."""
    recipes = gen.sample(spec, n)
    box = spec["targets"]
    hits = 0
    for r in recipes:
        x = np.array([r[k] for k in ("flux", "temp")])
        mean = np.atleast_1d(forward.predict(x).mean)
        ok = True
        for j, name in enumerate(OUTPUTS):
            if name in box:
                lo, hi = box[name]
                lo = -np.inf if lo is None else lo
                hi = np.inf if hi is None else hi
                if not (lo <= mean[j] <= hi):
                    ok = False
                    break
        hits += int(ok)
    return hits / n


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # cp1252 console guard
    except Exception:
        pass
    rng = np.random.default_rng(0)

    # 1) in-silico data → fit the real GP forward tier + train the real generator
    X = rng.uniform([0.0, 0.0], [4.0, 4.0], size=(220, 2))
    Y = in_silico(X) + rng.normal(0.0, 0.08, size=(220, 2))
    print("fitting GP forward tier on 220 in-silico runs ...")
    forward = GPForwardModel(seed=0).fit(X, Y)
    print("training amortized generator (zuko NSF ensemble) ...")
    gen = AmortizedInverseGenerator(
        VARIABLES,
        OUTPUTS,
        n_members=3,
        transforms=3,
        hidden=(96, 96),
        max_epochs=200,
        region_hw=(0.25, 2.0),
        seed=0,
    ).fit(X, Y)

    def make_solver(n_restarts: int) -> PessimisticInverseSolver:
        return PessimisticInverseSolver(
            forward,
            VARIABLES,
            OUTPUTS,
            X_train=X,
            delta_frac=0.0,
            n_restarts=n_restarts,
            seed=0,
        )

    HEAVY, LIGHT = 32, 1
    d2 = AmortizedRefiner(gen, make_solver(LIGHT), n_proposals=8)

    # 2) sweep target boxes (single- and joint-output specs)
    targets = [
        {"rate": (8.0, 12.0)},  # high rate → flux ≳ 2 (past plateau)
        {"rate": (9.0, 13.0)},  # higher rate
        {"rate": (8.0, 12.0), "uniformity": (7.0, 9.5)},  # joint spec
        {"rate": (7.0, 10.0), "uniformity": (7.5, 9.0)},  # joint, tighter uniformity
        {"uniformity": (8.5, 9.5)},  # uniformity-only (temp ≈ 2)
    ]

    rows = []
    for t in targets:
        spec = {"targets": t}
        cold_heavy = make_solver(HEAVY).solve(spec)
        cold_light = make_solver(LIGHT).solve(spec)
        d2_light = d2.solve(spec)
        row = {
            "target": t,
            "cold_heavy_feasible": _feasible(cold_heavy),
            "cold_light_feasible": _feasible(cold_light),
            "d2_light_feasible": _feasible(d2_light),
            "cold_heavy_conf": round(_best_conf(cold_heavy), 4),
            "cold_light_conf": round(_best_conf(cold_light), 4),
            "d2_light_conf": round(_best_conf(d2_light), 4),
            "raw_proposal_hit_rate": round(_raw_proposal_hit_rate(gen, forward, spec), 3),
            "cold_heavy_starts": HEAVY,
            "d2_light_starts": LIGHT + d2.n_proposals,
        }
        rows.append(row)
        print(
            f"  target={t}  cold_heavy={row['cold_heavy_conf']}(feas={row['cold_heavy_feasible']})"
            f"  cold_light_feas={row['cold_light_feasible']}"
            f"  d2_light={row['d2_light_conf']}(feas={row['d2_light_feasible']})"
            f"  raw_hit={row['raw_proposal_hit_rate']}"
        )

    # 3) verdict: D2-light matches cold-heavy quality at cold-light budget
    parity = [r for r in rows if r["cold_heavy_feasible"]]
    matched = sum(
        1
        for r in parity
        if r["d2_light_feasible"] and r["d2_light_conf"] >= r["cold_heavy_conf"] - 0.02
    )
    light_rescued = sum(1 for r in rows if r["d2_light_feasible"] and not r["cold_light_feasible"])
    gate_pass = matched == len(parity) and len(parity) > 0
    verdict = {
        "gate_pass": gate_pass,
        "n_targets": len(rows),
        "n_cold_heavy_feasible": len(parity),
        "n_d2_light_matches_cold_heavy": matched,
        "n_d2_light_rescued_over_cold_light": light_rescued,
        "cost_ratio_starts": round((LIGHT + d2.n_proposals) / HEAVY, 3),
    }

    RESULT_JSON.write_text(
        json.dumps({"verdict": verdict, "rows": rows}, indent=2), encoding="utf-8"
    )
    _write_markdown(verdict, rows)
    print(
        f"\nGATE {'PASS' if gate_pass else 'FAIL'}: D2-light matched cold-heavy on "
        f"{matched}/{len(parity)} feasible targets; rescued {light_rescued} that cold-light "
        f"missed; refinement cost {verdict['cost_ratio_starts']}x cold-heavy."
    )
    print(f"wrote {RESULT_MD.name} + {RESULT_JSON.name}")


def _write_markdown(verdict: dict, rows: list[dict]) -> None:
    lines = [
        "# M3 acceptance — amortized proposal matches per-query quality after refinement",
        "",
        "**Gate (implementation-plan §15.4):** *amortized proposal matches per-query quality "
        "after refinement.* End-to-end D2 (real GP forward tier + real zuko amortized "
        "generator + §8 pessimistic refinement via `AmortizedRefiner`) on a self-contained "
        "coupled in-silico process (plateau-shaped `rate`, so a cold box-centre start is "
        "trapped). Deterministic (seed 0). Generated by `examples/run_m3_acceptance.py`.",
        "",
        f"**Verdict: {'PASS' if verdict['gate_pass'] else 'FAIL'}.** "
        f"D2-light matched cold-heavy quality on "
        f"{verdict['n_d2_light_matches_cold_heavy']}/{verdict['n_cold_heavy_feasible']} "
        f"feasible targets; D2-light rescued "
        f"{verdict['n_d2_light_rescued_over_cold_light']}/{verdict['n_targets']} targets that "
        f"the light cold solver missed; refinement budget = "
        f"{verdict['cost_ratio_starts']}× cold-heavy (1 cold start + 8 amortized proposals "
        f"vs 32 Sobol starts).",
        "",
        "| target | cold-heavy conf (feas) | cold-light feas | **D2-light conf (feas)** | "
        "raw proposal hit-rate |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        t = ", ".join(f"{k}∈{v}" for k, v in r["target"].items())
        lines.append(
            f"| {t} | {r['cold_heavy_conf']} ({r['cold_heavy_feasible']}) | "
            f"{r['cold_light_feasible']} | **{r['d2_light_conf']} ({r['d2_light_feasible']})** | "
            f"{r['raw_proposal_hit_rate']} |"
        )
    lines += [
        "",
        "- **cold-heavy**: canonical §8 solver, 32-start cold Sobol multi-start (per-query "
        "gold standard).",
        "- **cold-light**: same solver, 1 start (box centre) — trapped on the plateau ⇒ the "
        "gap amortization fills.",
        "- **D2-light**: amortized generator proposes 8 recipes; the 1-start solver refines "
        "them (the D2 `AmortizedRefiner`).",
        "- **raw proposal hit-rate**: fraction of the generator's *unrefined* proposals whose "
        "GP-mean already lands in the spec box — the amortized quality before §8 refinement.",
        "",
        "**Calibration boundary (D2):** calibration attaches to the amortized proposal (the "
        "§14.6 SBC/TARP gate on the generator) and to the conformally re-validated selected "
        "set (the solver's `revalidation_model` + §13.2 gate) — never to the refined output.",
    ]
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
