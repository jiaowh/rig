"""M2 gate: powered head-to-head evaluation of two recipe-finding methods on the
cost-to-target survival metric (implementation-plan §12.2/§12.3).

This is the *measurement* that turns the M2 machinery (WP-D inverse + WP-F loop +
§12.3 warm-BO baseline + WP-G survival) into the M2 milestone verdict — "does the
method beat warm-started BO in-silico?" — over many seeds and several held-out
targets, with difference-in-RMST (Uno 2014) as the binding primary comparator.

Deliberately **method-agnostic**: the caller injects method *factories* and a
machine *factory*, so this module imports nothing from ``rig.active`` /
``rig.baselines`` (it evaluates them, it does not depend on them). Each method
factory returns an object exposing ``run() -> Trajectory``-like result with
``.hit: bool``, ``.cost_to_target: float``, ``.cumulative_cost: list[float]`` and
``.n_queries: int`` (the WP-F ``Trajectory`` contract).

Conventions (binding, WP-G §12): cost-to-target is survival data — event = spec
hit, censor = budget-exhausted, **smaller RMST is better**. A fresh machine per
(method, target, seed) gives common-random-numbers pairing (both methods see the
same seeded noise realization). Targets are reachable by construction, so a method
that fails to reach one within budget is a legitimate *censor* (not excluded).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from rig.eval.survival import rmst, rmst_difference_test


class _Runnable(Protocol):
    hit: bool
    cost_to_target: float
    cumulative_cost: list[float]
    n_queries: int


# machine(recipe) -> outcome vector
Machine = Callable[[dict[str, float]], np.ndarray]
# make_machine(seed) -> a FRESH, seeded machine (fresh so paired methods share the
# same run-indexed noise realization = common random numbers).
MachineFactory = Callable[[int], Machine]
# factory(machine=, in_spec=, spec=, seed=) -> a runnable campaign
MethodFactory = Callable[..., _Runnable]


@dataclass(frozen=True)
class Target:
    """One held-out spec: an id, the solver/loop ``spec`` dict, and the
    ground-truth ``in_spec(outcome) -> bool`` hit test."""

    id: str
    spec: dict[str, Any]
    in_spec: Callable[[np.ndarray], bool]


@dataclass
class Campaign:
    """One (method, target, seed) run reduced to survival data."""

    method: str
    target: str
    seed: int
    time: float  # cost-to-target if hit, else the censored (budget) cost
    event: bool  # True = hit (event), False = censored at budget
    n_queries: int


@dataclass
class TargetVerdict:
    target: str
    n_seeds: int
    rmst: dict[str, float]  # method -> RMST (smaller better)
    hit_rate: dict[str, float]  # method -> fraction of seeds hit
    median_cost: dict[str, float]
    delta_rmst: float  # RMST[reference] - RMST[other]  (< 0 => reference better)
    delta_se: float
    p_value: float
    win_rate: float  # paired: fraction of seeds where reference.time < other.time
    tie_rate: float
    reference: str
    other: str


@dataclass
class M2Report:
    reference: str
    other: str
    methods: list[str]
    n_targets: int
    n_seeds: int
    horizon: float
    per_target: list[TargetVerdict]
    # pooled across all (target, seed):
    pooled_rmst: dict[str, float]
    pooled_hit_rate: dict[str, float]
    pooled_delta_rmst: float
    pooled_delta_se: float
    pooled_p_value: float
    pooled_win_rate: float
    pooled_tie_rate: float
    bootstrap_ci95: tuple[float, float]  # paired bootstrap CI on pooled ΔRMST
    prob_reference_better: float  # bootstrap fraction with ΔRMST < 0
    verdict: str
    campaigns: list[Campaign] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bootstrap_ci95"] = list(self.bootstrap_ci95)
        return d


def _campaign_time(res: _Runnable) -> tuple[float, bool]:
    if res.hit:
        return float(res.cost_to_target), True
    censored = float(res.cumulative_cost[-1]) if res.cumulative_cost else float("inf")
    return censored, False


def _paired_bootstrap_delta(
    pairs: list[tuple[float, bool, float, bool]],
    horizon: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[tuple[float, float], float]:
    """Paired bootstrap over (ref_time, ref_event, other_time, other_event) units.
    Returns (95% CI on ΔRMST=RMST_ref-RMST_other, P(ΔRMST<0))."""
    if not pairs or n_bootstrap <= 0:
        return (float("nan"), float("nan")), float("nan")
    arr = np.array([(rt, re, ot, oe) for rt, re, ot, oe in pairs], dtype=float)
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    deltas = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        s = arr[idx]
        r = rmst(s[:, 0], s[:, 1].astype(bool), horizon).rmst
        o = rmst(s[:, 2], s[:, 3].astype(bool), horizon).rmst
        deltas[b] = r - o
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return (float(lo), float(hi)), float(np.mean(deltas < 0.0))


def run_m2_sweep(
    *,
    make_machine: MachineFactory,
    methods: dict[str, MethodFactory],
    targets: Sequence[Target],
    seeds: Sequence[int],
    horizon: float,
    reference: str | None = None,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
    progress: Callable[[str], None] | None = None,
) -> M2Report:
    """Run every (method, target, seed) campaign and score the head-to-head.

    ``methods`` maps name -> factory; exactly the ``reference`` method is compared
    against the single ``other`` (defaults: first key is reference, second is
    other). Difference-in-RMST uses delta = RMST[reference] - RMST[other], so a
    NEGATIVE pooled delta means the reference method reaches spec more cheaply.
    """
    names = list(methods)
    if len(names) != 2:
        raise ValueError(f"M2 sweep compares exactly two methods, got {names}")
    ref = reference or names[0]
    other = names[1] if names[0] == ref else names[0]

    campaigns: list[Campaign] = []
    for tgt in targets:
        for seed in seeds:
            for name in (ref, other):
                machine = make_machine(seed)  # fresh & seeded => common random numbers
                res = methods[name](machine=machine, in_spec=tgt.in_spec, spec=tgt.spec, seed=seed)
                run = res.run() if hasattr(res, "run") else res
                time, event = _campaign_time(run)
                campaigns.append(Campaign(name, tgt.id, seed, time, event, int(run.n_queries)))
            if progress is not None:
                progress(f"target={tgt.id} seed={seed} done")

    def _slice(method: str, target: str | None) -> tuple[np.ndarray, np.ndarray]:
        rows = [
            c for c in campaigns if c.method == method and (target is None or c.target == target)
        ]
        rows.sort(key=lambda c: c.seed)
        return (
            np.array([c.time for c in rows], dtype=float),
            np.array([c.event for c in rows], dtype=bool),
        )

    per_target: list[TargetVerdict] = []
    for tgt in targets:
        rt, re = _slice(ref, tgt.id)
        ot, oe = _slice(other, tgt.id)
        diff = rmst_difference_test(rt, re, ot, oe, horizon)
        wins = int(np.sum(rt < ot))
        ties = int(np.sum(rt == ot))
        per_target.append(
            TargetVerdict(
                target=tgt.id,
                n_seeds=len(seeds),
                rmst={ref: diff.rmst_a, other: diff.rmst_b},
                hit_rate={ref: float(np.mean(re)), other: float(np.mean(oe))},
                median_cost={ref: float(np.median(rt)), other: float(np.median(ot))},
                delta_rmst=diff.delta,
                delta_se=diff.se,
                p_value=diff.p_value,
                win_rate=wins / len(seeds),
                tie_rate=ties / len(seeds),
                reference=ref,
                other=other,
            )
        )

    # pooled across all (target, seed)
    prt, pre = _slice(ref, None)
    pot, poe = _slice(other, None)
    pooled = rmst_difference_test(prt, pre, pot, poe, horizon)
    pairs = list(zip(prt.tolist(), pre.tolist(), pot.tolist(), poe.tolist(), strict=True))
    ci, prob_better = _paired_bootstrap_delta(pairs, horizon, n_bootstrap, bootstrap_seed)
    pooled_wins = int(np.sum(prt < pot))
    pooled_ties = int(np.sum(prt == pot))
    n_pairs = len(prt)

    sig = pooled.p_value < 0.05
    direction = "cheaper" if pooled.delta < 0 else "costlier"
    verdict = (
        f"{ref} is {direction} than {other} by dRMST={pooled.delta:.3g} "
        f"(95% CI [{ci[0]:.3g}, {ci[1]:.3g}], p={pooled.p_value:.3g}, "
        f"win-rate {pooled_wins / max(n_pairs, 1):.0%}); "
        + ("SIGNIFICANT at 0.05." if sig else "NOT significant at 0.05.")
    )

    return M2Report(
        reference=ref,
        other=other,
        methods=names,
        n_targets=len(targets),
        n_seeds=len(seeds),
        horizon=float(horizon),
        per_target=per_target,
        pooled_rmst={ref: pooled.rmst_a, other: pooled.rmst_b},
        pooled_hit_rate={ref: float(np.mean(pre)), other: float(np.mean(poe))},
        pooled_delta_rmst=pooled.delta,
        pooled_delta_se=pooled.se,
        pooled_p_value=pooled.p_value,
        pooled_win_rate=pooled_wins / max(n_pairs, 1),
        pooled_tie_rate=pooled_ties / max(n_pairs, 1),
        bootstrap_ci95=ci,
        prob_reference_better=prob_better,
        verdict=verdict,
        campaigns=campaigns,
    )
