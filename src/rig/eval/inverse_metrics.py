"""Inverse-recipe-generation metrics (implementation-plan §12.2).

Scores a campaign's per-target outcomes against ground truth (the in-silico
machine, or real hardware). The distinctive, MFL-absent metrics live here:

- **target-hit-rate @ N** (PRIMARY): a target is "hit" iff the SINGLE top-ranked
  recipe lands in tolerance on ground truth; best-of-q is reported separately
  and never conflated (§12.2).
- **feasibility & abstention calibration:** the target set is pre-registered to
  include known-infeasible targets. ``false_success_rate`` (a "hit" on an
  infeasible target — must be ≈0; MFL's clip() manufactures these) and
  ``false_abstention_rate`` (a known-feasible-but-hard target wrongly refused —
  structurally induced by our own pessimism + trust region, invisible unless
  pre-registered).
- **constraint-satisfaction rate BEFORE any projection** (constraint-by-
  construction scores ≈1.0; a clip-heavy method is honestly penalized).
- **robust-hit-rate:** perturb inputs by actuation noise, fraction still in tol.

numpy only; ground-truth "in tolerance" is decided by the caller (τ floored at
Gage-R&R repeatability, §12.2) and passed in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from rig.constraints import ConstraintSet


@dataclass(frozen=True)
class TargetOutcome:
    """One pre-registered target's campaign result (ground-truth scored).

    ``feasible_truth`` is the pre-registered ground-truth feasibility of the
    target (known before the campaign). ``declared_infeasible`` is whether the
    solver returned INFEASIBLE. ``hit`` is whether the top-ranked recipe landed
    in tolerance on ground truth (False whenever ``declared_infeasible``).
    ``cost`` is cost-to-target (event: hit) or the exhausted budget (censored).
    """

    target_id: str
    feasible_truth: bool
    declared_infeasible: bool
    hit: bool
    cost: float

    def __post_init__(self) -> None:
        if self.declared_infeasible and self.hit:
            raise ValueError(f"{self.target_id}: declared_infeasible but hit=True is contradictory")


def target_hit_rate(outcomes: Sequence[TargetOutcome], *, feasible_only: bool = True) -> float:
    """Fraction of targets whose top-ranked recipe hit (§12.2 PRIMARY).

    ``feasible_only`` (default) restricts to ground-truth-feasible targets — a
    hit on an infeasible target is scored by :func:`false_success_rate`, not
    here. Returns nan if no target qualifies.
    """
    pool = [o for o in outcomes if (o.feasible_truth or not feasible_only)]
    if not pool:
        return float("nan")
    return sum(o.hit for o in pool) / len(pool)


def success_rate_at_budget(
    outcomes: Sequence[TargetOutcome], budget: float, *, feasible_only: bool = True
) -> float:
    """Fraction of (feasible) targets hit within ``budget`` (§12.2 success-vs-
    budget curve). A target counts only if it hit AND cost ≤ budget."""
    pool = [o for o in outcomes if (o.feasible_truth or not feasible_only)]
    if not pool:
        return float("nan")
    return sum(o.hit and o.cost <= budget for o in pool) / len(pool)


def false_success_rate(outcomes: Sequence[TargetOutcome]) -> float:
    """Among ground-truth-INFEASIBLE targets, the fraction reported as a hit
    (§12.2). MUST be ≈0 — a hit on an infeasible target is a bug (clip() manufactures
    these). nan if there are no infeasible targets in the set."""
    pool = [o for o in outcomes if not o.feasible_truth]
    if not pool:
        return float("nan")
    return sum(o.hit for o in pool) / len(pool)


def false_abstention_rate(outcomes: Sequence[TargetOutcome]) -> float:
    """Among ground-truth-FEASIBLE targets, the fraction wrongly declared
    infeasible (§12.2 Type-I). Pessimism + the trust region push the estimated
    reachable set inward, so this is the direct cost of "refuses-instead-of-
    clips" — headline-relevant, invisible unless measured. nan if no feasible
    targets."""
    pool = [o for o in outcomes if o.feasible_truth]
    if not pool:
        return float("nan")
    return sum(o.declared_infeasible for o in pool) / len(pool)


def feasibility_flag_accuracy(outcomes: Sequence[TargetOutcome]) -> float:
    """Fraction of targets whose feasibility was correctly flagged: infeasible
    truth ⇒ declared infeasible, feasible truth ⇒ not declared infeasible."""
    if not outcomes:
        return float("nan")
    correct = sum(o.declared_infeasible == (not o.feasible_truth) for o in outcomes)
    return correct / len(outcomes)


def constraint_satisfaction_rate(
    recipes: Sequence[Mapping[str, float]], constraints: ConstraintSet
) -> float:
    """Fraction of proposed recipes satisfying the hard constraints BEFORE any
    projection/clipping (§12.2). Constraint-by-construction (rig.transforms)
    scores ≈1.0 by design; a method that needs heavy clipping is penalized."""
    if not recipes:
        return float("nan")
    return sum(constraints.is_satisfied(dict(r)) for r in recipes) / len(recipes)


def robust_hit_rate(
    recipe: Mapping[str, float],
    machine: Callable[[Mapping[str, float]], np.ndarray],
    in_tolerance: Callable[[np.ndarray], bool],
    *,
    actuation_noise: Mapping[str, float],
    n_samples: int = 64,
    seed: int = 0,
) -> float:
    """Robust-hit-rate (§12.2): perturb the recipe by known per-variable
    actuation noise (Gaussian σ), fraction of realized outcomes still in
    tolerance. Rewards flat basins — MFL's untested "domain randomization"
    turned into a ranked metric.

    ``machine`` maps a (perturbed) recipe dict to an outcome vector;
    ``in_tolerance`` decides whether an outcome hits spec; ``actuation_noise``
    is the per-variable std in the variable's units (missing keys → no jitter).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_samples):
        perturbed = {
            k: float(v + rng.normal(0.0, actuation_noise.get(k, 0.0))) for k, v in recipe.items()
        }
        if in_tolerance(np.asarray(machine(perturbed), dtype=float)):
            hits += 1
    return hits / n_samples
