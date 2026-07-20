"""Evaluation & benchmarking harness (implementation-plan §12) — the measurement layer that
makes MFL's failures impossible to hide. numpy/scipy only.

- ``survival``: cost-to-target as right-censored survival (KM, RMST,
  difference-in-RMST; infeasible-exclusion) — the §12.2 primary comparator.
- ``inverse_metrics``: target-hit-rate, feasibility/abstention calibration,
  constraint-satisfaction, robust-hit-rate.
- ``exploitation``: the §12.1 surrogate-exploitation stress test (headline).
- ``diversity``: Vendi score + mode-count + pairwise-L2 (§12.2).
- ``calibration_gates``: SBC + TARP posterior-recovery gates (§12.2).
"""

from rig.eval.calibration_gates import (
    UniformityResult,
    sbc_ranks,
    sbc_test,
    tarp_credibilities,
    tarp_test,
    uniformity_test,
)
from rig.eval.diversity import mean_pairwise_l2, mode_count, vendi_score
from rig.eval.exploitation import ExploitationReport, exploitation_stress_test
from rig.eval.inverse_metrics import (
    TargetOutcome,
    constraint_satisfaction_rate,
    false_abstention_rate,
    false_success_rate,
    feasibility_flag_accuracy,
    robust_hit_rate,
    success_rate_at_budget,
    target_hit_rate,
)
from rig.eval.m2_sweep import Campaign, M2Report, Target, run_m2_sweep
from rig.eval.survival import (
    KaplanMeier,
    RMSTDifference,
    RMSTResult,
    kaplan_meier,
    rmst,
    rmst_difference_test,
    split_feasible,
)

__all__ = [
    "Campaign",
    "ExploitationReport",
    "KaplanMeier",
    "M2Report",
    "RMSTDifference",
    "RMSTResult",
    "Target",
    "TargetOutcome",
    "UniformityResult",
    "constraint_satisfaction_rate",
    "run_m2_sweep",
    "exploitation_stress_test",
    "false_abstention_rate",
    "false_success_rate",
    "feasibility_flag_accuracy",
    "kaplan_meier",
    "mean_pairwise_l2",
    "mode_count",
    "rmst",
    "rmst_difference_test",
    "robust_hit_rate",
    "sbc_ranks",
    "sbc_test",
    "split_feasible",
    "success_rate_at_budget",
    "tarp_credibilities",
    "tarp_test",
    "target_hit_rate",
    "uniformity_test",
    "vendi_score",
]
