"""Calibration — the guarantee layer (implementation-plan §5.6, decision D4)."""

from rig.calibration.conformal import (
    ACIController,
    ConformalForwardModel,
    JackknifePlusCalibrator,
    SplitConformalCalibrator,
)
from rig.calibration.mondrian import (
    MondrianConformalCalibrator,
    MondrianConformalForwardModel,
    finite_quantile_floor,
    predicted_magnitude_group_fn,
)
from rig.calibration.pid import ConformalPIDController

__all__ = [
    "ACIController",
    "ConformalForwardModel",
    "ConformalPIDController",
    "JackknifePlusCalibrator",
    "MondrianConformalCalibrator",
    "MondrianConformalForwardModel",
    "SplitConformalCalibrator",
    "finite_quantile_floor",
    "predicted_magnitude_group_fn",
]
