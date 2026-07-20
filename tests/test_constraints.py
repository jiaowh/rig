"""Declarative ConstraintSet checker tests (implementation-plan §3.1, §8.3)."""

from rig.constraints import (
    BoxConstraint,
    ConstraintSet,
    LinearConstraint,
    MonotoneConstraint,
    SimplexConstraint,
)
from rig.interfaces import ChangeCost

CS = ConstraintSet(
    box=(BoxConstraint(name="temp", lower=300.0, upper=900.0),),
    simplex=(SimplexConstraint(components=("alloy.ga", "alloy.al")),),
    linear=(LinearConstraint(coefficients={"flow_a": 1.0, "flow_b": 1.0}, upper=100.0),),
    monotone=(MonotoneConstraint(output="growth_rate", wrt_input="temp", direction="increasing"),),
    change_cost={"chamber": ChangeCost.HARD_TO_CHANGE, "temp": ChangeCost.EASY},
)

GOOD = {"temp": 500.0, "alloy.ga": 0.7, "alloy.al": 0.3, "flow_a": 40.0, "flow_b": 30.0}


def test_satisfied_recipe_has_no_violations():
    assert CS.validate(GOOD) == []
    assert CS.is_satisfied(GOOD)


def test_box_violation_reported():
    bad = GOOD | {"temp": 1000.0}
    assert any(v.startswith("box:") for v in CS.validate(bad))


def test_simplex_violations_reported():
    assert any(v.startswith("simplex:") for v in CS.validate(GOOD | {"alloy.ga": 0.9}))  # sum != 1
    assert any(
        v.startswith("simplex:")
        for v in CS.validate(GOOD | {"alloy.ga": -0.1, "alloy.al": 1.1})  # negative
    )


def test_linear_violation_reported():
    bad = GOOD | {"flow_a": 80.0, "flow_b": 30.0}
    assert any(v.startswith("linear:") for v in CS.validate(bad))


def test_missing_variable_reported():
    assert any("missing" in v for v in CS.validate({"temp": 500.0}))


def test_monotone_not_pointwise_checked():
    """Monotone constraints are declarations for the surrogate (§6.3), not
    pointwise-checkable - validate() must not flag them."""
    assert CS.is_satisfied(GOOD)


def test_change_cost_tags_first_class():
    assert CS.change_cost["chamber"] is ChangeCost.HARD_TO_CHANGE
    assert CS.change_cost["temp"] is ChangeCost.EASY
