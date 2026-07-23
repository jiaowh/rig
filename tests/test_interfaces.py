"""Canonical-name and D7 tests (implementation-plan §3.2, §13.3)."""

import dataclasses
import warnings

import numpy as np
import pytest

from rig.interfaces import (
    AdapterValidationError,
    ChangeCost,
    ContinuousVariable,
    CostModel,
    Infeasible,
    OutputSpec,
    PredictiveDistribution,
    RecipeCandidate,
    sobol_seed_design,
    validate_adapter,
)

# ---------------------------------------------------------------------------
# §3.2: exact canonical field set AND order - binding, verbatim everywhere
# ---------------------------------------------------------------------------


def test_predictive_distribution_canonical_fields_in_order():
    names = [f.name for f in dataclasses.fields(PredictiveDistribution)]
    assert names == ["mean", "aleatoric_sigma", "epistemic_sigma", "conformal_set"]


def test_recipe_candidate_canonical_fields():
    # The five §3.3 canonical fields are binding and come FIRST, in order; a sixth,
    # `calibration_status`, is the F1 (audit 2026-07-21) provenance extension. Both the
    # canonical set AND the extension are pinned here so future drift in either is caught.
    names = [f.name for f in dataclasses.fields(RecipeCandidate)]
    assert names == [
        "recipe",
        "confidence",
        "predicted_outcome_interval",
        "feasibility_flag",
        "support_score",
        "calibration_status",
    ]
    # the provenance tag defaults to the UNCALIBRATED label (raw-σ pessimism only), so a
    # candidate built without it is never silently presented as conformally accepted.
    assert RecipeCandidate.__dataclass_fields__["calibration_status"].default == "model-feasible"


def test_infeasible_is_explicit_verdict():
    v = Infeasible(nearest_achievable={"temp": 900.0}, distance_to_feasible=3.2)
    assert not isinstance(v, list)  # tagged union: never a clipped point/list
    assert v.distance_to_feasible == 3.2


# ---------------------------------------------------------------------------
# D7: physics plug-in must differ from the independent verifier
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal structural ProcessAdapter for validation tests."""

    def __init__(self, physics=None, verifier=None, process_id="stub"):
        self._physics = physics
        self._verifier = verifier
        self._process_id = process_id

    @property
    def process_id(self):
        return self._process_id

    @property
    def input_schema(self):
        return (ContinuousVariable(name="t", lower=0.0, upper=1.0, change_cost=ChangeCost.EASY),)

    @property
    def output_schema(self):
        return (OutputSpec(name="y", modality="scalar_vector"),)

    @property
    def cost_model(self):
        return CostModel(c_batch=100.0, c_recipe=lambda r: 1.0, batch_size=4)

    @property
    def expert_ranges(self):
        return {"t": (0.1, 0.9)}

    def seed_design(self, n_runs, seed):
        return sobol_seed_design(self.expert_ranges, n_runs, seed)

    @property
    def physics_plugin(self):
        return self._physics

    @property
    def independent_verifier(self):
        return self._verifier

    def encode_recipe(self, recipe):
        return np.array([recipe["t"]])

    def decode_recipe(self, x):
        return {"t": float(x[0])}


def test_d7_same_object_fails_validation():
    def shared_model(x):
        return x

    adapter = _StubAdapter(physics=shared_model, verifier=shared_model)
    with pytest.raises(AdapterValidationError, match="D7"):
        validate_adapter(adapter)


def test_d7_different_objects_pass():
    validate_adapter(_StubAdapter(physics=lambda x: x, verifier=lambda x: 2 * x))


def test_d7_absent_by_default_passes():
    validate_adapter(_StubAdapter())  # both None: fine (§3.1 - absent by default)


# ---------------------------------------------------------------------------
# DoE hook: scrambled Sobol seed design
# ---------------------------------------------------------------------------


def test_sobol_seed_design_in_ranges_and_deterministic():
    ranges = {"temp": (300.0, 900.0), "flow": (0.0, 10.0)}
    d1 = sobol_seed_design(ranges, n_runs=16, seed=7)
    d2 = sobol_seed_design(ranges, n_runs=16, seed=7)
    assert d1 == d2  # seeded determinism (implementation-plan §13.4)
    assert len(d1) == 16
    for row in d1:
        assert 300.0 <= row["temp"] <= 900.0
        assert 0.0 <= row["flow"] <= 10.0
    d3 = sobol_seed_design(ranges, n_runs=16, seed=8)
    assert d3 != d1


def test_sobol_non_power_of_two_is_silent_and_deterministic():
    # audit D8: a non-power-of-2 n_runs (typical DoE size) is a valid design;
    # scipy's balance UserWarning must be suppressed, and output stays
    # deterministic and in range.
    ranges = {"a": (0.0, 1.0), "b": (0.0, 2.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any UserWarning would fail the test
        d1 = sobol_seed_design(ranges, n_runs=10, seed=3)
        d2 = sobol_seed_design(ranges, n_runs=10, seed=3)
    assert len(d1) == 10
    assert d1 == d2
    for row in d1:
        assert 0.0 <= row["a"] <= 1.0
        assert 0.0 <= row["b"] <= 2.0


def test_sobol_empty_ranges_raises_clear_error():
    # audit D8: an empty ranges dict must raise a descriptive ValueError, not an
    # opaque numpy 'zero-size array to reduction' error.
    with pytest.raises(ValueError, match="non-empty"):
        sobol_seed_design({}, n_runs=4, seed=1)
