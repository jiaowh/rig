"""WP-B adapter tests: conformance, recipe-vs-config split (E2), cost model,
DoE hooks, encoders, and the D7 honesty state (implementation-plan §3.1, §15.2, §15.6)."""

from pathlib import Path

import pytest

from rig import registry
from rig.interfaces import ChangeCost, ContinuousVariable, validate_adapter
from rig.schema import Quantity, RecipeRecord
from rig_adapters.mbe import simlink
from rig_adapters.mbe.adapter import (
    MACHINE_CONFIG_BOUNDS,
    MACHINE_CONFIG_DEFAULTS,
    RECIPE_VARIABLE_NAMES,
    MBEAdapter,
    evaluate_physics,
    make_adapter,
)
from rig_adapters.mbe.outcomes import metrics_to_outcomes

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

RECIPE = {"T_heater": 1320.0, "film_thickness": 1e-6}


@pytest.fixture(scope="module")
def adapter() -> MBEAdapter:
    return make_adapter()


def _recipe_record(**overrides) -> RecipeRecord:
    values = dict(RECIPE, **overrides)
    units = {"T_heater": "K", "film_thickness": "m"}
    return RecipeRecord(values={k: Quantity(magnitude=v, unit=units[k]) for k, v in values.items()})


# -- conformance --------------------------------------------------------------


def test_validate_adapter_passes(adapter):
    validate_adapter(adapter)  # includes the D7 identity check


def test_registry_round_trip():
    registry.register_adapter_for_testing("mbe_test", make_adapter)
    try:
        loaded = registry.get_adapter("mbe_test")
        assert isinstance(loaded, MBEAdapter)
        assert loaded.process_id == "mbe"
    finally:
        registry.clear_test_registry()


def test_entry_point_declared_in_pyproject():
    """The packaging wiring for self-registration (implementation-plan §3) is present."""
    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mbe = "rig_adapters.mbe.adapter:make_adapter"' in text


# -- recipe-vs-config split (E2) ----------------------------------------------


def test_recipe_vs_config_change_cost_split(adapter):
    by_cost = {}
    for var in adapter.input_schema:
        assert isinstance(var, ContinuousVariable)
        by_cost.setdefault(var.change_cost, set()).add(var.name)
    assert by_cost[ChangeCost.EASY] == {"T_heater", "film_thickness"}
    assert by_cost[ChangeCost.HARD_TO_CHANGE] == {
        "heater_radius",
        "gap",
        "source_offset",
        "source_height",
        "aim_offset",
    }


def test_bounds_mirror_sim_default_knobs(adapter):
    """The hardcoded bound mirror must stay in sync with the sim repo."""
    sim = simlink.load_mbe_sim()
    knobs = sim.optimize.DEFAULT_KNOBS
    t = next(v for v in adapter.input_schema if v.name == "T_heater")
    assert (t.lower, t.upper) == (knobs["T_heater"][1], knobs["T_heater"][2])
    for name, (default, lo, hi) in MACHINE_CONFIG_BOUNDS.items():
        assert (default, lo, hi) == tuple(knobs[name][:3]), name
        assert MACHINE_CONFIG_DEFAULTS[name] == knobs[name][0]


def test_output_schema_scalar_vector(adapter):
    names = [o.name for o in adapter.output_schema]
    assert names == [
        "nonuniformity_pct",
        "T_center",
        "slip_max_ratio",
        "bow_cooldown_um",
        "thickness_grown",
    ]
    assert all(o.modality == "scalar_vector" for o in adapter.output_schema)


# -- recipe validation ---------------------------------------------------------


def test_in_bounds_recipe_validates(adapter):
    _recipe_record().validate_against(list(adapter.input_schema))


def test_out_of_bounds_t_heater_rejected(adapter):
    with pytest.raises(ValueError, match="T_heater"):
        _recipe_record(T_heater=2000.0).validate_against(list(adapter.input_schema))


def test_wrong_unit_rejected(adapter):
    bad = RecipeRecord(
        values={
            "T_heater": Quantity(magnitude=1320.0, unit="m"),  # wrong dimension
            "film_thickness": Quantity(magnitude=1e-6, unit="m"),
        }
    )
    with pytest.raises(ValueError, match="T_heater"):
        bad.validate_against(list(adapter.input_schema))


# -- cost model / DoE hooks -----------------------------------------------------


def test_cost_model_kanarik_defaults(adapter):
    cm = adapter.cost_model
    assert cm.c_batch == 1000.0
    assert cm.c_recipe(RECIPE) == 1000.0
    assert cm.batch_size == 4


def test_expert_ranges_are_recipe_vars_only(adapter):
    assert set(adapter.expert_ranges) == set(RECIPE_VARIABLE_NAMES)


def test_seed_design_in_bounds_and_recipe_only(adapter):
    design = adapter.seed_design(8, seed=1)
    assert len(design) == 8
    for point in design:
        assert set(point) == set(RECIPE_VARIABLE_NAMES)
        for name, value in point.items():
            lo, hi = adapter.expert_ranges[name]
            assert lo <= value <= hi
    assert adapter.seed_design(8, seed=1) == design  # seeded => reproducible


# -- encoders --------------------------------------------------------------------


def test_encode_decode_round_trip(adapter):
    x = adapter.encode_recipe(RECIPE)
    assert x.shape == (len(RECIPE_VARIABLE_NAMES),)
    assert adapter.decode_recipe(x) == RECIPE


def test_encode_accepts_quantities(adapter):
    rec = _recipe_record()
    x = adapter.encode_recipe(rec.values)
    assert adapter.decode_recipe(x) == RECIPE


# -- physics plug-in / D7 ---------------------------------------------------------


def test_physics_plugin_present_verifier_absent(adapter):
    assert adapter.physics_plugin is not None
    assert adapter.independent_verifier is None  # honest D7 state (E2)


def test_physics_plugin_matches_direct_evaluation(adapter):
    y = adapter.physics_plugin(RECIPE)
    metrics = evaluate_physics(RECIPE)
    assert y["T_center"] == metrics["T_center"]
    assert y["nonuniformity_pct"] == metrics["combined_nonuniformity_pct"]
    assert y["thickness_grown"] == RECIPE["film_thickness"]  # clean: no flux loss


# -- outcomes translation ----------------------------------------------------------


def test_metrics_to_outcomes_translation(adapter):
    outcomes = metrics_to_outcomes(evaluate_physics(RECIPE))
    by_name = {o.name: o for o in outcomes}
    assert set(by_name) == {o.name for o in adapter.output_schema}
    assert all(o.modality == "scalar_vector" for o in outcomes)
    # percent canonicalizes to a dimensionless fraction; um to metres.
    assert by_name["nonuniformity_pct"].value.unit == "dimensionless"
    assert by_name["bow_cooldown_um"].value.unit == "m"
    assert by_name["T_center"].value.unit == "K"


def test_evaluate_physics_missing_variable_raises():
    with pytest.raises(ValueError, match="film_thickness"):
        evaluate_physics({"T_heater": 1320.0})
