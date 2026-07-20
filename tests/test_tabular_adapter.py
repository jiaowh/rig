"""WP-H spec + adapter tests: spec parsing (happy path + every validation
error, incl. the E5 sccm-compositional rejection), ProcessAdapter conformance,
DoE feasibility, encoders, and the parameterized-factory pattern."""

import json
from pathlib import Path

import pytest

from rig import registry
from rig.interfaces import (
    CategoricalVariable,
    ChangeCost,
    CompositionalVariable,
    ContinuousVariable,
    validate_adapter,
)
from rig_adapters.tabular.adapter import SPEC_ENV_VAR, TabularAdapter, make_adapter
from rig_adapters.tabular.spec import ProcessSpec, SpecError, load_spec, parse_spec

REPO = Path(__file__).resolve().parents[1]
PECVD_SPEC = REPO / "examples" / "pecvd_example.toml"
MINIMAL_SPEC = REPO / "examples" / "tabular_minimal.toml"


def base_spec_dict() -> dict:
    """A small valid spec dict tests mutate to hit each validation error."""
    return {
        "process_id": "unit_test_proc",
        "inputs": {
            "temp": {"kind": "continuous", "unit": "degC", "lower": 100.0, "upper": 200.0},
            "mode": {"kind": "categorical", "levels": ["a", "b"]},
            "blend": {"kind": "compositional", "components": ["x", "y"]},
        },
        "outputs": {"thick": {"unit": "nm"}},
        "cost": {"c_batch": 500.0, "c_recipe": 250.0, "batch_size": 2},
    }


# -- spec happy path -----------------------------------------------------------


def test_minimal_example_spec_loads():
    spec = load_spec(MINIMAL_SPEC)
    assert spec.process_id == "demo_minimal"
    assert len(spec.variables) == 1 and len(spec.outputs) == 1
    # cost block omitted -> Kanarik defaults
    assert (spec.c_batch, spec.c_recipe, spec.batch_size) == (1000.0, 1000.0, 4)


def test_pecvd_example_spec_loads():
    spec = load_spec(PECVD_SPEC)
    assert spec.process_id == "pecvd_sin_demo"
    by_name = {v.name: v for v in spec.variables}
    temp = by_name["temperature"]
    assert isinstance(temp, ContinuousVariable)
    assert (temp.unit, temp.lower, temp.upper) == ("degC", 200.0, 400.0)
    blend = by_name["precursor_blend"]
    assert isinstance(blend, CompositionalVariable)
    assert blend.components == ("silane", "ammonia", "nitrogen")
    assert spec.flat_input_names == (
        "temperature",
        "pressure",
        "rf_power",
        "precursor_blend.silane",
        "precursor_blend.ammonia",
        "precursor_blend.nitrogen",
    )
    out = {o.name: o for o in spec.outputs}
    assert out["thickness"].target == 500.0
    assert out["nonuniformity"].upper_spec == 5.0
    assert all(o.modality == "scalar_vector" for o in spec.outputs)


def test_parse_spec_dict_and_defaults():
    data = base_spec_dict()
    del data["cost"]
    spec = parse_spec(data)
    assert (spec.c_batch, spec.c_recipe, spec.batch_size) == (1000.0, 1000.0, 4)
    assert isinstance(spec.variables[1], CategoricalVariable)
    assert spec.numeric_input_names == ("temp", "blend.x", "blend.y")


def test_json_spec_loads(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(base_spec_dict()), encoding="utf-8")
    spec = load_spec(path)
    assert spec.process_id == "unit_test_proc"
    assert spec.batch_size == 2


def test_change_cost_hard_parses():
    data = base_spec_dict()
    data["inputs"]["temp"]["change_cost"] = "hard"
    spec = parse_spec(data)
    assert spec.variables[0].change_cost is ChangeCost.HARD_TO_CHANGE
    assert spec.variables[1].change_cost is ChangeCost.EASY  # default


# -- spec validation errors ------------------------------------------------------


def _expect_spec_error(data: dict, match: str) -> None:
    with pytest.raises(SpecError, match=match):
        parse_spec(data)


def test_missing_process_id_rejected():
    data = base_spec_dict()
    del data["process_id"]
    _expect_spec_error(data, "process_id")


def test_unknown_top_level_key_rejected():
    data = base_spec_dict()
    data["inpts"] = data.pop("inputs")  # typo must fail loudly
    _expect_spec_error(data, "inpts")


def test_empty_inputs_rejected():
    data = base_spec_dict()
    data["inputs"] = {}
    _expect_spec_error(data, "inputs")


def test_empty_outputs_rejected():
    data = base_spec_dict()
    data["outputs"] = {}
    _expect_spec_error(data, "outputs")


def test_unknown_kind_rejected():
    data = base_spec_dict()
    data["inputs"]["temp"]["kind"] = "florp"
    _expect_spec_error(data, "florp")


def test_unknown_input_key_rejected():
    data = base_spec_dict()
    data["inputs"]["temp"]["upperr"] = 3.0  # typo must fail loudly
    _expect_spec_error(data, "upperr")


def test_bad_unit_rejected():
    data = base_spec_dict()
    data["inputs"]["temp"]["unit"] = "florps_per_zorp"
    _expect_spec_error(data, "not pint-parseable")


def test_inverted_bounds_rejected():
    data = base_spec_dict()
    data["inputs"]["temp"]["lower"] = 300.0  # > upper=200
    _expect_spec_error(data, "lower")


def test_single_level_categorical_rejected():
    data = base_spec_dict()
    data["inputs"]["mode"]["levels"] = ["a"]
    _expect_spec_error(data, "levels")


def test_single_component_compositional_rejected():
    data = base_spec_dict()
    data["inputs"]["blend"]["components"] = ["x"]
    _expect_spec_error(data, "components")


def test_compositional_with_flow_unit_rejected():
    """The E5-mandated tag: sccm MFC flows are NOT a simplex (implementation-plan §3.1)."""
    data = base_spec_dict()
    data["inputs"]["blend"]["unit"] = "sccm"
    with pytest.raises(SpecError) as excinfo:
        parse_spec(data)
    message = str(excinfo.value)
    assert "sccm" in message
    assert "NOT a simplex" in message
    assert "3.1" in message  # cites implementation-plan §3.1
    assert "continuous" in message  # tells the author what to do instead


def test_compositional_dimensionless_unit_allowed():
    data = base_spec_dict()
    data["inputs"]["blend"]["unit"] = "dimensionless"
    assert parse_spec(data).compositional[0].components == ("x", "y")


def test_compositional_scaled_dimensionless_unit_rejected():
    # audit D10: percent/ppm are dimensionless-DIMENSIONALITY but scaled, and
    # ingest ignores the unit and reads raw values as fractions — so they must be
    # rejected, not silently mis-scaled. Only a literally-dimensionless unit passes.
    for bad_unit in ("percent", "ppm"):
        data = base_spec_dict()
        data["inputs"]["blend"]["unit"] = bad_unit
        with pytest.raises(SpecError, match="dimensionless"):
            parse_spec(data)


def test_gp_input_keys_drops_one_component_per_composition_order_safe():
    # audit B4: gp_input_keys must drop exactly one component PER compositional
    # variable regardless of declaration order. Here 'blend' is declared BEFORE
    # 'temp', so numeric_input_names[:-1] would wrongly drop 'temp' and keep all
    # three collinear blend coords (a rank-deficient GP design).
    data = {
        "process_id": "order_test",
        "inputs": {
            "blend": {"kind": "compositional", "components": ["x", "y", "z"]},
            "temp": {"kind": "continuous", "unit": "degC", "lower": 100.0, "upper": 200.0},
        },
        "outputs": {"thick": {"unit": "nm"}},
    }
    spec = parse_spec(data)
    assert spec.numeric_input_names == ("blend.x", "blend.y", "blend.z", "temp")
    assert spec.gp_input_keys == ("blend.y", "blend.z", "temp")  # one blend comp dropped, temp kept
    assert len(spec.gp_input_keys) == len(spec.numeric_input_names) - 1


def test_continuous_si_offset_and_multiplicative_units():
    # Finding A: continuous_si must convert OFFSET units (degC) additively
    # (+273.15), NOT by a multiplicative scale factor, while still converting a
    # multiplicative unit (mtorr) correctly. degC bounds are absolute POINTS, so
    # 50 degC -> 323.15 K (a delta/width would differ, but these fields are points).
    from rig.schema import ureg

    data = {
        "process_id": "si_test",
        "inputs": {
            "temp": {"kind": "continuous", "unit": "degC", "lower": 50.0, "upper": 400.0},
            "press": {"kind": "continuous", "unit": "mtorr", "lower": 1.0, "upper": 43.0},
        },
        "outputs": {"thick": {"unit": "nm"}},
    }
    si = {v.name: v for v in parse_spec(data).continuous_si}
    assert si["temp"].unit == "K"
    assert si["temp"].lower == pytest.approx(323.15)
    assert si["temp"].upper == pytest.approx(673.15)
    assert si["press"].lower == pytest.approx(ureg.Quantity(1.0, "mtorr").to_base_units().magnitude)
    assert si["press"].upper == pytest.approx(
        ureg.Quantity(43.0, "mtorr").to_base_units().magnitude
    )


def test_pecvd_continuous_si_temperature_offset_corrected():
    # Finding A on the SHIPPED spec: temperature is degC 200..400 -> 473.15..673.15 K.
    spec = load_spec(PECVD_SPEC)
    si = {v.name: v for v in spec.continuous_si}
    assert si["temperature"].unit == "K"
    assert si["temperature"].lower == pytest.approx(473.15)
    assert si["temperature"].upper == pytest.approx(673.15)


def test_outputs_si_offset_safe():
    # Finding D: outputs_si mirrors continuous_si (offset-safe per finding A) so
    # OutputSpec target/lower_spec/upper_spec are available in SI (the SI-trap the
    # continuous_si note in CLAUDE.md warns about, applied to outputs).
    data = {
        "process_id": "out_si_test",
        "inputs": {"temp": {"kind": "continuous", "unit": "degC", "lower": 50.0, "upper": 400.0}},
        "outputs": {
            "substrate_temp": {
                "unit": "degC",
                "target": 300.0,
                "lower_spec": 250.0,
                "upper_spec": 350.0,
            },
            "thick": {"unit": "nm", "target": 500.0},  # unset specs must stay None
        },
    }
    out = {o.name: o for o in parse_spec(data).outputs_si}
    st = out["substrate_temp"]
    assert st.unit == "K"
    assert st.target == pytest.approx(573.15)
    assert st.lower_spec == pytest.approx(523.15)
    assert st.upper_spec == pytest.approx(623.15)
    thick = out["thick"]
    assert thick.unit == "m"
    assert thick.target == pytest.approx(5e-7)
    assert thick.lower_spec is None and thick.upper_spec is None


def test_curve_modality_not_yet_supported():
    data = base_spec_dict()
    data["outputs"]["thick"]["modality"] = "curve_1d"
    _expect_spec_error(data, "not yet supported")


def test_unknown_modality_rejected():
    data = base_spec_dict()
    data["outputs"]["thick"]["modality"] = "hologram"
    _expect_spec_error(data, "hologram")


def test_inverted_specs_rejected():
    data = base_spec_dict()
    data["outputs"]["thick"].update(lower_spec=10.0, upper_spec=5.0)
    _expect_spec_error(data, "lower_spec")


def test_target_outside_specs_rejected():
    data = base_spec_dict()
    data["outputs"]["thick"].update(lower_spec=1.0, upper_spec=5.0, target=9.0)
    _expect_spec_error(data, "target")


def test_bad_batch_size_rejected():
    data = base_spec_dict()
    data["cost"]["batch_size"] = 0
    _expect_spec_error(data, "batch_size")


def test_dot_in_name_rejected():
    data = base_spec_dict()
    data["inputs"]["a.b"] = {"kind": "continuous", "lower": 0.0, "upper": 1.0}
    _expect_spec_error(data, "reserved")


def test_dot_in_output_name_rejected():
    data = base_spec_dict()
    data["outputs"]["blend.x"] = {"unit": "nm"}  # would shadow a flattened input
    _expect_spec_error(data, "reserved")


def test_input_output_name_overlap_rejected():
    data = base_spec_dict()
    data["outputs"]["temp"] = {"unit": "nm"}
    _expect_spec_error(data, "both input and output")


def test_unsupported_suffix_rejected(tmp_path):
    path = tmp_path / "spec.yaml"
    path.write_text("process_id: x", encoding="utf-8")
    with pytest.raises(SpecError, match="suffix"):
        load_spec(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(SpecError, match="not found"):
        load_spec(tmp_path / "nope.toml")


def test_invalid_toml_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("process_id = ", encoding="utf-8")
    with pytest.raises(SpecError, match="invalid TOML"):
        load_spec(path)


# -- adapter conformance -----------------------------------------------------------


@pytest.fixture(scope="module")
def adapter() -> TabularAdapter:
    return TabularAdapter.from_spec(PECVD_SPEC)


def test_validate_adapter_passes(adapter):
    validate_adapter(adapter)  # includes the D7 identity check
    assert adapter.process_id == "pecvd_sin_demo"


def test_d7_honesty_no_physics_no_verifier(adapter):
    assert adapter.physics_plugin is None
    assert adapter.independent_verifier is None


def test_cost_model_from_spec(adapter):
    cm = adapter.cost_model
    assert cm.c_batch == 1000.0
    assert cm.c_recipe({}) == 1000.0
    assert cm.batch_size == 4


def test_seed_design_feasible_and_deterministic(adapter):
    design = adapter.seed_design(8, seed=3)
    assert len(design) == 8
    ranges = adapter.expert_ranges
    comps = ["precursor_blend.silane", "precursor_blend.ammonia", "precursor_blend.nitrogen"]
    for point in design:
        assert set(point) == set(ranges)
        for name in ("temperature", "pressure", "rf_power"):
            lo, hi = ranges[name]
            assert lo <= point[name] <= hi
        total = sum(point[c] for c in comps)
        assert total == pytest.approx(1.0, abs=1e-12)  # feasible simplex seed (E5)
        assert all(point[c] >= 0.0 for c in comps)
    assert adapter.seed_design(8, seed=3) == design  # seeded => reproducible


def test_encode_decode_round_trip(adapter):
    recipe = {
        "temperature": 300.0,
        "pressure": 2.0,
        "rf_power": 100.0,
        "precursor_blend.silane": 0.2,
        "precursor_blend.ammonia": 0.3,
        "precursor_blend.nitrogen": 0.5,
    }
    x = adapter.encode_recipe(recipe)
    assert x.shape == (6,)
    assert adapter.decode_recipe(x) == recipe


def test_encode_missing_variable_raises(adapter):
    with pytest.raises(ValueError, match="pressure"):
        adapter.encode_recipe({"temperature": 300.0})


def test_registry_round_trip():
    registry.register_adapter_for_testing(
        "tabular_test", lambda: TabularAdapter.from_spec(PECVD_SPEC)
    )
    try:
        loaded = registry.get_adapter("tabular_test")
        assert isinstance(loaded, TabularAdapter)
        assert loaded.process_id == "pecvd_sin_demo"
    finally:
        registry.clear_test_registry()


def test_entry_point_declared_in_pyproject():
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'tabular = "rig_adapters.tabular.adapter:make_adapter"' in text


# -- parameterized-factory pattern ---------------------------------------------------


def test_factory_with_spec_path_kwarg():
    adapter = make_adapter(spec_path=PECVD_SPEC)
    validate_adapter(adapter)
    assert adapter.process_id == "pecvd_sin_demo"


def test_factory_env_var_fallback(monkeypatch):
    monkeypatch.setenv(SPEC_ENV_VAR, str(PECVD_SPEC))
    adapter = make_adapter()
    assert adapter.process_id == "pecvd_sin_demo"


def test_factory_bare_call_raises_actionable_error(monkeypatch):
    monkeypatch.delenv(SPEC_ENV_VAR, raising=False)
    with pytest.raises(LookupError) as excinfo:
        make_adapter()
    message = str(excinfo.value)
    assert "spec_path" in message
    assert SPEC_ENV_VAR in message
    assert "new-process-onboarding" in message


def test_factory_rejects_unknown_kwargs():
    with pytest.raises(TypeError, match="unexpected"):
        make_adapter(spec_path=PECVD_SPEC, bogus=1)


def test_from_spec_accepts_parsed_dict_and_spec_object():
    spec = parse_spec(base_spec_dict())
    assert isinstance(spec, ProcessSpec)
    assert TabularAdapter.from_spec(spec).process_id == "unit_test_proc"
    assert TabularAdapter.from_spec(base_spec_dict()).process_id == "unit_test_proc"
