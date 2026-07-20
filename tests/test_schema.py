"""Data-contract tests (implementation-plan §3.5): SI canonicalization, typed values, provenance."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from rig.interfaces import CategoricalVariable, CompositionalVariable, ContinuousVariable
from rig.schema import (
    ArrayRef,
    CategoricalValue,
    Fraction,
    OutcomeRecord,
    Provenance,
    Quantity,
    RecipeRecord,
    RunRecord,
)

# ---------------------------------------------------------------------------
# Unit canonicalization (kills degC-vs-K / sccm-vs-slm silent bugs)
# ---------------------------------------------------------------------------


def test_degc_canonicalized_to_kelvin():
    q = Quantity(magnitude=100.0, unit="degC")
    assert q.unit == "K"
    assert q.magnitude == pytest.approx(373.15)


def test_sccm_canonicalized_to_si_flow():
    q = Quantity(magnitude=10.0, unit="sccm")  # 10 cm^3/min
    assert q.magnitude == pytest.approx(10e-6 / 60.0, rel=1e-9)  # m^3/s
    # slm and sccm land on the SAME canonical unit -> magnitudes comparable
    q2 = Quantity(magnitude=0.01, unit="slm")  # = 10 sccm
    assert q2.unit == q.unit
    assert q2.magnitude == pytest.approx(q.magnitude, rel=1e-9)


def test_unknown_unit_rejected():
    with pytest.raises(ValidationError):
        Quantity(magnitude=1.0, unit="not_a_unit_xyz")


# ---------------------------------------------------------------------------
# Categorical / Fraction typing
# ---------------------------------------------------------------------------


def test_invalid_categorical_rejected():
    with pytest.raises(ValidationError):
        CategoricalValue(value="chamber_Z", levels=("chamber_A", "chamber_B"))


def test_valid_categorical_accepted():
    v = CategoricalValue(value="chamber_A", levels=("chamber_A", "chamber_B"))
    assert v.value == "chamber_A"


def test_fraction_bounds():
    assert Fraction(value=0.25).value == 0.25
    with pytest.raises(ValidationError):
        Fraction(value=1.5)
    with pytest.raises(ValidationError):
        Fraction(value=-0.1)


def test_fraction_is_distinct_from_quantity():
    assert not isinstance(Fraction(value=0.5), Quantity)


# ---------------------------------------------------------------------------
# Provenance literal (headline metrics only on real_tool)
# ---------------------------------------------------------------------------


def test_provenance_literal_enforced():
    assert Provenance(source="physics_sim").source == "physics_sim"
    assert Provenance(source="real_tool").source == "real_tool"
    with pytest.raises(ValidationError):
        Provenance(source="my_notebook")


# ---------------------------------------------------------------------------
# Recipe vs adapter input schema
# ---------------------------------------------------------------------------

SCHEMA = [
    ContinuousVariable(name="substrate_temp", lower=300.0, upper=900.0, unit="degC"),
    ContinuousVariable(name="n2_flow", lower=0.0, upper=100.0, unit="sccm"),
    CategoricalVariable(name="chamber", levels=("A", "B")),
    CompositionalVariable(name="alloy", components=("ga", "al")),
]


def _recipe(**over):
    values = {
        "substrate_temp": Quantity(magnitude=500.0, unit="degC"),
        "n2_flow": Quantity(magnitude=50.0, unit="sccm"),
        "chamber": CategoricalValue(value="A", levels=("A", "B")),
        "alloy.ga": Fraction(value=0.7),
        "alloy.al": Fraction(value=0.3),
    }
    values.update(over)
    return RecipeRecord(values=values)


def test_valid_recipe_passes_adapter_ranges():
    _recipe().validate_against(SCHEMA)  # no raise


def test_out_of_bounds_numeric_rejected():
    bad = _recipe(substrate_temp=Quantity(magnitude=1000.0, unit="degC"))
    with pytest.raises(ValueError, match="outside declared range"):
        bad.validate_against(SCHEMA)


def test_bounds_compared_in_si_across_units():
    # 0.05 slm == 50 sccm -> inside [0, 100] sccm even though units differ
    _recipe(n2_flow=Quantity(magnitude=0.05, unit="slm")).validate_against(SCHEMA)
    # 0.2 slm == 200 sccm -> out of range
    with pytest.raises(ValueError, match="outside declared range"):
        _recipe(n2_flow=Quantity(magnitude=0.2, unit="slm")).validate_against(SCHEMA)


def test_undeclared_variable_rejected():
    with pytest.raises(ValueError, match="not declared"):
        _recipe(mystery_knob=Quantity(magnitude=1.0, unit="V")).validate_against(SCHEMA)


def test_bare_quantity_rejected_for_compositional():
    bad = _recipe(**{"alloy.ga": Quantity(magnitude=0.7, unit="dimensionless")})
    with pytest.raises(ValueError, match="Fraction"):
        bad.validate_against(SCHEMA)


def test_compositional_sum_not_one_rejected():
    # audit B1: simplex components must sum to 1 (implementation-plan §3.1). {ga:0.7, al:0.7}
    # = 1.4 is physically impossible and must not pass the data-contract gate.
    bad = _recipe(**{"alloy.ga": Fraction(value=0.7), "alloy.al": Fraction(value=0.7)})
    with pytest.raises(ValueError, match="sum to 1.4"):
        bad.validate_against(SCHEMA)


def test_compositional_missing_component_rejected():
    # audit B1: an incomplete simplex ({ga} with 'al' omitted) must be rejected,
    # not silently accepted as a valid composition.
    values = {
        "substrate_temp": Quantity(magnitude=500.0, unit="degC"),
        "n2_flow": Quantity(magnitude=50.0, unit="sccm"),
        "chamber": CategoricalValue(value="A", levels=("A", "B")),
        "alloy.ga": Fraction(value=0.7),  # 'alloy.al' deliberately omitted
    }
    with pytest.raises(ValueError, match="missing components"):
        RecipeRecord(values=values).validate_against(SCHEMA)


def test_compositional_sum_to_one_within_tolerance_passes():
    # floating-point drift within 1e-6 of 1.0 is accepted (not brittle).
    _recipe(
        **{"alloy.ga": Fraction(value=0.7000001), "alloy.al": Fraction(value=0.2999999)}
    ).validate_against(SCHEMA)


# ---------------------------------------------------------------------------
# Outcomes + full RunRecord
# ---------------------------------------------------------------------------


def test_outcome_modality_payload_contract():
    OutcomeRecord(
        name="growth_rate",
        modality="scalar_vector",
        value=Quantity(magnitude=1.0, unit="nm/s"),
    )
    OutcomeRecord(
        name="thickness_profile",
        modality="curve_1d",
        value=ArrayRef(hash="abc123", path="data/profiles/abc123.npy"),
    )
    with pytest.raises(ValidationError):
        OutcomeRecord(
            name="thickness_profile",
            modality="curve_1d",
            value=Quantity(magnitude=1.0, unit="nm"),
        )
    with pytest.raises(ValidationError):
        OutcomeRecord(
            name="growth_rate",
            modality="scalar_vector",
            value=ArrayRef(hash="abc123", path="x.npy"),
        )


def test_runrecord_roundtrip():
    rec = RunRecord(
        run_id=uuid4(),
        process_id="mbe_gaas",
        tool_id="chamber_A",
        timestamp=datetime(2026, 7, 15, tzinfo=UTC),
        recipe=_recipe(),
        outcomes=[
            OutcomeRecord(
                name="growth_rate",
                modality="scalar_vector",
                value=Quantity(magnitude=1.0, unit="angstrom/s"),
            )
        ],
        provenance=Provenance(source="physics_sim", git_sha="deadbeef"),
    )
    restored = RunRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    # SI canonicalization survived serialization
    assert restored.recipe.values["substrate_temp"].unit == "K"
