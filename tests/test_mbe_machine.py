"""InSilicoMachine tests (E3): determinism, pathology direction, RunRecord
validity (implementation-plan §10, §13.4, §15.6 E3)."""

import statistics

import pytest

from rig.schema import Quantity, RunRecord
from rig_adapters.mbe import simlink
from rig_adapters.mbe.machine import (
    DEFAULT_FIRST_WAFER_OFFSETS,
    InSilicoMachine,
    PathologyConfig,
)
from rig_adapters.mbe.outcomes import OUTPUT_UNITS

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

RECIPE = {"T_heater": 1320.0, "film_thickness": 1e-6}

ALL_ON = PathologyConfig(
    tool_perturbation=True, seasoning=True, first_wafer=True, metrology_noise=True
)


def _values(record: RunRecord) -> dict[str, float]:
    return {o.name: o.value.magnitude for o in record.outcomes}


# -- determinism (implementation-plan §13.4) -------------------------------------------------


@pytest.mark.parametrize("config", [PathologyConfig(), ALL_ON], ids=["clean", "all_on"])
def test_bit_identical_given_same_seed_config_sequence(config):
    def sequence(machine):
        out = [machine.run(RECIPE, "A"), machine.run(RECIPE, "B")]
        machine.clean("A")
        out.append(machine.run(RECIPE, "A"))
        return [r.model_dump_json() for r in out]

    a = sequence(InSilicoMachine(config=config, seed=42))
    b = sequence(InSilicoMachine(config=config, seed=42))
    assert a == b  # bit-identical serialized RunRecords


def test_different_seed_changes_stochastic_outputs():
    cfg = PathologyConfig(metrology_noise=True)
    r1 = InSilicoMachine(config=cfg, seed=1).run(RECIPE)
    r2 = InSilicoMachine(config=cfg, seed=2).run(RECIPE)
    assert _values(r1) != _values(r2)


def test_clean_machine_is_pathology_free_and_reproducible():
    machine = InSilicoMachine(seed=0)
    v1, v2 = _values(machine.run(RECIPE)), _values(machine.run(RECIPE))
    assert v1 == v2  # no noise, no drift, no state
    assert v1["thickness_grown"] == RECIPE["film_thickness"]


def test_different_tool_id_gives_different_outcomes():
    machine = InSilicoMachine(config=PathologyConfig(tool_perturbation=True), seed=0)
    va = _values(machine.run(RECIPE, "A"))
    vb = _values(machine.run(RECIPE, "B"))
    assert va != vb
    # ... and the perturbation is FIXED per tool: A again reproduces A.
    machine2 = InSilicoMachine(config=PathologyConfig(tool_perturbation=True), seed=0)
    assert _values(machine2.run(RECIPE, "A")) == va


# -- pathology direction tests ----------------------------------------------------


def test_seasoning_monotone_drift_and_clean_reset():
    machine = InSilicoMachine(config=PathologyConfig(seasoning=True), seed=0)
    thickness = [_values(machine.run(RECIPE))["thickness_grown"] for _ in range(6)]
    assert all(a > b for a, b in zip(thickness, thickness[1:], strict=False))  # monotone
    machine.clean()
    assert _values(machine.run(RECIPE))["thickness_grown"] == thickness[0]


def test_first_wafer_offset_after_clean():
    machine = InSilicoMachine(config=PathologyConfig(first_wafer=True), seed=0)
    first, second, third = (_values(machine.run(RECIPE)) for _ in range(3))
    assert second == third  # steady state
    for name, offset in DEFAULT_FIRST_WAFER_OFFSETS.items():
        # Offsets are declared-unit values; records are SI-canonical.
        offset_si = Quantity(magnitude=offset, unit=OUTPUT_UNITS[name]).magnitude
        assert first[name] == pytest.approx(second[name] + offset_si)
    machine.clean()
    assert _values(machine.run(RECIPE)) == first  # clean() re-arms the effect


def test_metrology_noise_increases_replicate_scatter():
    def scatter(config, seed=3, n=8):
        machine = InSilicoMachine(config=config, seed=seed)
        vals = [_values(machine.run(RECIPE))["T_center"] for _ in range(n)]
        return statistics.pstdev(vals)

    assert scatter(PathologyConfig()) == 0.0
    assert scatter(PathologyConfig(metrology_noise=True)) > 0.0


def test_censoring_flags_and_saturates():
    cfg = PathologyConfig(censor_ranges={"T_center": (0.0, 900.0)})
    record = InSilicoMachine(config=cfg, seed=0).run(RECIPE)
    assert record.extra["censored"] == {"T_center": "high"}
    assert _values(record)["T_center"] == 900.0
    # In-range outputs carry no censored flag.
    cfg2 = PathologyConfig(censor_ranges={"T_center": (0.0, 5000.0)})
    assert "censored" not in InSilicoMachine(config=cfg2, seed=0).run(RECIPE).extra


# -- RunRecord validity -------------------------------------------------------------


def test_run_record_contract():
    machine = InSilicoMachine(seed=5)
    r0 = machine.run(RECIPE, tool_id="chamber_X")
    r1 = machine.run(RECIPE, tool_id="chamber_X")
    for record in (r0, r1):
        assert record.process_id == "mbe"
        assert record.tool_id == "chamber_X"
        assert record.provenance.source == "physics_sim"
        assert {o.name for o in record.outcomes} == {
            "nonuniformity_pct",
            "T_center",
            "slip_max_ratio",
            "bow_cooldown_um",
            "thickness_grown",
        }
    assert r0.run_id != r1.run_id
    assert r0.timestamp < r1.timestamp
    assert r0.extra["run_index"] == 0 and r1.extra["run_index"] == 1
    # JSON round trip re-validates.
    assert RunRecord.model_validate_json(r0.model_dump_json()) == r0


def test_out_of_bounds_recipe_rejected_by_machine():
    with pytest.raises(ValueError, match="T_heater"):
        InSilicoMachine(seed=0).run({"T_heater": 900.0, "film_thickness": 1e-6})


def test_state_snapshot_tracks_hidden_state_without_leaking():
    machine = InSilicoMachine(config=PathologyConfig(seasoning=True), seed=0)
    record = machine.run(RECIPE, "A")
    machine.run(RECIPE, "B")
    snap = machine.state_snapshot()
    assert snap == {"run_index": 2, "runs_since_clean": {"A": 1, "B": 1}}
    assert "runs_since_clean" not in record.extra  # hidden state stays hidden
    assert record.provenance.calibration_state is None
