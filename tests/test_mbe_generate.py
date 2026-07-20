"""Data generation tests: Sobol design over recipe vars, split-plot
conditioning, JSONL round trip, CLI, and the checked-in smoke fixture."""

from pathlib import Path

import pytest

from rig_adapters.mbe import simlink
from rig_adapters.mbe.adapter import (
    MACHINE_CONFIG_DEFAULTS,
    RECIPE_VARIABLE_NAMES,
    make_adapter,
)
from rig_adapters.mbe.generate import generate_dataset, main, read_jsonl, write_jsonl
from rig_adapters.mbe.machine import PathologyConfig

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mbe_silico_smoke.jsonl"


def test_generate_dataset_shape_and_tools():
    records = generate_dataset(8, tool_ids=("A", "B"), seed=0)
    assert len(records) == 8
    assert [r.tool_id for r in records] == ["A", "B"] * 4
    adapter = make_adapter()
    ranges = adapter.expert_ranges
    for record in records:
        assert set(record.recipe.values) == set(RECIPE_VARIABLE_NAMES)
        record.recipe.validate_against(list(adapter.input_schema))
        for name, q in record.recipe.values.items():
            lo, hi = ranges[name]
            assert lo <= q.magnitude <= hi
        # Split-plot conditioning: machine config held at defaults.
        assert record.extra["machine_config"] == MACHINE_CONFIG_DEFAULTS


def test_generate_dataset_deterministic():
    kwargs = dict(
        tool_ids=("A", "B"),
        pathology_config=PathologyConfig(seasoning=True, metrology_noise=True),
        seed=11,
    )
    a = generate_dataset(4, **kwargs)
    b = generate_dataset(4, **kwargs)
    assert [r.model_dump_json() for r in a] == [r.model_dump_json() for r in b]


def test_jsonl_round_trip(tmp_path):
    records = generate_dataset(2, seed=2)
    path = write_jsonl(records, tmp_path / "out.jsonl")
    assert read_jsonl(path) == records


def test_cli_smoke(tmp_path, capsys):
    out = tmp_path / "cli.jsonl"
    assert main(["--n", "4", "--out", str(out), "--seed", "1", "--tools", "A,B"]) == 0
    assert "wrote 4 RunRecords" in capsys.readouterr().out
    records = read_jsonl(out)
    assert len(records) == 4
    assert {r.tool_id for r in records} == {"A", "B"}


def test_smoke_fixture_loads_and_validates():
    """The checked-in fixture (16 clean-machine runs, seed 0) stays loadable."""
    records = read_jsonl(FIXTURE)
    assert len(records) == 16
    adapter = make_adapter()
    for record in records:
        assert record.process_id == "mbe"
        assert record.provenance.source == "physics_sim"
        record.recipe.validate_against(list(adapter.input_schema))
        assert {o.name for o in record.outcomes} == {o.name for o in adapter.output_schema}


def test_smoke_fixture_is_regenerable():
    """Fixture == generate_dataset(16, seed=0) on a clean machine (bit-identical)."""
    expected = [r.model_dump_json() for r in generate_dataset(16, tool_ids=("A",), seed=0)]
    on_disk = [r.model_dump_json() for r in read_jsonl(FIXTURE)]
    assert on_disk == expected
