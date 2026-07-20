"""In-silico data generation: Sobol seed design -> InSilicoMachine -> RunRecords.

The seed design spans the RECIPE variables only (implementation-plan §3.1 DoE hook via
``rig.interfaces.sobol_seed_design``); machine-config variables are held at
their defaults — the split-plot whole-plot conditioning (§8.3). Runs are
assigned round-robin across ``tool_ids``.

CLI (JSONL, one Pydantic-serialized RunRecord per line)::

    python -m rig_adapters.mbe.generate --n 64 --out data/mbe_silico_v0.jsonl
    python -m rig_adapters.mbe.generate --n 16 --seed 0 \
        --out tests/fixtures/mbe_silico_smoke.jsonl

Pathology switches: ``--tools A,B --tool-perturbation --seasoning
--first-wafer --noise`` (all off by default = clean machine).
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

from rig.schema import Quantity, RecipeRecord, RunRecord
from rig_adapters.mbe.adapter import RECIPE_VARIABLES, MBEAdapter, make_adapter
from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

_RECIPE_UNITS = {v.name: v.unit for v in RECIPE_VARIABLES}


def generate_dataset(
    n_runs: int,
    tool_ids: Sequence[str] = ("A",),
    pathology_config: PathologyConfig | None = None,
    seed: int = 0,
    adapter: MBEAdapter | None = None,
) -> list[RunRecord]:
    """Generate ``n_runs`` RunRecords from a scrambled-Sobol recipe design.

    Deterministic in (n_runs, tool_ids, pathology_config, seed).
    """
    if not tool_ids:
        raise ValueError("need at least one tool_id")
    adapter = adapter if adapter is not None else make_adapter()
    machine = InSilicoMachine(config=pathology_config, seed=seed, adapter=adapter)
    design = adapter.seed_design(n_runs, seed)
    records = []
    for i, point in enumerate(design):
        recipe = RecipeRecord(
            values={
                name: Quantity(magnitude=float(value), unit=_RECIPE_UNITS[name])
                for name, value in point.items()
            }
        )
        records.append(machine.run(recipe, tool_id=tool_ids[i % len(tool_ids)]))
    return records


def write_jsonl(records: Iterable[RunRecord], path: str | Path) -> Path:
    """Serialize RunRecords to JSONL (one ``model_dump_json`` per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
    return path


def read_jsonl(path: str | Path) -> list[RunRecord]:
    """Load and re-validate RunRecords from a JSONL file."""
    out = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(RunRecord.model_validate_json(line))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m rig_adapters.mbe.generate",
        description="Generate in-silico MBE RunRecords (Sobol design, JSONL).",
    )
    parser.add_argument("--n", type=int, default=64, help="number of runs (default 64)")
    parser.add_argument("--out", required=True, help="output JSONL path")
    parser.add_argument("--seed", type=int, default=0, help="master seed (default 0)")
    parser.add_argument(
        "--tools", default="A", help="comma-separated tool_ids, round-robin (default 'A')"
    )
    parser.add_argument(
        "--tool-perturbation", action="store_true", help="enable per-tool hidden perturbation"
    )
    parser.add_argument("--seasoning", action="store_true", help="enable seasoning drift")
    parser.add_argument("--first-wafer", action="store_true", help="enable first-wafer offset")
    parser.add_argument("--noise", action="store_true", help="enable metrology noise")
    args = parser.parse_args(argv)

    config = PathologyConfig(
        tool_perturbation=args.tool_perturbation,
        seasoning=args.seasoning,
        first_wafer=args.first_wafer,
        metrology_noise=args.noise,
    )
    records = generate_dataset(
        args.n,
        tool_ids=tuple(t.strip() for t in args.tools.split(",") if t.strip()),
        pathology_config=config,
        seed=args.seed,
    )
    path = write_jsonl(records, args.out)
    print(f"wrote {len(records)} RunRecords -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
