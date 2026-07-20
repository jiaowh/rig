"""Generic tabular ProcessAdapter, built entirely from a declarative spec (WP-H).

Conforms structurally to :class:`rig.interfaces.ProcessAdapter` (passes
``validate_adapter``). D7 honesty: a config-driven adapter has no physics —
``physics_plugin`` and ``independent_verifier`` are both ``None``.

The parameterized-factory pattern (documented for future config-driven adapters)
==============================================================================

Entry points in the ``rig.adapters`` group must resolve to a zero-arg /
kwargs-only factory, but a config-driven adapter is meaningless without its
config. The convention established here:

1. The factory takes its config as an OPTIONAL kwarg (``spec_path=None``).
2. When the kwarg is absent, the factory falls back to a documented
   environment variable (here ``RIG_TABULAR_SPEC``).
3. When neither is provided (e.g. blind ``registry.get_adapter("tabular")``),
   the factory raises a helpful, actionable error — it never silently returns
   a default process nobody declared.

So the three supported call paths are::

    adapter = TabularAdapter.from_spec("examples/pecvd_example.toml")   # direct
    adapter = registry.get_adapter("tabular", spec_path="my_spec.toml")  # kwargs
    $env:RIG_TABULAR_SPEC = "my_spec.toml"; registry.get_adapter("tabular")

Future config-parameterized adapters should copy this pattern
(kwarg -> ``RIG_<NAME>_<PARAM>`` env var -> actionable error).
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rig.interfaces import (
    CompositionalVariable,
    ContinuousVariable,
    CostModel,
    OutputSpec,
    VariableSpec,
    sobol_seed_design,
)
from rig_adapters.tabular.spec import ProcessSpec, load_spec, parse_spec

SPEC_ENV_VAR = "RIG_TABULAR_SPEC"


class TabularAdapter:
    """ProcessAdapter for an arbitrary tabular process described by a spec."""

    def __init__(self, spec: ProcessSpec) -> None:
        self._spec = spec

    @classmethod
    def from_spec(cls, spec: ProcessSpec | Mapping[str, Any] | str | Path) -> TabularAdapter:
        """Build from a ProcessSpec, a spec file path (.toml/.json), or a
        parsed spec dict."""
        if isinstance(spec, ProcessSpec):
            return cls(spec)
        if isinstance(spec, (str, Path)):
            return cls(load_spec(spec))
        return cls(parse_spec(spec))

    @property
    def spec(self) -> ProcessSpec:
        return self._spec

    # -- identity -----------------------------------------------------------
    @property
    def process_id(self) -> str:
        return self._spec.process_id

    # -- input / output schema ------------------------------------------------
    @property
    def input_schema(self) -> Sequence[VariableSpec]:
        return self._spec.variables

    @property
    def output_schema(self) -> Sequence[OutputSpec]:
        return self._spec.outputs

    # -- cost model (implementation-plan §3.1, §11) -----------------------------------------
    @property
    def cost_model(self) -> CostModel:
        c_recipe = self._spec.c_recipe
        return CostModel(
            c_batch=self._spec.c_batch,
            c_recipe=lambda _recipe: c_recipe,
            batch_size=self._spec.batch_size,
        )

    # -- DoE / warm-start hooks --------------------------------------------------
    @property
    def expert_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Box ranges over the NUMERIC inputs, in declared units.

        Continuous variables use their declared bounds; compositional
        components get the raw [0, 1] box (the seed design renormalizes each
        composition onto the simplex). Categoricals are conditioning
        (implementation-plan §8.3), never part of the space-filling design.
        """
        ranges: dict[str, tuple[float, float]] = {}
        for var in self._spec.variables:
            if isinstance(var, ContinuousVariable):
                ranges[var.name] = (var.lower, var.upper)
            elif isinstance(var, CompositionalVariable):
                for comp in var.components:
                    ranges[f"{var.name}.{comp}"] = (0.0, 1.0)
        return ranges

    def seed_design(self, n_runs: int, seed: int) -> list[dict[str, Any]]:
        """Scrambled-Sobol seed design over the numeric inputs (declared units).

        Compositional groups are renormalized to sum to 1, so every emitted
        seed is FEASIBLE by construction (implementation-plan §15.6 E5: DoE hooks must
        emit feasible seeds).
        """
        design = sobol_seed_design(self.expert_ranges, n_runs, seed)
        for point in design:
            for var in self._spec.compositional:
                keys = [f"{var.name}.{comp}" for comp in var.components]
                total = sum(point[k] for k in keys)
                if total <= 0.0:  # pragma: no cover - scrambled Sobol never hits 0
                    for k in keys:
                        point[k] = 1.0 / len(keys)
                else:
                    for k in keys:
                        point[k] = point[k] / total
        return design

    # -- physics plug-in + independent verifier (D7) ------------------------------
    @property
    def physics_plugin(self) -> None:
        """None: a config-driven tabular adapter carries no physics prior."""
        return None

    @property
    def independent_verifier(self) -> None:
        """None: no verifier ships with a bare spec (honest D7 default)."""
        return None

    # -- encoders -------------------------------------------------------------
    def encode_recipe(self, recipe: Mapping[str, Any]) -> np.ndarray:
        """Recipe dict -> float vector over ``spec.numeric_input_names`` order.

        Accepts plain floats, rig ``Quantity``/``Fraction`` models, or pint
        quantities. Magnitudes are taken AS STORED (SI if they came from a
        validated RunRecord). Categoricals are conditioning (implementation-plan §8.3) and
        are not encoded; extra keys are ignored.
        """
        out = []
        for name in self._spec.numeric_input_names:
            if name not in recipe:
                raise ValueError(f"recipe is missing required variable {name!r}")
            value = recipe[name]
            magnitude = getattr(value, "magnitude", None)
            if magnitude is None:
                magnitude = getattr(value, "value", value)
            out.append(float(magnitude))
        return np.asarray(out, dtype=float)

    def decode_recipe(self, x: np.ndarray) -> dict[str, Any]:
        """Float vector -> recipe dict of magnitudes (inverse of encode)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        names = self._spec.numeric_input_names
        if x.shape[0] != len(names):
            raise ValueError(f"expected {len(names)} recipe components, got {x.shape[0]}")
        return {name: float(v) for name, v in zip(names, x, strict=True)}


def make_adapter(spec_path: str | Path | None = None, **kwargs: Any) -> TabularAdapter:
    """Entry-point factory (``rig.adapters`` group, name ``tabular``).

    Resolution order: ``spec_path`` kwarg -> ``RIG_TABULAR_SPEC`` env var ->
    actionable error (see the module docstring for the pattern).
    """
    if kwargs:
        raise TypeError(f"make_adapter got unexpected kwargs: {sorted(kwargs)}")
    if spec_path is None:
        spec_path = os.environ.get(SPEC_ENV_VAR) or None
    if spec_path is None:
        raise LookupError(
            "the generic 'tabular' adapter needs a process spec and none was "
            "provided. Either pass it explicitly — "
            "registry.get_adapter('tabular', spec_path='my_process.toml') or "
            "TabularAdapter.from_spec('my_process.toml') — or set the "
            f"{SPEC_ENV_VAR} environment variable to the spec path. "
            "See docs/new-process-onboarding.md for how to write a spec."
        )
    return TabularAdapter.from_spec(spec_path)
