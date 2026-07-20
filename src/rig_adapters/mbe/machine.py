"""InSilicoMachine — the E3 stand-in "machine" with injectable pathologies.

Wraps the FAST Arrhenius sim path (never the kMC ``ZoneEnsemble``) as "the
machine" for everything before real data (implementation-plan §10, §15.2, E3). Every
pathology is OFF by default (= clean machine) and independently switchable
via :class:`PathologyConfig`; everything is seeded and deterministic — given
the same config, seed, and run sequence, emitted RunRecords are bit-identical
(tested). This substrate feeds §10's drift/hidden-state tests and the §12.1
non-circular exploitation figure.

Pathologies
-----------

- **tool_id-keyed hidden-parameter perturbation**: each tool applies a fixed
  multiplicative perturbation vector (default +-3%) to hidden physics
  parameters (substrate emissivity, source ``cosine_n``, effective flux). A
  "second chamber" is just ``tool_id="B"``.
- **Seasoning drift**: hidden per-tool state ``runs_since_clean`` multiplies
  the effective flux by ``(1 - drift_rate * runs_since_clean)``; incremented
  every run, reset by :meth:`InSilicoMachine.clean`.
- **First-wafer effect**: an additive offset on declared outputs for the
  first run after ``clean()`` (a freshly cleaned chamber starts "first-wafer",
  including the machine's very first run).
- **Heteroscedastic metrology noise**: ``y_obs = y_true + N(0, a + b*|y|)``
  per output (in declared engineering units), plus optional censoring: an
  out-of-range reading saturates at the range bound and the RunRecord carries
  ``extra["censored"] = {output: "low"|"high"}``. (WP-A follow-up noted in
  the build log: OutcomeRecord has no first-class censored field, so the flag
  lives in RunRecord.extra — core schema untouched.)

Hidden state (runs_since_clean, perturbation vectors) is deliberately NOT
written into the RunRecords — the learning stack must detect it (§10.2).
Use :meth:`InSilicoMachine.state_snapshot` for ground truth in figures.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from rig.schema import Provenance, Quantity, RecipeRecord, RunRecord
from rig_adapters.mbe.adapter import (
    FAST_N_NODES,
    FAST_N_PHI,
    MACHINE_CONFIG_DEFAULTS,
    NOMINAL_COSINE_N,
    NOMINAL_EMISSIVITY,
    PROCESS_ID,
    RECIPE_VARIABLES,
    MBEAdapter,
    evaluate_physics,
    make_adapter,
)
from rig_adapters.mbe.outcomes import (
    OUTPUT_ORDER,
    metrics_to_output_values,
    output_values_to_outcomes,
)

_RECIPE_UNITS = {v.name: v.unit for v in RECIPE_VARIABLES}

# Additive first-wafer offsets, declared engineering units (percent, K).
DEFAULT_FIRST_WAFER_OFFSETS: dict[str, float] = {
    "nonuniformity_pct": 0.5,
    "T_center": -3.0,
}

# Metrology-noise floors `a` per output (sigma = a + b*|y|), declared units.
DEFAULT_NOISE_A: dict[str, float] = {
    "nonuniformity_pct": 0.05,
    "T_center": 0.3,
    "slip_max_ratio": 0.001,
    "bow_cooldown_um": 0.2,
    "thickness_grown": 1e-9,
}

# rng stream domain separators (arbitrary fixed constants).
_STREAM_TOOL = 101
_STREAM_NOISE = 202


@dataclass(frozen=True)
class PathologyConfig:
    """Injectable pathology switches — everything OFF = clean machine (E3)."""

    tool_perturbation: bool = False
    tool_perturbation_scale: float = 0.03  # +-3% on (emissivity, cosine_n, flux)
    seasoning: bool = False
    seasoning_drift_rate: float = 0.004  # flux_eff loss per run since clean
    first_wafer: bool = False
    first_wafer_offsets: Mapping[str, float] | None = None  # None -> defaults
    metrology_noise: bool = False
    noise_a: Mapping[str, float] | None = None  # None -> DEFAULT_NOISE_A
    noise_b: float = 0.002  # relative sigma component (0.2% of |y|)
    censor_ranges: Mapping[str, tuple[float, float]] | None = None


def _stable_tool_hash(tool_id: str) -> int:
    """Process-independent integer hash of a tool_id (NOT builtin hash())."""
    return int.from_bytes(hashlib.sha256(tool_id.encode("utf-8")).digest()[:8], "big")


class InSilicoMachine:
    """The in-silico "machine": fast sim path + seeded, switchable pathologies."""

    def __init__(
        self,
        config: PathologyConfig | None = None,
        seed: int = 0,
        adapter: MBEAdapter | None = None,
        machine_config: Mapping[str, float] | None = None,
        n_nodes: int = FAST_N_NODES,
        n_phi: int = FAST_N_PHI,
        base_timestamp: datetime | None = None,
    ) -> None:
        self.adapter = adapter if adapter is not None else make_adapter()
        self.config = config if config is not None else PathologyConfig()
        self.seed = int(seed)
        # Whole-plot factors: fixed for the machine's lifetime (split-plot).
        self.machine_config = dict(MACHINE_CONFIG_DEFAULTS, **(machine_config or {}))
        self._n_nodes = n_nodes
        self._n_phi = n_phi
        self._base_timestamp = base_timestamp or datetime(2026, 1, 1, tzinfo=UTC)
        self._run_index = 0
        self._runs_since_clean: dict[str, int] = {}

    # -- hidden state ---------------------------------------------------------
    def clean(self, tool_id: str | None = None) -> None:
        """Chamber clean: reset ``runs_since_clean`` (one tool, or all)."""
        if tool_id is None:
            self._runs_since_clean.clear()
        else:
            self._runs_since_clean[tool_id] = 0

    def state_snapshot(self) -> dict[str, Any]:
        """Ground-truth hidden state for figures/tests — never in RunRecords."""
        return {
            "run_index": self._run_index,
            "runs_since_clean": dict(self._runs_since_clean),
        }

    def _tool_factors(self, tool_id: str) -> tuple[float, float, float]:
        """Fixed multiplicative (emissivity, cosine_n, flux) factors per tool."""
        if not self.config.tool_perturbation:
            return (1.0, 1.0, 1.0)
        rng = np.random.default_rng([self.seed, _STREAM_TOOL, _stable_tool_hash(tool_id)])
        s = self.config.tool_perturbation_scale
        f = rng.uniform(1.0 - s, 1.0 + s, size=3)
        return (float(f[0]), float(f[1]), float(f[2]))

    # -- the run --------------------------------------------------------------
    def _coerce_recipe(self, recipe: RecipeRecord | Mapping[str, Any]) -> RecipeRecord:
        if isinstance(recipe, RecipeRecord):
            return recipe
        values = {}
        for name, value in recipe.items():
            if name not in _RECIPE_UNITS:
                raise ValueError(f"unknown recipe variable {name!r}")
            magnitude = getattr(value, "magnitude", value)
            values[name] = Quantity(magnitude=float(magnitude), unit=_RECIPE_UNITS[name])
        return RecipeRecord(values=values)

    def run(self, recipe: RecipeRecord | Mapping[str, Any], tool_id: str = "A") -> RunRecord:
        """Execute one run: validate recipe, apply pathologies, emit a RunRecord.

        Deterministic: same (config, seed, machine_config, run sequence) =>
        bit-identical RunRecords.
        """
        cfg = self.config
        record_recipe = self._coerce_recipe(recipe)
        record_recipe.validate_against(list(self.adapter.input_schema))
        recipe_si = {name: q.magnitude for name, q in record_recipe.values.items()}

        # Hidden physics parameters (E3 pathology surface).
        eps_f, cos_f, flux_f = self._tool_factors(tool_id)
        runs_since_clean = self._runs_since_clean.get(tool_id, 0)
        flux_eff = flux_f
        if cfg.seasoning:
            flux_eff *= max(0.0, 1.0 - cfg.seasoning_drift_rate * runs_since_clean)

        metrics = evaluate_physics(
            recipe_si,
            self.machine_config,
            emissivity=NOMINAL_EMISSIVITY * eps_f,
            cosine_n=NOMINAL_COSINE_N * cos_f,
            flux_eff=flux_eff,
            n_nodes=self._n_nodes,
            n_phi=self._n_phi,
        )
        y = metrics_to_output_values(metrics)

        # First-wafer effect: additive offset on the run right after a clean.
        if cfg.first_wafer and runs_since_clean == 0:
            offsets = (
                cfg.first_wafer_offsets
                if cfg.first_wafer_offsets is not None
                else DEFAULT_FIRST_WAFER_OFFSETS
            )
            for name, off in offsets.items():
                if name in y:
                    y[name] += float(off)

        # Heteroscedastic metrology noise: sigma(y) = a + b*|y| per output,
        # drawn in fixed OUTPUT_ORDER from a per-run-index stream.
        if cfg.metrology_noise:
            rng = np.random.default_rng([self.seed, _STREAM_NOISE, self._run_index])
            a = cfg.noise_a if cfg.noise_a is not None else DEFAULT_NOISE_A
            for name in OUTPUT_ORDER:
                sigma = float(a.get(name, 0.0)) + cfg.noise_b * abs(y[name])
                y[name] += float(rng.normal(0.0, sigma))

        # Optional censoring: readings saturate at the metrology range bound.
        censored: dict[str, str] = {}
        if cfg.censor_ranges:
            for name, (lo, hi) in cfg.censor_ranges.items():
                if name not in y:
                    continue
                if y[name] < lo:
                    y[name] = float(lo)
                    censored[name] = "low"
                elif y[name] > hi:
                    y[name] = float(hi)
                    censored[name] = "high"

        run_index = self._run_index
        extra: dict[str, Any] = {
            "run_index": run_index,
            "machine_config": dict(self.machine_config),
        }
        if censored:
            extra["censored"] = censored

        record = RunRecord(
            run_id=uuid5(NAMESPACE_URL, f"rig://mbe-silico/{self.seed}/{run_index}/{tool_id}"),
            process_id=PROCESS_ID,
            tool_id=tool_id,
            timestamp=self._base_timestamp + timedelta(hours=run_index),
            recipe=record_recipe,
            outcomes=output_values_to_outcomes(y),
            provenance=Provenance(source="physics_sim", operator="in_silico_machine"),
            extra=extra,
        )

        # Advance hidden state AFTER the run (a clean chamber's first run
        # sees runs_since_clean == 0).
        self._runs_since_clean[tool_id] = runs_since_clean + 1
        self._run_index += 1
        return record
