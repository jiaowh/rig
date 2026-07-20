"""MBE ProcessAdapter (Phase 0, implementation-plan §3.1, §15.2) — resolves E2.

The recipe-vs-config split (E2: "no unified recipe vector")
============================================================

The sim's ``optimize.DEFAULT_KNOBS`` mixes two kinds of quantity behind one
knob dict ("All knobs are machine-settable quantities"). This adapter is where
that ambiguity is resolved, once, for everything downstream:

**RECIPE variables** (per-run settable, :attr:`ChangeCost.EASY` — the vector
the inverse solver searches over):

- ``T_heater`` [K] — heater setpoint; bounds straight from
  ``optimize.DEFAULT_KNOBS``.
- ``film_thickness`` [m] — target film thickness for the run (the growth-time
  proxy: thickness = rate x time at nominal flux). The sim genuinely consumes
  it (``UniformityProblem(film_thickness=...)`` — it drives the cooldown-bow
  metric), and the in-silico machine's flux pathologies act on the *achieved*
  thickness, so it is the flux-scale-sensitive KPI channel.

**MACHINE-CONFIG variables** (:attr:`ChangeCost.HARD_TO_CHANGE` — split-plot
whole-plot factors, conditioning-not-free for the inverse per implementation-plan §8.3):
``heater_radius``, ``gap``, ``source_offset``, ``source_height``,
``aim_offset`` — chamber build/geometry, changed with a wrench, not a recipe
editor. Bounds from ``optimize.DEFAULT_KNOBS`` (mirrored in
:data:`MACHINE_CONFIG_BOUNDS`; a test asserts the mirror stays in sync with
the sim). ``seed_design`` deliberately spans ONLY the recipe variables:
machine config is held at :data:`MACHINE_CONFIG_DEFAULTS` (split-plot
conditioning), and campaigns that vary whole-plot factors must schedule them
explicitly across lots.

Output schema (modality ``scalar_vector``), extracted from
``UniformityProblem.evaluate()`` — see :mod:`rig_adapters.mbe.outcomes` for
the exact metric -> OutcomeRecord mapping:

- ``nonuniformity_pct`` — the sim's combined flux x incorporation
  nonuniformity [percent, canonicalized to a dimensionless fraction].
- ``T_center`` [K] — wafer centre temperature.
- ``slip_max_ratio`` [-] — the sim's slip criterion (tau/CRSS; >= 1 slips).
- ``bow_cooldown_um`` [um] — ex-situ cooldown bow (film-thickness sensitive).
- ``thickness_grown`` [m] — achieved thickness (= target on a clean machine;
  the channel where flux drift/seasoning becomes observable).

Cost model: Kanarik defaults (implementation-plan §11.1) — c_batch = $1000,
c_recipe = $1000/run, batch_size = 4.

D7 honesty: ``physics_plugin`` is the fast Arrhenius-path evaluator;
``independent_verifier`` is ``None`` — the reduced-order path shares physics
lineage with the kMC, so neither may pose as the independent verifier. The
different-physics ROM verifier is a future work package.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from rig.interfaces import (
    ChangeCost,
    ContinuousVariable,
    CostModel,
    OutputSpec,
    VariableSpec,
    sobol_seed_design,
)
from rig_adapters.mbe import simlink

PROCESS_ID = "mbe"

# Fast-path evaluation resolution (implementation-plan §15.6 E3: tests/datagen must be
# cheap — UniformityProblem's display default is 40-120 nodes).
FAST_N_NODES = 24
FAST_N_PHI = 24

# --- RECIPE variables (per-run settable, EASY) ------------------------------
# T_heater bounds mirror optimize.DEFAULT_KNOBS (asserted in tests).
RECIPE_VARIABLES: tuple[ContinuousVariable, ...] = (
    ContinuousVariable("T_heater", 1150.0, 1500.0, unit="K", change_cost=ChangeCost.EASY),
    ContinuousVariable("film_thickness", 2e-7, 5e-6, unit="m", change_cost=ChangeCost.EASY),
)

# --- MACHINE-CONFIG variables (whole-plot, HARD_TO_CHANGE) -------------------
# (default, lower, upper) mirroring optimize.DEFAULT_KNOBS; units all metres.
MACHINE_CONFIG_BOUNDS: dict[str, tuple[float, float, float]] = {
    "heater_radius": (0.018, 0.010, 0.045),
    "gap": (0.012, 0.006, 0.030),
    "source_offset": (0.10, 0.02, 0.25),
    "source_height": (0.20, 0.08, 0.40),
    "aim_offset": (0.0, -0.10, 0.10),
}

MACHINE_CONFIG_VARIABLES: tuple[ContinuousVariable, ...] = tuple(
    ContinuousVariable(name, lo, hi, unit="m", change_cost=ChangeCost.HARD_TO_CHANGE)
    for name, (_default, lo, hi) in MACHINE_CONFIG_BOUNDS.items()
)

MACHINE_CONFIG_DEFAULTS: dict[str, float] = {
    name: default for name, (default, _lo, _hi) in MACHINE_CONFIG_BOUNDS.items()
}

RECIPE_VARIABLE_NAMES: tuple[str, ...] = tuple(v.name for v in RECIPE_VARIABLES)

OUTPUT_SPECS: tuple[OutputSpec, ...] = (
    OutputSpec("nonuniformity_pct", "scalar_vector", unit="percent", target=0.0),
    OutputSpec("T_center", "scalar_vector", unit="K"),
    OutputSpec("slip_max_ratio", "scalar_vector", unit="dimensionless", upper_spec=1.0),
    OutputSpec("bow_cooldown_um", "scalar_vector", unit="micrometer"),
    OutputSpec("thickness_grown", "scalar_vector", unit="m"),
)

# Nominal substrate card for the fast path (Si-like: k, CTE, 500 um, ~180 GPa
# biaxial). The in-silico machine perturbs `emissivity` per tool_id (E3).
NOMINAL_EMISSIVITY = 0.7
NOMINAL_COSINE_N = 1.0
_NOMINAL_CARD_THERMAL = {"k": 80.0, "alpha": 2.6e-6}
_NOMINAL_CARD_MECHANICAL = {"thickness": 5e-4, "biaxial_GPa": 180.0}
MATERIAL = "GaN"


def evaluate_physics(
    recipe: Mapping[str, float],
    machine_config: Mapping[str, float] | None = None,
    *,
    emissivity: float = NOMINAL_EMISSIVITY,
    cosine_n: float = NOMINAL_COSINE_N,
    flux_eff: float = 1.0,
    n_nodes: int = FAST_N_NODES,
    n_phi: int = FAST_N_PHI,
) -> dict[str, float]:
    """One deterministic fast-Arrhenius-path evaluation (NEVER the kMC).

    ``recipe`` maps recipe-variable names to SI magnitudes (K, m).
    ``machine_config`` overrides :data:`MACHINE_CONFIG_DEFAULTS` (whole-plot
    factors). The keyword-only hidden parameters (``emissivity``,
    ``cosine_n``, ``flux_eff``) are the E3 pathology injection surface — the
    adapter itself always calls this with nominal values.

    Returns the sim's ``UniformityProblem.evaluate()`` metric dict plus
    ``thickness_grown_m`` (= target thickness x ``flux_eff``: a seasoned/
    depleted source deposits a thinner film in the same growth time).
    """
    sim = simlink.load_mbe_sim()
    missing = [k for k in RECIPE_VARIABLE_NAMES if k not in recipe]
    if missing:
        raise ValueError(f"recipe is missing required variables: {missing}")
    knobs = dict(MACHINE_CONFIG_DEFAULTS)
    knobs.update(machine_config or {})
    knobs["T_heater"] = float(recipe["T_heater"])
    thickness_grown = float(recipe["film_thickness"]) * float(flux_eff)
    card = {
        "thermal": dict(_NOMINAL_CARD_THERMAL, emissivity=float(emissivity)),
        "mechanical": dict(_NOMINAL_CARD_MECHANICAL),
    }
    problem = sim.optimize.UniformityProblem(
        sim.cards.load_material(MATERIAL),
        substrate_card=card,
        n_nodes=n_nodes,
        n_phi=n_phi,
        cosine_n=float(cosine_n),
        film_thickness=thickness_grown,
    )
    metrics = dict(problem.evaluate(knobs))
    metrics["thickness_grown_m"] = thickness_grown
    return metrics


class _PhysicsPlugin:
    """The optional f_physics(x) prior (implementation-plan §3.1): clean-machine fast path.

    A distinct object from any future verifier by construction (D7)."""

    def __init__(self, n_nodes: int, n_phi: int) -> None:
        self._n_nodes = n_nodes
        self._n_phi = n_phi

    def __call__(self, recipe: Mapping[str, float]) -> dict[str, float]:
        from rig_adapters.mbe.outcomes import metrics_to_output_values

        metrics = evaluate_physics(recipe, n_nodes=self._n_nodes, n_phi=self._n_phi)
        return metrics_to_output_values(metrics)


class MBEAdapter:
    """ProcessAdapter for the MBE fast-path sim (structural Protocol conform)."""

    def __init__(self, n_nodes: int = FAST_N_NODES, n_phi: int = FAST_N_PHI) -> None:
        self._n_nodes = n_nodes
        self._n_phi = n_phi
        self._physics_plugin = _PhysicsPlugin(n_nodes, n_phi)

    # -- identity -----------------------------------------------------------
    @property
    def process_id(self) -> str:
        return PROCESS_ID

    # -- input / output schema ---------------------------------------------
    @property
    def input_schema(self) -> Sequence[VariableSpec]:
        return RECIPE_VARIABLES + MACHINE_CONFIG_VARIABLES

    @property
    def output_schema(self) -> Sequence[OutputSpec]:
        return OUTPUT_SPECS

    # -- cost model (§11.1 Kanarik defaults) ---------------------------------
    @property
    def cost_model(self) -> CostModel:
        return CostModel(c_batch=1000.0, c_recipe=lambda _recipe: 1000.0, batch_size=4)

    # -- DoE / warm-start hooks ----------------------------------------------
    @property
    def expert_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Expert-constrained ranges over the RECIPE variables only.

        Machine config is split-plot conditioning (held at defaults), never
        part of the space-filling seed design.
        """
        return {v.name: (v.lower, v.upper) for v in RECIPE_VARIABLES}

    def seed_design(self, n_runs: int, seed: int) -> list[dict[str, Any]]:
        return sobol_seed_design(self.expert_ranges, n_runs, seed)

    # -- physics plug-in + independent verifier (D7) --------------------------
    @property
    def physics_plugin(self) -> _PhysicsPlugin:
        return self._physics_plugin

    @property
    def independent_verifier(self) -> None:
        """None: the honest D7 state — the fast path shares physics lineage
        with the kMC, so no in-repo path qualifies as independent. The
        different-physics ROM verifier is a future work package."""
        return None

    # -- encoders -------------------------------------------------------------
    def encode_recipe(self, recipe: Mapping[str, Any]) -> np.ndarray:
        """Recipe dict -> float vector in :data:`RECIPE_VARIABLE_NAMES` order.

        Accepts SI floats, rig ``Quantity`` models, or pint quantities; the
        declared units (K, m) are already SI base units, so magnitudes pass
        through unchanged.
        """
        out = []
        for name in RECIPE_VARIABLE_NAMES:
            if name not in recipe:
                raise ValueError(f"recipe is missing required variable {name!r}")
            value = recipe[name]
            magnitude = getattr(value, "magnitude", value)
            out.append(float(magnitude))
        return np.asarray(out, dtype=float)

    def decode_recipe(self, x: np.ndarray) -> dict[str, Any]:
        """Float vector -> recipe dict of SI magnitudes (inverse of encode)."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != len(RECIPE_VARIABLE_NAMES):
            raise ValueError(
                f"expected {len(RECIPE_VARIABLE_NAMES)} recipe components, got {x.shape[0]}"
            )
        return {name: float(v) for name, v in zip(RECIPE_VARIABLE_NAMES, x, strict=True)}


def make_adapter(**kwargs: Any) -> MBEAdapter:
    """Entry-point factory (``rig.adapters`` group, name ``mbe``)."""
    return MBEAdapter(**kwargs)
