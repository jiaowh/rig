"""WP-F in-silico integration: cost-to-target on the WP-B MBE machine (implementation-plan §9,
validated per §15.2 — machinery proof, not a real-data headline; gated on M0).

Runs the closed active-learning loop against the in-silico MBE machine for a
reachable ``thickness_grown`` spec and checks it reaches the target within
budget, producing a finite cost-to-target the WP-G survival stats can consume.

Sim-gated; may import rig_adapters (same exemption as WP-B/D/G tests).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rig.active import ActiveLearningLoop
from rig.forward import records_to_arrays
from rig_adapters.mbe import simlink

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

INPUT_KEYS = ["T_heater", "film_thickness"]
OUTPUT_KEYS = ["thickness_grown"]


def test_loop_reaches_thickness_spec_in_silico():
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    adapter = make_adapter()
    sim = InSilicoMachine(config=PathologyConfig(), seed=0, adapter=adapter)

    # calibrate a reachable mid-range target from a quick Sobol probe.
    probe = [sim.run(p, tool_id="A") for p in adapter.seed_design(16, 7)]
    _, Yp = records_to_arrays(probe, INPUT_KEYS, OUTPUT_KEYS)
    lo, hi = float(np.min(Yp)), float(np.max(Yp))
    target = 0.5 * (lo + hi)
    tol = 0.15 * (hi - lo)

    def machine(recipe):
        rec = sim.run(recipe, tool_id="A")
        _, y = records_to_arrays([rec], INPUT_KEYS, OUTPUT_KEYS)
        return y[0]

    def in_spec(y):
        return abs(float(y[0]) - target) <= tol

    loop = ActiveLearningLoop(
        machine=machine,
        in_spec=in_spec,
        variables=list(RECIPE_VARIABLES),
        input_keys=INPUT_KEYS,
        output_keys=OUTPUT_KEYS,
        spec={"targets": {"thickness_grown": {"target": target, "tol": tol}}},
        cost_recipe=lambda r: 1000.0,
        c_batch=1000.0,  # Kanarik cost model
        budget=32,
        q=4,
        n_seed=8,
        n_pool=96,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    traj = loop.run()
    assert traj.hit is True, traj.stop_reason
    assert math.isfinite(traj.cost_to_target)
    assert traj.n_queries <= 32
