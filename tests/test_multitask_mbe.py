"""WP-I in-silico MBE integration: tool-aware surrogate on the WP-B pathology
machine (implementation-plan §10.4 level (a), validated per §15.2 — machinery proof, not a
real-data headline; those stay gated on M0).

Tool A (n=32) and tool B (n=8 train + 8 held-out) come from InSilicoMachine
with tool_perturbation ON — the per-tool hidden (emissivity, cosine_n,
flux_eff) perturbation. `thickness_grown` is the perturbation-sensitive KPI
(WP-B handoff: a pure flux-scale change is invisible in the normalized
uniformity outputs), so that is where a tool-blind pool must lose.

May import rig_adapters (same exemption as WP-B's tests); skips when the
sibling sim repo is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from rig.forward import GPForwardModel, MultiToolGPForwardModel, records_to_arrays_with_tools
from rig_adapters.mbe import simlink

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

INPUT_KEYS = ["T_heater", "film_thickness"]
OUTPUT_KEYS = ["thickness_grown"]
SEED = 0


@pytest.fixture(scope="module")
def mbe_two_tool_data():
    """One machine (seed discipline as WP-B: everything deterministic in the
    (config, seed, sequence) triple), Sobol recipe designs per slice."""
    from rig_adapters.mbe.adapter import make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    adapter = make_adapter()
    machine = InSilicoMachine(
        config=PathologyConfig(tool_perturbation=True), seed=SEED, adapter=adapter
    )

    def runs(n: int, tool_id: str, design_seed: int):
        return [
            machine.run(point, tool_id=tool_id) for point in adapter.seed_design(n, design_seed)
        ]

    rec_a = runs(32, "A", design_seed=0)
    rec_b_train = runs(8, "B", design_seed=1)
    rec_b_test = runs(8, "B", design_seed=2)
    return rec_a, rec_b_train, rec_b_test


def test_multitool_beats_pooled_blind_on_perturbed_chamber(mbe_two_tool_data):
    rec_a, rec_b_train, rec_b_test = mbe_two_tool_data
    X, Y, tools = records_to_arrays_with_tools(rec_a + rec_b_train, INPUT_KEYS, OUTPUT_KEYS)
    X_test, Y_test, _ = records_to_arrays_with_tools(rec_b_test, INPUT_KEYS, OUTPUT_KEYS)

    multi = MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(X, Y, tools)
    pooled = GPForwardModel(n_restarts=2, seed=0).fit(X, Y)

    mu_multi = np.asarray(multi.predict(X_test, tool_id="B").mean)
    mu_pooled = np.asarray(pooled.predict(X_test).mean)
    rmse_multi = float(np.sqrt(np.mean((mu_multi - Y_test) ** 2)))
    rmse_pooled = float(np.sqrt(np.mean((mu_pooled - Y_test) ** 2)))
    assert rmse_multi < rmse_pooled, (rmse_multi, rmse_pooled)

    # §5.8 leave-one-tool-out epistemic check on the same fitted model: a
    # never-trained chamber must carry more epistemic than the fitted ones.
    X_a, _, _ = records_to_arrays_with_tools(rec_a, INPUT_KEYS, OUTPUT_KEYS)
    epi_unknown = float(np.mean(multi.predict(X_test, tool_id="C").epistemic_sigma))
    epi_known_a = float(np.mean(multi.predict(X_a, tool_id="A").epistemic_sigma))
    epi_known_b = float(np.mean(multi.predict(X_test, tool_id="B").epistemic_sigma))
    assert epi_unknown > epi_known_a
    assert epi_unknown > epi_known_b
