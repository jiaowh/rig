"""WP-G in-silico integration: the §12.1 surrogate-exploitation stress test on
the WP-B pathology machine (validated per §15.2 — machinery proof, not a
real-data headline; those stay gated on M0).

The non-circular test: fit the surrogate on the CLEAN machine, let the WP-D
inverse propose recipes, then run those proposals on a PATHOLOGY-perturbed tool
whose structure the surrogate never saw. The optimism gap and interval-violation
rate must be worse on the OOD tool than on the in-distribution clean tool — that
is the exploitation the metric is built to expose (§12.1).

Sim-gated; may import rig_adapters (same exemption as WP-B/D tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from rig.eval.exploitation import exploitation_stress_test
from rig.forward import GPForwardModel, records_to_arrays
from rig.interfaces import Infeasible
from rig.inverse import PessimisticInverseSolver
from rig_adapters.mbe import simlink

pytestmark = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

INPUT_KEYS = ["T_heater", "film_thickness"]
OUTPUT_KEYS = ["thickness_grown"]
SEED = 0


def test_exploitation_gap_worse_on_ood_tool():
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    adapter = make_adapter()
    clean = InSilicoMachine(config=PathologyConfig(), seed=SEED, adapter=adapter)
    # surrogate trained ONLY on the clean tool A.
    train = [clean.run(p, tool_id="A") for p in adapter.seed_design(40, 0)]
    X, Y = records_to_arrays(train, INPUT_KEYS, OUTPUT_KEYS)
    gp = GPForwardModel(n_restarts=3, seed=0, max_iter=100).fit(X, Y)

    lo, hi = float(np.min(Y)), float(np.max(Y))
    target = 0.5 * (lo + hi)
    solver = PessimisticInverseSolver(
        gp,
        list(RECIPE_VARIABLES),
        OUTPUT_KEYS,
        X_train=X,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    res = solver.solve({"targets": {"thickness_grown": {"target": target, "tol": 0.2 * (hi - lo)}}})
    assert not isinstance(res, Infeasible) and res

    proposals = [np.array([c.recipe[k] for k in INPUT_KEYS]) for c in res]
    Xp = np.vstack(proposals)

    # run the SAME proposals on the clean tool (in-distribution) and on a
    # pathology-perturbed tool (OOD structure the surrogate never trained on).
    ood = InSilicoMachine(
        config=PathologyConfig(tool_perturbation=True), seed=SEED, adapter=adapter
    )

    def realized(machine, tool_id):
        recs = [machine.run(c.recipe, tool_id=tool_id) for c in res]
        _, Yr = records_to_arrays(recs, INPUT_KEYS, OUTPUT_KEYS)
        return Yr

    Y_clean = realized(clean, "A")
    Y_ood = realized(ood, "B_perturbed")

    def score(y):  # closeness to target thickness (higher = better)
        return -abs(float(np.asarray(y)[0]) - target)

    rep_clean = exploitation_stress_test(gp, Xp, Y_clean, score_fn=score, alpha=0.1)
    rep_ood = exploitation_stress_test(gp, Xp, Y_ood, score_fn=score, alpha=0.1)

    # the surrogate is more wrong on the perturbed tool: larger optimism-gap
    # magnitude AND at least as many interval violations.
    assert abs(rep_ood.optimism_gap) > abs(rep_clean.optimism_gap)
    assert rep_ood.interval_violation_fraction >= rep_clean.interval_violation_fraction
