"""WP-D in-silico MBE integration: pessimistic inverse on the WP-B machine
(implementation-plan §8, validated per §15.2 — machinery proof, not a real-data headline;
those stay gated on M0).

End-to-end claim exercised here: fit the GP forward surrogate on in-silico MBE
runs, invert a REACHABLE ``thickness_grown`` spec, then run the machine at the
recommended recipe and confirm the achieved outcome lands inside the
solver's pessimistic (worst-case credited) interval — i.e. the pessimism is
honest, not decorative. An UNREACHABLE spec must return an explicit
:class:`Infeasible`.

May import rig_adapters (same exemption as WP-B/WP-I tests); skips when the
sibling sim repo is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.fixture(scope="module")
def mbe_gp_and_machine():
    """Clean (pathology-free) in-silico machine + a GP fit on 40 Sobol runs.

    Pathologies OFF so the surrogate ≈ machine and the coverage claim is a
    clean check of the interval, not of drift handling (that is WP-F)."""
    from rig_adapters.mbe.adapter import RECIPE_VARIABLES, make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    adapter = make_adapter()
    machine = InSilicoMachine(config=PathologyConfig(), seed=SEED, adapter=adapter)
    train = [machine.run(p, tool_id="A") for p in adapter.seed_design(40, 0)]
    X, Y = records_to_arrays(train, INPUT_KEYS, OUTPUT_KEYS)
    gp = GPForwardModel(n_restarts=3, seed=0, max_iter=100).fit(X, Y)
    return adapter, machine, gp, X, Y, RECIPE_VARIABLES


def _achieved_thickness(machine, recipe):
    rec = machine.run(recipe, tool_id="A")
    x, y = records_to_arrays([rec], INPUT_KEYS, OUTPUT_KEYS)
    return float(y[0, 0])


def test_mbe_reachable_spec_interval_covers_machine(mbe_gp_and_machine):
    adapter, machine, gp, X, Y, recipe_vars = mbe_gp_and_machine
    lo, hi = float(np.min(Y)), float(np.max(Y))
    target = 0.5 * (lo + hi)
    tol = 0.15 * (hi - lo)  # a comfortably reachable band mid-range

    kappa = 1.0
    solver = PessimisticInverseSolver(
        gp,
        list(recipe_vars),
        OUTPUT_KEYS,
        X_train=X,
        kappa=kappa,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    res = solver.solve({"targets": {"thickness_grown": {"target": target, "tol": tol}}})
    assert isinstance(res, list) and res, "reachable mid-range spec must be feasible"

    covered = 0
    for c in res:
        x = np.array([c.recipe[k] for k in INPUT_KEYS])
        sig_ale = float(np.atleast_1d(gp.predict(x).aleatoric_sigma)[0])
        ilo, ihi = c.predicted_outcome_interval["thickness_grown"]
        # the interval is HONEST, not decorative: its half-width is at least the
        # credited κ·σ_ale band (a bug that dropped s or shrank κ would fail
        # here, since a clean-machine outcome sits at the interval centre and
        # would be "covered" regardless of width otherwise — review finding).
        assert ihi - ilo >= 2.0 * kappa * sig_ale - 1e-18, (ihi - ilo, sig_ale)
        achieved = _achieved_thickness(machine, c.recipe)
        if ilo <= achieved <= ihi:
            covered += 1
    # clean machine ⇒ the pessimistic interval must cover EVERY recommended
    # recipe's real outcome.
    assert covered == len(res), (covered, len(res))


def test_mbe_unreachable_spec_is_infeasible(mbe_gp_and_machine):
    adapter, machine, gp, X, Y, recipe_vars = mbe_gp_and_machine
    unreachable = 10.0 * float(np.max(Y))  # 10x the thickest achievable film
    solver = PessimisticInverseSolver(
        gp,
        list(recipe_vars),
        OUTPUT_KEYS,
        X_train=X,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.01,
        seed=0,
    )
    res = solver.solve({"targets": {"thickness_grown": {"target": unreachable, "tol": 1e-9}}})
    assert isinstance(res, Infeasible)
    assert res.distance_to_feasible > 0.0
    assert res.nearest_achievable  # a concrete nearest recipe is reported
