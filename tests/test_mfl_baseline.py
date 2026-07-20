"""Tests for the MFL baseline (Gu et al. 2025, Alg. 1) + the bake-off scorer.

Written FIRST, per the build spec (docs/mfl-bakeoff-build-spec-2026-07-18.md §Tests).
The six tests map one-to-one to the spec's numbered requirements:

1. MFL recovers a known linear inverse (M(x)=Ax+b) on HELD-OUT targets.
2. Loop B improves on Loop A alone when E is deliberately biased from M (their Fig 9).
3. The conservative-LR gate actually fires (and does NOT fire when it should not).
4. The forward-difference Jacobian of a quadratic matches analytic to 1e-4.
5. Ledger exactness: one deployable Loop-B step on n' targets adds the documented count.
6. The scorer flags a PLANTED miss and counts RIG abstentions as non-presented.

All configs here are deliberately SMALL (few epochs / short loops) so the suite stays
fast; the Table-10 defaults live on the class and are exercised by the full run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rig.baselines.mfl import MFLLedger, ModelFeedbackLearning, fd_jacobian, spectral_norm

# make the example runner importable for the scorer test (test 6).
_BAKEOFF_DIR = Path(__file__).resolve().parents[1] / "examples" / "mfl_bakeoff"
if str(_BAKEOFF_DIR) not in sys.path:
    sys.path.insert(0, str(_BAKEOFF_DIR))


_X_LO = np.array([-2.0, -2.0])
_X_HI = np.array([2.0, 2.0])


def _sobol_box(n: int, lo: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    from scipy.stats import qmc

    s = qmc.Sobol(d=len(lo), scramble=True, seed=seed)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = s.random(n)
    return qmc.scale(u, lo, hi)


def _z_norm_err(mfl: ModelFeedbackLearning, machine, targets_z: np.ndarray) -> np.ndarray:
    """Per-target ‖M(R(z')) − z'‖ normalized by the target-space std."""
    X = mfl.propose(targets_z)
    Z_hat = machine(X)
    scale = np.maximum(targets_z.std(axis=0), 1e-9)
    return np.linalg.norm((Z_hat - targets_z) / scale, axis=1)


# --------------------------------------------------------------------------- #
# 1. linear inverse recovery                                                  #
# --------------------------------------------------------------------------- #


def test_mfl_recovers_known_linear_inverse():
    rng = np.random.default_rng(0)
    A = np.array([[1.3, -0.4], [0.2, 0.9]])  # well-conditioned, invertible
    b = np.array([0.5, -1.0])

    def machine(X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ A.T + b

    X = _sobol_box(256, _X_LO, _X_HI, seed=1)
    Z = machine(X)

    mfl = ModelFeedbackLearning(
        x_dim=2,
        z_dim=2,
        x_lower=_X_LO,
        x_upper=_X_HI,
        emulator_epochs=500,
        T=1500,
        T0=2000,  # keep the sensitivity gate out of this test
        tau=0,  # Loop-A only — no machine needed
        alpha1=0.05,
        seed=3,
    )
    mfl.fit_emulator(X, Z)

    train_z = machine(_sobol_box(96, _X_LO, _X_HI, seed=5))
    held_out_z = machine(_sobol_box(24, _X_LO, _X_HI, seed=7))
    mfl.train_reverse(train_z, machine=None)

    err = _z_norm_err(mfl, machine, held_out_z)
    # generalizes to unseen targets: median normalized round-trip error is small.
    assert np.median(err) < 0.25, f"median held-out inverse error {np.median(err):.3f} too large"


# --------------------------------------------------------------------------- #
# 2. Loop B corrects a biased emulator                                        #
# --------------------------------------------------------------------------- #


def test_loop_b_improves_over_loop_a_when_emulator_biased():
    A = np.array([[1.0, 0.3], [-0.2, 1.1]])
    b = np.array([0.0, 0.0])
    bias = np.array([1.5, -1.2])  # E is trained on a machine offset by this

    def true_machine(X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ A.T + b

    def biased_machine(X: np.ndarray) -> np.ndarray:
        return true_machine(X) + bias

    X = _sobol_box(256, _X_LO, _X_HI, seed=11)
    Z_biased = biased_machine(X)
    targets_z = true_machine(_sobol_box(16, _X_LO, _X_HI, seed=13))

    common = dict(
        x_dim=2,
        z_dim=2,
        x_lower=_X_LO,
        x_upper=_X_HI,
        emulator_epochs=500,
        T=1200,
        T0=2000,
        tau0=0,
        alpha1=0.05,
        delta=1e9,  # neutralize the gate; isolate the Loop-A-vs-B effect
        seed=17,
    )

    # Loop A only (same seed ⇒ identical E + identical Loop A as the A+B run).
    only_a = ModelFeedbackLearning(tau=0, **common)
    only_a.fit_emulator(X, Z_biased)
    only_a.train_reverse(targets_z, machine=None)
    err_a = np.median(_z_norm_err(only_a, true_machine, targets_z))

    # Loop A THEN Loop B on the TRUE machine.
    a_and_b = ModelFeedbackLearning(tau=120, **common)
    a_and_b.fit_emulator(X, Z_biased)
    a_and_b.train_reverse(targets_z, machine=true_machine, ledger=MFLLedger())
    err_ab = np.median(_z_norm_err(a_and_b, true_machine, targets_z))

    assert err_ab < 0.9 * err_a, f"Loop B did not improve: err_A={err_a:.3f} err_AB={err_ab:.3f}"


# --------------------------------------------------------------------------- #
# 3. conservative-LR gate fires (and only when it should)                     #
# --------------------------------------------------------------------------- #


def test_conservative_lr_gate_fires():
    A = np.array([[1.2, 0.0], [0.0, 1.1]])

    def machine(X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X) @ A.T

    X = _sobol_box(256, _X_LO, _X_HI, seed=19)
    Z = machine(X)
    targets_z = machine(_sobol_box(8, _X_LO, _X_HI, seed=23))

    common = dict(
        x_dim=2,
        z_dim=2,
        x_lower=_X_LO,
        x_upper=_X_HI,
        emulator_epochs=400,
        T=60,
        T0=50,  # sensitivity computed for the last 10 steps
        tau=0,
        seed=29,
    )

    # a well-fit emulator has standardized-Jacobian spectral norm ≈ 1 ≥ δ=0.5 ⇒ fires.
    fires = ModelFeedbackLearning(delta=0.5, **common)
    fires.fit_emulator(X, Z)
    fires.train_reverse(targets_z, machine=None)
    assert fires.alpha2_count_loopA > 0, "conservative LR never engaged despite high sensitivity"

    # an unreachable threshold must NEVER select α2 — proves the gate is a real test.
    never = ModelFeedbackLearning(delta=1e9, **common)
    never.fit_emulator(X, Z)
    never.train_reverse(targets_z, machine=None)
    assert never.alpha2_count_loopA == 0, "gate fired against an unreachable δ"


# --------------------------------------------------------------------------- #
# 4. finite-difference Jacobian of a quadratic                                #
# --------------------------------------------------------------------------- #


def test_fd_jacobian_matches_analytic_on_quadratic():
    def f(x: np.ndarray) -> np.ndarray:
        x0, x1 = x
        return np.array([x0**2, x1**2, x0 * x1])

    x = np.array([0.7, -1.3])
    analytic = np.array([[2 * x[0], 0.0], [0.0, 2 * x[1]], [x[1], x[0]]])
    J, n_probes = fd_jacobian(f, x, f(x), h=1e-6)

    assert n_probes == x.size  # exactly d probes (forward differences)
    assert np.max(np.abs(J - analytic)) < 1e-4
    # spectral_norm is the largest singular value.
    assert np.isclose(spectral_norm(analytic), np.linalg.svd(analytic, compute_uv=False)[0])


# --------------------------------------------------------------------------- #
# 5. ledger exactness                                                         #
# --------------------------------------------------------------------------- #


def test_ledger_counts_loop_b_machine_touches_exactly():
    def machine(X: np.ndarray) -> np.ndarray:
        return np.atleast_2d(X)  # identity is enough; we only count touches

    X = _sobol_box(64, _X_LO, _X_HI, seed=31)
    Z = machine(X)
    n_targets = 5
    targets_z = machine(_sobol_box(n_targets, _X_LO, _X_HI, seed=37))

    mfl = ModelFeedbackLearning(
        x_dim=2,
        z_dim=2,
        x_lower=_X_LO,
        x_upper=_X_HI,
        emulator_epochs=50,
        T=1,
        T0=1150,
        tau=1,  # exactly ONE Loop-B step
        tau0=1150,
        seed=41,
    )
    mfl.fit_emulator(X, Z)
    ledger = MFLLedger(seed_runs=30)
    mfl.train_reverse(targets_z, machine=machine, ledger=ledger)

    d = 2
    assert ledger.loopB_evals == n_targets  # one base value per target
    assert ledger.fd_probe_evals == n_targets * d  # d FD probes per target
    # charitable (their setting) does NOT count FD probes; deployable does — the
    # difference is EXACTLY the probe count (prereg §3 ledger split).
    assert ledger.deployable_total - ledger.charitable_total == ledger.fd_probe_evals
    assert ledger.charitable_total == 30 + n_targets
    assert ledger.deployable_total == 30 + n_targets + n_targets * d


# --------------------------------------------------------------------------- #
# 6. scorer: planted miss + abstentions are non-presented                     #
# --------------------------------------------------------------------------- #


def test_scorer_flags_planted_miss_and_excludes_abstentions():
    import run_bakeoff as rb

    spec = {"nonuniformity_pct": (0.0, 0.05), "T_center": (1400.0, 1600.0)}

    planted_miss = rb.ArmTargetResult(
        target_id="t_miss",
        presented=True,
        recipe={"T_heater": 1300.0, "film_thickness": 1e-6},
        ground_truth={"nonuniformity_pct": 0.20, "T_center": 1500.0},  # nonunif OUT of spec
        spec=spec,
        feasible_truth=True,
    )
    good_hit = rb.ArmTargetResult(
        target_id="t_hit",
        presented=True,
        recipe={"T_heater": 1350.0, "film_thickness": 1e-6},
        ground_truth={"nonuniformity_pct": 0.03, "T_center": 1500.0},  # in spec
        spec=spec,
        feasible_truth=True,
    )
    rig_abstention = rb.ArmTargetResult(
        target_id="t_abstain",
        presented=False,  # RIG returned INFEASIBLE — NOT presented
        recipe=None,
        ground_truth=None,
        spec=spec,
        feasible_truth=True,  # and the target WAS feasible ⇒ a false abstention
    )

    results = [planted_miss, good_hit, rig_abstention]

    # the planted miss is flagged, and the denominator is PRESENTED recipes only:
    # 1 miss / 2 presented = 0.5. The abstention is neither a miss nor a denominator.
    assert rb.certified_miss_rate(results) == pytest.approx(0.5)

    # the abstention on a feasible target counts as a false abstention (1 of 3 feasible).
    assert rb.false_abstention_rate(results) == pytest.approx(1.0 / 3.0)

    # all-abstained ⇒ no presented recipes ⇒ miss rate is undefined (nan), never a miss.
    all_abstained = [rig_abstention]
    assert np.isnan(rb.certified_miss_rate(all_abstained))

    # one-sided spec handling (slip ≤ 1.0, no lower edge).
    one_sided = {"slip_max_ratio": (None, 1.0)}
    assert rb.in_spec({"slip_max_ratio": 0.4}, one_sided) is True
    assert rb.in_spec({"slip_max_ratio": 1.5}, one_sided) is False
