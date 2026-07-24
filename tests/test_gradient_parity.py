"""Tests for the analytic-gradient vs FD parity study
(examples/run_gradient_parity_study.py).

Covers the agreement-scoring core (:func:`score_pair`) — including a disagreement case
scored by GROUND TRUTH, never the model's own opinion — a smoke end-to-end +
determinism run, and a RED-PROOF that the scorer distinguishes a genuine disagreement
outcome from its opposite (a scorer that scored ground truth backwards would label a
false success as the "right" arm)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))

import run_gradient_parity_study as gp  # noqa: E402

from rig.interfaces import Infeasible, RecipeCandidate  # noqa: E402
from rig.inverse.pessimistic import PessimisticInverseSolver  # noqa: E402

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _linear_truth(dim: int):
    """y0 = y1 = sum(x): trivial, deterministic, dim-agnostic."""

    def f(X):
        X = np.atleast_2d(X)
        s = X.sum(axis=1)
        return np.stack([s, s], axis=-1)

    return f


class _FakePD:
    def __init__(self, mean, ale, epi):
        self.mean = np.asarray(mean, float)
        self.aleatoric_sigma = np.asarray(ale, float)
        self.epistemic_sigma = np.asarray(epi, float)
        self.conformal_set = None


class _FakeModel:
    """A public-API-only ForwardModel stand-in: mean = sum(x) on both outputs, fixed
    aleatoric/epistemic sigma, zero Jacobian (so ``public_margin`` reduces to a simple
    closed form we can check by hand)."""

    def __init__(self, dim: int, ale=0.2, epi=0.0):
        self.dim = dim
        self.ale = ale
        self.epi = epi

    def predict(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        s = float(np.sum(x))
        return _FakePD([s, s], [self.ale, self.ale], [self.epi, self.epi])

    def jacobian(self, x):
        return np.ones((2, self.dim))

    def support_score(self, x):
        return 0.0


def _cand(recipe: dict, confidence: float = 0.99) -> RecipeCandidate:
    return RecipeCandidate(
        recipe=recipe,
        confidence=confidence,
        predicted_outcome_interval={"y0": (0.0, 0.0), "y1": (0.0, 0.0)},
        feasibility_flag=True,
        support_score=0.0,
        calibration_status="model-feasible",
    )


def _score_kwargs(dim, model, lower, upper):
    return dict(
        dim=dim,
        lower=lower,
        upper=upper,
        model=model,
        kappa=2.0,
        z_epi=2.0,
        delta_frac=0.0,  # zero Jacobian contribution -> margin is a plain z-score
        delta_raw=np.zeros(dim),
        flat_scale=np.full(dim, 4.0),
    )


# --------------------------------------------------------------------------
# public_margin — checked against a hand-computed value
# --------------------------------------------------------------------------


def test_public_margin_matches_hand_computation():
    """mu=0 at x=(0,0) (2-D), sigma_ale=0.2, box +-0.8, delta_frac=0 -> z_epi term is 0
    (epi=0 in the fake model) -> u_hi = u_lo = (0.8 - 0) / 0.2 = 4.0."""
    model = _FakeModel(dim=2, ale=0.2, epi=0.0)
    m = gp.public_margin(
        model,
        {"x0": 0.0, "x1": 0.0},
        2,
        gp.OUT_IDX,
        np.array([-0.8, -0.8]),
        np.array([0.8, 0.8]),
        kappa=2.0,
        z_epi=2.0,
        delta_frac=0.0,
        delta_raw=np.zeros(2),
    )
    assert m == pytest.approx(4.0, abs=1e-9)


def test_public_margin_reflects_epistemic_displacement():
    """Same box, but epi=0.1 and z_epi=2.0 -> s=0.2 displacement subtracted from the
    (0.8) headroom before standardizing: u = (0.8 - 0.2) / 0.2 = 3.0 < the epi=0 case."""
    model = _FakeModel(dim=2, ale=0.2, epi=0.1)
    m = gp.public_margin(
        model,
        {"x0": 0.0, "x1": 0.0},
        2,
        gp.OUT_IDX,
        np.array([-0.8, -0.8]),
        np.array([0.8, 0.8]),
        kappa=2.0,
        z_epi=2.0,
        delta_frac=0.0,
        delta_raw=np.zeros(2),
    )
    assert m == pytest.approx(3.0, abs=1e-9)


# --------------------------------------------------------------------------
# score_pair — verdict agreement
# --------------------------------------------------------------------------


def test_score_pair_both_infeasible_agree():
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    fd_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    an_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    rec = gp.score_pair(
        fd_res, an_res, truth=truth, **_score_kwargs(dim, model, [-0.8, -0.8], [0.8, 0.8])
    )
    assert rec["verdict_agree"] is True
    assert rec["verdict_fd"] == rec["verdict_an"] == "INFEASIBLE"
    assert "recipe_distance_normalized" not in rec  # only computed for agreeing FEASIBLE


def test_score_pair_agreeing_feasible_records_distance_and_ground_truth():
    dim = 2
    model = _FakeModel(dim, ale=0.2)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = [_cand({"x0": 0.1, "x1": -0.1})]  # y=(0,0): a genuine hit
    an_res = [_cand({"x0": 0.15, "x1": -0.15})]  # y=(0,0): also a genuine hit, different recipe
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is True
    assert rec["hit_fd"] is True and rec["hit_an"] is True
    assert rec["gt_both_hit"] is True
    assert rec["gt_split"] is False
    # different recipes but both genuinely hit: distance > 0 is FINE (per spec).
    assert rec["recipe_distance_normalized"] > 0.0
    assert rec["margin_diff"] == pytest.approx(0.0, abs=1e-9)  # same margin formula, same mu here


def test_score_pair_agreeing_feasible_can_split_on_ground_truth():
    """Agreeing verdicts (both FEASIBLE) does not imply agreeing ground-truth outcomes:
    one arm's top candidate can hit while the other's misses. `gt_split` must catch it —
    this is exactly why ground-truth agreement is a SEPARATE endpoint from verdict
    agreement, not implied by it."""
    dim = 2
    model = _FakeModel(dim, ale=0.2)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = [_cand({"x0": 0.1, "x1": -0.1})]  # y=(0,0): hit
    an_res = [_cand({"x0": 2.0, "x1": 2.0})]  # y=(4,4): a genuine miss
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is True
    assert rec["hit_fd"] is True and rec["hit_an"] is False
    assert rec["gt_split"] is True
    assert rec["gt_both_hit"] is False and rec["gt_both_miss"] is False


# --------------------------------------------------------------------------
# score_pair — disagreement, scored by GROUND TRUTH (the endpoint that matters)
# --------------------------------------------------------------------------


def test_score_pair_disagreement_fd_false_success_favors_flip():
    """FD certifies a recipe that MISSES ground truth (a false success); analytic
    abstains. This is evidence FOR flipping the default: analytic did not ship the
    false success FD did."""
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = [_cand({"x0": 2.0, "x1": 2.0})]  # y=(4,4): certified but MISSES truth
    an_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is False
    assert rec["disagreement_type"] == "fd_feasible_an_infeasible"
    assert rec["hit_fd"] is False
    assert rec["verdict_favors"] == "fd_false_success"


def test_score_pair_disagreement_an_false_abstention_favors_keep_fd():
    """FD certifies a recipe that genuinely HITS ground truth; analytic abstains
    needlessly. This is evidence AGAINST flipping: flipping would have lost a genuine
    hit FD found."""
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = [_cand({"x0": 0.1, "x1": -0.1})]  # y=(0,0): genuine hit
    an_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is False
    assert rec["hit_fd"] is True
    assert rec["verdict_favors"] == "an_false_abstention"


def test_score_pair_disagreement_fd_false_abstention_favors_flip():
    """analytic certifies a recipe that genuinely HITS ground truth; FD abstains
    needlessly. Evidence FOR flipping: analytic found a hit FD missed."""
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    an_res = [_cand({"x0": 0.1, "x1": -0.1})]  # y=(0,0): genuine hit
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is False
    assert rec["hit_an"] is True
    assert rec["verdict_favors"] == "fd_false_abstention"


def test_score_pair_disagreement_an_false_success_favors_keep_fd():
    """analytic certifies a recipe that MISSES ground truth (a false success); FD
    abstains. Evidence AGAINST flipping: flipping would have shipped a false success
    FD's abstention avoided."""
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    fd_res = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="epistemic")
    an_res = [_cand({"x0": 2.0, "x1": 2.0})]  # y=(4,4): certified but MISSES truth
    rec = gp.score_pair(fd_res, an_res, truth=truth, **_score_kwargs(dim, model, lower, upper))
    assert rec["verdict_agree"] is False
    assert rec["hit_an"] is False
    assert rec["verdict_favors"] == "an_false_success"


# --------------------------------------------------------------------------
# aggregate — disagreement tallying rolls up correctly
# --------------------------------------------------------------------------


def test_aggregate_tallies_disagreement_direction():
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = [-0.8, -0.8], [0.8, 0.8]
    kw = _score_kwargs(dim, model, np.array(lower), np.array(upper))

    # one "for flip" case (fd_false_success) + one "against flip" case (an_false_abstention)
    fd_miss = [_cand({"x0": 2.0, "x1": 2.0})]
    an_abstain = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="x")
    fd_hit = [_cand({"x0": 0.1, "x1": -0.1})]

    r1 = gp.score_pair(fd_miss, an_abstain, truth=truth, **kw)
    r1.update(d=2, n=24, seed=0, target_class="hard", t_fit=0.0, t_fd=0.1, t_an=0.01)
    r2 = gp.score_pair(fd_hit, an_abstain, truth=truth, **kw)
    r2.update(d=2, n=24, seed=1, target_class="hard", t_fit=0.0, t_fd=0.1, t_an=0.01)

    agg = gp.aggregate([r1, r2])
    assert agg["overall"]["n_disagreements"] == 2
    assert agg["overall"]["n_disagreements_favoring_flip"] == 1
    assert agg["overall"]["n_disagreements_favoring_keep_fd"] == 1
    assert agg["overall"]["disagreement_breakdown"] == {
        "fd_false_success": 1,
        "an_false_abstention": 1,
    }


# --------------------------------------------------------------------------
# RED-PROOF: a ground-truth-blind scorer would mislabel the disagreement
# --------------------------------------------------------------------------


def test_red_proof_ground_truth_polarity_matters():
    """Prove the disagreement label actually depends on GROUND TRUTH, not merely on
    which arm certified. Same certifying recipe, only ITS GROUND-TRUTH OUTCOME differs
    (hit vs miss) — the label must flip with it. A scorer that ignored ground truth
    (e.g. always labelling 'the certifying arm was right') would give the SAME label in
    both cases; we assert they differ, which is exactly what such a broken scorer would
    get wrong."""
    dim = 2
    model = _FakeModel(dim)
    truth = _linear_truth(dim)
    lower, upper = np.array([-0.8, -0.8]), np.array([0.8, 0.8])
    kw = _score_kwargs(dim, model, lower, upper)
    an_abstain = Infeasible(nearest_achievable={}, distance_to_feasible=1.0, reason="x")

    fd_hit = [_cand({"x0": 0.1, "x1": -0.1})]  # y=(0,0): genuine hit
    fd_miss = [_cand({"x0": 2.0, "x1": 2.0})]  # y=(4,4): genuine miss

    rec_hit = gp.score_pair(fd_hit, an_abstain, truth=truth, **kw)
    rec_miss = gp.score_pair(fd_miss, an_abstain, truth=truth, **kw)

    # a ground-truth-blind scorer ("FD certified => FD is right") would label BOTH
    # 'an_false_abstention'. The real scorer must NOT: the miss case must flip.
    assert rec_hit["verdict_favors"] == "an_false_abstention"
    assert rec_miss["verdict_favors"] == "fd_false_success"
    assert rec_hit["verdict_favors"] != rec_miss["verdict_favors"], (
        "ground-truth-blind scoring would give the same label regardless of whether "
        "the certifying recipe actually hits truth — this must not happen"
    )


# --------------------------------------------------------------------------
# the documented analytic_grad + delta_mode='pgd' construction-time boundary
# --------------------------------------------------------------------------


def test_analytic_grad_and_pgd_delta_mode_raise_at_construction():
    """The boundary this study must respect: analytic_grad forms its delta term from
    the Taylor/Hessian expansion, so combining it with delta_mode='pgd' (a
    finite-difference-only inner max) is an inconsistent solve and raises at
    construction (src/rig/inverse/pessimistic.py `__init__`). This study never
    constructs that combination — this test pins the boundary so a future change to
    either side is caught here rather than silently in the grid."""
    from rig.forward import GPForwardModel
    from rig.interfaces import ContinuousVariable

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(10, 2))
    Y = rng.normal(size=(10, 2))
    model = GPForwardModel(n_restarts=1, seed=0).fit(X, Y)
    with pytest.raises(ValueError, match="delta_mode='pgd'"):
        PessimisticInverseSolver(
            model,
            variables=[ContinuousVariable("x0", -1.0, 1.0), ContinuousVariable("x1", -1.0, 1.0)],
            output_keys=["y0", "y1"],
            X_train=X,
            analytic_grad=True,
            delta_mode="pgd",
        )


# --------------------------------------------------------------------------
# smoke end-to-end + determinism (a couple of real solves)
# --------------------------------------------------------------------------


def test_smoke_end_to_end_and_determinism():
    g1 = gp.run_grid(2, d_values=(2,), verbose=False)
    g2 = gp.run_grid(2, d_values=(2,), verbose=False)
    assert gp.deterministic_view(g1) == gp.deterministic_view(g2)
    assert len(g1["per_seed"]) == 2 * 2  # 2 seeds x 2 target classes
    agg = gp.aggregate(g1["per_seed"])
    assert agg["overall"]["n_pairs"] == 4
    assert 0.0 <= agg["overall"]["verdict_agreement_rate"] <= 1.0
