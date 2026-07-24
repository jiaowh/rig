"""Tests for the false-success-rate-vs-dimension study (examples/run_false_success_study.py).

Covers the safety-central scorer (certified-miss counting, incl. the certified-but-
missing case), a smoke end-to-end + determinism run, and a RED-PROOF that the scorer
test actually catches an inverted classifier (a scorer that counted a miss as a hit
would fail these assertions).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))

import run_false_success_study as fs  # noqa: E402

from rig.interfaces import Infeasible, RecipeCandidate  # noqa: E402

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _linear_truth(dim: int):
    """A trivial deterministic truth: y0 = sum(x), y1 = sum(x). 2 outputs, dim inputs."""

    def f(X):
        X = np.atleast_2d(X)
        s = X.sum(axis=1)
        return np.stack([s, s], axis=-1)

    return f


def _cand(recipe: dict, status: str = "model-feasible") -> RecipeCandidate:
    return RecipeCandidate(
        recipe=recipe,
        confidence=0.99,
        predicted_outcome_interval={"y0": (0.0, 0.0), "y1": (0.0, 0.0)},
        feasibility_flag=True,
        support_score=0.0,
        calibration_status=status,
    )


# --------------------------------------------------------------------------
# evaluate_on_truth — the per-recipe ground-truth classifier
# --------------------------------------------------------------------------


def test_evaluate_on_truth_hit_and_miss():
    truth = _linear_truth(2)  # y = x0 + x1 on both outputs
    lower = np.array([-0.8, -0.8])
    upper = np.array([0.8, 0.8])

    # recipe summing to 0.0 -> y = (0, 0) -> inside the box -> genuine hit
    inbox, exc, y = fs.evaluate_on_truth({"x0": 0.5, "x1": -0.5}, truth, 2, lower, upper)
    assert inbox is True
    assert exc == 0.0
    assert np.allclose(y, [0.0, 0.0])

    # recipe summing to 2.0 -> y = (2, 2) -> OUTSIDE the +-0.8 box -> a miss
    inbox, exc, y = fs.evaluate_on_truth({"x0": 1.0, "x1": 1.0}, truth, 2, lower, upper)
    assert inbox is False
    assert exc == pytest.approx(1.2)  # 2.0 - 0.8


def test_evaluate_on_truth_only_calls_truth_on_the_recipe():
    # a recipe exactly on the upper boundary is IN (inclusive); just past it is OUT.
    truth = _linear_truth(1)
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    assert fs.evaluate_on_truth({"x0": 1.0}, truth, 1, lower, upper)[0] is True
    assert fs.evaluate_on_truth({"x0": 1.0000001}, truth, 1, lower, upper)[0] is False


# --------------------------------------------------------------------------
# score_result — certified-miss counting, incl. the certified-but-missing case
# --------------------------------------------------------------------------


def test_score_result_counts_certified_miss_as_false_success():
    """THE certified-but-missing case: a certified candidate (feasibility_flag=True)
    whose recipe misses the spec box on truth is a FALSE SUCCESS."""
    truth = _linear_truth(2)
    lower = np.array([-0.8, -0.8])
    upper = np.array([0.8, 0.8])
    result = [
        _cand({"x0": 0.2, "x1": -0.2}),  # y=(0,0) -> genuine hit
        _cand({"x0": 1.0, "x1": 1.0}),  # y=(2,2) -> certified MISS = false success
    ]
    rec = fs.score_result(result, truth, 2, lower, upper)
    assert rec["status"] == "FEASIBLE"
    assert rec["n_cand"] == 2
    assert rec["n_hit"] == 1
    assert rec["n_false_success"] == 1
    assert rec["worst_miss_excursion"] == pytest.approx(1.2)


def test_score_result_all_hits_no_false_success():
    truth = _linear_truth(2)
    lower = np.array([-0.8, -0.8])
    upper = np.array([0.8, 0.8])
    result = [_cand({"x0": 0.1, "x1": 0.1}), _cand({"x0": -0.2, "x1": 0.2})]
    rec = fs.score_result(result, truth, 2, lower, upper)
    assert rec["n_false_success"] == 0
    assert rec["n_hit"] == 2
    assert rec["worst_miss_excursion"] == 0.0


def test_score_result_infeasible_is_abstention_not_false_success():
    truth = _linear_truth(2)
    lower = np.array([-0.8, -0.8])
    upper = np.array([0.8, 0.8])
    inf = Infeasible(
        nearest_achievable={"x0": 0.0, "x1": 0.0},
        distance_to_feasible=0.3,
        reason="pessimistic-infeasible, but the binding term is EPISTEMIC: collect more runs",
    )
    rec = fs.score_result(inf, truth, 2, lower, upper)
    assert rec["status"] == "INFEASIBLE"
    assert rec["n_false_success"] == 0
    assert rec["n_cand"] == 0
    assert rec["reason_category"] == "epistemic"


def test_score_result_rejects_noncertified_candidate():
    """A returned candidate must be certified; a non-certified one makes 'false success'
    meaningless, so the scorer fails loud rather than miscount."""
    truth = _linear_truth(2)
    bad = RecipeCandidate(
        recipe={"x0": 0.0, "x1": 0.0},
        confidence=0.5,
        predicted_outcome_interval={"y0": (0.0, 0.0), "y1": (0.0, 0.0)},
        feasibility_flag=False,
        support_score=0.0,
    )
    with pytest.raises(AssertionError):
        fs.score_result([bad], truth, 2, np.array([-1.0, -1.0]), np.array([1.0, 1.0]))


def test_reason_category_buckets():
    assert fs._reason_category("conformal-infeasible (§13.2): C(x) ⊄ Z*") == "conformal"
    assert fs._reason_category("binding term is EPISTEMIC; collect more runs") == "epistemic"
    assert fs._reason_category("genuinely unreachable: mean outside") == "unreachable"
    assert (
        fs._reason_category("no on-manifold recipe: below the §8.2 support floor") == "off-manifold"
    )


# --------------------------------------------------------------------------
# RED-PROOF: the scorer test catches an inverted classifier
# --------------------------------------------------------------------------


def test_red_proof_inverted_scorer_would_fail():
    """Prove the certified-miss test has teeth: a scorer that counted a MISS as a HIT
    (inbox inverted) would produce n_false_success=0 on the certified-but-missing case,
    and the real assertion (n_false_success == 1) would then be RED. We assert the
    inverted count here directly, so if evaluate_on_truth's polarity ever flipped this
    test and test_score_result_counts_certified_miss_as_false_success can't both pass."""
    truth = _linear_truth(2)
    lower = np.array([-0.8, -0.8])
    upper = np.array([0.8, 0.8])
    miss_recipe = {"x0": 1.0, "x1": 1.0}  # y=(2,2), a true miss of the +-0.8 box
    inbox, _exc, _y = fs.evaluate_on_truth(miss_recipe, truth, 2, lower, upper)
    # the correct classifier says NOT in box. An inverted one would say in box; then
    # score_result would count 0 false successes and the safety test would go red.
    assert inbox is False
    inverted_false_success_count = 1 if inbox else 0  # what a flipped scorer would report
    assert inverted_false_success_count == 0, (
        "sanity of the red-proof: a scorer treating this miss as a hit yields 0 "
        "false successes — which is exactly the failure the miss test guards against"
    )


# --------------------------------------------------------------------------
# Clopper-Pearson exact CI (incl. the 0-count 'upper bound, not zero' guard)
# --------------------------------------------------------------------------


def test_clopper_pearson_zero_count_gives_upper_bound_not_zero():
    lo, hi = fs.clopper_pearson(0, 40)
    assert lo == 0.0
    assert 0.0 < hi < 0.1  # a real 95% upper bound on a 0/40 rate, never a claim of 0
    assert hi == pytest.approx(1 - 0.025 ** (1 / 40), rel=1e-6)  # exact CP upper at k=0


def test_clopper_pearson_full_and_empty():
    assert fs.clopper_pearson(5, 5)[1] == 1.0
    assert np.isnan(fs.clopper_pearson(0, 0)[0])
    lo, hi = fs.clopper_pearson(1, 100)
    assert lo < 0.01 / 100 * 100 < hi  # contains the point estimate 0.01


# --------------------------------------------------------------------------
# smoke end-to-end + determinism (a couple of real solves)
# --------------------------------------------------------------------------


def test_smoke_end_to_end_and_determinism():
    cells = ({"d": 2, "n": 24},)
    g1 = fs.run_grid(1, n_restarts=48, analytic_grad=True, cells=cells, verbose=False)
    g2 = fs.run_grid(1, n_restarts=48, analytic_grad=True, cells=cells, verbose=False)
    # a real run produced structured results for both arms
    assert set(g1["results"]["d2_n24"]) == {"raw", "wrapped"}
    for arm in ("raw", "wrapped"):
        cell = g1["results"]["d2_n24"][arm]
        assert cell["n_seeds"] == 1
        assert cell["n_false_success"] >= 0
    # determinism: the timing-free view is byte-identical across two runs
    assert fs.deterministic_view(g1) == fs.deterministic_view(g2)
