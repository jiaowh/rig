"""Tests for the conditional / per-region conformal-coverage study
(examples/real_data/empa_hipims/run_conditional_coverage.py).

Two tiers:

  * PURE-FUNCTION tests (no data files): the pre-stated group-assignment
    machinery -- tertile edges are deterministic, the density statistic uses the
    TRAINING set as its ONLY neighbour reference (no test-point leakage), the
    binomial-CI is literally the recorded gate's function, underpowered groups
    are flagged, and the stream-position thirds are contiguous.

  * FIDELITY tests (local Empa files, sim-free, like tests/test_empa_ingest.py):
    on one real campaign/split the reproduced pooled + per-output PICP is
    byte-equal to the recorded results/m1_empa.json, and the fidelity gate is
    RED-PROOFED -- a one-step perturbation of the reproduced indicator sequence
    makes the assertion fire, and restoring it makes the gate pass again.

The runner is imported BY PATH (examples/ is not a package -- same loader trick
as tests/test_empa_ingest.py / tests/test_empa_pooled.py). Importing it reads no
data (the baseline JSON and CSVs are only touched inside main()/the fixtures).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "real_data" / "empa_hipims"
CSV_DIR = EX / "csv"
BASELINE_JSON = EX / "results" / "m1_empa.json"


def _load_by_path(name: str, path: Path):
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


mod = _load_by_path("run_conditional_coverage", EX / "run_conditional_coverage.py")
runner = mod.runner  # the imported run_m1_empa module (protocol single source of truth)

DATA_PRESENT = CSV_DIR.exists() and BASELINE_JSON.exists()
needs_data = pytest.mark.skipif(
    not DATA_PRESENT, reason="local Empa CSVs / recorded baseline not present"
)


# ============================================================================
# (1) tertile edges are deterministic
# ============================================================================


def test_tertile_labels_deterministic_and_correct():
    v = np.arange(1.0, 10.0)  # 1..9
    labels1, edges1 = mod.tertile_labels(v, ("low", "mid", "high"))
    labels2, edges2 = mod.tertile_labels(v, ("low", "mid", "high"))
    # deterministic: identical edges + labels on repeat calls
    assert edges1 == edges2
    np.testing.assert_array_equal(labels1, labels2)
    # edges are the 1/3, 2/3 empirical quantiles
    assert edges1[0] == pytest.approx(float(np.quantile(v, 1.0 / 3.0)))
    assert edges1[1] == pytest.approx(float(np.quantile(v, 2.0 / 3.0)))
    # digitize: 1,2,3 low ; 4,5,6 mid ; 7,8,9 high  (balanced thirds here)
    assert list(labels1) == ["low"] * 3 + ["mid"] * 3 + ["high"] * 3


def test_tertile_labels_shuffle_invariant_edges():
    """Edges depend on the value MULTISET, not on row order -- a reshuffle of the
    same values gives the same edges (a determinism guard against order effects)."""
    v = np.array([5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0])
    _, edges_a = mod.tertile_labels(v, ("low", "mid", "high"))
    _, edges_b = mod.tertile_labels(np.sort(v), ("low", "mid", "high"))
    assert edges_a == edges_b


# ============================================================================
# (2) density statistic: TRAINING reference only, no test-point leakage
# ============================================================================


def test_density_reference_is_training_only_no_leakage():
    rng = np.random.default_rng(0)
    X_train = rng.normal(0.0, 1.0, (60, 3))  # a cluster near the origin
    # two test points FAR from the training cluster but very close to EACH OTHER
    far = np.array([[100.0, 100.0, 100.0], [100.05, 100.0, 100.0]])
    d = mod.knn_distance_to_train(far, X_train, k=1)
    # if test points leaked into the neighbour reference, each far point would be
    # the other's 1-NN at standardized distance ~0.05. Because the reference is
    # TRAINING-only, both must instead sit at their (large) distance to the
    # training cluster -- proving no test point is usable as a neighbour.
    assert np.all(d > 10.0), d
    # and the reference REALLY is the training set: a test point coinciding with
    # a training row finds that row at distance 0 (correct, not leakage).
    coincident = X_train[[7]].copy()
    d0 = mod.knn_distance_to_train(coincident, X_train, k=1)
    assert d0[0] == pytest.approx(0.0, abs=1e-9)


def test_knn_distance_is_the_kth_nearest_standardized():
    X_train = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    X_test = np.array([[0.0]])
    d = mod.knn_distance_to_train(X_test, X_train, k=3)
    # standardize by TRAIN stats (mean 2, std sqrt(2)); 3rd-nearest by |Δ|
    s = float(np.std([0, 1, 2, 3, 4]))
    tr = (np.array([0, 1, 2, 3, 4]) - 2) / s
    xt = (0 - 2) / s
    third_nearest = np.sort(np.abs(tr - xt))[2]
    assert d[0] == pytest.approx(third_nearest)


def test_standardize_by_train_uses_train_stats():
    X_train = np.array([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]])  # col1 zero-variance
    X_test = np.array([[20.0, 5.0]])
    xt_std, xtr_std = mod.standardize_by_train(X_test, X_train)
    # col0: (20-20)/std -> 0 at the train mean; col1: zero-variance -> scale 1
    assert xt_std[0, 0] == pytest.approx(0.0)
    assert xt_std[0, 1] == pytest.approx(5.0)  # (5 - 0)/1
    assert xtr_std.mean(axis=0)[0] == pytest.approx(0.0)


# ============================================================================
# (3) the binomial-CI is the recorded gate's exact function (reuse, not re-impl)
# ============================================================================


def test_binom_ci_is_the_recorded_gate_function():
    assert mod.binom_ci is runner.binom_ci  # literally reused
    for k, n in [(90, 100), (15, 25), (0, 10), (40, 40)]:
        lo, hi = mod.binom_ci(k, n)
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        assert lo == pytest.approx(float(ref.low))
        assert hi == pytest.approx(float(ref.high))


# ============================================================================
# (4) per-group coverage: underpowered flag + directional flag
# ============================================================================


def test_group_coverage_underpowered_flag():
    hits = np.ones(60, dtype=bool)
    small = np.zeros(60, dtype=bool)
    small[:19] = True  # n=19 < 20 -> UNDERPOWERED
    g_small = mod.group_coverage(hits, small)
    assert g_small["n"] == 19 and g_small["underpowered"] is True

    big = np.zeros(60, dtype=bool)
    big[:20] = True  # n=20 -> powered (boundary)
    g_big = mod.group_coverage(hits, big)
    assert g_big["n"] == 20 and g_big["underpowered"] is False

    empty = mod.group_coverage(hits, np.zeros(60, dtype=bool))
    assert empty["n"] == 0 and empty["picp"] is None and empty["underpowered"] is True


def test_group_coverage_direction_under_ok_over():
    mask25 = np.zeros(60, dtype=bool)
    mask25[:25] = True

    under = np.zeros(60, dtype=bool)
    under[:15] = True  # 15/25 = 0.60 -> CI upper < 0.90
    gu = mod.group_coverage(under, mask25)
    assert gu["picp"] == pytest.approx(15 / 25)
    assert gu["direction"] == "under" and gu["nominal_in_ci"] is False

    ok = np.zeros(60, dtype=bool)
    ok[:23] = True  # 23/25 = 0.92 -> CI straddles 0.90
    gok = mod.group_coverage(ok, mask25)
    assert gok["direction"] == "ok" and gok["nominal_in_ci"] is True

    # over-coverage: 40/40 covered -> CI lower ~0.912 > 0.90
    mask40 = np.zeros(60, dtype=bool)
    mask40[:40] = True
    over = np.ones(60, dtype=bool)
    go = mod.group_coverage(over, mask40)
    assert go["picp"] == 1.0 and go["direction"] == "over" and go["nominal_in_ci"] is False


# ============================================================================
# (5) stream-position thirds are contiguous early/mid/late
# ============================================================================


def test_stream_position_labels_contiguous_thirds():
    assert list(mod.stream_position_labels(12)) == ["early"] * 4 + ["mid"] * 4 + ["late"] * 4
    labs = mod.stream_position_labels(10)  # array_split -> 4,3,3
    assert list(labs) == ["early"] * 4 + ["mid"] * 3 + ["late"] * 3
    assert labs[0] == "early" and labs[-1] == "late"


# ============================================================================
# FIDELITY: reproduced coverage == recorded, on one real campaign/split
# ============================================================================


@pytest.fixture(scope="module")
def one_cell():
    if not DATA_PRESENT:
        pytest.skip("local Empa CSVs / recorded baseline not present")
    slug = "al_250w_low_duty"  # smallest campaign (n=401) -> fastest fit
    campaign = next(c for c in mod.CAMPAIGNS if c.slug == slug)
    campaign_index = [c.slug for c in mod.CAMPAIGNS].index(slug)
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    gp_restarts = int(baseline["meta"]["gp_restarts"])
    X, Y, ik, ok, units, n, _degen = mod.campaign_arrays(campaign)
    split_name = "random"
    fit_idx, cal_idx, test_idx = mod.reconstruct_splits(n, campaign_index)[split_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, model, _ = runner.fit_and_eval(
            X[fit_idx],
            Y[fit_idx],
            X[cal_idx],
            Y[cal_idx],
            X[test_idx],
            Y[test_idx],
            ik,
            ok,
            units,
            gp_restarts,
        )
    hits = mod.path_indicators(model, X[cal_idx], Y[cal_idx], X[test_idx], Y[test_idx], ok, units)
    return {
        "slug": slug,
        "split": split_name,
        "hits": hits,
        "output_keys": ok,
        "recorded": baseline["campaigns"][slug]["splits"][split_name],
    }


@needs_data
def test_fidelity_reproduces_recorded_pooled_and_per_output(one_cell):
    rep = mod.fidelity_check(
        one_cell["slug"],
        one_cell["split"],
        one_cell["hits"],
        one_cell["output_keys"],
        one_cell["recorded"],
    )
    rec = one_cell["recorded"]
    # pooled PICP reproduced byte-equal (static / ACI / PID)
    assert rep["static"]["pooled"]["reproduced_picp"] == pytest.approx(rec["pooled"]["picp"])
    assert rep["aci"]["pooled"]["reproduced_picp"] == pytest.approx(rec["aci"]["pooled"]["picp"])
    assert rep["pid"]["pooled"]["reproduced_picp"] == pytest.approx(rec["pid"]["pooled"]["picp"])
    # per-output k_covered reproduced exactly for every path
    for path, recblock in (
        ("static", rec),
        ("aci", rec["aci"]),
        ("pid", rec["pid"]),
    ):
        for key in one_cell["output_keys"]:
            assert (
                rep[path]["per_output"][key]["k_covered"]
                == recblock["per_output"][key]["k_covered"]
            )


@needs_data
@pytest.mark.parametrize("path", ["static", "aci", "pid"])
def test_fidelity_gate_is_red_proofed_one_step_perturbation(one_cell, path):
    """RED-PROOF: the gate PASSES as reproduced; flipping ONE step of a path's
    indicator sequence makes it FIRE; restoring makes it PASS again."""
    args = (
        one_cell["slug"],
        one_cell["split"],
        one_cell["hits"],
        one_cell["output_keys"],
        one_cell["recorded"],
    )

    mod.fidelity_check(*args)  # baseline: passes (no raise)

    seq = one_cell["hits"][path]["hits"]
    original = bool(seq[0, 0])
    seq[0, 0] = not original  # perturb exactly one step
    try:
        with pytest.raises(RuntimeError, match="FIDELITY FAIL"):
            mod.fidelity_check(*args)
    finally:
        seq[0, 0] = original  # restore

    mod.fidelity_check(*args)  # restored: passes again
