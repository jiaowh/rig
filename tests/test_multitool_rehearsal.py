"""Tests for the in-silico multi-tool M4 dress rehearsal (examples/run_multitool_rehearsal.py).

The end-to-end / determinism / auto-qualification tests are SIM-GATED (they drive
the WP-B ``InSilicoMachine``). The EPIG>0-on-unknown-tool assertion is a pure-numpy
regression guard for the posterior_cov unknown-tool bug (audit 2026-07-17) and needs
NO sim and NO torch, so it runs unconditionally. Nothing on this whole path imports
torch (numpy/scipy GP tier), so there is deliberately no torch gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rig_adapters.mbe import simlink

requires_sim = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

# `examples/` is not an installed package; put the repo root on sys.path so the
# rehearsal runner is importable (same idiom the runners use for sibling modules).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The runner imports rig_adapters (same sim exemption as the WP-B/WP-I tests).
from examples.run_multitool_rehearsal import (  # noqa: E402
    SMOKE,
    _min_n_cal,
    main,
    phase4b_conformal_wrap,
    run_rehearsal,
    strip_volatile,
)


@pytest.fixture(scope="module")
def smoke_payload():
    """One smoke run of the whole rehearsal, shared across the sim-gated tests."""
    return run_rehearsal(SMOKE)


# ---------------------------------------------------------------------------
# end-to-end smoke: every phase produces its verdict
# ---------------------------------------------------------------------------


@requires_sim
def test_smoke_end_to_end_all_phases(smoke_payload):
    p = smoke_payload
    assert p["meta"]["provenance"] == "physics_sim"
    assert p["meta"]["REHEARSAL"] is True and p["meta"]["headline_eligible"] is False

    # Phase 1: fleet built with 3 tools; tool signal recorded
    assert set(p["phase1_fleet"]["tools"]) == {"toolA", "toolB", "toolC"}
    assert p["phase1_fleet"]["fixed_recipe_signal"]["signal_to_noise"]["thickness_grown"] > 3.0

    # Phase 2: §5.8 LOTO domination on all 3 folds; a pooling verdict is recorded
    assert p["phase2_pooling"]["loto_zero_shot_domination"]["all_folds_dominate"] is True
    assert p["phase2_pooling"]["pooling_verdict"] in {"HELPS", "HURTS", "WASH"}

    # Phase 3: EPIG > 0 on the unknown-tool path; onboarding verdict present
    assert p["phase3_onboarding"]["epig_positive_on_unknown_tool"] is True
    assert p["phase3_onboarding"]["epig_unknown_tool_max_nats"] > 0.0
    assert p["phase3_onboarding"]["onboarding_verdict"]

    # Phase 4: reachable spec certified; unreachable -> NothingToQualify, ZERO calls
    ph4 = p["phase4_solve_qualify"]
    assert ph4["reachable"]["solve"] == "FEASIBLE"
    assert ph4["reachable"]["campaign"]["n_certified"] >= 1
    assert ph4["reachable"]["campaign"]["provenance_source"] == "physics_sim"
    assert ph4["reachable"]["campaign"]["headline_eligible"] is False
    assert ph4["unreachable"]["solve"] == "INFEASIBLE"
    assert ph4["unreachable"]["nothing_to_qualify"] is True
    assert ph4["unreachable"]["machine_calls_fired"] == 0
    assert ph4["qualification_ok"] is True


# ---------------------------------------------------------------------------
# determinism: two smoke runs are byte-identical modulo wall-clock timings
# ---------------------------------------------------------------------------


@requires_sim
def test_smoke_is_deterministic(smoke_payload):
    second = run_rehearsal(SMOKE)
    a = json.dumps(strip_volatile(smoke_payload), sort_keys=True)
    b = json.dumps(strip_volatile(second), sort_keys=True)
    assert a == b


@requires_sim
def test_main_writes_json(tmp_path):
    out = tmp_path / "rehearsal.json"
    rc = main(["--smoke", "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["verdicts"]["loto_domination_all_folds"] is True
    assert payload["verdicts"]["auto_qualification_ok"] is True
    assert payload["verdicts"]["conformal_wrap_upgraded_to_conformal_checked"] is True
    assert "phase4b_conformal_wrap" in payload


# ---------------------------------------------------------------------------
# auto-qualification fires AND charges budget (the F2-remainder loop hook)
# ---------------------------------------------------------------------------


@requires_sim
def test_auto_qualification_fires_and_charges_budget(smoke_payload):
    hook = smoke_payload["phase4_solve_qualify"]["loop_qualification_hook"]
    assert hook["qualification_fired"] is True
    # a confirmation batch actually ran and its runs were charged to the loop budget
    assert hook["confirmation_runs_charged"] > 0
    assert hook["n_queries_total"] >= hook["confirmation_runs_charged"]

    # and the DIRECT solve->campaign flow charged real confirmation runs too
    camp = smoke_payload["phase4_solve_qualify"]["reachable"]["campaign"]
    assert camp["n_machine_calls"] == camp["n_candidates"] * SMOKE.gate_n_runs


# ---------------------------------------------------------------------------
# PHASE 4b -- conformal wrap of the onboarded tool (§5.6 D4 / §13.2 upgrade)
# ---------------------------------------------------------------------------


def test_min_n_cal_matches_the_conformal_quantile_boundary():
    """Regression pin: the smallest n admitting a FINITE alpha=0.1 split-conformal
    quantile is 9 -- ceil(0.9*(n+1)) <= n first holds at n=9. If this ever drifts
    (e.g. a different alpha default), phase 4b's "minimal viable n_cal" claim would
    silently be wrong, so it is pinned as its own guard."""
    assert _min_n_cal(0.1) == 9
    # a sharper coverage target needs MORE calibration points, never fewer.
    assert _min_n_cal(0.05) > _min_n_cal(0.1)


def test_conformal_quantile_infinite_at_tiny_n_cal_synthetic():
    """Unit-scale, synthetic, UNGATED (no sim, no torch): the honest tiny-n_cal
    outcome -- a calibration split too small for alpha=0.1 gives a +inf band, and
    ConformalForwardModel.predict faithfully carries that +inf through to
    conformal_set (never silently substituting a finite guess). Mirrors the
    "honest-infinite-band" branch phase4b hits naturally at n_cal=8 (full) / 4 (smoke)."""
    from rig.calibration import ConformalForwardModel, SplitConformalCalibrator
    from rig.interfaces import PredictiveDistribution

    class _StubModel:
        """Deterministic 1-D-input, 1-output stub: predict(x) = (2x, sigma=1, 0)."""

        def predict(self, x):
            x = np.atleast_1d(np.asarray(x, dtype=float))
            single = x.ndim == 0
            xq = np.atleast_1d(x).reshape(-1)
            mean = 2.0 * xq
            sig = np.ones_like(mean)
            if single or mean.shape[0] == 1:
                mean, sig = float(mean[0]), float(sig[0])
            return PredictiveDistribution(
                mean=mean, aleatoric_sigma=sig, epistemic_sigma=0.0, conformal_set=None
            )

        def support_score(self, x):
            return 0.0

        def jacobian(self, x):
            return np.array([[2.0]])

    model = _StubModel()
    rng = np.random.default_rng(0)
    n_cal = 4  # well below the alpha=0.1 minimum of 9
    X_cal = rng.uniform(0.0, 1.0, size=(n_cal, 1))
    Y_cal = 2.0 * X_cal[:, 0] + rng.normal(0.0, 0.05, size=n_cal)

    calibrator = SplitConformalCalibrator(alpha=0.1)
    calibrator.fit(model, X_cal, Y_cal)
    kappa = calibrator.kappa()
    assert not np.all(np.isfinite(kappa)), "n_cal=4 at alpha=0.1 MUST be honestly infinite"

    wrapped = ConformalForwardModel(model, calibrator)
    cs = np.atleast_2d(wrapped.predict(np.array([0.5])).conformal_set)
    assert not np.isfinite(cs[0, 0]) or not np.isfinite(cs[0, 1])

    # and with enough calibration points (n_cal=9, the pinned minimum) the SAME
    # data-generating process gives a FINITE band.
    n_cal2 = 9
    X_cal2 = rng.uniform(0.0, 1.0, size=(n_cal2, 1))
    Y_cal2 = 2.0 * X_cal2[:, 0] + rng.normal(0.0, 0.05, size=n_cal2)
    calibrator2 = SplitConformalCalibrator(alpha=0.1)
    calibrator2.fit(model, X_cal2, Y_cal2)
    assert np.all(np.isfinite(calibrator2.kappa()))


@requires_sim
def test_phase4b_conformal_wrap_smoke(smoke_payload):
    """The recorded smoke run: natural split is honestly TOO SMALL (n_cal=4 <
    n_cal_min=9) -> +inf band -> the §13.2 gate rejects everything; the labelled
    extra-collection variant charges the runs needed to reach n_cal_min and
    upgrades to calibration_status='conformal-checked'."""
    p4b = smoke_payload["phase4b_conformal_wrap"]

    # split rule is honestly reported and self-consistent
    natural = p4b["honest_natural_split"]
    extended = p4b["extra_collection_variant"]
    assert natural["n_cal"] < p4b["n_cal_min_for_finite_quantile"]
    assert extended["n_cal"] >= p4b["n_cal_min_for_finite_quantile"]
    assert extended["extra_runs_collected"] == (
        p4b["n_cal_min_for_finite_quantile"] - natural["n_cal"]
    )

    # the raw (unwrapped) baseline is explicitly NOT conformal-checked -- proves the
    # upgrade below is a real effect of the wrap, not a status every candidate gets
    assert p4b["raw_unwrapped_reduced_data_model"]["calibration_status"] == "model-feasible"

    # honest branch: too-small n_cal -> infinite band -> gate rejects every candidate
    assert natural["finite_band"] is False
    assert natural["solve"] == "INFEASIBLE"
    assert natural["campaign"]["nothing_to_qualify"] is True

    # THE UPGRADE: enough calibration data -> finite band -> conformal-checked
    assert extended["finite_band"] is True
    assert extended["solve"] == "FEASIBLE"
    assert extended["first_candidate_calibration_status"] == "conformal-checked"
    assert p4b["upgraded_to_conformal_checked"] is True

    # band widths are recorded for both variants and the raw pessimistic interval
    widths = p4b["band_width_comparison"]
    assert widths is not None
    for key in (
        "raw_pessimistic_interval_width",
        "conformal_width_natural",
        "conformal_width_extended",
    ):
        assert set(widths[key]) == {"thickness_grown", "T_center"}


@requires_sim
def test_phase4b_wired_into_run_rehearsal(smoke_payload):
    """phase4b_conformal_wrap is reachable both via the full run_rehearsal payload
    AND directly (the function signature the tests import), and both agree with
    each other on a fresh call given the SAME campaigns/C_X/C_Y -- i.e. phase 4b
    does not depend on hidden global state beyond its declared inputs."""
    from examples.run_multitool_rehearsal import (
        SMOKE as _SMOKE,
    )
    from examples.run_multitool_rehearsal import (
        phase1_fleet,
        phase2_pooling,
        phase3_onboarding,
    )

    _, campaigns = phase1_fleet(_SMOKE)
    _, X_test, _, y_truth = phase2_pooling(_SMOKE, campaigns)
    _, warm_model, C_X, C_Y = phase3_onboarding(_SMOKE, campaigns, X_test, y_truth)
    direct = phase4b_conformal_wrap(_SMOKE, campaigns, C_X, C_Y)
    a = json.dumps(direct, sort_keys=True)
    b = json.dumps(smoke_payload["phase4b_conformal_wrap"], sort_keys=True)
    assert a == b


# ---------------------------------------------------------------------------
# EPIG > 0 on the unknown-tool path — standalone unit test, tiny fleet, no sim
# ---------------------------------------------------------------------------


def _tiny_fleet():
    """Two related tools (a genuine offset), 2 outputs — enough tool disagreement
    that the §5.8 unknown-tool inflation terms are non-trivial (mirrors
    test_multitask_gp._fit_multitool)."""
    from rig.forward import MultiToolGPForwardModel

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 1.0, size=(36, 2))
    tools = np.array(["A"] * 18 + ["B"] * 18)
    offs = np.where(tools == "A", 0.0, 0.4)
    Y = np.stack([np.sin(3.0 * X[:, 0]) + offs, 0.5 * np.cos(2.0 * X[:, 1]) + offs], axis=-1)
    Y = Y + rng.normal(0.0, 0.01, size=(36, 2))
    return MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(X, Y, tools)


def test_epig_positive_on_unknown_tool_path():
    """The §10.4 chamber-onboarding acquisition MUST report positive EPIG on the
    UNKNOWN-tool path — the exact regression the posterior_cov/predict law mismatch
    silently broke (EPIG collapsed to ~0 nats). Guards it with a tiny fleet, no sim."""
    from rig.active.acquisition import epig

    model = _tiny_fleet()
    view = model.for_tool("NEVER-SEEN")  # unknown tool -> §5.8 population fallback
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, 1.0, size=(6, 2))  # candidate pool
    X_star = rng.uniform(0.0, 1.0, size=(3, 2))  # target pre-image points (distinct)
    e = epig(view, X, X_star)
    assert e.shape == (6,)
    assert np.all(np.isfinite(e))
    assert np.all(e >= 0.0)
    assert float(e.max()) > 1e-3  # NOT collapsed to ~0


def test_epig_equals_bald_at_query_point_unknown_tool():
    """EPIG(x; {x}) == BALD(x) for ANY self-consistent joint law — the sharpest
    end-to-end check that predict and posterior_cov describe the SAME law on the
    unknown-tool branch (pre-fix: EPIG 1.01 vs BALD 19.06)."""
    from rig.active.acquisition import bald, epig

    view = _tiny_fleet().for_tool("NEVER-SEEN")
    x = np.array([[0.4, 0.6]])
    np.testing.assert_allclose(epig(view, x, x), bald(view, x), rtol=1e-6)
