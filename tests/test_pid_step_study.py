"""Tests for the PID decaying-step side study
(examples/real_data/empa_hipims/run_pid_step_study.py).

The runner module is imported by PATH (examples/ is not a package -- same
trick as tests/test_empa_ingest.py and tests/test_empa_pooled.py). Covers:

  (a) the FIDELITY GATE on one real campaign (sim-free, local Empa CSV/spec
      files -- the test_empa_ingest.py pattern): this study's step="fixed"
      reproduction must exactly match results/m1_empa.json's recorded PID
      numbers, checked via the runner's own ``check_fidelity`` on BOTH the
      imported ``pid_eval`` reference and the runner's own ``pid_stream``
      tracer;
  (b) ``volatility_stats`` on hand-built traces (deterministic, no model);
  (c) decaying-vs-fixed late-stream volatility direction on a synthetic
      stable (exchangeable) stream, adapting tests/test_pid.py's pattern
      through the runner's own ``pid_stream`` wrapper;
  (d) a red-proof of the fidelity gate: perturb one covered-indicator in a
      synthetic recorded/candidate pair, confirm ``check_fidelity`` flags it,
      then restore and confirm it passes again -- all in-memory, no repo
      files touched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from rig.interfaces import PredictiveDistribution

REPO = Path(__file__).resolve().parents[1]
EX = REPO / "examples" / "real_data" / "empa_hipims"


def _load_runner():
    """Import run_pid_step_study by path. It puts its own directory on
    sys.path (for `from run_m1_empa import ...`) at exec time, so a plain
    spec-exec is enough; register in sys.modules first (the PEP-563 /
    dataclass-resolution trap prepare_empa.py can trip, same guard as the
    other empa test files)."""
    module_spec = importlib.util.spec_from_file_location(
        "run_pid_step_study", EX / "run_pid_step_study.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


runner = _load_runner()


# ---------------------------------------------------------------------------
# (a) fidelity gate on one real campaign -- fixed-step must reproduce the
# recorded results/m1_empa.json exactly. Fastest campaign (401 rows, ~2.5s/
# split GP fit) so the test stays quick while still exercising real ingest +
# real GP fit + the real online protocol, not a stub.
# ---------------------------------------------------------------------------

FIDELITY_SLUG = "al_250w_low_duty"


def test_fixed_step_reproduces_recorded_m1_empa_json():
    recorded = runner.load_recorded()
    idx = next(i for i, c in enumerate(runner.CAMPAIGNS) if c.slug == FIDELITY_SLUG)
    campaign = runner.CAMPAIGNS[idx]

    cell = runner.run_cell(campaign, idx, gp_restarts=runner.GP_RESTARTS)
    assert set(cell) == {"temporal", "random"}

    for split_name, data in cell.items():
        # reference (a): the recorded runner's OWN pid_eval, imported directly
        mism_a = runner.check_fidelity(
            FIDELITY_SLUG, split_name, recorded, data["recorded_form_fixed"]
        )
        assert mism_a == [], (split_name, "pid_eval", mism_a)
        # reference (b): this study's own tracer at step="fixed"
        mism_b = runner.check_fidelity(FIDELITY_SLUG, split_name, recorded, data["fixed"])
        assert mism_b == [], (split_name, "pid_stream", mism_b)
        # the anti-infinite-width contract, on real data
        for po in data["fixed"]["per_output"].values():
            assert po["n_infinite_width"] == 0
        for po in data["decaying"]["per_output"].values():
            assert po["n_infinite_width"] == 0


# ---------------------------------------------------------------------------
# (b) volatility_stats: deterministic, correct on a hand-built trace
# ---------------------------------------------------------------------------


def test_volatility_stats_hand_built_values():
    # last 4 of 8 points oscillate 1,2,1,2: mean 1.5, population std 0.5;
    # |delta| sequence [1,1,1] -> mean 1.0. Worked by hand, not by the impl.
    trace = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0])
    std, mad = runner.volatility_stats(trace, tail_frac=0.5)
    assert std == pytest.approx(0.5)
    assert mad == pytest.approx(1.0)


def test_volatility_stats_constant_tail_is_zero():
    trace = np.full(20, 3.5)
    std, mad = runner.volatility_stats(trace, tail_frac=0.5)
    assert std == 0.0
    assert mad == 0.0


def test_volatility_stats_single_point_tail_is_zero_not_nan():
    trace = np.array([7.0])
    std, mad = runner.volatility_stats(trace, tail_frac=1.0)
    assert std == 0.0
    assert mad == 0.0  # a single point has no step-to-step delta -> 0, not NaN


def test_volatility_stats_whole_trace_frac_1():
    trace = np.array([1.0, 3.0, 1.0, 3.0])
    std, mad = runner.volatility_stats(trace, tail_frac=1.0)
    assert std == pytest.approx(1.0)  # mean 2, deviations +-1 -> population std 1
    assert mad == pytest.approx(2.0)  # |diff| = [2,2,2] -> mean 2


# ---------------------------------------------------------------------------
# (c) decaying-vs-fixed volatility direction on a synthetic stable stream,
# through the runner's own pid_stream (adapts tests/test_pid.py's pattern).
# ---------------------------------------------------------------------------


class _ConstModel:
    """Fixed-prediction stub (mu, sigma) -- exchangeable stream by construction
    (no drift), single output. Mirrors tests/test_pid.py's ``_Const``."""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0) -> None:
        self.mu, self.sigma = mu, sigma

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        x = np.asarray(x, dtype=float)
        shape = (1,) if x.ndim == 1 else (x.shape[0], 1)
        return PredictiveDistribution(
            mean=np.full(shape, self.mu),
            aleatoric_sigma=np.full(shape, self.sigma),
            epistemic_sigma=np.zeros(shape),
            conformal_set=None,
        )


def _synthetic_cal_and_stream(seed: int, n_cal: int = 100, n_test: int = 1500):
    rng = np.random.default_rng(seed)
    Xc = np.zeros((n_cal, 1))
    Yc = rng.normal(0.0, 1.0, size=(n_cal, 1))
    Xt = np.zeros((n_test, 1))
    Yt = rng.normal(0.0, 1.0, size=(n_test, 1))
    return Xc, Yc, Xt, Yt


def test_decaying_reduces_late_stream_volatility_on_stable_stream():
    model = _ConstModel(mu=0.0, sigma=1.0)
    Xc, Yc, Xt, Yt = _synthetic_cal_and_stream(seed=42)

    fixed = runner.pid_stream(model, Xc, Yc, Xt, Yt, ["y"], ["unit"], "fixed")
    decaying = runner.pid_stream(model, Xc, Yc, Xt, Yt, ["y"], ["unit"], "decaying")

    # coverage holds for both on the exchangeable stream (the pre-stated
    # hypothesis's "without losing coverage" half) -- the same loose
    # coverage-range convention tests/test_pid.py uses for this same
    # comparison (a strict nominal-in-CI check is noisier seed to seed at
    # n=1500 for the online per-step trials, for BOTH step modes).
    assert 0.85 <= fixed["pooled"]["picp"] <= 0.95, fixed["pooled"]["picp"]
    assert 0.85 <= decaying["pooled"]["picp"] <= 0.95, decaying["pooled"]["picp"]

    # decaying's late-stream threshold is materially quieter than fixed's
    f_std = fixed["per_output"]["y"]["q_late_stream_std"]
    d_std = decaying["per_output"]["y"]["q_late_stream_std"]
    f_mad = fixed["per_output"]["y"]["q_late_stream_mean_abs_delta"]
    d_mad = decaying["per_output"]["y"]["q_late_stream_mean_abs_delta"]
    assert d_std < f_std, (d_std, f_std)
    assert d_mad < f_mad, (d_mad, f_mad)
    assert d_std < 0.5 * f_std, (d_std, f_std)


def test_pid_stream_is_deterministic():
    model = _ConstModel(mu=0.3, sigma=1.2)
    Xc, Yc, Xt, Yt = _synthetic_cal_and_stream(seed=7, n_cal=50, n_test=300)
    r1 = runner.pid_stream(model, Xc, Yc, Xt, Yt, ["y"], ["unit"], "decaying")
    r2 = runner.pid_stream(model, Xc, Yc, Xt, Yt, ["y"], ["unit"], "decaying")
    assert r1 == r2


# ---------------------------------------------------------------------------
# (d) red-proof the fidelity gate: perturb one covered-indicator -> fails;
# restore -> passes. Fully synthetic (no repo files touched).
# ---------------------------------------------------------------------------


def _fake_pid_block(k: int, n: int) -> dict:
    picp = k / n
    return {
        "pooled": {"k_covered": k, "n_trials": n, "picp": picp},
        "per_output": {"y": {"k_covered": k, "n_test": n, "picp": picp}},
    }


def test_fidelity_gate_red_proof():
    recorded = {"campaigns": {"slug": {"splits": {"temporal": {"pid": _fake_pid_block(90, 100)}}}}}

    # GREEN: matching candidate -> no mismatches
    matching = _fake_pid_block(90, 100)
    assert runner.check_fidelity("slug", "temporal", recorded, matching) == []

    # RED: perturb one covered-indicator (one hit flipped to a miss)
    perturbed = _fake_pid_block(89, 100)
    mismatches = runner.check_fidelity("slug", "temporal", recorded, perturbed)
    assert mismatches != [], "red-proof failed: the gate did not notice the perturbation"
    fields = {m[0] for m in mismatches}
    assert "pooled.k_covered" in fields
    assert "y.k_covered" in fields

    # RESTORE: back to the matching value -> passes again
    restored = _fake_pid_block(90, 100)
    assert runner.check_fidelity("slug", "temporal", recorded, restored) == []
