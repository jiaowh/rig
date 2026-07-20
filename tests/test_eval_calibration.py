"""WP-G: SBC + TARP posterior-recovery gates (implementation-plan §12.2)."""

from __future__ import annotations

import numpy as np
import pytest

from rig.eval.calibration_gates import (
    sbc_ranks,
    sbc_test,
    tarp_credibilities,
    tarp_test,
    uniformity_test,
)

L, M, DIM = 300, 200, 2


def _calibrated(seed=0):
    # theta_true and posterior samples all iid N(0,1) => marginally consistent,
    # so SBC ranks are Uniform{0..M} (the canonical calibrated case).
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((L, DIM))
    post = rng.standard_normal((L, M, DIM))
    return theta, post


def _overconfident(seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal((L, DIM))
    post = 0.3 * rng.standard_normal((L, M, DIM))  # too narrow
    return theta, post


def test_uniformity_test_calibration_over_seeds():
    # a calibration gate rejects a truly-uniform sample at ~ (1-confidence) by
    # definition, so validate the PASS-RATE over seeds, not a single draw.
    passes = sum(
        uniformity_test(np.random.default_rng(s).random(500), seed=1000 + s).passed
        for s in range(40)
    )
    assert passes / 40 >= 0.85  # expect ~0.95
    # a skewed sample is rejected essentially always.
    skew_passes = sum(
        uniformity_test(np.random.default_rng(s).random(500) ** 3, seed=1000 + s).passed
        for s in range(20)
    )
    assert skew_passes == 0


def test_sbc_ranks_shape_and_range():
    theta, post = _calibrated()
    ranks = sbc_ranks(theta, post)
    assert ranks.shape == (L, DIM)
    assert ranks.min() >= 0 and ranks.max() <= M


def _sbc_pass_rate(make_data, n_seeds=20, n_sim=800):
    passes = 0
    for s in range(n_seeds):
        theta, post = make_data(seed=s)
        ranks = sbc_ranks(theta, post)
        res = sbc_test(ranks, M, n_sim=n_sim, seed=500 + s)
        passes += all(r.passed for r in res)
    return passes / n_seeds


def test_sbc_calibration_over_seeds():
    # calibrated: both dims pass at 95% ⇒ joint ~0.90; overconfident: ~never.
    assert _sbc_pass_rate(_calibrated) >= 0.7
    assert _sbc_pass_rate(_overconfident) <= 0.05


def _tarp_pass_rate(make_data, n_seeds=20, n_sim=800):
    passes = 0
    for s in range(n_seeds):
        theta, post = make_data(seed=s)
        passes += tarp_test(theta, post, n_sim=n_sim, seed=500 + s).passed
    return passes / n_seeds


def test_tarp_calibration_over_seeds():
    assert _tarp_pass_rate(_calibrated) >= 0.7
    assert _tarp_pass_rate(_overconfident) <= 0.05


def test_tarp_calibration_error_small_when_calibrated_large_when_not():
    theta_c, post_c = _calibrated(seed=2)
    assert tarp_test(theta_c, post_c, seed=3).max_calibration_error < 0.15
    theta_o, post_o = _overconfident(seed=2)
    assert tarp_test(theta_o, post_o, seed=3).max_calibration_error > 0.2


def test_tarp_coarse_posterior_not_over_rejected():
    # regression guard for the discreteness fix (review finding): a calibrated
    # but COARSE (small-M) posterior — the advertised GP-tier candidate-set feed
    # — must NOT be systematically rejected. Without the (K+U)/(M+1) continuity
    # correction the pass-rate here was ~0.0 (100% false-reject).
    def cal(seed, M):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((300, DIM)), rng.standard_normal((300, M, DIM))

    for M in (5, 10):
        passes = sum(tarp_test(*cal(s, M), n_sim=600, seed=1000 + s).passed for s in range(20))
        assert passes / 20 >= 0.7, (M, passes / 20)


def test_sbc_coarse_posterior_not_over_rejected():
    # audit C4: symmetric to test_tarp_coarse_posterior_not_over_rejected. The
    # (K+U)/(M+1) continuity correction in sbc_test is load-bearing at small M
    # (the advertised coarse GP-tier candidate-set feed); all other SBC tests use
    # M=200 where it's inert. Without it, a calibrated coarse posterior is
    # systematically rejected (pass-rate ~0.0).
    def cal(seed, m):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((300, DIM)), rng.standard_normal((300, m, DIM))

    for m in (5, 10):
        passes = 0
        for s in range(20):
            theta, post = cal(s, m)
            ranks = sbc_ranks(theta, post)
            res = sbc_test(ranks, m, n_sim=600, seed=1000 + s)
            passes += all(r.passed for r in res)
        assert passes / 20 >= 0.6, (m, passes / 20)


def test_tarp_credibilities_uniform_when_calibrated():
    theta, post = _calibrated(seed=5)
    cred = tarp_credibilities(theta, post, seed=6)
    assert cred.shape == (L,)
    assert 0.0 <= cred.min() and cred.max() <= 1.0
    assert abs(float(np.mean(cred)) - 0.5) < 0.1  # uniform mean ~ 0.5


def test_uniformity_rejects_out_of_range():
    with pytest.raises(ValueError):
        uniformity_test(np.array([0.5, 1.5]))


def test_sbc_1d_input_accepted():
    # 1-D theta/posterior are promoted to a single param column and run.
    rng = np.random.default_rng(7)
    theta = rng.standard_normal(L)  # 1-D
    post = rng.standard_normal((L, M))  # 1-D params
    ranks = sbc_ranks(theta, post)
    assert ranks.shape == (L, 1)
    res = sbc_test(ranks, M, seed=1)
    assert len(res) == 1 and 0.0 <= res[0].statistic <= 1.0
