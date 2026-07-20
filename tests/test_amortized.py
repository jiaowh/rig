"""WP-E / M3: amortized NPE inverse-posterior generator tests (§14.3, §14.6, D2).

The zuko conditional-neural-spline-flow ensemble that learns q(recipe | box) as the
D2 proposal service, gated by SBC/TARP (§14.6, WP-G calibration_gates). Tests pin:
the flow recovers a CALIBRATED posterior (right mean AND width), samples are feasible
BY CONSTRUCTION (box + simplex), and the SBC/TARP gate both PASSES a calibrated flow
and BITES a miscalibrated one. torch/zuko are the optional extra → module skips when
absent."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from rig.interfaces import CompositionalVariable, ContinuousVariable  # noqa: E402
from rig.inverse import AmortizedInverseGenerator, CalibrationGate  # noqa: E402


def _make_simulator(seed):
    """y = x + 0.3·noise — a genuine posterior width (0.3) for SBC to certify."""
    r = np.random.default_rng(seed)

    def sim(recipe):
        return np.array([recipe["x"] + 0.3 * r.standard_normal()])

    return sim


def _make_prior(seed):
    r = np.random.default_rng(seed)

    def prior():
        return {"x": float(r.uniform(0.0, 4.0))}

    return prior


def _train_data(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 4.0, size=(n, 1))
    Y = X + 0.3 * rng.standard_normal((n, 1))
    return X, Y


@pytest.fixture(scope="module")
def calibrated_gen():
    X, Y = _train_data(1500, seed=0)
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 4.0)],
        ["y"],
        n_members=2,
        transforms=3,
        hidden=(96, 96),
        max_epochs=180,
        region_hw=(0.1, 2.5),
        seed=0,
    )
    gen.fit(X, Y)
    return gen


@pytest.fixture(scope="module")
def two_output_gen():
    """A 2-output model (y0=x, y1=independent noise) for partial-spec context tests —
    training quality is irrelevant here, so it trains cheaply."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 4.0, size=(300, 1))
    Y = np.column_stack([X[:, 0], rng.standard_normal(300)])
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 4.0)],
        ["y0", "y1"],
        n_members=1,
        transforms=1,
        hidden=(16,),
        max_epochs=4,
        region_hw=(0.25, 2.0),
        seed=0,
    )
    gen.fit(X, Y)
    return gen


# --- posterior quality ------------------------------------------------------


def test_sample_recovers_posterior_mean_and_width(calibrated_gen):
    """The proposal for a box around y=2 concentrates near x=2 with ~the true
    posterior spread (0.3) — not a collapsed point, not garbage."""
    S = calibrated_gen.sample_array({"targets": {"y": (1.6, 2.4)}}, 400)[:, 0]
    assert abs(float(S.mean()) - 2.0) < 0.2
    assert 0.15 < float(S.std()) < 0.7  # genuine posterior width, not collapsed


def test_sample_and_log_prob_api(calibrated_gen):
    spec = {"targets": {"y": (1.6, 2.4)}}
    recipes = calibrated_gen.sample(spec, 5)
    assert len(recipes) == 5 and all("x" in r for r in recipes)
    arr = calibrated_gen.sample_array(spec, 7)
    assert arr.shape == (7, 1)
    lp = calibrated_gen.log_prob({"x": 2.0}, spec)
    assert np.isfinite(lp)
    # a recipe near the box centre should be more probable than one far away
    assert calibrated_gen.log_prob({"x": 2.0}, spec) > calibrated_gen.log_prob({"x": 3.8}, spec)


# --- constraint-by-construction (§14.4) -------------------------------------


def test_samples_are_feasible_by_construction(calibrated_gen):
    """Every proposal is inside the recipe box — the flow samples u-space and maps
    through the box-sigmoid, so a bound violation is structurally impossible."""
    S = calibrated_gen.sample_array({"targets": {"y": (0.5, 3.5)}}, 500)[:, 0]
    assert float(S.min()) >= 0.0 and float(S.max()) <= 4.0


# --- serve-time conditioning context (§14.4 / D2) ---------------------------


def test_spec_context_unconstrained_output_stays_in_trained_width(two_output_gen):
    """Regression (finding 3, MED): an output the spec leaves UNCONSTRAINED is served
    the widest trained box (width ``2·region_hw[1]``) centered at the outcome mean —
    not a 6σ, width-12 box far off the region-augmentation manifold the flow saw."""
    gen = two_output_gen
    ctx = gen._spec_context({"targets": {"y0": (1.6, 2.4)}})
    lo, hi = ctx[: gen.m], ctx[gen.m :]
    w_max = 2.0 * gen.region_hw[1]
    assert abs((hi[1] - lo[1]) - w_max) < 1e-9  # unconstrained y1: exactly the max width
    assert abs(hi[1] + lo[1]) < 1e-9  # centered at 0 (= standardized y_mean)
    assert lo[0] < hi[0]  # constrained y0 well-ordered


def test_spec_context_extreme_one_sided_not_inverted(two_output_gen):
    """Regression (finding 4, LOW): a finite one-sided bound far outside the
    achievable range must NOT invert the box (lower > upper) — the open side extends
    by the max trained width, keeping ``lower <= upper`` structural, and it still
    runs end-to-end."""
    gen = two_output_gen
    for spec in ({"targets": {"y0": {"lower": 30.0}}}, {"targets": {"y0": {"upper": -30.0}}}):
        ctx = gen._spec_context(spec)
        lo, hi = ctx[: gen.m], ctx[gen.m :]
        assert np.all(lo <= hi)  # no inverted box anywhere (was lo>hi before the fix)
        assert abs((hi[0] - lo[0]) - 2.0 * gen.region_hw[1]) < 1e-9
    arr = gen.sample_array({"targets": {"y0": {"lower": 30.0}}}, 8)
    assert arr.shape == (8, 1) and np.all(np.isfinite(arr))


def test_compositional_samples_sum_to_one():
    """A simplex recipe variable → proposals live on the simplex (components sum to
    1) by construction."""
    rng = np.random.default_rng(1)
    n = 400
    # 2-component alloy fraction ga∈(0,1); outcome y = 5·ga + noise
    ga = rng.uniform(0.1, 0.9, size=n)
    X = np.stack([ga, 1 - ga], axis=-1)  # ga, in (flat-key order)
    Y = (5.0 * ga + 0.2 * rng.standard_normal(n))[:, None]
    gen = AmortizedInverseGenerator(
        [CompositionalVariable("alloy", ("ga", "in"))],
        ["y"],
        n_members=1,
        transforms=2,
        hidden=(48,),
        max_epochs=40,
        seed=0,
    )
    gen.fit(X, Y)
    arr = gen.sample_array({"targets": {"y": (2.0, 3.0)}}, 200)  # (n, 2): ga, in
    assert np.allclose(arr.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(arr >= 0.0) and np.all(arr <= 1.0)


# --- §14.6 SBC/TARP blocking gate -------------------------------------------


def test_gate_passes_calibrated_flow(calibrated_gen):
    """A well-trained flow passes the §14.6 SBC + TARP gate (matched prior +
    simulator)."""
    gate = calibrated_gen.validate(
        _make_simulator(7),
        prior_sampler=_make_prior(11),
        n_sim=200,
        n_posterior=100,
        seed=0,
    )
    assert isinstance(gate, CalibrationGate)
    assert gate.passed and gate.sbc_passed and gate.tarp_passed


def test_gate_bites_miscalibrated_flow():
    """The gate is meaningful: an UNDERTRAINED (overconfident) flow FAILS SBC/TARP.
    'No posterior ships until SBC/TARP passes' — a failed gate must be detectable."""
    X, Y = _train_data(1500, seed=0)
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 4.0)],
        ["y"],
        n_members=2,
        transforms=3,
        hidden=(96, 96),
        max_epochs=2,  # undertrained
        region_hw=(0.1, 2.5),
        seed=0,
    )
    gen.fit(X, Y)
    gate = gen.validate(
        _make_simulator(7),
        prior_sampler=_make_prior(11),
        n_sim=200,
        n_posterior=100,
        seed=0,
    )
    assert not gate.passed  # the gate bites


def test_gate_posterior_draws_the_mixture_not_one_member():
    """Regression (finding 1, HIGH): the SBC/TARP posterior draw must SPLIT across
    every ensemble member (the shipped mixture ``sample_array`` uses), not draw all
    n_posterior from a single member — else the gate certifies a NARROWER object
    than ships and false-fails a calibrated mixture."""
    gen = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], n_members=5)

    class _TagFlow:  # each member emits its own index, so we can see who contributed
        def __init__(self, tag):
            self.tag = tag

        def __call__(self, ctx):
            return self

        def sample(self, shape):
            return torch.full((shape[0], gen.d_u), float(self.tag))

    gen._flows = [_TagFlow(k) for k in range(5)]
    draw = gen._draw_u_std_mixture(torch.zeros(2 * gen.m), 10, base_seed=0)
    assert draw.shape == (10, 1)
    assert set(np.unique(draw).astype(int).tolist()) == {0, 1, 2, 3, 4}  # all members drawn


def test_default_prior_bootstraps_training_recipes():
    """Regression (finding 2, MED): the default SBC prior (prior_sampler=None)
    BOOTSTRAPS the empirical training-u rows — the exact prior the flow trained under
    — not a moment-matched Gaussian (which mis-shapes a non-Gaussian marginal and
    silently invalidates the blocking gate)."""
    X, Y = _train_data(60, seed=0)
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 4.0)],
        ["y"],
        n_members=1,
        transforms=1,
        hidden=(16,),
        max_epochs=2,
        seed=0,
    )
    gen.fit(X, Y)
    assert gen._U_train.shape == (60, 1)
    seen: list[float] = []

    def sim(recipe):
        seen.append(recipe["x"])
        return np.array([recipe["x"]])

    gen.validate(sim, prior_sampler=None, n_sim=25, n_posterior=8, seed=1)
    train_x = X[:, 0]  # every default-prior recipe is an EXACT training row (bootstrap)
    assert all(float(np.min(np.abs(train_x - x))) < 1e-9 for x in seen)


# --- lifecycle / determinism ------------------------------------------------


def test_determinism_same_seed():
    X, Y = _train_data(600, seed=2)
    kw = dict(n_members=2, transforms=2, hidden=(48,), max_epochs=30, seed=5)
    a = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], **kw)
    b = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], **kw)
    a.fit(X, Y)
    b.fit(X, Y)
    spec = {"targets": {"y": (1.6, 2.4)}}
    assert np.allclose(a.sample_array(spec, 20), b.sample_array(spec, 20))


def test_not_fitted_raises():
    gen = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], n_members=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        gen.sample({"targets": {"y": (1.0, 2.0)}}, 3)


def test_sample_zero_returns_empty(calibrated_gen):
    """Regression (finding 5, LOW): n==0 yields a clean empty result, not an opaque
    'need at least one array to concatenate' numpy error from deep inside."""
    spec = {"targets": {"y": (1.6, 2.4)}}
    arr = calibrated_gen.sample_array(spec, 0)
    assert arr.shape == (0, 1)
    assert calibrated_gen.sample(spec, 0) == []


def test_sample_negative_raises(calibrated_gen):
    with pytest.raises(ValueError, match="n must be"):
        calibrated_gen.sample_array({"targets": {"y": (1.6, 2.4)}}, -1)


def test_sample_and_sample_array_are_consistent(calibrated_gen):
    """Regression (finding 6, LOW): sample() and sample_array() share ONE draw path,
    so the dicts and the matrix are the SAME recipes — no redundant inverse->forward
    round-trip and no boundary drift between them.

    NB (audit 2026-07-17) this asserts the consistency of ONE draw, via the shared
    `_draw_recipes`. It used to call `sample()` and `sample_array()` separately and
    compare them, which only passed because the sampler re-used a frozen seed on
    every call — i.e. the test was pinning the stream-advance bug in place. Two
    calls SHOULD now differ (see test_repeated_calls_draw_fresh_samples).
    """
    spec = {"targets": {"y": (1.6, 2.4)}}
    recipes, arr = calibrated_gen._draw_recipes(spec, 12)
    from_dicts = np.array([[r[k] for k in calibrated_gen._flat_keys] for r in recipes])
    assert np.array_equal(from_dicts, arr)


def test_repeated_calls_draw_fresh_samples(calibrated_gen):
    """Regression (audit 2026-07-17, finding 6): a posterior SAMPLER must advance its
    stream. `_draw_recipes` used to pass `self.seed` on every call, so repeated calls
    returned bit-identical rows — `2× sample(spec, 3)` was 3 duplicated pairs, and any
    MC estimate or pooled proposal set built by looping silently collapsed onto copies.
    """
    spec = {"targets": {"y": (1.6, 2.4)}}
    a = calibrated_gen.sample_array(spec, 6)
    b = calibrated_gen.sample_array(spec, 6)
    assert not np.allclose(a, b), "repeated sample_array calls returned identical draws"


def test_same_seed_replays_the_whole_draw_sequence():
    """The §13.4 determinism that matters: stream advance must not cost reproducibility.
    A fresh generator with the same seed replays the identical SEQUENCE of draws."""
    X, Y = _train_data(600, seed=2)
    kw = dict(n_members=2, transforms=2, hidden=(48,), max_epochs=30, seed=5)
    spec = {"targets": {"y": (1.6, 2.4)}}
    a = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], **kw)
    b = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], **kw)
    a.fit(X, Y)
    b.fit(X, Y)
    for _ in range(3):
        assert np.allclose(a.sample_array(spec, 5), b.sample_array(spec, 5))


def test_member_counts_are_an_even_mixture_for_every_n(calibrated_gen):
    """Regression (audit 2026-07-17): the small-n mirror of the Session-6 HIGH gate bug.

    `_member_counts` used to hand the divmod remainder to the LOW-INDEX members every
    time, so the shipped draw was a skewed — and for n < K truncated — sub-mixture while
    `log_prob` and the §14.6 gate both assume the even one. n=3,K=5 gave [1,1,1,0,0]:
    members 3 and 4 could never be drawn. AmortizedRefiner's default n_proposals=8 sat
    exactly on the skewed case (weights .25/.25/.25/.125/.125).
    """
    gen = AmortizedInverseGenerator([ContinuousVariable("x", 0.0, 4.0)], ["y"], n_members=5, seed=0)
    for n in (3, 8):
        totals = np.zeros(5)
        for s in range(2000):
            totals += gen._member_counts(n, np.random.default_rng(s))
        weights = totals / totals.sum()
        assert np.abs(weights - 0.2).max() < 0.02, f"n={n} skewed mixture: {weights}"
    # K | n keeps the gate's exact stratification (and its lower variance)
    assert gen._member_counts(100, np.random.default_rng(0)) == [20] * 5


def test_log_prob_is_a_normalized_recipe_space_density():
    """Regression (audit 2026-07-17, finding 1): log_prob returned the u-space density
    while advertising `log q(recipe | box)`, excusing it as "a monotone reparam". A
    monotone reparam preserves the ordering of the VARIABLE, not the DENSITY: the
    box-sigmoid |du/dx| was never applied, so the values did not integrate to 1 (3.36
    over this box) and re-ordered the posterior. Anything ranking proposals by density
    was silently wrong.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 10.0, size=(600, 1))
    Y = (0.5 * X[:, 0] + rng.normal(0.0, 0.05, 600))[:, None]
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x0", 0.0, 10.0)],
        ["y"],
        n_members=2,
        transforms=2,
        hidden=(48,),
        max_epochs=40,
        seed=0,
    )
    gen.fit(X, Y)
    spec = {"targets": {"y": (2.4, 2.6)}}
    grid = np.linspace(1e-4, 10.0 - 1e-4, 4000)
    dens = np.exp([gen.log_prob({"x0": float(v)}, spec) for v in grid])
    assert np.trapezoid(dens, grid) == pytest.approx(1.0, abs=0.02)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_path_runs():
    X, Y = _train_data(400, seed=3)
    gen = AmortizedInverseGenerator(
        [ContinuousVariable("x", 0.0, 4.0)],
        ["y"],
        n_members=1,
        transforms=2,
        hidden=(48,),
        max_epochs=20,
        seed=0,
        device="cuda",
    )
    gen.fit(X, Y)
    arr = gen.sample_array({"targets": {"y": (1.6, 2.4)}}, 16)
    assert arr.shape == (16, 1) and np.all(np.isfinite(arr))
