"""WP-E: ensemble distribution distillation tests (implementation-plan §5.7 option A).

The `/invert` serving path: one net distilled from the K-member deep ensemble,
same canonical §3.2 surface. The contract tests mirror ``test_ensemble.py`` so
the tier is provably interchangeable, and the rest of the file exists for ONE
reason — the thing a naive distillation destroys is the ALEATORIC/EPISTEMIC
SPLIT, and a student matching only the mean and the TOTAL variance would pass a
casual "it tracks the teacher" test while silently breaking §8 pessimism (which
displaces by ``z_epi·σ_epi``) and §9 BALD/EPIG (functions of the σ_epi/σ_ale
RATIO). So the split is tested directly and from both sides: each σ is tracked
against the teacher's own value, the ratio is tracked, and the OOD asymmetry
(epistemic explodes, aleatoric does not) is asserted as a pair.

Every tolerance below is a MEASURED figure with headroom, quoted in the test.
torch is an optional extra, so the whole module skips without it.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip the module if the [torch] extra is absent

from rig.calibration.conformal import SplitConformalCalibrator  # noqa: E402
from rig.forward import (  # noqa: E402
    DeepEnsembleForwardModel,
    DistilledForwardModel,
    distill_ensemble,
)
from rig.interfaces import (  # noqa: E402
    ContinuousVariable,
    ForwardModel,
    PredictiveDistribution,
)
from rig.inverse import PessimisticInverseSolver  # noqa: E402

RNG_SEED = 20260717

# x=8.5 is past the training range [0, 2π] but INSIDE the distilled transfer box
# — so epistemic inflation there is earned by distillation, not by the
# out-of-box guard (test_ood_guard_is_a_noop_inside_the_transfer_box pins that).
X_OOD_IN_BOX = 8.5
# far outside the transfer box: the ood_inflation guard rail's territory
X_FAR_OOD = 400.0


def _sin_data(n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    y = np.sin(X[:, 0]) + noise * rng.standard_normal(n)
    return X, y[:, None]


def _rel_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b) / np.maximum(np.abs(a), 1e-12)


@pytest.fixture(scope="module")
def distilled():
    """The §5.7 pair: a K=3 deep-ensemble teacher and its distilled student.

    The student takes the SHIPPED n_transfer/max_epochs defaults on purpose — the
    fidelity bounds below are what those defaults buy, so pinning them here would
    let the defaults regress without a test noticing. (Only ``width`` is cut, to
    match the small teacher; it was measured not to matter.)
    """
    X, Y = _sin_data(n=300, noise=0.1, seed=RNG_SEED)
    teacher = DeepEnsembleForwardModel(
        n_members=3, width=64, n_blocks=2, d_rff=128, max_epochs=200, seed=0
    ).fit(X, Y)
    student = distill_ensemble(teacher, X, seed=0, width=64, n_blocks=2)
    return teacher, student, X, Y


@pytest.fixture(scope="module")
def x_in_dist():
    return np.linspace(0.2, 2.0 * np.pi - 0.2, 100)[:, None]


# --- protocol / contract (mirrors test_ensemble.py) ------------------------


def test_is_forward_model(distilled):
    _, student, _, _ = distilled
    assert isinstance(student, ForwardModel)


def test_predict_returns_canonical_distribution(distilled):
    _, student, _, _ = distilled
    pred = student.predict(np.array([1.0]))
    assert isinstance(pred, PredictiveDistribution)
    # canonical field ORDER (§3.2): mean, aleatoric_sigma, epistemic_sigma, conformal_set
    assert list(vars(pred).keys()) == [
        "mean",
        "aleatoric_sigma",
        "epistemic_sigma",
        "conformal_set",
    ]
    assert pred.conformal_set is None  # filled only by the §5.6 wrapper


def test_shape_contract(distilled):
    """(d,) -> (m,); (n,d) -> (n,m); jacobian (m,d); support float / (n,)."""
    _, student, _, _ = distilled
    single = student.predict(np.array([1.0]))
    assert single.mean.shape == (1,)
    assert single.aleatoric_sigma.shape == (1,)
    assert single.epistemic_sigma.shape == (1,)
    batch = student.predict(np.linspace(0.5, 5.5, 7)[:, None])
    assert batch.mean.shape == (7, 1)
    assert batch.aleatoric_sigma.shape == (7, 1)
    assert batch.epistemic_sigma.shape == (7, 1)
    assert student.jacobian(np.array([2.0])).shape == (1, 1)
    assert isinstance(student.support_score(np.array([1.0])), float)
    assert student.support_score(np.linspace(0.5, 5.5, 5)[:, None]).shape == (5,)


def test_jacobian_rejects_batch(distilled):
    _, student, _, _ = distilled
    with pytest.raises(ValueError, match="single point"):
        student.jacobian(np.array([[1.0], [2.0]]))


# --- THE crux: the two-component split survives distillation ---------------


def test_student_tracks_teacher_mean_and_both_sigmas(distilled, x_in_dist):
    """§5.7: the student must reproduce the teacher's mean AND BOTH σ's — not the
    mean and a total. Tolerances are measured figures (see the module docstring)
    with headroom; the mean is scored against the output's own scale."""
    teacher, student, _, Y = distilled
    t, s = teacher.predict(x_in_dist), student.predict(x_in_dist)

    # mean: max abs error < 5% of the output std (MEASURED 0.14%)
    mean_err = float(np.abs(t.mean - s.mean).max())
    assert mean_err < 0.05 * float(Y.std()), mean_err
    # aleatoric: median relative error < 2% (MEASURED 0.003%)
    ale_err = float(np.median(_rel_err(t.aleatoric_sigma, s.aleatoric_sigma)))
    assert ale_err < 0.02, ale_err
    # epistemic: the hardest channel, and the one that pays for n_transfer —
    # median relative error < 10% (MEASURED 0.98% at the default 8192 transfer
    # points; it was 21.7% at 4096, which is why the default moved. This bound is
    # deliberately tight enough that the 4096 setting would FAIL it).
    epi_err = float(np.median(_rel_err(t.epistemic_sigma, s.epistemic_sigma)))
    assert epi_err < 0.10, epi_err


def test_epistemic_aleatoric_ratio_tracks_teacher(distilled, x_in_dist):
    """THE guard against a collapsed split. §9's BALD is 0.5·log(1+σ_epi²/σ_ale²)
    and §8 divides by σ_ale after displacing by z_epi·σ_epi: both read the RATIO.
    A student that matched the mean + TOTAL variance and split it by any fixed
    rule would still track the total here and fail this."""
    teacher, student, _, _ = distilled
    x = np.vstack([x_in_dist, np.array([[7.0], [X_OOD_IN_BOX]])])  # in-dist + near-OOD
    t, s = teacher.predict(x), student.predict(x)
    r_t = (t.epistemic_sigma / t.aleatoric_sigma).ravel()
    r_s = (s.epistemic_sigma / s.aleatoric_sigma).ravel()
    # the ratio really does span ~0.13 (in-dist) to ~7.0 (OOD) here — a collapsed
    # split would pin it to a constant, so the span is what makes this test bite
    assert r_t.max() / r_t.min() > 10.0, (r_t.min(), r_t.max())
    # MEASURED: median relative error 0.96%, Pearson on log-ratio 0.9989
    assert float(np.median(_rel_err(r_t, r_s))) < 0.10
    assert float(np.corrcoef(np.log(r_t), np.log(r_s))[0, 1]) > 0.95


def test_ood_epistemic_inflation_survives_distillation(distilled, x_in_dist):
    """§5.9 invariant 1 through the distillation — the property most likely to be
    lost. The student must still blow epistemic up off the training manifold, and
    it must be EARNED: X_OOD_IN_BOX sits inside the transfer box, where the
    out-of-box guard contributes exactly zero."""
    teacher, student, _, _ = distilled
    x_ood = np.array([[X_OOD_IN_BOX]])
    assert student.in_transfer_box(x_ood[0]) is True  # the guard is inactive here

    t_in = float(teacher.predict(x_in_dist).epistemic_sigma.mean())
    s_in = float(student.predict(x_in_dist).epistemic_sigma.mean())
    t_ood = float(teacher.predict(x_ood).epistemic_sigma[0, 0])
    s_ood = float(student.predict(x_ood).epistemic_sigma[0, 0])

    # the teacher has the property (guards the fixture, not the student):
    # MEASURED 13.7x
    assert t_ood > 3.0 * t_in, (t_ood, t_in)
    # ... and so does the student: MEASURED 13.6x, i.e. the inflation factor is
    # reproduced to 0.3%, not merely "present"
    assert s_ood > 3.0 * s_in, (s_ood, s_in)
    assert 0.75 < (s_ood / s_in) / (t_ood / t_in) < 1.33, (s_ood / s_in, t_ood / t_in)


def test_aleatoric_does_not_inflate_ood(distilled, x_in_dist):
    """The other half of the split, and the sharpest single collapse detector: OOD
    the teacher's EPISTEMIC explodes while its ALEATORIC stays put (process noise
    does not care that we have no data there). A student that folded the two
    together would inflate both and fail this while still passing the epistemic
    test above."""
    teacher, student, _, _ = distilled
    x_ood = np.array([[X_OOD_IN_BOX]])
    factors = {}
    for name, model in (("teacher", teacher), ("student", student)):
        p_in, p_ood = model.predict(x_in_dist), model.predict(x_ood)
        ale_f = float(p_ood.aleatoric_sigma[0, 0]) / float(p_in.aleatoric_sigma.mean())
        epi_f = float(p_ood.epistemic_sigma[0, 0]) / float(p_in.epistemic_sigma.mean())
        # MEASURED: aleatoric 0.49x (it does not inflate), epistemic 13.7x
        assert ale_f < 2.0, (name, ale_f)
        assert epi_f / ale_f > 5.0, (name, epi_f, ale_f)  # MEASURED ~27.9
        factors[name] = (ale_f, epi_f)
    # and the student reproduces the teacher's ASYMMETRY, not just its direction
    for j in (0, 1):
        assert factors["student"][j] == pytest.approx(factors["teacher"][j], rel=0.25)


def test_ood_guard_is_a_noop_inside_the_transfer_box(distilled, x_in_dist):
    """Pins the honesty of the two tests above: ``ood_inflation`` must contribute
    EXACTLY nothing inside the transfer box, so in-box OOD inflation is distilled
    behaviour and not the guard rail manufacturing a pass."""
    _, student, _, _ = distilled
    x = np.vstack([x_in_dist, np.array([[X_OOD_IN_BOX]])])
    Xs = (x - student._x_mean) / student._x_scale
    assert np.all(student._box_excess(Xs) == 0.0)
    assert np.all(student.in_transfer_box(x))


# --- the out-of-box guard rail (honest extrapolation, not a distilled value) --


def test_predict_off_box_is_exactly_edge_plus_guard(distilled):
    """REGRESSION GUARD on a real defect this module hit, and the exact statement
    of the shipped off-box law: heads evaluated at the NEAREST IN-BOX point, plus
    ``ood_inflation·d`` on the log-variance. Deterministic — it pins the design,
    not the net's arbitrary extrapolation.

    WHY: the first cut evaluated the heads wherever it was asked. At x=400 (box
    [-6.2, 12.5]) they returned σ_epi = 0.2x their IN-DISTRIBUTION value, σ_ale =
    5e-14 and a mean of 26.7 on a sine — a confident, precise, wrong answer in a
    far-OOD hole, which is the §8.2 failure mode itself. Delete the clip in
    _eval_inputs and this test goes red on the mean alone.
    """
    _, student, _, _ = distilled
    far = np.array([[X_FAR_OOD]])
    p_far = student.predict(far)
    p_edge = student.predict(student.transfer_box[1][None, :])
    d = float(student._box_excess((far - student._x_mean) / student._x_scale)[0])
    assert d > 0.0

    # mean and aleatoric are the box-edge values, untouched by the guard
    assert float(p_far.mean[0, 0]) == pytest.approx(float(p_edge.mean[0, 0]), rel=1e-5)
    assert float(p_far.aleatoric_sigma[0, 0]) == pytest.approx(
        float(p_edge.aleatoric_sigma[0, 0]), rel=1e-5
    )
    # epistemic is the box-edge value inflated by exactly exp(gain·d/2)
    expected = float(p_edge.epistemic_sigma[0, 0]) * np.exp(0.5 * student.ood_inflation * d)
    assert float(p_far.epistemic_sigma[0, 0]) == pytest.approx(expected, rel=1e-5)
    # support is the box-edge score penalized by exactly gain·d (clipped at 0)
    assert student.support_score(far[0]) == pytest.approx(
        min(student.support_score(student.transfer_box[1]) - student.ood_inflation * d, 0.0),
        rel=1e-5,
    )


def test_far_outside_box_guard_inflates_and_rejects(distilled, x_in_dist):
    """Outside the transfer box the shipped law is "the nearest distilled point +
    a monotone ignorance flag": epistemic blows up, support collapses (so §8.2
    fails closed), the mean stays sane, and everything stays finite."""
    _, student, _, _ = distilled
    far = np.array([[X_FAR_OOD]])
    assert student.in_transfer_box(far[0]) is False
    p_in = student.predict(x_in_dist)
    s_in = float(p_in.epistemic_sigma.mean())
    p_far = student.predict(far)
    assert float(p_far.epistemic_sigma[0, 0]) > 100.0 * s_in
    assert np.all(np.isfinite(p_far.epistemic_sigma))  # capped, never inf/nan
    assert np.all(np.isfinite(p_far.aleatoric_sigma))
    assert np.all(np.isfinite(p_far.mean))
    # aleatoric must NOT collapse to zero out here (the raw head's value did)
    assert float(p_far.aleatoric_sigma[0, 0]) > 0.1 * float(p_in.aleatoric_sigma.mean())
    assert student.support_score(far[0]) < student.support_score(np.array([1.0]))


def test_out_of_box_epistemic_is_monotone_in_distance(distilled):
    """The guard TERM's defining property: further out => strictly more epistemic,
    strictly less support.

    Scope, stated honestly: this tests the ``ood_inflation`` term, NOT the clip.
    Verified by injection — deleting the clip in ``_eval_inputs`` leaves this test
    GREEN (the guard term dominates over these modest distances), so it must not
    be read as a clip regression guard. ``test_predict_off_box_is_exactly_edge_
    plus_guard`` is the one that bites on the clip.
    """
    _, student, _, _ = distilled
    hi = student.transfer_box[1][0]
    xs = hi + np.array([0.5, 2.0, 6.0, 15.0])
    epi = student.predict(xs[:, None]).epistemic_sigma[:, 0]
    assert np.all(np.diff(epi) > 0.0), epi
    sup = student.support_score(xs[:, None])
    assert np.all(np.diff(sup) < 0.0), sup


def test_box_excess_zero_inside_monotone_outside_capped(distilled):
    """The guard term's three stated properties, tested directly."""
    from rig.forward.distill import _MAX_BOX_EXCESS

    _, student, _, _ = distilled
    lo, hi = student.transfer_box
    inside = np.linspace(lo[0], hi[0], 22)[1:-1, None]  # strictly inside
    Xs_in = (inside - student._x_mean) / student._x_scale
    assert np.all(student._box_excess(Xs_in) == 0.0)  # EXACTLY zero, not merely small
    assert np.all(student._eval_inputs(Xs_in)[0] == Xs_in)  # the clip is a no-op inside
    outside = hi[0] + np.array([0.1, 1.0, 5.0, 50.0])
    Xs_out = (outside[:, None] - student._x_mean) / student._x_scale
    d = student._box_excess(Xs_out)
    assert np.all(np.diff(d) > 0.0)  # monotone in distance
    assert np.all(d <= _MAX_BOX_EXCESS)  # capped -> exp() stays finite
    absurd = (np.array([[1e9]]) - student._x_mean) / student._x_scale
    assert float(student._box_excess(absurd)[0]) == _MAX_BOX_EXCESS


def test_box_excess_fires_below_the_box_symmetrically_not_only_above(distilled):
    """The out-of-box guard is SYMMETRIC: a point equally far BELOW the transfer box
    must get the same excess as one above it. Every other OOD test probes only the
    upper face (``hi + delta``); dropping the lower-face term
    ``np.maximum(self._box_lo - Xs, 0.0)`` in ``_box_excess`` leaves them all green
    while a far-OOD point BELOW the box gets ZERO guard inflation — the §8.2
    fail-closed gate silently not firing in the unsafe direction. Verified: that
    one-line drop turns this red while the whole rest of the suite stays green."""
    _, student, _, _ = distilled
    lo, hi = student.transfer_box
    for delta in (0.1, 1.0, 5.0):  # below the _MAX_BOX_EXCESS cap on both faces
        below = (np.array([[lo[0] - delta]]) - student._x_mean) / student._x_scale
        above = (np.array([[hi[0] + delta]]) - student._x_mean) / student._x_scale
        d_below = float(student._box_excess(below)[0])
        d_above = float(student._box_excess(above)[0])
        assert d_below > 0.0, f"guard does not fire {delta} below the box (lower face)"
        # symmetric: standardized distance to the box is |delta|/x_scale either side.
        assert d_below == pytest.approx(d_above, rel=1e-9)


def test_ood_inflation_zero_disables_only_the_guard():
    """The guard is a knob: at the same seed, ood_inflation=0 and =1 are the SAME
    net, so they must agree EXACTLY in-box and differ only outside. Pins that the
    guard has no in-box side effect and that it is what buys far-OOD inflation."""
    X, Y = _sin_data(n=120, noise=0.1, seed=1)
    teacher = DeepEnsembleForwardModel(
        n_members=2, width=32, n_blocks=1, d_rff=64, max_epochs=40, seed=11
    ).fit(X, Y)
    kw = dict(seed=5, width=32, n_blocks=1, n_transfer=512, max_epochs=30)
    guarded = distill_ensemble(teacher, X, ood_inflation=1.0, **kw)
    bare = distill_ensemble(teacher, X, ood_inflation=0.0, **kw)
    x_in = np.array([[3.0]])
    assert guarded.in_transfer_box(x_in[0]) is True
    assert float(guarded.predict(x_in).epistemic_sigma[0, 0]) == float(
        bare.predict(x_in).epistemic_sigma[0, 0]
    )
    assert guarded.support_score(x_in[0]) == bare.support_score(x_in[0])
    far = np.array([[X_FAR_OOD]])
    assert guarded.in_transfer_box(far[0]) is False
    assert float(guarded.predict(far).epistemic_sigma[0, 0]) > float(
        bare.predict(far).epistemic_sigma[0, 0]
    )
    assert guarded.support_score(far[0]) < bare.support_score(far[0])


# --- support / jacobian -----------------------------------------------------


def test_support_score_discriminates_and_keeps_teacher_scale(distilled):
    """The support head is distilled from the teacher's OWN score, so a §8.2
    support_floor computed against either model means the same thing — that is
    what makes the student a drop-in for the fail-closed gate. Contract: negative
    Mahalanobis, max 0."""
    teacher, student, _, _ = distilled
    s_in = student.support_score(np.array([3.0]))
    s_out = student.support_score(np.array([X_OOD_IN_BOX]))
    assert isinstance(s_in, float)
    assert s_in > s_out
    assert s_in <= 0.0 and s_out <= 0.0  # never positive (max 0 at the train mean)
    # same scale as the teacher's, on-manifold (MEASURED median abs err 0.099 on
    # scores of magnitude ~1.3 — a re-scaled score would blow this out)
    xg = np.linspace(0.3, 6.0, 40)[:, None]
    err = np.abs(np.asarray(teacher.support_score(xg)) - np.asarray(student.support_score(xg)))
    assert float(np.median(err)) < 0.3, float(np.median(err))


def test_jacobian_tracks_teacher_and_finite_difference(distilled):
    """The mean head's autograd Jacobian is what §8's δ box consumes. It is NOT
    Sobolev-supervised (§6.1) — it is inherited from the mean fit — so this is a
    measurement of that inheritance, not a guarantee."""
    teacher, student, _, _ = distilled
    for x0 in ([1.0], [2.0], [4.0]):
        x0 = np.array(x0)
        J = student.jacobian(x0)
        assert J.shape == (1, 1)
        eps = 1e-3
        fd = (student.predict(x0 + eps).mean[0] - student.predict(x0 - eps).mean[0]) / (2 * eps)
        assert abs(J[0, 0] - fd) < 0.15, (J[0, 0], fd)  # self-consistent with its own mean
        assert abs(J[0, 0] - teacher.jacobian(x0)[0, 0]) < 0.25, (J, teacher.jacobian(x0))


# --- determinism / lifecycle -------------------------------------------------


def test_determinism_same_seed():
    """§13.4: same seed + same teacher (CPU) -> bit-identical predictions."""
    X, Y = _sin_data(n=120, noise=0.1, seed=1)
    teacher = DeepEnsembleForwardModel(
        n_members=2, width=32, n_blocks=1, d_rff=64, max_epochs=40, seed=11
    ).fit(X, Y)
    kw = dict(seed=5, width=32, n_blocks=1, n_transfer=512, max_epochs=30)
    a = distill_ensemble(teacher, X, **kw).predict(np.array([1.0]))
    b = distill_ensemble(teacher, X, **kw).predict(np.array([1.0]))
    assert a.mean[0] == b.mean[0]
    assert a.aleatoric_sigma[0] == b.aleatoric_sigma[0]
    assert a.epistemic_sigma[0] == b.epistemic_sigma[0]


def test_not_fitted_raises():
    student = DistilledForwardModel()
    with pytest.raises(RuntimeError, match="not fitted"):
        student.predict(np.array([1.0]))
    with pytest.raises(RuntimeError, match="not fitted"):
        student.support_score(np.array([1.0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"box_fraction": 1.5}, "box_fraction"),
        ({"dilate": -1.0}, "dilate"),
        ({"ood_inflation": -0.1}, "ood_inflation"),
    ],
)
def test_constructor_validates(kwargs, match):
    with pytest.raises(ValueError, match=match):
        DistilledForwardModel(**kwargs)


def test_multi_output():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, size=(200, 2))
    Y = np.stack([X[:, 0] ** 2, np.sin(2 * X[:, 1])], axis=-1) + 0.05 * rng.standard_normal(
        (200, 2)
    )
    teacher = DeepEnsembleForwardModel(
        n_members=2, width=48, n_blocks=1, d_rff=96, max_epochs=120, seed=3
    ).fit(X, Y)
    student = distill_ensemble(
        teacher, X, seed=0, width=48, n_blocks=1, n_transfer=4096, max_epochs=200
    )
    pred = student.predict(X[:10])
    assert pred.mean.shape == (10, 2)
    assert pred.aleatoric_sigma.shape == (10, 2)
    assert pred.epistemic_sigma.shape == (10, 2)
    assert student.jacobian(X[0]).shape == (2, 2)
    assert student.support_score(X[:10]).shape == (10,)
    # This is a PLUMBING test at a cut transfer budget, not a fidelity claim (the
    # 1-D fixture owns fidelity at the full budget). What it must catch is a
    # per-output wiring bug: the two outputs' σ's are genuinely distinct, and each
    # tracks its OWN teacher channel — a broadcast/shared-scalar bug passes the
    # shape asserts above and dies here.
    t, s = teacher.predict(X[:50]), student.predict(X[:50])
    assert not np.allclose(s.aleatoric_sigma[:, 0], s.aleatoric_sigma[:, 1])
    for j in range(2):
        assert float(np.median(_rel_err(t.mean[:, j], s.mean[:, j]))) < 0.3
        assert float(np.median(_rel_err(t.aleatoric_sigma[:, j], s.aleatoric_sigma[:, j]))) < 0.3
        assert float(np.median(_rel_err(t.epistemic_sigma[:, j], s.epistemic_sigma[:, j]))) < 0.5


def test_update_requires_keys(distilled):
    _, student, _, _ = distilled
    with pytest.raises(ValueError, match="input_keys/output_keys"):
        student.update([])


def test_no_posterior_cov_is_deliberate(distilled):
    """The honest boundary (§5.7): a single net has no joint epistemic law over
    pairs of inputs, so the distilled tier is NOT _JointModel-conformant and
    cannot drive §9 EPIG. It must not grow a fake posterior_cov — use the
    ensemble or sngp_member_view for the active loop."""
    _, student, _, _ = distilled
    assert not hasattr(student, "posterior_cov")


# --- drop-in behind the §5.6 wrapper and the §8 solver -----------------------


def test_conformal_wrapper_accepts_the_student():
    """Drop-in behind the §5.6 calibrator: distilled σ's feed the conformal layer
    and recover ~nominal coverage on a fresh split."""
    X, Y = _sin_data(n=400, noise=0.1, seed=7)
    Xtr, Ytr = X[:250], Y[:250]
    Xcal, Ycal = X[250:325], Y[250:325]
    Xte, Yte = X[325:], Y[325:]
    teacher = DeepEnsembleForwardModel(
        n_members=3, width=64, n_blocks=2, d_rff=128, max_epochs=200, seed=0
    ).fit(Xtr, Ytr)
    student = distill_ensemble(
        teacher, Xtr, seed=0, width=64, n_blocks=2, n_transfer=2048, max_epochs=200
    )
    cal = SplitConformalCalibrator(alpha=0.1)
    cal.fit(student, Xcal, Ycal)
    lo_hi = cal.interval(Xte)  # (n, m, 2)
    covered = (Yte[:, 0] >= lo_hi[:, 0, 0]) & (Yte[:, 0] <= lo_hi[:, 0, 1])
    assert 0.82 <= float(covered.mean()) <= 1.0, float(covered.mean())


def test_solver_drop_in_with_full_ensemble_revalidation():
    """§5.7's actual prescription end-to-end: search the §8 inverse on the CHEAP
    distilled model, re-validate survivors against the FULL ensemble."""
    rng = np.random.default_rng(11)
    X = rng.uniform(0.0, 2.0, size=(200, 1))
    Y = (0.5 * X[:, 0] + 0.02 * rng.standard_normal(200))[:, None]
    teacher = DeepEnsembleForwardModel(
        n_members=3, width=48, n_blocks=1, d_rff=96, max_epochs=150, seed=0
    ).fit(X, Y)
    student = distill_ensemble(
        teacher, X, seed=0, width=48, n_blocks=1, n_transfer=2048, max_epochs=200
    )
    solver = PessimisticInverseSolver(
        student,
        [ContinuousVariable(name="x", lower=0.0, upper=2.0)],
        ["y"],
        X_train=X,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.0,
        seed=0,
        revalidation_model=teacher,
    )
    res = solver.solve({"targets": {"y": (0.3, 0.7)}})
    assert isinstance(res, list) and len(res) >= 1
    for cand in res:
        # every survivor is certified by the FULL ensemble, per §5.7
        assert 0.3 <= float(teacher.predict(np.array([cand.recipe["x"]])).mean[0]) <= 0.7


# --- the §5.7 cost claim: measure it, do not assert a slogan -----------------


def _time_predict(model, B, reps):
    for _ in range(3):  # warm up
        model.predict(B)
    t0 = time.perf_counter()
    for _ in range(reps):
        model.predict(B)
    return (time.perf_counter() - t0) / reps


def test_distilled_is_faster_than_the_ensemble(distilled, capsys):
    """The whole point of §5.7 option A. The K=3 / d_rff=128 fixture teacher is a
    DEV-small ensemble, so this understates the K=10 production win — see
    test_speedup_at_production_teacher_size for that figure. Asserted bound is
    deliberately modest; the measured number is printed."""
    teacher, student, _, _ = distilled
    B = np.linspace(0.2, 6.0, 256)[:, None]
    t_teacher = _time_predict(teacher, B, 15)
    t_student = _time_predict(student, B, 15)
    speedup = t_teacher / t_student
    print(f"\n[§5.7] distilled predict speedup (K=3, d_rff=128, m=1): {speedup:.1f}x")
    assert speedup > 1.5, speedup


def test_speedup_at_production_teacher_size(capsys):
    """The honest production-scale cost figure. ``predict`` cost is independent of
    how well the teacher was TRAINED, so the teacher here is fitted with
    max_epochs=1 — this measures cost only and makes no accuracy claim. Sizes are
    the §5.2/§5.4 production defaults (K=10 final, d_rff=256, width=128)."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(200, 4))
    Y = np.stack([X[:, 0], X[:, 1] ** 2, np.sin(X[:, 2])], axis=-1)
    teacher = DeepEnsembleForwardModel(
        n_members=10, width=128, n_blocks=2, d_rff=256, max_epochs=1, seed=0
    ).fit(X, Y)
    student = distill_ensemble(
        teacher, X, seed=0, width=128, n_blocks=2, n_transfer=256, max_epochs=1
    )
    B = rng.uniform(-1.0, 1.0, size=(64, 4))
    speedup = _time_predict(teacher, B, 10) / _time_predict(student, B, 10)
    print(f"\n[§5.7] distilled predict speedup (K=10, d_rff=256, m=3): {speedup:.1f}x")
    assert speedup > 1.5, speedup
