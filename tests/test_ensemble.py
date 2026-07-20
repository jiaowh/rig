"""WP-E: deep-ensemble β-NLL + SNGP forward tier tests (implementation-plan §5.4, §5.7, §5.9).

Backend B — the large-data D3 primary. Mirrors the GP tier's contract tests
(``test_gp.py``) so the two backends are provably interchangeable behind the
canonical ``PredictiveDistribution`` (§3.2), and adds the backend-B-specific
gates: OOD epistemic inflation must come from the spectral/SNGP term (§5.9
invariant 1), β-NLL must not collapse variance (§5.4), and the conformal wrapper
must recover near-nominal coverage on a held-out split (§5.6). Synthetic +
in-silico only; torch is an optional extra so the whole module is skipped when it
is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip the module if the [torch] extra is absent

from rig.active import ActiveLearningLoop, Trajectory  # noqa: E402
from rig.active.acquisition import bald, epig  # noqa: E402
from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator  # noqa: E402
from rig.forward import DeepEnsembleForwardModel  # noqa: E402
from rig.interfaces import (  # noqa: E402
    ContinuousVariable,
    ForwardModel,
    PredictiveDistribution,
    RecipeCandidate,
)
from rig.inverse import PessimisticInverseSolver, parse_targets  # noqa: E402

RNG_SEED = 20260717


def _sin_data(n: int, noise: float, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    y = np.sin(X[:, 0]) + noise * rng.standard_normal(n)
    return X, y[:, None]


@pytest.fixture(scope="module")
def sin_model():
    # >~300-run regime is backend B's home (D3); train on 300 points.
    X, Y = _sin_data(n=300, noise=0.1, seed=RNG_SEED)
    model = DeepEnsembleForwardModel(
        n_members=3, width=64, n_blocks=2, d_rff=128, max_epochs=200, seed=0
    )
    model.fit(X, Y)
    return model, X, Y


# --- protocol / contract ---------------------------------------------------


def test_is_forward_model(sin_model):
    model, _, _ = sin_model
    assert isinstance(model, ForwardModel)


def test_predict_returns_canonical_distribution(sin_model):
    model, _, _ = sin_model
    pred = model.predict(np.array([1.0]))
    assert isinstance(pred, PredictiveDistribution)
    # canonical field ORDER (§3.2): mean, aleatoric_sigma, epistemic_sigma, conformal_set
    assert list(vars(pred).keys()) == [
        "mean",
        "aleatoric_sigma",
        "epistemic_sigma",
        "conformal_set",
    ]
    assert pred.conformal_set is None  # filled only by the §5.6 wrapper


def test_shape_contract(sin_model):
    """(d,) -> (m,); (n,d) -> (n,m) — the binding WP-C shape contract."""
    model, _, _ = sin_model
    single = model.predict(np.array([1.0]))
    assert single.mean.shape == (1,)
    assert single.aleatoric_sigma.shape == (1,)
    assert single.epistemic_sigma.shape == (1,)
    batch = model.predict(np.linspace(0.5, 5.5, 7)[:, None])
    assert batch.mean.shape == (7, 1)
    assert batch.epistemic_sigma.shape == (7, 1)


def test_beats_predict_the_mean_baseline(sin_model):
    model, X, Y = sin_model
    x_test = np.linspace(0.2, 2.0 * np.pi - 0.2, 200)[:, None]
    y_true = np.sin(x_test[:, 0])
    rmse = float(np.sqrt(np.mean((model.predict(x_test).mean[:, 0] - y_true) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((float(Y.mean()) - y_true) ** 2)))
    assert rmse * 4.0 < rmse_baseline, (rmse, rmse_baseline)


# --- §5.4 / §5.9 backend-B-specific gates ----------------------------------


def test_beta_nll_recovers_noise_and_does_not_collapse_variance(sin_model):
    """§5.4: β-NLL must keep a calibrated, POSITIVE aleatoric σ (no collapse to 0,
    no blow-up) while the mean still fits — the whole reason we use β-NLL over
    plain Gaussian NLL."""
    model, _, _ = sin_model
    x_in = np.linspace(0.3, 2.0 * np.pi - 0.3, 150)[:, None]
    ale = model.predict(x_in).aleatoric_sigma[:, 0]
    assert np.all(ale > 0.0)
    # true observation noise is 0.1; a collapsed or blown-up σ fails this band
    assert 0.03 <= float(ale.mean()) <= 0.3, float(ale.mean())


def test_ood_epistemic_growth(sin_model):
    """§5.9 invariant 1 (THE disqualifying test): epistemic MUST inflate off the
    training manifold. For backend B this is the spectral-trunk + SNGP-Laplace
    term doing its job — a flat epistemic would mean the OOD gate is blind."""
    model, _, _ = sin_model
    x_in = np.linspace(0.2, 2.0 * np.pi - 0.2, 100)[:, None]
    in_range = float(np.mean(model.predict(x_in).epistemic_sigma))
    far = model.predict(np.array([18.0]))
    assert float(far.epistemic_sigma[0]) > 3.0 * in_range, (
        float(far.epistemic_sigma[0]),
        in_range,
    )


def test_support_score_discriminates_and_typed(sin_model):
    """§8.2/§11: negative-Mahalanobis support in the spectral latent — higher
    in-distribution; float for a point, (n,) for a batch."""
    model, _, _ = sin_model
    s_in = model.support_score(np.array([3.0]))
    s_out = model.support_score(np.array([18.0]))
    assert isinstance(s_in, float)
    assert s_in > s_out  # closer to the training manifold scores higher
    batch = model.support_score(np.linspace(0.5, 5.5, 5)[:, None])
    assert batch.shape == (5,)


def test_jacobian_matches_finite_difference(sin_model):
    """Autograd jacobian ~ central finite difference of the mixture mean, and has
    the (m, d) shape the §8 inverse consumes."""
    model, _, _ = sin_model
    x0 = np.array([2.0])
    J = model.jacobian(x0)
    assert J.shape == (1, 1)
    eps = 1e-3
    fd = (model.predict(x0 + eps).mean[0] - model.predict(x0 - eps).mean[0]) / (2 * eps)
    assert np.isfinite(J[0, 0])
    assert abs(J[0, 0] - fd) < 0.15, (J[0, 0], fd)


# --- determinism / lifecycle ------------------------------------------------


def test_determinism_same_seed():
    """§13.4: same seed + same data (CPU) -> bit-identical prediction."""
    X, Y = _sin_data(n=120, noise=0.1, seed=1)
    kw = dict(n_members=2, width=32, n_blocks=1, d_rff=64, max_epochs=40, seed=11)
    a = DeepEnsembleForwardModel(**kw).fit(X, Y).predict(np.array([1.0])).mean[0]
    b = DeepEnsembleForwardModel(**kw).fit(X, Y).predict(np.array([1.0])).mean[0]
    assert a == b


def test_not_fitted_raises():
    model = DeepEnsembleForwardModel(n_members=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.array([1.0]))


def test_multi_output():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1.0, 1.0, size=(200, 2))
    Y = np.stack([X[:, 0] ** 2, np.sin(2 * X[:, 1])], axis=-1) + 0.05 * rng.standard_normal(
        (200, 2)
    )
    model = DeepEnsembleForwardModel(
        n_members=2, width=48, n_blocks=1, d_rff=96, max_epochs=120, seed=3
    )
    model.fit(X, Y)
    pred = model.predict(X[:10])
    assert pred.mean.shape == (10, 2)
    assert pred.epistemic_sigma.shape == (10, 2)
    assert model.jacobian(X[0]).shape == (2, 2)


# --- §5.6 conformal integration --------------------------------------------


def test_conformal_wrapper_recovers_nominal_coverage():
    """The tier is a drop-in for the split-conformal calibrator, and calibrated
    intervals reach ~1-alpha empirical coverage on a fresh split (§5.6)."""
    X, Y = _sin_data(n=400, noise=0.1, seed=7)
    Xtr, Ytr = X[:250], Y[:250]
    Xcal, Ycal = X[250:325], Y[250:325]
    Xte, Yte = X[325:], Y[325:]
    model = DeepEnsembleForwardModel(
        n_members=3, width=64, n_blocks=2, d_rff=128, max_epochs=200, seed=0
    )
    model.fit(Xtr, Ytr)
    cal = SplitConformalCalibrator(alpha=0.1)
    cal.fit(model, Xcal, Ycal)
    lo_hi = cal.interval(Xte)  # (n, m, 2)
    covered = (Yte[:, 0] >= lo_hi[:, 0, 0]) & (Yte[:, 0] <= lo_hi[:, 0, 1])
    picp = float(covered.mean())
    # split conformal is marginally valid; allow finite-sample slack around 0.90
    assert 0.82 <= picp <= 1.0, picp


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_path_runs():
    """The sm_120 (RTX 5050) GPU path fits and predicts without error."""
    X, Y = _sin_data(n=150, noise=0.1, seed=2)
    model = DeepEnsembleForwardModel(
        n_members=2, width=32, n_blocks=1, d_rff=64, max_epochs=40, seed=0, device="cuda"
    )
    model.fit(X, Y)
    pred = model.predict(np.linspace(0.5, 5.5, 10)[:, None])
    assert pred.mean.shape == (10, 1)
    assert np.all(np.isfinite(pred.mean))


# --- §9.4 EPIG-capability: posterior_cov joint epistemic covariance ---------


def test_posterior_cov_consistent_symmetric_psd(sin_model):
    """posterior_cov's diagonal MUST equal epistemic_sigma² (else EPIG's σ²(x*|x)
    reduction is silently mis-scaled); it must also be symmetric + PSD."""
    model, _, _ = sin_model
    Xq = np.linspace(0.4, 6.0, 8)[:, None]
    cov = model.posterior_cov(Xq, Xq)
    assert cov.shape == (1, 8, 8)
    epi2 = model.predict(Xq).epistemic_sigma[:, 0] ** 2  # variance, matches cov diagonal
    assert np.allclose(np.diag(cov[0]), epi2, rtol=1e-6), (np.diag(cov[0]), epi2)
    assert np.allclose(cov[0], cov[0].T, atol=1e-8)
    assert float(np.linalg.eigvalsh(cov[0]).min()) > -1e-8


def test_ensemble_drives_epig_and_bald(sin_model):
    """The ensemble is a full _JointModel: EPIG (needs posterior_cov) and BALD
    both compute finite, non-negative nats — so the §9 AL loop runs on backend B."""
    model, _, _ = sin_model
    Xc = np.linspace(0.5, 5.5, 6)[:, None]
    Xstar = np.array([[2.0], [3.0]])
    e = epig(model, Xc, Xstar)
    b = bald(model, Xc)
    assert e.shape == (6,) and b.shape == (6,)
    assert np.all(np.isfinite(e)) and np.all(e >= -1e-9)
    assert np.all(np.isfinite(b)) and np.all(b >= -1e-9)


# --- §5.7 fast inner-loop surrogate (SNGP single member) --------------------


def test_sngp_member_view_is_forwardmodel_and_inflates_ood(sin_model):
    """The fast view is ForwardModel-conformant and keeps a distance-aware
    epistemic (its SNGP-Laplace term) so it inflates OOD — a screening surrogate,
    not a blind one."""
    model, _, _ = sin_model
    view = model.inner_loop_surrogate()
    assert isinstance(view, ForwardModel)
    x_in = np.linspace(0.3, 6.0, 60)[:, None]
    in_epi = float(np.mean(view.predict(x_in).epistemic_sigma))
    far = float(view.predict(np.array([18.0])).epistemic_sigma[0])
    assert far > 3.0 * in_epi
    assert view.jacobian(np.array([1.0])).shape == (1, 1)
    assert view.posterior_cov(x_in[:4], x_in[:4]).shape == (1, 4, 4)


def test_sngp_member_view_is_faster(sin_model):
    """The single-member view is materially cheaper than the K-member mixture on a
    batch predict (the §8 inner-loop-budget win; ~K× at K members)."""
    import time

    model, _, _ = sin_model
    view = model.inner_loop_surrogate()
    B = np.linspace(0.2, 6.0, 256)[:, None]
    for _ in range(2):  # warm up
        model.predict(B)
        view.predict(B)
    t0 = time.perf_counter()
    for _ in range(15):
        model.predict(B)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(15):
        view.predict(B)
    t_view = time.perf_counter() - t0
    assert t_view < t_full  # strictly cheaper (K=3 fixture → ~3×)


def test_inner_loop_surrogate_mode_guard(sin_model):
    model, _, _ = sin_model
    with pytest.raises(NotImplementedError, match="distilled"):
        model.inner_loop_surrogate(mode="distilled")


# --- §5.7 / §13.2 full-ensemble re-validation on the solver -----------------


@pytest.fixture(scope="module")
def linear_model():
    """y = 0.5x on x∈[0,2] — an unambiguous, on-support inverse for solver tests."""
    rng = np.random.default_rng(11)
    X = rng.uniform(0.0, 2.0, size=(200, 1))
    Y = (0.5 * X[:, 0] + 0.02 * rng.standard_normal(200))[:, None]
    model = DeepEnsembleForwardModel(
        n_members=3, width=48, n_blocks=1, d_rff=96, max_epochs=150, seed=0
    )
    model.fit(X, Y)
    return model, X


def _solver(model, X, **kw):
    return PessimisticInverseSolver(
        model,
        [ContinuousVariable(name="x", lower=0.0, upper=2.0)],
        ["y"],
        X_train=X,
        kappa=1.0,
        z_epi=1.0,
        delta_frac=0.0,
        seed=0,
        **kw,
    )


def test_revalidation_none_is_identity(linear_model):
    """revalidation_model=None (the default) must be byte-for-byte the plain
    single-model solve — the M2/WP-D path is untouched."""
    model, X = linear_model
    view = model.inner_loop_surrogate()
    spec = {"targets": {"y": (0.3, 0.7)}}
    a = _solver(view, X).solve(spec)
    b = _solver(view, X, revalidation_model=None).solve(spec)
    assert isinstance(a, list) and isinstance(b, list)
    assert [c.recipe["x"] for c in a] == [c.recipe["x"] for c in b]
    assert [c.confidence for c in a] == [c.confidence for c in b]


def test_revalidation_on_full_ensemble_certifies(linear_model):
    """Search on the fast view, re-validate on the full ensemble: the survivors
    are certified by the FULL model (their confidence/interval come from it)."""
    model, X = linear_model
    view = model.inner_loop_surrogate()
    spec = {"targets": {"y": (0.3, 0.7)}}
    res = _solver(view, X, revalidation_model=model).solve(spec)
    assert isinstance(res, list) and len(res) >= 1
    for cand in res:
        assert isinstance(cand, RecipeCandidate)
        # each survivor is feasible under the FULL ensemble (re-scored)
        pred = model.predict(np.array([cand.recipe["x"]]))
        assert 0.3 <= float(pred.mean[0]) <= 0.7


def test_revalidation_conformal_gate(linear_model):
    """§13.2 gate C(x')⊆Z*: a box narrower than the full model's conformal
    interval fails the gate; a wide box passes; a non-conformal model leaves the
    gate inactive (returns True)."""
    model, X = linear_model
    rng = np.random.default_rng(3)
    Xc = rng.uniform(0.0, 2.0, size=(60, 1))
    Yc = (0.5 * Xc[:, 0] + 0.02 * rng.standard_normal(60))[:, None]
    cal = SplitConformalCalibrator(alpha=0.1)
    cal.fit(model, Xc, Yc)
    conformal = ConformalForwardModel(model, cal)
    view = model.inner_loop_surrogate()
    solver = _solver(view, X, revalidation_model=conformal)
    out_idx = np.array([0])
    x1 = np.array([1.0])  # y ≈ 0.5
    narrow = parse_targets({"y": (0.499, 0.501)}, ["y"])  # ≪ conformal band
    wide = parse_targets({"y": (-5.0, 5.0)}, ["y"])
    assert solver._conformal_in_box(x1, narrow, out_idx) is False
    assert solver._conformal_in_box(x1, wide, out_idx) is True
    # a non-conformal re-validation model (conformal_set None) leaves it inactive
    solver_nc = _solver(view, X, revalidation_model=model)
    assert solver_nc._conformal_in_box(x1, narrow, out_idx) is True


# --- end-to-end: the AL loop runs on backend B (EPIG via posterior_cov) ------


def test_active_learning_loop_runs_on_ensemble():
    """The §9 closed loop drives the deep-ensemble surrogate end-to-end: it fits
    each batch, solves the §8 inverse on the fast view, scores EPIG/BALD via
    posterior_cov, and returns a well-formed Trajectory. Proves backend B is a
    drop-in for the AL loop (structural validity, not a performance claim)."""

    def machine(recipe):
        return np.array([0.5 * recipe["x"]])  # deterministic linear machine

    def in_spec(y):
        return bool(0.45 <= y[0] <= 0.55)

    loop = ActiveLearningLoop(
        machine=machine,
        in_spec=in_spec,
        variables=[ContinuousVariable(name="x", lower=0.0, upper=2.0)],
        input_keys=["x"],
        output_keys=["y"],
        spec={"targets": {"y": (0.45, 0.55)}},
        budget=12,
        q=3,
        n_seed=6,
        n_pool=64,
        surrogate_factory=lambda: DeepEnsembleForwardModel(
            n_members=2, width=32, n_blocks=1, d_rff=64, max_epochs=30, seed=0
        ),
        kappa=1.0,
        z_epi=1.0,
        seed=0,
    )
    traj = loop.run()
    assert isinstance(traj, Trajectory)
    assert traj.n_queries >= 6
    assert traj.cumulative_cost == sorted(traj.cumulative_cost)  # monotone
    assert traj.stop_reason
