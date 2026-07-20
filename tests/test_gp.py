"""WP-C: exact-GP forward surrogate tests (implementation-plan §5.2, §5.9). Synthetic only."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from rig.forward import GPForwardModel, records_to_arrays
from rig.interfaces import ForwardModel, PredictiveDistribution
from rig.schema import (
    OutcomeRecord,
    Provenance,
    Quantity,
    RecipeRecord,
    RunRecord,
)

RNG_SEED = 20260715


def _sin_data(n: int = 40, noise: float = 0.1, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    y = np.sin(X[:, 0]) + noise * rng.standard_normal(n)
    return X, y[:, None]


@pytest.fixture(scope="module")
def sin_model():
    X, Y = _sin_data()
    model = GPForwardModel(n_restarts=5, seed=0)
    model.fit(X, Y)
    return model, X, Y


def test_gp_beats_predict_the_mean_baseline(sin_model):
    model, X, Y = sin_model
    x_test = np.linspace(0.1, 2.0 * np.pi - 0.1, 200)[:, None]
    y_true = np.sin(x_test[:, 0])
    pred = model.predict(x_test)
    rmse_gp = float(np.sqrt(np.mean((pred.mean[:, 0] - y_true) ** 2)))
    rmse_baseline = float(np.sqrt(np.mean((float(Y.mean()) - y_true) ** 2)))
    assert rmse_gp * 5.0 < rmse_baseline, (rmse_gp, rmse_baseline)


def test_gp_recovers_noise_level(sin_model):
    model, _, _ = sin_model
    true_sigma = 0.1
    fitted = float(model.noise_std_[0])
    assert 0.5 * true_sigma <= fitted <= 1.5 * true_sigma, fitted


def test_ood_epistemic_growth(sin_model):
    """§5.9 invariant 1: epistemic MUST inflate off the training manifold.

    A flat epistemic is disqualifying (the plan's own words) — this is THE
    test that rules out MC-dropout-style backbones.
    """
    model, _, _ = sin_model
    x_in = np.linspace(0.2, 2.0 * np.pi - 0.2, 100)[:, None]
    in_range_avg = float(np.mean(model.predict(x_in).epistemic_sigma))
    far = model.predict(np.array([15.0]))
    assert float(far.epistemic_sigma[0]) > 3.0 * in_range_avg


def test_predict_returns_canonical_distribution(sin_model):
    model, _, _ = sin_model
    assert isinstance(model, ForwardModel)  # runtime-checkable protocol
    dist = model.predict(np.array([1.0]))
    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (1,)
    assert dist.aleatoric_sigma.shape == (1,)
    assert dist.epistemic_sigma.shape == (1,)
    assert dist.conformal_set is None  # unwrapped: the §5.6 wrapper fills it
    batch = model.predict(np.array([[1.0], [2.0]]))
    assert batch.mean.shape == (2, 1)
    assert batch.epistemic_sigma.shape == (2, 1)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(RNG_SEED)
    n = 60
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    Y = np.column_stack(
        [
            np.sin(X[:, 0]) * X[:, 1],
            X[:, 0] + 0.5 * X[:, 1] ** 2,
        ]
    ) + 0.01 * rng.standard_normal((n, 2))
    model = GPForwardModel(n_restarts=3, seed=1).fit(X, Y)

    x0 = np.array([0.3, -0.7])
    J = model.jacobian(x0)
    assert J.shape == (2, 2)

    h = 1e-4
    J_fd = np.empty((2, 2))
    for d in range(2):
        e = np.zeros(2)
        e[d] = h
        J_fd[:, d] = (model.predict(x0 + e).mean - model.predict(x0 - e).mean) / (2 * h)
    np.testing.assert_allclose(J, J_fd, rtol=1e-4, atol=1e-6)


def test_support_score_monotone_along_ray(sin_model):
    model, X, _ = sin_model
    center = X.mean(axis=0)
    direction = np.array([1.0])
    scores = [model.support_score(center + t * direction) for t in [0.0, 2.0, 5.0, 10.0, 20.0]]
    assert all(a > b for a, b in zip(scores[:-1], scores[1:], strict=True)), scores
    # far point scores lower than every training point
    train_scores = model.support_score(X)
    assert scores[-1] < float(train_scores.min())


def test_support_score_types(sin_model):
    model, X, _ = sin_model
    assert isinstance(model.support_score(np.array([1.0])), float)
    batch = model.support_score(X[:5])
    assert batch.shape == (5,)


def _make_record(temp_k: float, flow: float, rate: float, thick: float) -> RunRecord:
    return RunRecord(
        run_id=uuid4(),
        process_id="synthetic",
        tool_id="tool-A",
        timestamp=datetime(2026, 7, 15, tzinfo=UTC),
        recipe=RecipeRecord(
            values={
                "temperature": Quantity(magnitude=temp_k, unit="K"),
                "flow": Quantity(magnitude=flow, unit="m^3/s"),
            }
        ),
        outcomes=[
            OutcomeRecord(
                name="growth_rate",
                modality="scalar_vector",
                value=Quantity(magnitude=rate, unit="m/s"),
            ),
            OutcomeRecord(
                name="thickness",
                modality="scalar_vector",
                value=Quantity(magnitude=thick, unit="m"),
            ),
        ],
        provenance=Provenance(source="physics_sim"),
    )


def test_records_to_arrays_and_update():
    rng = np.random.default_rng(RNG_SEED)
    records = []
    for _ in range(25):
        t = rng.uniform(700.0, 900.0)
        f = rng.uniform(1e-7, 5e-7)
        rate = 1e-10 * (t / 800.0) + 1e-11 * rng.standard_normal()
        records.append(_make_record(t, f, rate, rate * 3600.0))

    X, Y = records_to_arrays(records, ["temperature", "flow"], ["growth_rate", "thickness"])
    assert X.shape == (25, 2)
    assert Y.shape == (25, 2)
    assert X[0, 0] == records[0].recipe.values["temperature"].magnitude

    keys = dict(input_keys=["temperature", "flow"], output_keys=["growth_rate", "thickness"])
    model = GPForwardModel(n_restarts=2, seed=0, **keys)
    model.update(records[:20])  # fit-from-scratch path
    assert model.n_train_ == 20
    model.update(records[20:])  # refit including new data
    assert model.n_train_ == 25
    dist = model.predict(X[0])
    assert dist.mean.shape == (2,)


def test_records_to_arrays_rejects_missing_and_nonscalar():
    rec = _make_record(800.0, 2e-7, 1e-10, 3.6e-7)
    with pytest.raises(ValueError, match="no outcome named"):
        records_to_arrays([rec], ["temperature"], ["nonexistent"])
    with pytest.raises(ValueError, match="no value for input key"):
        records_to_arrays([rec], ["pressure"], ["growth_rate"])
    with pytest.raises(ValueError, match="at least one record"):
        records_to_arrays([], ["temperature"], ["growth_rate"])


def test_update_requires_keys():
    model = GPForwardModel()
    with pytest.raises(ValueError, match="input_keys"):
        model.update([])


def test_fit_is_deterministic():
    X, Y = _sin_data(n=25)
    m1 = GPForwardModel(n_restarts=3, seed=7).fit(X, Y)
    m2 = GPForwardModel(n_restarts=3, seed=7).fit(X, Y)
    x = np.array([2.5])
    np.testing.assert_array_equal(m1.predict(x).mean, m2.predict(x).mean)
    np.testing.assert_array_equal(m1.noise_std_, m2.noise_std_)


def test_single_output_nlml_gradient_matches_finite_differences():
    # audit C1: the analytic NLML gradient drives L-BFGS-B hyperparameter
    # optimization and was untested (its multitask sibling IS FD-tested). Compare
    # analytic vs central-difference on a small random multi-dim dataset; an
    # injected gradient bug (sign flip, dropped term, scaling) breaks this.
    from rig.forward.gp import _SingleOutputGP

    rng = np.random.default_rng(11)
    n, d = 7, 3
    X = rng.standard_normal((n, d))
    y = rng.standard_normal(n)
    gp = _SingleOutputGP(X, y)
    theta = np.array([0.3, -0.2, 0.1, np.log(0.8), np.log(0.05)])  # [log ell..., log sf2, log sn2]
    _, grad = gp._nlml_and_grad(theta)

    h = 1e-6
    fd = np.empty_like(theta)
    for k in range(theta.size):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        fp, _ = gp._nlml_and_grad(tp)
        fm, _ = gp._nlml_and_grad(tm)
        fd[k] = (fp - fm) / (2.0 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-6)


def test_posterior_cov_invariant_diagonal_symmetry_psd():
    # audit C2: bind the posterior_cov contract — its per-output diagonal must
    # equal epistemic_sigma**2 (raw units), it must be symmetric, and PSD.
    # Dropping the ys*ys raw-unit factor (a plausible regression) breaks the
    # diagonal==epistemic^2 equality even though ordering-only EPIG tests pass.
    rng = np.random.default_rng(3)
    n, d = 30, 3
    X = rng.uniform(-1.0, 1.0, size=(n, d))
    Y = np.stack([np.sin(X[:, 0]) + 0.3 * X[:, 1], np.cos(X[:, 2])], axis=-1)
    model = GPForwardModel(n_restarts=3, seed=1).fit(X, Y)

    Xq = rng.uniform(-1.0, 1.0, size=(6, d))
    cov = model.posterior_cov(Xq, Xq)  # (m, nq, nq)
    epi = model.predict(Xq).epistemic_sigma  # (nq, m)
    assert cov.shape == (Y.shape[1], 6, 6)
    for j in range(Y.shape[1]):
        diag = np.diagonal(cov[j])
        np.testing.assert_allclose(diag, epi[:, j] ** 2, rtol=1e-6, atol=1e-10)
        np.testing.assert_allclose(cov[j], cov[j].T, atol=1e-10)  # symmetry
        eigmin = float(np.min(np.linalg.eigvalsh(cov[j])))
        assert eigmin >= -1e-9, eigmin  # PSD (up to numerical slack)
