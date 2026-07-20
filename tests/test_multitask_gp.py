"""WP-I: multi-tool ICM GP surrogate tests (implementation-plan §10.4 level (a), §5.8).

Synthetic only — two "tools" sharing one latent function with a small
multiplicative + additive tool discrepancy: tool A = f(x), tool B =
f(x)*(1+eps) + delta. These tests live with core and MUST NOT import
rig_adapters (import-linter guards the src side; this file keeps the
discipline on the test side too). The in-silico MBE integration test is
tests/test_multitask_mbe.py.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pytest

from rig.calibration.conformal import ConformalForwardModel, SplitConformalCalibrator
from rig.forward import (
    GPForwardModel,
    MultiToolGPForwardModel,
    records_to_arrays_with_tools,
)
from rig.forward.multitask import _ICMSingleOutputGP
from rig.interfaces import ForwardModel, PredictiveDistribution
from rig.schema import OutcomeRecord, Provenance, Quantity, RecipeRecord, RunRecord

RNG_SEED = 20260715
N_A = 40
EPS, DELTA, NOISE = 0.1, 0.15, 0.05


def _two_tool_data(k_b: int, seed: int = RNG_SEED, n_test: int = 60):
    """Tool A: sin(x); tool B: sin(x)*(1+EPS)+DELTA — small, related tools."""
    rng = np.random.default_rng(seed)
    Xa = rng.uniform(0.0, 2.0 * np.pi, (N_A, 1))
    ya = np.sin(Xa[:, 0]) + NOISE * rng.standard_normal(N_A)
    Xb = rng.uniform(0.0, 2.0 * np.pi, (k_b, 1))
    yb = np.sin(Xb[:, 0]) * (1.0 + EPS) + DELTA + NOISE * rng.standard_normal(k_b)
    Xt = np.linspace(0.1, 2.0 * np.pi - 0.1, n_test)[:, None]
    yt_true = np.sin(Xt[:, 0]) * (1.0 + EPS) + DELTA  # noise-free B truth
    return Xa, ya, Xb, yb, Xt, yt_true


def _rmse(model, Xt: np.ndarray, y_true: np.ndarray, **predict_kwargs) -> float:
    mu = np.asarray(model.predict(Xt, **predict_kwargs).mean).ravel()
    return float(np.sqrt(np.mean((mu - y_true) ** 2)))


@pytest.fixture(scope="module", params=[4, 8], ids=["k=4", "k=8"])
def few_shot_setup(request):
    k = request.param
    Xa, ya, Xb, yb, Xt, yt_true = _two_tool_data(k)
    X_all = np.vstack([Xa, Xb])
    y_all = np.concatenate([ya, yb])
    tools = ["A"] * N_A + ["B"] * k
    multi = MultiToolGPForwardModel(rank=1, n_restarts=3, seed=0).fit(X_all, y_all, tools)
    return k, multi, X_all, y_all, tools, Xb, yb, Xt, yt_true


# -- (i)/(ii) few-shot transfer beats pooled-blind AND scratch --------------------


def test_multitool_beats_pooled_blind_and_scratch(few_shot_setup):
    """§10.4's whole point: k runs of tool B + 40 of tool A must beat both a
    tool-ignorant pool and a from-scratch fit on B's k runs alone."""
    k, multi, X_all, y_all, _, Xb, yb, Xt, yt_true = few_shot_setup
    pooled = GPForwardModel(n_restarts=3, seed=0).fit(X_all, y_all[:, None])
    scratch = GPForwardModel(n_restarts=3, seed=0).fit(Xb, yb[:, None])
    rmse_multi = _rmse(multi, Xt, yt_true, tool_id="B")
    rmse_pooled = _rmse(pooled, Xt, yt_true)
    rmse_scratch = _rmse(scratch, Xt, yt_true)
    assert rmse_multi < rmse_pooled, (k, rmse_multi, rmse_pooled)
    assert rmse_multi < rmse_scratch, (k, rmse_multi, rmse_scratch)


# -- (iii) unknown tool: inflated epistemic, never a silent known-tool ------------


def test_unknown_tool_epistemic_dominates_known(few_shot_setup):
    _, multi, *_rest, Xt, _ = few_shot_setup
    epi_unknown = multi.predict(Xt, tool_id="never-seen").epistemic_sigma
    epi_a = multi.predict(Xt, tool_id="A").epistemic_sigma
    epi_b = multi.predict(Xt, tool_id="B").epistemic_sigma
    assert np.all(epi_unknown >= epi_a) and np.all(epi_unknown >= epi_b)
    assert float(epi_unknown.mean()) > float(epi_a.mean())
    # tool_id=None takes the same population-fallback path (documented)
    np.testing.assert_array_equal(
        multi.predict(Xt).mean, multi.predict(Xt, tool_id="never-seen").mean
    )


def test_leave_one_tool_out_epistemic_check():
    """§5.8 OOD check: mean epistemic on a held-out (never-trained) tool must
    exceed mean in-distribution epistemic."""
    Xa, ya, _, _, Xt, _ = _two_tool_data(8)
    model = MultiToolGPForwardModel(rank=1, n_restarts=3, seed=0).fit(Xa, ya, ["A"] * N_A)
    epi_in = float(model.predict(Xt, tool_id="A").epistemic_sigma.mean())
    epi_out = float(model.predict(Xt, tool_id="B").epistemic_sigma.mean())
    assert epi_out > epi_in, (epi_in, epi_out)


# -- (iv) B recovers the inter-tool correlation ------------------------------------


def test_tool_covariance_recovers_high_correlation(few_shot_setup):
    """A and B differ by a small affine discrepancy => their latent functions
    are highly correlated; B[s,t] must say so (loose sanity bound)."""
    _, multi, *_ = few_shot_setup
    corr = multi.tool_correlation_
    assert corr.shape == (1, 2, 2)
    assert float(corr[0, 0, 1]) > 0.5, corr


# -- (v) WP-C shape contract --------------------------------------------------------


def test_shape_contract_and_protocol(few_shot_setup):
    _, multi, *_ = few_shot_setup
    assert isinstance(multi, ForwardModel)  # runtime-checkable protocol
    view = multi.for_tool("B")
    assert isinstance(view, ForwardModel)
    for m in (multi, view):
        dist = m.predict(np.array([1.0]))
        assert isinstance(dist, PredictiveDistribution)
        assert dist.mean.shape == (1,)
        assert dist.aleatoric_sigma.shape == (1,)
        assert dist.epistemic_sigma.shape == (1,)
        assert dist.conformal_set is None  # unwrapped: the §5.6 wrapper fills it
        batch = m.predict(np.array([[1.0], [2.0], [3.0]]))
        assert batch.mean.shape == (3, 1)
        assert batch.epistemic_sigma.shape == (3, 1)
        assert isinstance(m.support_score(np.array([1.0])), float)
        assert m.support_score(np.array([[1.0], [2.0]])).shape == (2,)
        J = m.jacobian(np.array([1.0]))
        assert J.shape == (1, 1)


def test_jacobian_matches_finite_differences(few_shot_setup):
    _, multi, *_ = few_shot_setup
    x0 = np.array([2.5])
    for tool in ("A", "B", None):
        J = multi.jacobian(x0, tool_id=tool)
        h = 1e-5
        fd = (
            np.asarray(multi.predict(x0 + h, tool_id=tool).mean)
            - np.asarray(multi.predict(x0 - h, tool_id=tool).mean)
        ) / (2 * h)
        np.testing.assert_allclose(J[:, 0], fd, rtol=1e-4, atol=1e-6)


def test_support_score_per_tool_vs_global(few_shot_setup):
    k, multi, *_ = few_shot_setup
    # tool A has 40 >= d+2 runs -> per-tool stats; scores are negative-distance
    s_center = multi.support_score(np.array([np.pi]), tool_id="A")
    s_far = multi.support_score(np.array([50.0]), tool_id="A")
    assert s_far < s_center <= 0.0
    if k < 3:  # k=4 and k=8 both have >= d+2 = 3 runs, so exercise the fallback
        assert multi.support_score(np.array([np.pi]), tool_id="B") == multi.support_score(
            np.array([np.pi])
        )
    # unknown tool always falls back to the global cloud
    s_unknown = multi.support_score(np.array([np.pi]), tool_id="ghost")
    assert s_unknown == multi.support_score(np.array([np.pi]))


# -- (vi) conformal wrapping via the tool-bound view --------------------------------


def test_conformal_wrapping_via_for_tool(few_shot_setup):
    """ConformalForwardModel is tool-blind; binding the tool with for_tool()
    slots the multi-tool model under it unchanged (no wrapper edits)."""
    _, multi, *_ = few_shot_setup
    rng = np.random.default_rng(RNG_SEED + 1)
    X_cal = rng.uniform(0.0, 2.0 * np.pi, (30, 1))
    y_cal = np.sin(X_cal[:, 0]) + NOISE * rng.standard_normal(30)  # tool A data
    view = multi.for_tool("A")
    calibrator = SplitConformalCalibrator(alpha=0.1)
    calibrator.fit(view, X_cal, y_cal)
    wrapped = ConformalForwardModel(view, calibrator)
    dist = wrapped.predict(np.array([1.0]))
    assert dist.conformal_set is not None
    assert dist.conformal_set.shape == (1, 2)
    assert np.all(np.isfinite(dist.conformal_set))
    lo, hi = dist.conformal_set[0]
    assert lo < float(dist.mean[0]) < hi
    batch = wrapped.predict(np.array([[1.0], [2.0]]))
    assert batch.conformal_set.shape == (2, 1, 2)


# -- NLML gradient correctness (the machinery the whole WP rests on) ----------------


def test_icm_nlml_gradient_matches_finite_differences():
    rng = np.random.default_rng(RNG_SEED)
    n, d, n_tools, rank = 12, 2, 2, 2
    X = rng.uniform(-1.0, 1.0, (n, d))
    ix = rng.integers(0, n_tools, n)
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(n)
    gp = _ICMSingleOutputGP(X, y, ix, n_tools, rank)
    theta = np.concatenate(
        [
            rng.uniform(-0.5, 0.5, d),
            0.5 * rng.standard_normal(n_tools * rank),
            rng.uniform(-2.0, 0.0, n_tools),
            [np.log(0.05)],
        ]
    )
    _, grad = gp._nlml_and_grad(theta)
    h = 1e-6
    fd = np.empty_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = h
        fd[i] = (gp._nlml_and_grad(theta + e)[0] - gp._nlml_and_grad(theta - e)[0]) / (2 * h)
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-7)


# -- update() / add_tool() / adapt_to_tool() -----------------------------------------


def _make_record(tool_id: str, x: float, y: float) -> RunRecord:
    return RunRecord(
        run_id=uuid4(),
        process_id="synthetic",
        tool_id=tool_id,
        timestamp=datetime(2026, 7, 15, tzinfo=UTC),
        recipe=RecipeRecord(values={"knob": Quantity(magnitude=x, unit="K")}),
        outcomes=[
            OutcomeRecord(
                name="kpi", modality="scalar_vector", value=Quantity(magnitude=y, unit="m")
            )
        ],
        provenance=Provenance(source="physics_sim"),
    )


def _records_for(tool_id: str, X: np.ndarray, y: np.ndarray) -> list[RunRecord]:
    return [_make_record(tool_id, float(x), float(v)) for x, v in zip(X[:, 0], y, strict=True)]


def test_records_to_arrays_with_tools():
    Xa, ya, Xb, yb, *_ = _two_tool_data(4)
    records = _records_for("A", Xa, ya) + _records_for("B", Xb, yb)
    X, Y, tools = records_to_arrays_with_tools(records, ["knob"], ["kpi"])
    assert X.shape == (N_A + 4, 1)
    assert Y.shape == (N_A + 4, 1)
    assert tools == ["A"] * N_A + ["B"] * 4
    with pytest.raises(ValueError, match="at least one record"):
        records_to_arrays_with_tools([], ["knob"], ["kpi"])


def test_update_registers_unseen_tools_implicitly():
    Xa, ya, Xb, yb, Xt, yt_true = _two_tool_data(8)
    model = MultiToolGPForwardModel(
        input_keys=["knob"], output_keys=["kpi"], rank=1, n_restarts=2, seed=0
    )
    model.update(_records_for("A", Xa, ya))  # fit-from-scratch path
    assert model.tools_ == ["A"]
    model.update(_records_for("B", Xb, yb))  # unseen tool id -> implicit add_tool
    assert model.tools_ == ["A", "B"]
    assert model.tool_counts_ == {"A": N_A, "B": 8}
    assert model.n_train_ == N_A + 8
    # B is now a KNOWN tool: its prediction differs from the population fallback
    assert not np.allclose(
        model.predict(Xt, tool_id="B").mean, model.predict(Xt, tool_id="ghost").mean
    )


def test_add_tool_without_data_stays_on_fallback():
    Xa, ya, _, _, Xt, _ = _two_tool_data(4)
    model = MultiToolGPForwardModel(rank=1, n_restarts=2, seed=0).fit(Xa, ya, ["A"] * N_A)
    model.add_tool("C")  # declared, zero runs
    assert "C" not in model.tools_  # not a fitted tool
    pred_c = model.predict(Xt, tool_id="C")
    pred_ghost = model.predict(Xt, tool_id="ghost")
    np.testing.assert_array_equal(pred_c.mean, pred_ghost.mean)
    np.testing.assert_array_equal(pred_c.epistemic_sigma, pred_ghost.epistemic_sigma)


def test_adapt_to_tool_few_shot_improves_over_fallback(caplog):
    Xa, ya, Xb, yb, Xt, yt_true = _two_tool_data(8)
    model = MultiToolGPForwardModel(rank=1, n_restarts=3, seed=0).fit(Xa, ya, ["A"] * N_A)
    rmse_before = _rmse(model, Xt, yt_true, tool_id="B")  # unknown-tool fallback
    with caplog.at_level(logging.INFO, logger="rig.forward.multitask"):
        model.adapt_to_tool("B", Xb, yb)
    assert any(
        "adapt_to_tool" in r.getMessage() and "8 run" in r.getMessage() for r in caplog.records
    )
    assert model.tool_counts_["B"] == 8
    rmse_after = _rmse(model, Xt, yt_true, tool_id="B")
    assert rmse_after < rmse_before, (rmse_before, rmse_after)


# -- hygiene ---------------------------------------------------------------------


def test_fit_is_deterministic():
    Xa, ya, Xb, yb, *_ = _two_tool_data(4)
    X = np.vstack([Xa, Xb])
    y = np.concatenate([ya, yb])
    tools = ["A"] * N_A + ["B"] * 4
    m1 = MultiToolGPForwardModel(rank=2, n_restarts=3, seed=7).fit(X, y, tools)
    m2 = MultiToolGPForwardModel(rank=2, n_restarts=3, seed=7).fit(X, y, tools)
    x = np.array([2.5])
    np.testing.assert_array_equal(m1.predict(x, tool_id="B").mean, m2.predict(x, tool_id="B").mean)
    np.testing.assert_array_equal(m1.tool_covariance_, m2.tool_covariance_)


def test_fit_validation_errors():
    with pytest.raises(ValueError, match="rank"):
        MultiToolGPForwardModel(rank=0)
    with pytest.raises(ValueError, match="disagree"):
        MultiToolGPForwardModel().fit(np.zeros((3, 1)), np.zeros(3), ["A", "B"])
    with pytest.raises(RuntimeError, match="not fitted"):
        MultiToolGPForwardModel().predict(np.array([1.0]))
    with pytest.raises(ValueError, match="input_keys"):
        MultiToolGPForwardModel().update([])


# -- (vi) posterior_cov: EPIG/batch AL can drive the chamber model (audit B5) -----


def test_posterior_cov_invariant_and_acquisition(few_shot_setup):
    # audit B5: the ICM model (and its for_tool view) must expose posterior_cov
    # so EPIG / BatchBALD can onboard a new chamber (§10.4). Known-tool diagonal
    # must equal predict's epistemic_sigma**2 (acquisition-consistency), and the
    # unknown-tool path must return finite covariances (not crash).
    from rig.active.acquisition import epig
    from rig.active.batch import select_batch

    _, multi, *_ = few_shot_setup
    Xq = np.linspace(0.5, 6.0, 5)[:, None]
    Xstar = np.array([[1.5], [4.5]])

    # known tool: diagonal == epistemic^2, symmetric
    cov = multi.posterior_cov(Xq, Xq, tool_id="B")
    epi = multi.predict(Xq, tool_id="B").epistemic_sigma  # (5, 1)
    assert cov.shape == (1, 5, 5)
    np.testing.assert_allclose(np.diagonal(cov[0]), epi[:, 0] ** 2, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(cov[0], cov[0].T, atol=1e-10)

    # unknown tool: finite, no crash (population mixture)
    cov_u = multi.posterior_cov(Xq, Xstar, tool_id="never-seen")
    assert cov_u.shape == (1, 5, 2)
    assert np.all(np.isfinite(cov_u))

    # the §10.4 use case: EPIG + diverse batch through the for_tool view — these
    # previously raised AttributeError on the missing posterior_cov.
    view = multi.for_tool("never-seen")
    e = epig(view, Xq, Xstar)
    assert e.shape == (5,) and np.all(np.isfinite(e)) and np.all(e >= 0.0)
    idx = select_batch(e, Xq, 2, model=view)
    assert len(idx) == 2 and len(set(idx)) == 2


# -- (vii) multi-output prediction VALUES, not just shapes (audit C11) -------------


def _two_output_two_tool(seed: int = RNG_SEED):
    """Two outputs with DISTINCT means/scales so a per-output de-standardization
    misalignment produces visibly wrong values."""
    rng = np.random.default_rng(seed)
    Xa = rng.uniform(0.0, 2.0 * np.pi, (N_A, 1))
    Xb = rng.uniform(0.0, 2.0 * np.pi, (12, 1))
    X = np.vstack([Xa, Xb])
    tools = ["A"] * N_A + ["B"] * 12

    def outs(Xin, tool_b):
        base = np.sin(Xin[:, 0])
        o0 = 100.0 + 10.0 * base  # large mean/scale
        o1 = -3.0 + 0.5 * np.cos(Xin[:, 0])  # small, negative-mean
        if tool_b is not None:
            o0 = o0 * (1.0 + EPS) + DELTA
        return np.stack([o0, o1], axis=-1)

    Ya = outs(Xa, None) + NOISE * rng.standard_normal((N_A, 2))
    Yb = outs(Xb, True) + NOISE * rng.standard_normal((12, 2))
    Y = np.vstack([Ya, Yb])
    return X, Y, tools


def test_multi_output_prediction_values_and_jacobian():
    # audit C11: every other correctness test is single-output; a 2-output model
    # with distinct per-output means/scales pins that de-standardization is
    # applied per output (a swap would move o0~100 and o1~-3 wildly).
    X, Y, tools = _two_output_two_tool()
    model = MultiToolGPForwardModel(rank=1, n_restarts=3, seed=0).fit(X, Y, tools)

    Xt = np.linspace(0.3, 6.0, 40)[:, None]
    pred = model.predict(Xt, tool_id="A")
    assert pred.mean.shape == (40, 2)
    # output 0 lives near +100, output 1 near -3: a per-output mixup is impossible
    # to pass here.
    assert 80.0 < float(pred.mean[:, 0].mean()) < 120.0
    assert -4.0 < float(pred.mean[:, 1].mean()) < -2.0

    # known-tool RMSE is small for both outputs (values are actually right)
    Ya_true = np.stack([100.0 + 10.0 * np.sin(Xt[:, 0]), -3.0 + 0.5 * np.cos(Xt[:, 0])], axis=-1)
    for j in range(2):
        rmse = float(np.sqrt(np.mean((pred.mean[:, j] - Ya_true[:, j]) ** 2)))
        assert rmse < 2.0, (j, rmse)

    # jacobian (m, d) vs finite differences, per output, known tool
    x0 = np.array([2.0])
    J = model.jacobian(x0, tool_id="A")
    assert J.shape == (2, 1)
    h = 1e-5
    fd = (model.predict(x0 + h, tool_id="A").mean - model.predict(x0 - h, tool_id="A").mean) / (
        2.0 * h
    )
    np.testing.assert_allclose(J[:, 0], fd, rtol=1e-3, atol=1e-3)


def _fit_multitool(seed: int = 0) -> MultiToolGPForwardModel:
    """Two related tools with a genuine offset, 2 outputs — enough tool disagreement
    that the unknown-tool inflation terms are non-trivial."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(36, 2))
    tools = np.array(["A"] * 18 + ["B"] * 18)
    offs = np.where(tools == "A", 0.0, 0.4)
    Y = np.stack(
        [np.sin(3.0 * X[:, 0]) + offs, 0.5 * np.cos(2.0 * X[:, 1]) + offs], axis=-1
    ) + rng.normal(0.0, 0.01, size=(36, 2))
    return MultiToolGPForwardModel().fit(X, Y, tools)


@pytest.mark.parametrize("tool_id", ["A", "ZZZ", None])
def test_posterior_cov_diagonal_matches_predict_for_known_and_unknown_tools(tool_id):
    """Audit 2026-07-17 (MED): posterior_cov's diagonal MUST equal predict's
    epistemic_sigma**2 for the SAME tool_id — the invariant test_gp.py and
    test_ensemble.py both pin for their tiers, and which multitask lacked.

    It was violated on the UNKNOWN-tool branch: predict inflates
    (max_t var_t + spread + (1-rho^2)*mean diag B) but posterior_cov returned only
    the mixture Sum_t w_t*Cov_t. epig() takes var_f_star from predict and
    Cov(f(x*),f(x)) from posterior_cov, so the two laws broke Cauchy-Schwarz and
    the info-gain log-ratio collapsed (~19x under-report; exactly 0.0 nats across a
    batch) — silently disabling EPIG on precisely the §10.4 chamber-onboarding path
    posterior_cov advertises, and precisely as lambda anneals 0.2->0.9 to let EPIG
    dominate.
    """
    model = _fit_multitool()
    x = np.array([[0.4, 0.6]])
    pd_ = model.predict(x[0], tool_id=tool_id)
    cov = model.posterior_cov(x, x, tool_id=tool_id)
    diag = np.array([cov[j, 0, 0] for j in range(cov.shape[0])])
    np.testing.assert_allclose(diag, pd_.epistemic_sigma**2, rtol=1e-9)


@pytest.mark.parametrize("tool_id", ["A", "ZZZ"])
def test_posterior_cov_is_symmetric_and_psd(tool_id):
    """The congruence rescale that fixes the unknown-tool diagonal must not cost
    PSD-ness (a non-PSD 'covariance' would make EPIG's log-determinant meaningless)."""
    model = _fit_multitool()
    rng = np.random.default_rng(1)
    Q = rng.uniform(0.0, 1.0, size=(10, 2))
    cov = model.posterior_cov(Q, Q, tool_id=tool_id)
    for j in range(cov.shape[0]):
        S = cov[j]
        np.testing.assert_allclose(S, S.T, atol=1e-12)
        assert np.linalg.eigvalsh((S + S.T) / 2.0).min() > -1e-8


@pytest.mark.parametrize("tool_id", ["A", "ZZZ"])
def test_epig_at_the_query_point_equals_bald(tool_id):
    """The identity EPIG(x; {x}) == BALD(x) holds exactly for ANY self-consistent
    joint model, so it is the sharpest end-to-end check that predict and
    posterior_cov describe the SAME law. Pre-fix the unknown-tool arm read
    EPIG=1.01 vs BALD=19.06."""
    from rig.active.acquisition import bald, epig

    model = _fit_multitool()
    view = model.for_tool(tool_id if tool_id is not None else "UNSEEN")
    x = np.array([[0.4, 0.6]])
    np.testing.assert_allclose(epig(view, x, x), bald(view, x), rtol=1e-6)


def test_unknown_tool_epistemic_dominates_known_tools():
    """The §5.8 LOTO invariant (BUILD_STATE WP-I, binding): unknown-tool epistemic
    dominates every known tool's ELEMENTWISE. This is why predict credits max_t var_t
    rather than the mixture's Sum_t w_t var_t — so posterior_cov is what had to be
    reconciled to predict, not the reverse."""
    model = _fit_multitool()
    rng = np.random.default_rng(2)
    Q = rng.uniform(0.0, 1.0, size=(12, 2))
    unknown = model.predict(Q, tool_id="ZZZ").epistemic_sigma
    for tool in ("A", "B"):
        assert np.all(unknown >= model.predict(Q, tool_id=tool).epistemic_sigma - 1e-12)
