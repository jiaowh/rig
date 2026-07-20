"""WP-F: acquisition + batch selection (implementation-plan §9.4/§9.5)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from rig.active.acquisition import (
    anneal,
    bald,
    cost_cooled_acquisition,
    epig,
    qlognehvi_phase2,
)
from rig.active.batch import select_batch
from rig.forward import GPForwardModel


def _fit_gp(seed=0, n=25):
    rng = np.random.default_rng(seed)
    X = np.linspace(0.0, 1.0, n)[:, None]
    y = np.sin(6.0 * X[:, 0])[:, None] + 0.02 * rng.standard_normal((n, 1))
    return GPForwardModel(n_restarts=2, seed=seed).fit(X, y), X


# ---------------------------------------------------------------------------
# BALD
# ---------------------------------------------------------------------------


def test_bald_higher_off_data():
    gp, _ = _fit_gp()
    # in-data (0.5) vs far-OOD (3.0): OOD epistemic huge => BALD huge.
    b = bald(gp, np.array([[0.5], [3.0]]))
    assert b[1] > 10 * b[0]
    assert np.all(b >= 0.0)


def test_bald_is_epistemic_closed_form():
    gp, _ = _fit_gp()
    x = np.array([[0.42]])
    d = gp.predict(x)
    ale = float(np.ravel(d.aleatoric_sigma)[0])
    epi = float(np.ravel(d.epistemic_sigma)[0])
    expected = 0.5 * np.log1p(epi**2 / ale**2)
    assert bald(gp, x)[0] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# EPIG — prediction-targeted
# ---------------------------------------------------------------------------


def test_epig_localizes_to_target():
    gp, _ = _fit_gp()
    # info about x*=0.55: a candidate NEXT to it beats a far one (per nat),
    # even though the far one has larger BALD (global info).
    Xc = np.array([[0.53], [3.0]])
    e = epig(gp, Xc, np.array([[0.55]]))
    assert e[0] > e[1]  # near-target candidate is more prediction-informative
    assert np.all(e >= 0.0)


def test_epig_vs_bald_distinction():
    gp, _ = _fit_gp()
    far = np.array([[3.0]])
    # far OOD: BALD large (global), EPIG about an in-data target ~0 (uninformative).
    assert bald(gp, far)[0] > 1.0
    assert epig(gp, far, np.array([[0.5]]))[0] < 0.05


def test_epig_matches_closed_form_mutual_information():
    # audit C6: EPIG's magnitude was never pinned (only ordering/sign), so a
    # squared-covariance or units bug survived. Pin it to the Gaussian closed form
    # info = -0.5·log(1 - ρ²), ρ² = Cov(f_x,f_x*)² / (σ_f²(x*)·(σ_f²(x)+σ_ale²(x))).
    gp, _ = _fit_gp()
    xc = np.array([[0.5]])
    xstar = np.array([[0.6]])
    d_c, d_s = gp.predict(xc), gp.predict(xstar)
    var_f_x = float(np.ravel(d_c.epistemic_sigma)[0]) ** 2
    ale2 = float(np.ravel(d_c.aleatoric_sigma)[0]) ** 2
    var_f_star = float(np.ravel(d_s.epistemic_sigma)[0]) ** 2
    c = float(gp.posterior_cov(xc, xstar)[0, 0, 0])
    rho2 = c**2 / (var_f_star * (var_f_x + ale2))
    expected = -0.5 * np.log(1.0 - rho2)
    assert epig(gp, xc, xstar)[0] == pytest.approx(expected, rel=1e-9)


def test_blend_weights_epig_by_lambda_not_bald():
    # audit C7: every blend test uses lam=0.5 (symmetric), so swapping the
    # EPIG/BALD weights was invisible. Pin the DIRECTION: at beta=0,
    # acq = lam·EPIG + (1-lam)·BALD, so lam multiplies EPIG (not BALD).
    gp, _ = _fit_gp()
    Xc = np.array([[0.53], [3.0]])  # near-target (EPIG-favored) vs far-OOD (BALD-favored)
    star = np.array([[0.55]])
    e = epig(gp, Xc, star)
    b = bald(gp, Xc)
    assert not np.allclose(e, b)  # families differ here, so direction is observable
    hi = cost_cooled_acquisition(gp, Xc, star, lam=0.9, beta=0.0)
    lo = cost_cooled_acquisition(gp, Xc, star, lam=0.1, beta=0.0)
    np.testing.assert_allclose(hi, 0.9 * e + 0.1 * b, rtol=1e-9)  # NOT 0.1·e + 0.9·b
    np.testing.assert_allclose(lo, 0.1 * e + 0.9 * b, rtol=1e-9)


# ---------------------------------------------------------------------------
# cost-cooled blend + anneal
# ---------------------------------------------------------------------------


def test_cost_cooling_downweights_expensive():
    gp, _ = _fit_gp()
    Xc = np.array([[0.5], [0.9]])
    star = np.array([[0.55]])
    recipes = [{"x": 0.5}, {"x": 0.9}]
    cheap = cost_cooled_acquisition(
        gp, Xc, star, cost_fn=lambda r: 1.0, recipes=recipes, lam=0.5, beta=1.0
    )
    expensive = cost_cooled_acquisition(
        gp,
        Xc,
        star,
        cost_fn=lambda r: 10.0 if r["x"] > 0.7 else 1.0,
        recipes=recipes,
        lam=0.5,
        beta=1.0,
    )
    # the x=0.9 candidate is 10x costlier => its acquisition drops ~10x; x=0.5 unchanged.
    assert expensive[1] == pytest.approx(cheap[1] / 10.0, rel=1e-9)
    assert expensive[0] == pytest.approx(cheap[0], rel=1e-9)


def test_beta_zero_is_cost_agnostic():
    gp, _ = _fit_gp()
    Xc = np.array([[0.5], [0.9]])
    star = np.array([[0.55]])
    recipes = [{"x": 0.5}, {"x": 0.9}]
    a = cost_cooled_acquisition(
        gp, Xc, star, cost_fn=lambda r: 5.0, recipes=recipes, lam=0.5, beta=0.0
    )
    b = cost_cooled_acquisition(gp, Xc, star, lam=0.5, beta=0.0)  # no cost
    np.testing.assert_allclose(a, b)  # cost^0 = 1


def test_anneal_endpoints_and_clamp():
    assert anneal(0.0, 0.2, 0.9) == pytest.approx(0.2)
    assert anneal(1.0, 0.2, 0.9) == pytest.approx(0.9)
    assert anneal(0.5, 1.0, 0.0) == pytest.approx(0.5)
    assert anneal(-1.0, 0.2, 0.9) == pytest.approx(0.2)  # clamped
    assert anneal(2.0, 1.0, 0.0) == pytest.approx(0.0)  # clamped


# ---------------------------------------------------------------------------
# Phase II — feasibility-weighted qLogNEHVI toward the spec box (§9.4/§11.3)
# ---------------------------------------------------------------------------


def _truth(X):
    """Two competing KPIs + a one-sided constraint channel, all linear/known."""
    X = np.atleast_2d(X)
    return np.stack([X[:, 0] + 0.3 * X[:, 1], X[:, 1] - 0.3 * X[:, 0], X[:, 0]], axis=1)


def _fit_mo_gp(seed=0, n=30, scale=(1.0, 1.0, 1.0), n_out=3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, 2))
    Y = (_truth(X) + 0.02 * rng.standard_normal((n, 3))) * np.asarray(scale)
    Y = Y[:, :n_out]
    return GPForwardModel(n_restarts=2, seed=seed).fit(X, Y), X


def _pool(n=16, seed=7):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 2))


_KEYS = ["a", "b", "c"]


def _spec(tol=0.2, center=(0.0, 0.0), scale=(1.0, 1.0), c_upper=None):
    t = {
        "a": {"target": center[0] * scale[0], "tol": tol * scale[0]},
        "b": {"target": center[1] * scale[1], "tol": tol * scale[1]},
    }
    if c_upper is not None:
        t["c"] = {"upper": c_upper}
    return {"targets": t}


def test_phase2_prefers_recipes_that_reach_the_spec_box():
    # Scored against GROUND TRUTH (_truth), never the GP's own opinion of its pick.
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool()
    s = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], X_baseline=Xb, seed=0)
    assert s.shape == (pool.shape[0],)
    assert np.all(np.isfinite(s))
    best, worst = _truth(pool[int(np.argmax(s))])[0], _truth(pool[int(np.argmin(s))])[0]
    # the top-ranked candidate genuinely lands in/near the |y| <= 0.2 spec box;
    # the bottom-ranked one is far outside it on the true function.
    assert max(abs(best[0]), abs(best[1])) < 0.3
    assert max(abs(worst[0]), abs(worst[1])) > 0.6


def test_phase2_is_deterministic_under_seed():
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool()
    kw = dict(X_baseline=Xb, seed=3)
    a = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], **kw)
    b = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], **kw)
    np.testing.assert_array_equal(a, b)  # §13.4: bit-identical, not merely close


def test_phase2_ranking_is_invariant_to_output_units():
    # THE reference-point/scale trap (§8.7: "hypervolume is acutely sensitive to
    # it"). The HV coordinates must be TOLERANCE-NORMALIZED margins ((w-|f-c|-b)/w).
    # Feed raw margins instead and the hypervolume is silently dominated by whichever
    # KPI carries the largest units - here 'b' in milli-units would outvote 'a' 1000:1
    # and the Pareto front would be a lie. Scaling an output AND its spec box by 1000
    # is a pure change of unit and must not move a single rank.
    pytest.importorskip("botorch")
    gp1, Xb1 = _fit_mo_gp(scale=(1.0, 1.0, 1.0), n_out=2)
    gp2, Xb2 = _fit_mo_gp(scale=(1.0, 1000.0, 1.0), n_out=2)
    pool = _pool()
    s1 = qlognehvi_phase2(gp1, pool, _spec(), _KEYS[:2], X_baseline=Xb1, seed=0)
    s2 = qlognehvi_phase2(gp2, pool, _spec(scale=(1.0, 1000.0)), _KEYS[:2], X_baseline=Xb2, seed=0)
    np.testing.assert_array_equal(np.argsort(-s1), np.argsort(-s2))  # the real claim
    # not bit-identical: the two GPs are separate L-BFGS-B fits on rescaled y, so
    # ~1e-6 of numerical residue survives the standardization. Ranks do not move.
    np.testing.assert_allclose(s1, s2, rtol=1e-5)


def test_phase2_cost_cooling_is_subtractive_in_log_space():
    # §9.4 writes Phase II as "qLogNEHVI(x)/cost(x)^beta", but the acquisition is a
    # LOG and is normally NEGATIVE, so dividing by cost^beta>1 RAISES it: taken
    # literally, cost-cooling would reward expensive runs. CArBO's quantity is
    # HVI/cost^beta, whose log is logHVI - beta*log cost. Pin the exact identity.
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool()
    base = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], X_baseline=Xb, seed=0)
    assert np.all(base < 0.0)  # the premise: a log acquisition, below 1 => negative
    recipes = [{"i": i} for i in range(pool.shape[0])]
    cooled = qlognehvi_phase2(
        gp,
        pool,
        _spec(),
        _KEYS[:2],
        X_baseline=Xb,
        seed=0,
        beta=1.0,
        cost_fn=lambda r: 10.0 if r["i"] % 2 == 0 else 1.0,
        recipes=recipes,
    )
    even, odd = np.arange(0, pool.shape[0], 2), np.arange(1, pool.shape[0], 2)
    np.testing.assert_allclose(cooled[even], base[even] - np.log(10.0), rtol=1e-9)
    np.testing.assert_allclose(cooled[odd], base[odd], rtol=1e-9)
    assert np.all(cooled[even] < base[even])  # 10x costlier => PENALIZED, not rewarded


def test_phase2_beta_zero_is_cost_agnostic():
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool()
    recipes = [{"i": i} for i in range(pool.shape[0])]
    a = qlognehvi_phase2(
        gp,
        pool,
        _spec(),
        _KEYS[:2],
        X_baseline=Xb,
        seed=0,
        beta=0.0,
        cost_fn=lambda r: 5.0,
        recipes=recipes,
    )
    b = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], X_baseline=Xb, seed=0)
    np.testing.assert_allclose(a, b)  # cost^0 = 1 => log cost term vanishes


def test_phase2_reference_point_tracks_the_spec_box_not_the_data():
    # §11.3: "reference point from the spec-box nadir, NOT a default". The origin of
    # the hypervolume is the SPEC, so with data/model/pool held fixed, moving the box
    # must move which candidate wins - toward the new box, on the true function.
    # Scope, honestly: this bites the compound "HV origin is the spec box" - verified
    # RED against an objective that ignores the box center AND against a ref point
    # detached from the nadir (-5, or a data-anchored -50). It does NOT pin the +10%
    # pad: 0.10 vs 0.0 is behaviourally invisible under the Log form (see the
    # qlognehvi_phase2 docstring). Nothing here justifies 0.10 over 0.0.
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool(n=40, seed=11)
    lo = qlognehvi_phase2(gp, pool, _spec(center=(-0.6, -0.6)), _KEYS[:2], X_baseline=Xb, seed=0)
    hi = qlognehvi_phase2(gp, pool, _spec(center=(0.6, 0.6)), _KEYS[:2], X_baseline=Xb, seed=0)
    y_lo, y_hi = _truth(pool[int(np.argmax(lo))])[0], _truth(pool[int(np.argmax(hi))])[0]
    assert int(np.argmax(lo)) != int(np.argmax(hi))
    assert y_lo[0] < 0.0 and y_lo[1] < 0.0  # picked for the box at (-0.6, -0.6)
    assert y_hi[0] > 0.0 and y_hi[1] > 0.0  # picked for the box at (+0.6, +0.6)


def test_phase2_one_sided_target_is_a_feasibility_constraint():
    # §11.3 native outcome-constraint form. A one-sided target has no finite
    # half-width, so it cannot be an HV axis; it must down-weight the acquisition
    # where it is violated. Adding `c <= -0.3` to the spec must cost the c-violating
    # candidates far more than the c-satisfying ones (c = x0 on the true function).
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=3)
    pool = _pool(n=40, seed=11)
    free = qlognehvi_phase2(gp, pool, _spec(), _KEYS, X_baseline=Xb, seed=0)
    gated = qlognehvi_phase2(gp, pool, _spec(c_upper=-0.3), _KEYS, X_baseline=Xb, seed=0)
    delta = gated - free
    violates = _truth(pool)[:, 2] > 0.0  # c well above the -0.3 limit
    satisfies = _truth(pool)[:, 2] < -0.6  # c well below it
    assert violates.any() and satisfies.any()
    assert delta[violates].max() < delta[satisfies].min()
    assert delta[violates].mean() < -1.0  # a real feasibility weight, not a rounding


def test_phase2_credited_band_shrinks_the_acquisition_with_kappa():
    # The margin carries the §8.4 credited band kappa*sigma_ale, so Phase II chases
    # the same feasibility the §8 gate will certify. A bigger band = less room to
    # spec = strictly less hypervolume to win. Drop the band and kappa is inert.
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    pool = _pool()
    tight = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], X_baseline=Xb, seed=0, kappa=0.0)
    wide = qlognehvi_phase2(gp, pool, _spec(), _KEYS[:2], X_baseline=Xb, seed=0, kappa=3.0)
    assert wide.max() < tight.max()


def test_phase2_rejects_a_single_objective_spec():
    pytest.importorskip("botorch")
    gp, Xb = _fit_mo_gp(n_out=2)
    with pytest.raises(ValueError, match="MULTI-objective"):
        qlognehvi_phase2(
            gp,
            _pool(),
            {"targets": {"a": {"target": 0.0, "tol": 0.2}}},
            _KEYS[:2],
            X_baseline=Xb,
        )


def test_phase2_requires_x_baseline():
    # qLogNEHVI is the NOISY form: without the observed set there is no posterior
    # Pareto front to price improvement against. Fail loud, never a silent default.
    pytest.importorskip("botorch")
    gp, _ = _fit_mo_gp(n_out=2)
    with pytest.raises(ValueError, match="X_baseline"):
        qlognehvi_phase2(gp, _pool(), _spec(), _KEYS[:2], X_baseline=np.empty((0, 2)))


def test_phase2_view_posterior_is_the_models_own_law():
    # The WP-I standing decision, applied to the BoTorch view: the law handed to
    # qLogNEHVI must be the model's law. Variance MUST equal predict's
    # epistemic_sigma**2 (latent/epistemic only - it is what EHVI takes the
    # expectation over), the joint block MUST be posterior_cov, and band_shift must
    # touch ONLY the outputs it is given (objective outputs get their band from the
    # objective callable, so a leak there would double-count).
    torch = pytest.importorskip("torch")
    pytest.importorskip("botorch")
    from rig.active.acquisition import _pessimistic_view

    gp, _ = _fit_mo_gp(n_out=3)
    X = _pool(n=5)
    shift = np.array([0.0, 0.0, 2.0])  # only output 'c' is band-shifted
    view = _pessimistic_view(gp, shift, 3)
    post = view.posterior(torch.as_tensor(X, dtype=torch.double).unsqueeze(-2))  # (5,1,3)
    d = gp.predict(X)
    mean = np.asarray(post.mean.detach().numpy()).reshape(5, 3)
    var = np.asarray(post.variance.detach().numpy()).reshape(5, 3)
    expected = np.asarray(d.mean) + shift[None, :] * np.asarray(d.aleatoric_sigma)
    np.testing.assert_allclose(mean, expected, rtol=1e-10)
    np.testing.assert_allclose(var, np.asarray(d.epistemic_sigma) ** 2, rtol=1e-6)
    # The JOINT across q is what makes qLogNEHVI *noisy* - it samples candidates and
    # X_baseline together, so a diagonal-only law would silently turn the baseline
    # Pareto front into fixed points and destroy the "N" in NEHVI.
    joint = view.posterior(torch.as_tensor(X, dtype=torch.double))  # q=5, event (5,3)
    assert joint.distribution._interleaved is False  # from_independent_mvns: task-major
    cov = np.asarray(joint.distribution.covariance_matrix.detach().numpy())
    ref = gp.posterior_cov(X, X)  # (3, 5, 5)
    q = X.shape[0]
    for j in range(3):
        block = cov[j * q : (j + 1) * q, j * q : (j + 1) * q]
        np.testing.assert_allclose(block, ref[j], rtol=1e-6, atol=1e-9)
        off = cov[j * q : (j + 1) * q, ((j + 1) % 3) * q : ((j + 1) % 3 + 1) * q]
        np.testing.assert_allclose(off, 0.0, atol=1e-14)  # outputs stay independent


def test_phase2_bands_constraint_outputs_only(monkeypatch):
    # Each output must be banded by exactly ONE mechanism: two-sided (objective)
    # outputs get kappa*sigma_ale inside the margin callable, one-sided (constraint)
    # outputs get it as a mean displacement toward their limit. Band an objective
    # output in the view as well and it is double-counted AND spuriously asymmetric
    # (the box is two-sided; a signed shift is meaningless there). Spy on the wiring,
    # because the view itself cannot know which outputs it should have been given.
    pytest.importorskip("botorch")
    import rig.active.acquisition as acq

    seen = {}
    real = acq._pessimistic_view

    def spy(model, band_shift, m):
        seen["shift"] = np.array(band_shift, copy=True)
        return real(model, band_shift, m)

    monkeypatch.setattr(acq, "_pessimistic_view", spy)
    gp, Xb = _fit_mo_gp(n_out=3)
    acq.qlognehvi_phase2(gp, _pool(), _spec(c_upper=-0.3), _KEYS, X_baseline=Xb, seed=0, kappa=2.0)
    # 'a','b' are HV axes -> no mean displacement; 'c' is upper-bounded -> +kappa,
    # pushing its mean UP toward the limit it must not cross.
    np.testing.assert_array_equal(seen["shift"], np.array([0.0, 0.0, 2.0]))


def test_phase2_import_rig_stays_torch_free():
    # Binding: `import rig` must not drag in torch. Phase II lives in an eagerly
    # imported module, so its botorch imports have to stay inside the function.
    # Checked in a SUBPROCESS - torch is already in sys.modules by this point.
    code = (
        "import sys, rig, rig.active.acquisition;"
        "assert 'torch' not in sys.modules, 'rig imported torch';"
        "assert 'botorch' not in sys.modules, 'rig imported botorch'"
    )
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# batch selection (§9.5)
# ---------------------------------------------------------------------------


def test_select_batch_avoids_duplicates():
    gp, _ = _fit_gp()
    # a pool with a cluster of near-duplicates (all ~0.5) and spread points.
    pool = np.array([[0.50], [0.501], [0.502], [0.1], [0.9]])
    acq = np.array([1.0, 0.99, 0.98, 0.5, 0.5])  # cluster has highest acq
    idx = select_batch(acq, pool, q=3, model=gp, w_div=0.8)
    chosen = pool[idx, 0]
    # must NOT pick 3 near-duplicates: the spread points get in despite lower acq.
    assert len(set(np.round(chosen, 2))) >= 2


def test_select_batch_respects_q_and_bounds():
    gp, _ = _fit_gp()
    pool = np.array([[0.1], [0.5], [0.9]])
    acq = np.array([0.3, 0.9, 0.6])
    assert len(select_batch(acq, pool, q=2, model=gp)) == 2
    assert len(select_batch(acq, pool, q=10, model=gp)) == 3  # capped at pool size
    assert select_batch(acq, pool, q=0, model=gp) == []
    assert select_batch(acq, pool, q=2)[0] == 1  # top-acq first, input fallback


def test_select_batch_top_acquisition_is_first():
    pool = np.array([[0.1], [0.5], [0.9], [0.7]])
    acq = np.array([0.2, 0.4, 0.95, 0.3])
    assert select_batch(acq, pool, q=2)[0] == 2  # argmax acq
