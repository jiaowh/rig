"""WP-E §8 hardening (2026-07-22): the two owed robust-objective features, opt-in and
default-off byte-identical.

1. PGD δ-box (``delta_mode="pgd"``) — the §8.5 ``max_{δ∈Δ}`` inner problem by projected
   gradient ascent, REPLACING the first-order ``Σ|J|·Δ`` Taylor term. Benchmarked
   against a brute-force box maximum on a CURVED response where the linearization
   under-estimates the worst case (the audit's "benchmark those elements").
2. Flow typicality (``typicality=``) — the §8.2 normalizing-flow off-manifold screen,
   applied ALONGSIDE the Mahalanobis floor. Proves it closes the multimodal hole a
   unimodal Mahalanobis distance has, and that the TYPICALITY-set formulation
   (``−|log p − E|``) rejects an atypically-high-density point that RAW log-likelihood
   thresholding would admit (Nalisnick et al. 2019).

The PGD half is torch-free (numpy GP-tier). The typicality half needs the ``[torch]``
extra; those tests ``importorskip`` it. A subprocess test pins that ``import rig`` stays
torch-free regardless.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from rig.interfaces import (
    ContinuousVariable,
    Infeasible,
    PredictiveDistribution,
)
from rig.inverse import PessimisticInverseSolver

# ---------------------------------------------------------------------------
# controllable model with a chooseable mean CURVATURE (drives PGD-vs-Taylor)
# ---------------------------------------------------------------------------


class _Model:
    """ForwardModel whose mean/jacobian come from callables, so we can dial in a
    convex response the first-order Taylor δ under-estimates. Constant aleatoric, zero
    epistemic, callable support."""

    def __init__(self, mean_fn, jac_fn, ale, m=1, support_fn=None):
        self.mean_fn = mean_fn
        self.jac_fn = jac_fn
        self.ale = np.asarray(ale, float)
        self._m = m
        self.support_fn = support_fn or (lambda x: 0.0)

    def predict(self, x) -> PredictiveDistribution:
        x = np.asarray(x, float)
        single = x.ndim == 1
        Xq = np.atleast_2d(x)
        mean = np.stack([np.atleast_1d(np.asarray(self.mean_fn(xi), float)) for xi in Xq])
        ale = np.broadcast_to(self.ale, mean.shape).copy()
        epi = np.zeros_like(mean)
        if single:
            return PredictiveDistribution(mean[0], ale[0], epi[0], None)
        return PredictiveDistribution(mean, ale, epi, None)

    def jacobian(self, x) -> np.ndarray:
        return np.atleast_2d(np.asarray(self.jac_fn(np.asarray(x, float)), float))

    def support_score(self, x):
        x = np.asarray(x, float)
        if x.ndim == 1:
            return float(self.support_fn(x))
        return np.array([float(self.support_fn(xi)) for xi in x])

    def update(self, records) -> None:  # pragma: no cover - not exercised
        pass


def _var(name, lo, hi):
    return ContinuousVariable(name, lo, hi)


def _quadratic_model(c: float, m_out: int = 1):
    """μ(x) = c·Σ x_i²  (convex); jac_j = 2c·x_j replicated across ``m_out`` outputs."""

    def mean_fn(xi):
        return np.full(m_out, c * float(np.sum(xi**2)))

    def jac_fn(xi):
        return np.tile(2.0 * c * xi, (m_out, 1))

    return _Model(mean_fn, jac_fn, ale=[0.05] * m_out, m=m_out)


def _brute_force_box_dev(model, x, delta_raw, j=0, n=41):
    """Ground-truth max |μ_j(x+δ) − μ_j(x)| over the ℓ∞ box, by dense grid (the honest
    reference the PGD lower bound is measured against). Grid per axis, full product for
    small dim."""
    x = np.asarray(x, float)
    mu0 = float(np.atleast_1d(model.predict(x).mean)[j])
    axes = [np.linspace(-dr, dr, n) for dr in delta_raw]
    grids = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([g.ravel() for g in grids], axis=1)  # (N, d)
    best = 0.0
    for p in pts:
        mu = float(np.atleast_1d(model.predict(x + p).mean)[j])
        best = max(best, abs(mu - mu0))
    return best


# ---------------------------------------------------------------------------
# PGD δ-box (§8.5) — torch-free
# ---------------------------------------------------------------------------


def _pgd_solver(model, var, *, delta_mode, delta_frac=0.1, **kw):
    return PessimisticInverseSolver(
        model,
        [var],
        ["y"],
        support_floor=-100.0,
        z_epi=0.0,
        delta_frac=delta_frac,
        delta_mode=delta_mode,
        seed=0,
        **kw,
    )


def test_pgd_recovers_taylor_on_a_linear_response():
    """The sanity anchor: on a LINEAR μ the box worst case IS the first-order term, and
    PGD (δ=0 start, step Δ/4, marching to the corner) reproduces ``Σ|J|·Δ`` to rounding.
    So the two modes agree exactly where there is no curvature, and any later divergence
    is a curvature signal, not a discretization artifact."""
    model = _Model(
        lambda xi: np.array([3.0 * xi[0] - 2.0 * xi[1]]),
        lambda xi: np.array([[3.0, -2.0]]),
        ale=[0.1],
        m=1,
    )
    solver = PessimisticInverseSolver(
        model,
        [_var("a", 0.0, 4.0), _var("b", 0.0, 4.0)],
        ["y"],
        support_floor=-100.0,
        z_epi=0.0,
        delta_frac=0.1,
        delta_mode="pgd",
        seed=0,
    )
    x = np.array([2.0, 1.5])
    taylor = np.abs(model.jacobian(x)) @ solver._delta_raw  # Σ|J|·Δ
    pgd = solver._pgd_delta(x, model)
    np.testing.assert_allclose(pgd, taylor, atol=1e-9)


def test_pgd_catches_curvature_that_taylor_underestimates():
    """THE benchmark (audit F5 "benchmark those elements"): a convex μ=c·x² whose true
    box worst case EXCEEDS the linear extrapolation at x. Taylor UNDER-estimates it; PGD
    catches it and matches the brute-force box maximum. This is the load-bearing PGD
    guard — red-proofed by hand (BUILD_LOG): stubbing `_pgd_delta` to return the Taylor
    term makes the `pgd > taylor` and `pgd ≈ brute` assertions fail."""
    c = 1.5
    model = _quadratic_model(c)
    solver = _pgd_solver(model, _var("x", 0.5, 4.0), delta_mode="pgd", delta_frac=0.1)
    x = np.array([2.0])
    Δ = solver._delta_raw  # 0.1*(4-0.5) = 0.35
    taylor = float((np.abs(model.jacobian(x)) @ Δ)[0])  # 2c·x·Δ
    pgd = float(solver._pgd_delta(x, model)[0])
    brute = _brute_force_box_dev(model, x, Δ)
    # Taylor STRICTLY under-estimates the true worst case; PGD does not.
    assert taylor < brute, (taylor, brute)
    assert pgd > taylor + 1e-6, "PGD failed to catch curvature Taylor missed"
    # PGD is not merely bigger — it lands on the true box maximum (the +Δ corner).
    assert abs(pgd - brute) < 1e-6, (pgd, brute)
    # closed-form check: brute = 2c·x·Δ + c·Δ² (the extra cΔ² is the curvature term).
    np.testing.assert_allclose(brute, taylor + c * float(Δ[0]) ** 2, atol=1e-6)


def test_pgd_matches_brute_force_on_a_2d_box():
    """The anisotropic multi-axis box: PGD marches every coordinate to its own corner
    and matches the brute-force maximum on a 2-D convex response (differing per-axis Δ)."""
    c = 1.2
    model = _quadratic_model(c)
    solver = PessimisticInverseSolver(
        model,
        [_var("a", 0.0, 4.0), _var("b", 0.0, 2.0)],
        ["y"],
        support_floor=-100.0,
        z_epi=0.0,
        delta_frac=0.1,
        delta_mode="pgd",
        seed=0,
    )
    x = np.array([2.5, 1.0])
    pgd = float(solver._pgd_delta(x, model)[0])
    brute = _brute_force_box_dev(model, x, solver._delta_raw, n=31)
    assert abs(pgd - brute) < 1e-3, (pgd, brute)


def test_pgd_is_deterministic():
    """No RNG — the fixed δ=0 start makes two calls, and two constructions, bit-identical
    (§13.4)."""
    model = _quadratic_model(1.5)
    s1 = _pgd_solver(model, _var("x", 0.5, 4.0), delta_mode="pgd")
    s2 = _pgd_solver(model, _var("x", 0.5, 4.0), delta_mode="pgd")
    x = np.array([1.7])
    a = s1._pgd_delta(x, model)
    b = s1._pgd_delta(x, model)
    c = s2._pgd_delta(x, model)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_pgd_default_off_is_byte_identical_to_taylor():
    """DEFAULT-OFF byte-identity, proven by test not asserted: a full seeded solve with
    `delta_mode` LEFT DEFAULT must return exactly what an explicit `delta_mode="taylor"`
    solve returns — recipe values, confidences, feasibility. The M2 sweep constructs the
    solver without the knob; this pins that the knob's mere existence moved nothing."""
    model = _quadratic_model(0.5, m_out=1)
    var = [_var("x", 0.5, 4.0)]
    spec = {"targets": {"y": (2.0, 6.0)}, "max_candidates": 3}
    common = dict(support_floor=-100.0, z_epi=0.0, delta_frac=0.02, seed=7)
    default = PessimisticInverseSolver(model, var, ["y"], **common).solve(spec)
    taylor = PessimisticInverseSolver(model, var, ["y"], delta_mode="taylor", **common).solve(spec)
    assert isinstance(default, list) and isinstance(taylor, list) and default
    assert [c.recipe["x"] for c in default] == [c.recipe["x"] for c in taylor]
    assert [c.confidence for c in default] == [c.confidence for c in taylor]


def test_pgd_tightens_the_margin_end_to_end():
    """The feature must MATTER, not merely differ: on a steep convex response a box the
    Taylor margin certifies FEASIBLE becomes INFEASIBLE under PGD, because the curvature
    deviation PGD catches eats the credited margin. This is the §8.5 pessimism working —
    a recipe robust only to the LINEARIZED tolerance is not robust to the real box.

    Sizing: at the margin-balance optimum the solver picks μ = box centre (here x=2,
    μ=c·4), where the δ displacement cancels between the two boundary margins, so the
    credited margin is ``(tol − s_δ)/σ``. With Δ = 0.15·3 = 0.45 the Taylor term is
    ``4cΔ = 3.6`` and the PGD term is ``4cΔ + cΔ² = 4.005``; tol=3.85 sits strictly
    between them (Taylor: margin +5σ, PGD: −3.1σ), so the curvature term ``cΔ²`` is
    exactly what flips the verdict."""
    c = 2.0
    model = _quadratic_model(c)  # μ = 2x², convex
    var = [_var("x", 0.5, 3.5)]
    spec = {"targets": {"y": {"target": 8.0, "tol": 3.85}}}  # 8 = 2·2², x≈2
    common = dict(support_floor=-100.0, z_epi=0.0, delta_frac=0.15, kappa=1.0, seed=0)
    taylor = PessimisticInverseSolver(model, var, ["y"], delta_mode="taylor", **common).solve(spec)
    pgd = PessimisticInverseSolver(model, var, ["y"], delta_mode="pgd", **common).solve(spec)
    assert isinstance(taylor, list) and taylor, "Taylor should certify this box"
    assert isinstance(pgd, Infeasible), "PGD must catch the curvature the linearization missed"


def test_pgd_rejects_the_analytic_grad_combination():
    """Fail loud: the analytic gradient forms δ from Taylor+Hessian, so pairing it with a
    PGD value path would descend one objective and certify against another. Refuse at
    construction (a fitted GP is needed to even reach the analytic_grad builder, so use
    one) rather than silently ship the inconsistency."""
    from rig.forward import GPForwardModel

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(24, 2))
    gp = GPForwardModel(n_restarts=1, seed=0).fit(X, np.tanh(X.sum(1, keepdims=True)))
    with pytest.raises(ValueError, match="incompatible"):
        PessimisticInverseSolver(
            gp,
            [_var("x0", -1, 1), _var("x1", -1, 1)],
            ["y"],
            X_train=X,
            delta_mode="pgd",
            analytic_grad=True,
        )


def test_delta_mode_validation():
    model = _quadratic_model(1.0)
    with pytest.raises(ValueError, match="delta_mode must be"):
        _pgd_solver(model, _var("x", 0.0, 1.0), delta_mode="nope")
    with pytest.raises(ValueError, match="pgd_steps must be"):
        PessimisticInverseSolver(
            model,
            [_var("x", 0.0, 1.0)],
            ["y"],
            support_floor=-1.0,
            delta_mode="pgd",
            pgd_steps=0,
        )


# ---------------------------------------------------------------------------
# Flow typicality (§8.2) — needs the [torch] extra
# ---------------------------------------------------------------------------


def _maha_score_and_floor(X_train, x):
    """A plain UNIMODAL Mahalanobis support (mean + pooled covariance) and its 5th-pct
    floor — the cheap §8.2 fallback, computed independently so the test does not lean on
    the solver's own implementation to make its point."""
    mu = X_train.mean(0)
    cov = np.cov(X_train.T) + 1e-6 * np.eye(X_train.shape[1])
    inv = np.linalg.inv(cov)

    def score(z):
        z = np.atleast_2d(z) - mu
        return -np.sqrt(np.einsum("ij,jk,ik->i", z, inv, z))

    return float(score(x)[0]), float(np.percentile(score(X_train), 5.0))


def _bimodal(seed=0):
    """Two well-separated 2-D clusters at (±4, 0). The gap centre (0, 0) sits at the
    POOLED mean (Mahalanobis distance ≈ 0) yet is ~8σ from every training point — the
    canonical multimodal hole."""
    rng = np.random.default_rng(seed)
    A = rng.normal([-4.0, 0.0], 0.5, (200, 2))
    B = rng.normal([4.0, 0.0], 0.5, (200, 2))
    return np.vstack([A, B])


@pytest.fixture(scope="module")
def bimodal_flow():
    pytest.importorskip("zuko")
    from rig.inverse.typicality import FlowTypicalityScore

    X = _bimodal(seed=0)
    flow = FlowTypicalityScore(transforms=2, hidden=(48, 48), max_epochs=120, seed=1).fit(X)
    return X, flow


def test_typicality_closes_the_multimodal_hole_mahalanobis_misses(bimodal_flow):
    """THE §8.2 hole this feature exists to close, proven closed. The gap centre (0,0) of
    a bimodal training set is admitted by the UNIMODAL Mahalanobis score (it sits at the
    pooled mean) but is a far-OOD hole no data is near. The flow-typicality screen
    REJECTS it while still ACCEPTING a genuine on-mode point.

    Red-proofed by hand (BUILD_LOG): replacing the typicality score with a raw-Mahalanobis
    proxy — i.e. dropping the flow — makes the `REJECT` assertion fail, which is exactly
    the hole. This is the load-bearing typicality guard."""
    X, flow = bimodal_flow
    gap = np.array([0.0, 0.0])
    on_mode = np.array([4.0, 0.0])

    # (i) the flow-typicality screen: gap REJECTED, an on-mode point ACCEPTED.
    assert flow.score(gap) < flow.floor, (flow.score(gap), flow.floor)
    assert flow.score(on_mode) >= flow.floor
    # (ii) the unimodal Mahalanobis fallback WRONGLY ADMITS the same gap.
    m_gap, m_floor = _maha_score_and_floor(X, gap)
    assert m_gap >= m_floor, "precondition: Mahalanobis must (wrongly) accept the gap"
    # (iii) and the gap really is far from all data — not a labeling accident.
    assert np.min(np.linalg.norm(X - gap, axis=1)) > 2.0


def test_typicality_rejects_high_raw_density_point_that_raw_loglik_would_admit():
    """WHY the score is ``−|log p − E|`` and NOT raw log-likelihood (Nalisnick et al.
    2019: deep generative models over-assign likelihood). A sharp minority spike inside a
    broad majority has a peak whose RAW log-density sits far ABOVE the 5th-percentile raw
    floor — so a raw-log-likelihood screen would ADMIT the spike centre — yet the
    TYPICALITY-set formulation rejects it as atypically high. The two-sidedness is the
    same property that flags the high-density-but-atypical OOD points raw density cannot.

    Red-proofed by hand (BUILD_LOG): switching the screen to ``log p ≥ raw_floor`` makes
    the REJECT assertion fail — the raw rule admits the spike."""
    pytest.importorskip("zuko")
    from rig.inverse.typicality import FlowTypicalityScore

    rng = np.random.default_rng(0)
    broad = rng.normal([0.0, 0.0], 1.6, (450, 2))
    spike = rng.normal([5.0, 0.0], 0.09, (50, 2))
    X = np.vstack([broad, spike])
    flow = FlowTypicalityScore(transforms=3, hidden=(64, 64), max_epochs=180, seed=2).fit(X)

    spike_c = np.array([5.0, 0.0])
    raw_floor = float(np.percentile(flow.log_density(X), 5.0))
    lp = float(flow.log_density(spike_c))
    # a RAW-log-likelihood screen (accept if log p ≥ 5th-pct floor) would ADMIT it:
    assert lp >= raw_floor, "precondition: the spike centre has high raw density"
    # the TYPICALITY screen rejects it (atypically high — far from E_train[log p]):
    assert flow.score(spike_c) < flow.floor, (flow.score(spike_c), flow.floor)
    # a genuinely typical broad-cluster point still passes.
    assert flow.score(rng.normal([0.0, 0.0], 1.6)) >= flow.floor


def test_typicality_floor_fail_closed_when_unfitted():
    """Fail-closed, mirroring the solver's support_floor contract: an UNFITTED screen has
    no calibrated floor and must raise on `.floor`/`.score`, never silently admit
    everything. Also pins that the solver refuses an unfitted screen at construction."""
    pytest.importorskip("zuko")
    from rig.inverse.typicality import FlowTypicalityScore

    ft = FlowTypicalityScore()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = ft.floor
    with pytest.raises(RuntimeError, match="not fitted"):
        ft.score(np.array([0.0, 0.0]))

    model = _quadratic_model(1.0, m_out=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        PessimisticInverseSolver(
            model, [_var("x", 0.0, 1.0)], ["y"], support_floor=-1.0, typicality=ft
        )


def test_solver_typicality_rejects_offmanifold_optimum_mahalanobis_admits(bimodal_flow):
    """End-to-end: a spec whose only pre-image is the multimodal GAP. The model's own
    (unimodal Mahalanobis) support ADMITS the gap, so WITHOUT the flow screen the solver
    certifies an off-manifold recipe FEASIBLE. WITH the flow screen the gap is rejected
    and the solver abstains with the §8.2 manifold verdict. The with/without contrast IS
    the red-proof: the flow is the only thing that changes the answer."""
    X, flow = bimodal_flow

    # y = x0 (target forces x0≈0 = the gap); support = the REAL unimodal Mahalanobis on X,
    # so the floor (5th pct) admits the pooled-mean gap exactly as §8.2 warns.
    mu = X.mean(0)
    inv = np.linalg.inv(np.cov(X.T) + 1e-6 * np.eye(2))

    def maha(x):
        z = np.asarray(x, float) - mu
        return -float(np.sqrt(z @ inv @ z))

    model = _Model(
        lambda xi: np.array([xi[0]]),
        lambda xi: np.array([[1.0, 0.0]]),
        ale=[0.05],
        m=1,
        support_fn=maha,
    )
    variables = [_var("x0", -6.0, 6.0), _var("x1", -6.0, 6.0)]
    spec = {"targets": {"y": (-0.5, 0.5)}, "max_candidates": 3}
    common = dict(output_keys=["y"], X_train=X, z_epi=0.0, delta_frac=0.0, kappa=1.0, seed=0)

    # WITHOUT the flow screen: Mahalanobis admits the gap -> FEASIBLE (the §8.2 hole).
    without = PessimisticInverseSolver(model, variables, **common).solve(spec)
    assert isinstance(without, list) and without, "Mahalanobis alone should admit the gap"
    assert all(abs(c.recipe["x0"]) < 0.5 for c in without)  # sits in the gap
    assert (
        np.min(
            np.linalg.norm(X - np.array([without[0].recipe["x0"], without[0].recipe["x1"]]), axis=1)
        )
        > 1.5
    )

    # WITH the flow screen: the gap is off-manifold -> INFEASIBLE, manifold verdict.
    with_flow = PessimisticInverseSolver(model, variables, typicality=flow, **common).solve(spec)
    assert isinstance(with_flow, Infeasible)
    assert "manifold" in with_flow.reason.lower() or "typicality" in with_flow.reason.lower()


def test_solver_typicality_none_is_byte_identical(bimodal_flow):
    """Default-off byte-identity for the typicality knob: passing `typicality=None`
    explicitly returns exactly what omitting it returns, on an on-manifold spec that IS
    feasible (so the comparison is over a non-empty candidate list)."""
    X, _flow = bimodal_flow
    mu = X.mean(0)
    inv = np.linalg.inv(np.cov(X.T) + 1e-6 * np.eye(2))

    def maha(x):
        z = np.asarray(x, float) - mu
        return -float(np.sqrt(z @ inv @ z))

    model = _Model(
        lambda xi: np.array([xi[0]]),
        lambda xi: np.array([[1.0, 0.0]]),
        ale=[0.05],
        m=1,
        support_fn=maha,
    )
    variables = [_var("x0", -6.0, 6.0), _var("x1", -6.0, 6.0)]
    spec = {"targets": {"y": (3.5, 4.5)}, "max_candidates": 3}  # on-mode (x0≈4)
    common = dict(output_keys=["y"], X_train=X, z_epi=0.0, delta_frac=0.0, kappa=1.0, seed=0)
    omitted = PessimisticInverseSolver(model, variables, **common).solve(spec)
    explicit = PessimisticInverseSolver(model, variables, typicality=None, **common).solve(spec)
    assert isinstance(omitted, list) and omitted
    assert [c.recipe["x0"] for c in omitted] == [c.recipe["x0"] for c in explicit]
    assert [c.recipe["x1"] for c in omitted] == [c.recipe["x1"] for c in explicit]


def test_typicality_score_is_deterministic():
    """A same-seeded fit replays the identical flow, so scores/floor are bit-reproducible
    (§13.4)."""
    pytest.importorskip("zuko")
    from rig.inverse.typicality import FlowTypicalityScore

    X = _bimodal(seed=3)
    a = FlowTypicalityScore(transforms=2, hidden=(32, 32), max_epochs=60, seed=5).fit(X)
    b = FlowTypicalityScore(transforms=2, hidden=(32, 32), max_epochs=60, seed=5).fit(X)
    probe = np.array([[0.0, 0.0], [4.0, 0.0], [1.0, 1.0]])
    np.testing.assert_array_equal(a.score(probe), b.score(probe))
    assert a.floor == b.floor


# ---------------------------------------------------------------------------
# torch-free base import (extend the established subprocess pattern)
# ---------------------------------------------------------------------------


def test_import_rig_stays_torch_free_with_the_hardening_wired():
    """Binding invariant: the PGD path and the `typicality=` parameter live in the
    eagerly-imported `rig.inverse.pessimistic`, so `import rig` — and constructing a
    solver — must NOT drag in torch. The flow screen's torch import is lazy
    (`rig.inverse.__init__.__getattr__` / TYPE_CHECKING only). Checked in a subprocess
    because torch is already in this process's sys.modules."""
    code = (
        "import sys, rig, rig.inverse, rig.inverse.pessimistic;"
        "assert 'torch' not in sys.modules, 'rig imported torch';"
        "assert 'zuko' not in sys.modules, 'rig imported zuko'"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
