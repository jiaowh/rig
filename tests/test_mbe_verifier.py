"""D7 different-physics ROM verifier tests (implementation-plan D7, §6.6, §15.2 iv, R10).

Five guarantees, per the Phase-0 owed-verifier brief:

(a) D7 identity — ``validate_adapter`` passes with the real geometric verifier
    attached, and BITES (raises) if the verifier is wired to the fast-path
    physics plug-in instead.
(b) nominal agreement — on seeded recipes at the default machine build the
    verifier's ``thickness_grown`` prediction matches the machine within a
    physics/noise-derived band.
(c) independent disagreement — a hidden flux-scale pathology (the true flux is
    corrupted while the verifier is told only the nominal recipe/config) is
    flagged beyond the band; a within-band drift is NOT flagged.
(d) determinism — pure geometry, no RNG: identical inputs, identical outputs.
(e) different physics is MECHANICALLY checkable — the verifier module imports and
    references NOTHING from the fast Arrhenius path / sim, and predicts with the
    sim unavailable.

Only the machine-comparison tests need the sim; the D7-identity, determinism, and
different-physics tests run everywhere (they are the ones that must hold in CI).
"""

import ast
from pathlib import Path

import pytest

import rig_adapters.mbe.verifier as verifier_module
from rig.interfaces import AdapterValidationError, validate_adapter
from rig_adapters.mbe import simlink
from rig_adapters.mbe.adapter import (
    MACHINE_CONFIG_DEFAULTS,
    NOMINAL_COSINE_N,
    MBEAdapter,
    evaluate_physics,
    make_adapter,
)
from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig
from rig_adapters.mbe.outcomes import metrics_to_output_values
from rig_adapters.mbe.verifier import (
    DEFAULT_REL_BAND,
    KNUDSEN_COSINE_M,
    NOMINAL_MACHINE_CONFIG,
    WAFER_RADIUS_M,
    GeometricDepositionVerifier,
    VerifierReport,
    outcome_values,
    verifier_imports,
)

requires_sim = pytest.mark.skipif(
    not simlink.sim_available(),
    reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})",
)

RECIPE = {"T_heater": 1320.0, "film_thickness": 1e-6}


# ---------------------------------------------------------------------------
# constants stay mirrored with the adapter (the verifier keeps its OWN copy so
# it imports nothing from the fast path — see test group (e))
# ---------------------------------------------------------------------------


def test_nominal_config_mirrors_adapter():
    assert NOMINAL_MACHINE_CONFIG == MACHINE_CONFIG_DEFAULTS
    # The ROM uses ideal-Knudsen (Lambertian) emission; at the nominal build the
    # cell's tunable exponent also happens to be 1.0, so agreement holds.
    assert KNUDSEN_COSINE_M == 1.0
    assert NOMINAL_COSINE_N == 1.0
    assert WAFER_RADIUS_M == 0.0254  # the fast path's UniformityProblem default R


@requires_sim
def test_wafer_radius_matches_the_sim_default():
    sim = simlink.load_mbe_sim()
    # UniformityProblem(material, substrate_card, R=0.0254, ...): R is the 3rd
    # positional (after self is bound, 2nd default). Assert the mirror holds.
    r_default = sim.optimize.UniformityProblem.__init__.__defaults__[1]
    assert r_default == WAFER_RADIUS_M


# ---------------------------------------------------------------------------
# (a) D7 identity — passes with a real verifier, BITES if it is the machine path
# ---------------------------------------------------------------------------


def test_d7_identity_real_verifier_passes():
    adapter = make_adapter()
    validate_adapter(adapter)  # must not raise
    assert adapter.independent_verifier is not None
    assert isinstance(adapter.independent_verifier, GeometricDepositionVerifier)
    # the whole point of D7: verifier and physics prior are DIFFERENT objects
    assert adapter.physics_plugin is not adapter.independent_verifier


class _CircularAdapter(MBEAdapter):
    """A D7 violation on purpose: the verifier IS the fast-path physics plug-in."""

    @property
    def independent_verifier(self):
        return self.physics_plugin


def test_d7_check_bites_when_verifier_is_the_machine_path():
    # Proves the guard is not vacuous: wiring the fast Arrhenius path in as its
    # own verifier is exactly the circularity D7/R10 forbids, and it is rejected.
    with pytest.raises(AdapterValidationError, match="D7"):
        validate_adapter(_CircularAdapter())


# ---------------------------------------------------------------------------
# (b) nominal agreement — verifier vs machine thickness_grown, within band
# ---------------------------------------------------------------------------


@requires_sim
def test_nominal_agreement_clean_machine_is_exact():
    machine = InSilicoMachine(seed=0)  # clean: no noise, no drift
    verifier = GeometricDepositionVerifier()
    for recipe in make_adapter().seed_design(4, seed=3):
        run = machine.run(recipe)
        report = verifier.check(recipe, run)
        assert report.agrees
        assert report.delivery_ratio == pytest.approx(1.0)
        # clean machine grows exactly the target thickness -> zero gap.
        assert report.thickness_rel_error == pytest.approx(0.0, abs=1e-12)


@requires_sim
def test_nominal_agreement_within_band_under_metrology_noise():
    # BAND JUSTIFICATION (physics/noise, NOT tuned to an observed gap — the
    # verifier-vs-clean-machine gap is 0 at nominal): the in-silico metrology
    # noise on thickness_grown is sigma = a + b|y| with a=1e-9 m, b=2e-3, i.e.
    # <=~0.7% rel at the thinnest films tested and ~0.3% at 1 um (≈2% at 3 sigma).
    # A meaningful flux pathology (seasoning/depletion/tool drift) is >=~10%.
    # The 5% band sits cleanly between: a clean+noisy machine never trips it.
    machine = InSilicoMachine(config=PathologyConfig(metrology_noise=True), seed=11)
    verifier = GeometricDepositionVerifier()
    abs_errs = []
    for recipe in make_adapter().seed_design(4, seed=5):
        run = machine.run(recipe)
        report = verifier.check(recipe, run)
        assert report.agrees
        assert abs(report.thickness_rel_error) < DEFAULT_REL_BAND
        abs_errs.append(abs(report.thickness_rel_error))
    assert max(abs_errs) > 0.0  # metrology noise actually perturbed the readings


# ---------------------------------------------------------------------------
# (c) independent disagreement — hidden flux-scale loss is flagged
# ---------------------------------------------------------------------------


@requires_sim
@pytest.mark.parametrize("flux_eff", [0.5, 0.7, 0.85])
def test_disagreement_flagged_on_hidden_flux_scale_loss(flux_eff):
    # The machine's TRUE flux is scaled down (a depleted / mis-calibrated source);
    # the verifier is told only the nominal recipe + nominal build, so it predicts
    # the target thickness. Gap = (1 - flux_eff), far outside the 5% band.
    observed = metrics_to_output_values(evaluate_physics(RECIPE, flux_eff=flux_eff))
    report = GeometricDepositionVerifier().check(RECIPE, observed)
    assert not report.agrees
    assert report.thickness_rel_error == pytest.approx(flux_eff - 1.0, rel=1e-6)


@requires_sim
def test_disagreement_flagged_on_seasoning_drift_through_the_machine():
    # Realistic pathology via the machine wrapper: seasoning depletes the source
    # run-over-run. The verifier (told the nominal recipe each time) flags the
    # run once the accumulated flux loss exceeds the band.
    machine = InSilicoMachine(
        config=PathologyConfig(seasoning=True, seasoning_drift_rate=0.05), seed=0
    )
    verifier = GeometricDepositionVerifier()
    verdicts = [verifier.check(RECIPE, machine.run(RECIPE)).agrees for _ in range(6)]
    assert verdicts[0] is True  # fresh source: delivers the target
    assert verdicts[-1] is False  # ~25% depleted: flagged


@requires_sim
def test_within_band_drift_is_not_flagged():
    # Guard on the guard: a drift SMALLER than the band must NOT trip the verifier
    # (else `not agrees` above would be vacuously true). 2% loss < 5% band.
    observed = metrics_to_output_values(evaluate_physics(RECIPE, flux_eff=0.98))
    report = GeometricDepositionVerifier().check(RECIPE, observed)
    assert report.agrees
    assert abs(report.thickness_rel_error) < DEFAULT_REL_BAND


def test_disagreement_needs_no_sim_direct_observation():
    # The verifier's decision is sim-free: hand it a corrupted observation and it
    # flags it with no machine at all (thickness 0.6 um vs 1.0 um target).
    report = GeometricDepositionVerifier().check(RECIPE, {"thickness_grown": 0.6e-6})
    assert not report.agrees
    assert report.thickness_rel_error == pytest.approx(-0.4, rel=1e-9)


# ---------------------------------------------------------------------------
# the geometric physics is genuinely load-bearing (not a stub returning 1)
# ---------------------------------------------------------------------------


def test_delivery_ratio_follows_line_of_sight_geometry():
    verifier = GeometricDepositionVerifier()
    assert verifier.delivery_ratio() == 1.0
    assert verifier.delivery_ratio(dict(NOMINAL_MACHINE_CONFIG)) == pytest.approx(1.0)
    # source receding halves-ish the delivered flux (~1/H^2 far-field), softened
    # by the off-axis offset and cosine incidence; approaching raises it.
    far = verifier.delivery_ratio({"source_height": 0.40})  # 2x nominal height
    near = verifier.delivery_ratio({"source_height": 0.14})
    assert far < 1.0 < near
    assert 0.20 < far < 0.45  # ~ (0.20/0.40)^2 = 0.25, geometry-softened


def test_predict_reports_scope_and_diagnostics():
    pred = GeometricDepositionVerifier().predict(RECIPE)
    assert pred["thickness_grown"] == pytest.approx(RECIPE["film_thickness"])
    assert pred["delivery_ratio"] == pytest.approx(1.0)
    # geometric flux nonuniformity is exposed as a DIAGNOSTIC only (the machine's
    # combined nonuniformity is ~98% thermal — out of scope for a flux ROM).
    assert 0.0 < pred["flux_nonuniformity_pct"] < 5.0


def test_report_surfaces_unbounded_channels():
    report = GeometricDepositionVerifier().check(RECIPE, {"thickness_grown": 1e-6})
    assert report.bounded_channels == ("thickness_grown",)
    # a consumer can never mistake a pass for blanket certification.
    assert "combined_nonuniformity_pct" in report.unbounded_channels
    assert "slip_max_ratio" in report.unbounded_channels


def test_check_missing_thickness_fails_closed():
    with pytest.raises(ValueError, match="thickness_grown"):
        GeometricDepositionVerifier().check(RECIPE, {"T_center": 900.0})


# ---------------------------------------------------------------------------
# (d) determinism
# ---------------------------------------------------------------------------


def test_predict_is_deterministic():
    verifier = GeometricDepositionVerifier()
    first = verifier.predict(RECIPE)
    assert verifier.predict(RECIPE) == first
    # independent instances agree bit-for-bit (no hidden state, no RNG).
    assert GeometricDepositionVerifier().predict(RECIPE) == first


def test_check_report_is_deterministic():
    observed = {"thickness_grown": 0.7e-6}
    verifier = GeometricDepositionVerifier()
    report = verifier.check(RECIPE, observed)
    assert isinstance(report, VerifierReport)
    assert verifier.check(RECIPE, observed) == report  # frozen dataclass equality


# ---------------------------------------------------------------------------
# (e) different physics is mechanically checkable
# ---------------------------------------------------------------------------

_ALLOWED_IMPORTS = {"__future__", "ast", "math", "collections", "dataclasses", "typing", "numpy"}

# Fast-path / sim call targets that must NEVER appear in the verifier's CODE
# (docstrings may mention them to explain the independence — the AST checks below
# ignore string constants, so they are not fooled by the documentation).
_FORBIDDEN_SYMBOLS = {
    "evaluate_physics",
    "UniformityProblem",
    "ThermalModel",
    "SourceCell",
    "flux_profile",
    "growth_rate_profile",
    "growth_rate_regime",
    "load_mbe_sim",
    "sim_available",
}


def _verifier_source() -> str:
    return Path(verifier_module.__file__).read_text(encoding="utf-8")


def _code_names(source: str) -> set[str]:
    """All identifiers actually USED in code (Name ids + Attribute attrs).

    Ignores docstrings/comments/string constants — so a docstring that mentions
    ``evaluate_physics`` to explain the independence does not trip the check.
    """
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_verifier_imports_are_stdlib_and_numpy_only():
    imported = verifier_imports(_verifier_source())
    assert imported <= _ALLOWED_IMPORTS, f"unexpected imports: {imported - _ALLOWED_IMPORTS}"
    # explicitly: nothing from the sim, the adapter, or any fast-path module.
    assert "mbe_sim" not in imported
    assert "rig_adapters" not in imported
    assert "rig" not in imported


def test_verifier_references_no_fast_path_symbols():
    used = _code_names(_verifier_source())
    leaked = used & _FORBIDDEN_SYMBOLS
    assert not leaked, f"verifier code references fast-path symbols: {leaked}"


def test_verifier_predicts_with_the_sim_unavailable(monkeypatch):
    # The verifier never touches the sim: even if loading it raises, predict works.
    def _boom(*_a, **_k):
        raise RuntimeError("sim must not be loaded by the verifier")

    monkeypatch.setattr(simlink, "load_mbe_sim", _boom)
    pred = GeometricDepositionVerifier().predict(RECIPE)
    assert pred["thickness_grown"] == pytest.approx(RECIPE["film_thickness"])


def test_outcome_values_normalizes_runrecord_and_mapping():
    # duck-typed extraction works for a mapping and for an outcomes-bearing object.
    assert outcome_values({"thickness_grown": 1e-6}) == {"thickness_grown": 1e-6}

    class _Val:
        def __init__(self, m):
            self.magnitude = m

    class _Outcome:
        def __init__(self, name, m):
            self.name = name
            self.value = _Val(m)

    class _Run:
        outcomes = [_Outcome("thickness_grown", 2e-6), _Outcome("T_center", 900.0)]

    assert outcome_values(_Run()) == {"thickness_grown": 2e-6, "T_center": 900.0}
