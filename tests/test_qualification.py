"""Tests for the §3.4 confirmation-batch QualificationGate (rig.qualification)."""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from rig import qualification
from rig.interfaces import QualificationGate, QualificationRecord
from rig.qualification import (
    ConfirmationBatchGate,
    clopper_pearson_lower,
    min_runs_for_claim,
)

# A 0.90 claim at 95% confidence needs 29 flawless runs (see min_runs_for_claim).
TARGETS = {"thickness": {"target": 100.0, "tol": 5.0}}
N_FOR_90 = 29


def _always_in_spec(value: float = 100.0):
    def verifier(recipe):
        return {"thickness": value}

    return verifier


def _fails_k_of_n(k: int, n: int):
    """Deterministic verifier: exactly ``k`` of the first ``n`` runs land out of spec."""
    calls = {"i": 0}

    def verifier(recipe):
        i = calls["i"]
        calls["i"] += 1
        return {"thickness": 200.0 if i < k else 100.0}

    return verifier


def _noisy_verifier(seed: int, sigma: float):
    rng = np.random.default_rng(seed)

    def verifier(recipe):
        return {"thickness": float(rng.normal(100.0, sigma))}

    return verifier


# ---------------------------------------------------------------------------
# the multi-output in-spec conjunction (np.all across outputs)
# ---------------------------------------------------------------------------


def test_a_run_is_in_spec_only_if_every_output_is_in_spec_not_just_one():
    """A confirmation run counts in-spec iff EVERY spec output is in spec
    (``np.all`` across outputs in ``certify``), not merely one of them. With a
    single-output box the ALL-vs-ANY distinction is invisible, and every other test
    in this file uses a single output — so this is the ONLY guard on the conjunction.
    A verifier that nails one output and blows the other must score the run OUT of
    spec; otherwise the gate would certify recipes that violate a spec target.
    Flipping ``np.all`` -> ``np.any`` in ``certify`` turns this red (verified)."""
    targets = {
        "thickness": {"target": 100.0, "tol": 5.0},  # always in spec below
        "slip": {"target": 50.0, "tol": 2.0},  # always OUT of spec below
    }

    def verifier(recipe):
        return {"thickness": 100.0, "slip": 60.0}  # slip 60 is far outside [48, 52]

    gate = ConfirmationBatchGate(
        verifier,
        targets,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"x": 1.0})
    # np.all: one output out of spec fails the whole run, so ZERO runs are in spec.
    assert record.evidence["n_in_spec"] == 0
    assert not record.passed
    # sanity: the failure is the slip output, not thickness — i.e. the conjunction
    # actually looked at both, it did not just default to fail.
    assert record.evidence["worst_margin_per_output"]["thickness"] >= 0.0
    assert record.evidence["worst_margin_per_output"]["slip"] < 0.0


# ---------------------------------------------------------------------------
# the binomial rule itself
# ---------------------------------------------------------------------------


def test_clopper_pearson_lower_matches_closed_form_at_perfect_batch():
    # Beta(n, 1) has CDF x**n, so the exact one-sided bound at k == n is alpha**(1/n).
    for n in (1, 8, 29, 100):
        assert clopper_pearson_lower(n, n, 0.95) == pytest.approx(0.05 ** (1 / n))


def test_perfect_batch_of_eight_does_not_prove_ninety_percent():
    # THE headline honesty property: "0 of 8 failed" is not a 99% yield, nor even 90%.
    assert clopper_pearson_lower(8, 8, 0.95) == pytest.approx(0.6877, abs=1e-4)
    assert clopper_pearson_lower(8, 8, 0.95) < 0.90


def test_clopper_pearson_zero_successes_is_zero_and_is_monotone():
    assert clopper_pearson_lower(0, 10, 0.95) == 0.0
    bounds = [clopper_pearson_lower(k, 29, 0.95) for k in range(30)]
    assert bounds == sorted(bounds)


def test_clopper_pearson_rejects_bad_arguments():
    with pytest.raises(ValueError):
        clopper_pearson_lower(3, 2, 0.95)
    with pytest.raises(ValueError):
        clopper_pearson_lower(-1, 2, 0.95)
    with pytest.raises(ValueError):
        clopper_pearson_lower(1, 0, 0.95)
    with pytest.raises(ValueError):
        clopper_pearson_lower(1, 2, 1.0)


def test_min_runs_for_claim_is_the_exact_tipping_point():
    n = min_runs_for_claim(0.90, 0.95)
    assert n == N_FOR_90
    assert clopper_pearson_lower(n, n, 0.95) >= 0.90
    assert clopper_pearson_lower(n - 1, n - 1, 0.95) < 0.90


@pytest.mark.parametrize(
    ("rate", "confidence"), [(0.80, 0.95), (0.90, 0.95), (0.95, 0.95), (0.99, 0.95), (0.90, 0.99)]
)
def test_min_runs_for_claim_is_self_consistent(rate, confidence):
    n = min_runs_for_claim(rate, confidence)
    assert clopper_pearson_lower(n, n, confidence) >= rate
    if n > 1:
        assert clopper_pearson_lower(n - 1, n - 1, confidence) < rate


# ---------------------------------------------------------------------------
# protocol conformance + D7 structure
# ---------------------------------------------------------------------------


def test_gate_conforms_to_the_canonical_protocol():
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    assert isinstance(gate, QualificationGate)
    record = gate.certify({"T": 1200.0})
    assert isinstance(record, QualificationRecord)


def test_module_imports_nothing_from_the_forward_model_d7():
    # D7 non-circularity is structural, not a promise: the accept/reject decision cannot
    # consult the surrogate if the module never imports it. Parsed, not grepped -- the
    # docstring says "rig.forward" in prose and must not satisfy this guard.
    tree = ast.parse(pathlib.Path(qualification.__file__).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert imported, "expected to parse some imports"
    assert not [m for m in imported if m.startswith(("rig.forward", "rig.active", "rig.baselines"))]


def test_import_stays_torch_free():
    code = "import rig.qualification, sys; assert 'torch' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# fail-closed construction
# ---------------------------------------------------------------------------


def test_refuses_to_construct_without_a_verifier():
    with pytest.raises(ValueError, match="requires a verifier"):
        ConfirmationBatchGate(
            None, TARGETS, n_runs=N_FOR_90, min_in_spec_rate=0.90, provenance_source="real_tool"
        )


def test_refuses_a_forward_model_as_its_own_verifier():
    class _Surrogate:
        def predict(self, x):  # pragma: no cover - never called
            raise AssertionError("the gate must not consult a model")

        def support_score(self, x):  # pragma: no cover - never called
            raise AssertionError("the gate must not consult a model")

    with pytest.raises(TypeError, match="ForwardModel"):
        ConfirmationBatchGate(
            _Surrogate(),
            TARGETS,
            n_runs=N_FOR_90,
            min_in_spec_rate=0.90,
            provenance_source="real_tool",
        )


def test_refuses_a_verifier_that_can_neither_be_called_nor_run():
    with pytest.raises(TypeError, match="neither callable"):
        ConfirmationBatchGate(
            object(),
            TARGETS,
            n_runs=N_FOR_90,
            min_in_spec_rate=0.90,
            provenance_source="real_tool",
        )


def test_accepts_an_object_with_a_run_method():
    class _Machine:
        def run(self, recipe):
            return {"thickness": 100.0}

    gate = ConfirmationBatchGate(
        _Machine(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    assert gate.certify({"T": 1200.0}).passed


def test_provenance_source_is_required_and_validated():
    with pytest.raises(TypeError):
        ConfirmationBatchGate(  # type: ignore[call-arg]
            _always_in_spec(), TARGETS, n_runs=N_FOR_90, min_in_spec_rate=0.90
        )
    with pytest.raises(ValueError, match="provenance_source"):
        ConfirmationBatchGate(
            _always_in_spec(),
            TARGETS,
            n_runs=N_FOR_90,
            min_in_spec_rate=0.90,
            provenance_source="simulator",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_runs": 0, "min_in_spec_rate": 0.90},
        {"n_runs": N_FOR_90, "min_in_spec_rate": 0.0},
        {"n_runs": N_FOR_90, "min_in_spec_rate": 1.0},
    ],
)
def test_rejects_out_of_range_configuration(kwargs):
    with pytest.raises(ValueError):
        ConfirmationBatchGate(_always_in_spec(), TARGETS, provenance_source="real_tool", **kwargs)


# ---------------------------------------------------------------------------
# the verdict
# ---------------------------------------------------------------------------


def test_calibrated_in_spec_verifier_passes():
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    assert record.passed
    assert record.evidence["n_in_spec"] == N_FOR_90
    assert record.evidence["binomial_lower_bound"] >= 0.90
    assert record.evidence["underpowered"] is False


def test_verifier_failing_a_fraction_of_runs_fails():
    gate = ConfirmationBatchGate(
        _noisy_verifier(seed=0, sigma=6.0),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    assert not record.passed
    assert 0 < record.evidence["n_in_spec"] < N_FOR_90
    assert record.evidence["binomial_lower_bound"] < 0.90


def test_a_single_failure_sinks_a_marginal_claim():
    # 28/29 is a 96.6% point estimate and still cannot certify 90% at 95% confidence.
    # The gate must score the BOUND, not the observed rate.
    gate = ConfirmationBatchGate(
        _fails_k_of_n(1, N_FOR_90),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    assert record.evidence["n_in_spec"] == N_FOR_90 - 1
    assert record.evidence["n_in_spec"] / N_FOR_90 > 0.90  # the naive rate would have passed
    assert record.evidence["binomial_lower_bound"] == pytest.approx(0.8466, abs=1e-3)
    assert not record.passed


def test_too_small_a_batch_cannot_pass_even_when_flawless():
    # The core statistical-honesty guard: 8/8 is a perfect batch and still not evidence
    # for a 90% yield. Nothing special-cases this -- the bound simply never reaches the
    # threshold, so there is no "pass by default" path.
    with pytest.warns(UserWarning, match="UNDERPOWERED"):
        gate = ConfirmationBatchGate(
            _always_in_spec(),
            TARGETS,
            n_runs=8,
            min_in_spec_rate=0.90,
            provenance_source="real_tool",
        )
    assert gate.underpowered
    record = gate.certify({"T": 1200.0})
    assert not record.passed
    assert record.evidence["n_in_spec"] == 8  # a FLAWLESS batch
    assert record.evidence["underpowered"] is True
    assert record.evidence["max_achievable_lower_bound"] == pytest.approx(0.6877, abs=1e-4)
    assert record.evidence["min_runs_for_claim"] == N_FOR_90
    assert "UNDERPOWERED" in record.evidence["reason"]


def test_no_batch_size_below_the_minimum_can_pass():
    for n in (1, 2, 5, 8, 16, N_FOR_90 - 1):
        with pytest.warns(UserWarning, match="UNDERPOWERED"):
            gate = ConfirmationBatchGate(
                _always_in_spec(),
                TARGETS,
                n_runs=n,
                min_in_spec_rate=0.90,
                provenance_source="real_tool",
            )
        assert not gate.certify({"T": 1200.0}).passed


def test_a_lower_claim_is_certifiable_by_a_small_batch():
    # The gate is not merely conservative: a modest claim clears on a modest batch.
    n = min_runs_for_claim(0.70, 0.95)
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=n,
        min_in_spec_rate=0.70,
        provenance_source="real_tool",
    )
    assert n < N_FOR_90
    assert gate.certify({"T": 1200.0}).passed


# ---------------------------------------------------------------------------
# measurement semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (102.0, 3.0),  # upper side binds: min(105-102, 102-95)
        (97.0, 2.0),  # LOWER side binds -- a distance-to-upper-only margin would say 8.0
        (100.0, 5.0),  # dead centre: both sides equal
        (106.0, -1.0),  # out of spec above: margin is signed, not clipped
        (94.0, -1.0),  # out of spec below
    ],
)
def test_margins_are_distance_to_the_nearest_spec_boundary(value, expected):
    gate = ConfirmationBatchGate(
        _always_in_spec(value),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    ev = gate.certify({"T": 1200.0}).evidence
    assert ev["spec_box"] == {"thickness": [95.0, 105.0]}
    assert ev["per_run_margins"][0]["thickness"] == pytest.approx(expected)
    assert ev["worst_margin_per_output"]["thickness"] == pytest.approx(expected)
    assert ev["in_spec_flags"][0] is (expected >= 0)


def test_one_sided_spec_uses_the_finite_side_only():
    gate = ConfirmationBatchGate(
        lambda r: {"slip": 2.0},
        {"slip": {"upper": 3.0}},
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    ev = gate.certify({"T": 1200.0}).evidence
    assert ev["spec_box"]["slip"][0] == -np.inf
    assert ev["per_run_margins"][0]["slip"] == pytest.approx(1.0)
    assert ev["n_in_spec"] == N_FOR_90


def test_nan_outcome_counts_as_out_of_spec():
    # A crashed run / failed metrology is fail-CLOSED: it is not in spec.
    gate = ConfirmationBatchGate(
        _always_in_spec(float("nan")),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    assert record.evidence["n_in_spec"] == 0
    assert not any(record.evidence["in_spec_flags"])
    assert not record.passed


def test_missing_spec_output_raises_rather_than_scoring_a_miss():
    gate = ConfirmationBatchGate(
        lambda r: {"other": 1.0},
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    with pytest.raises(KeyError, match="thickness"):
        gate.certify({"T": 1200.0})


def test_pint_quantity_outcome_raises_instead_of_guessing_a_unit():
    class _Q:
        magnitude = 100.0
        units = "nm"

    gate = ConfirmationBatchGate(
        lambda r: {"thickness": _Q()},
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    with pytest.raises(TypeError, match="pint Quantity"):
        gate.certify({"T": 1200.0})


def test_non_mapping_verifier_return_raises_with_the_runrecord_fix():
    gate = ConfirmationBatchGate(
        lambda r: [100.0],
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    with pytest.raises(TypeError, match="Mapping"):
        gate.certify({"T": 1200.0})


def test_verifier_cannot_mutate_the_callers_recipe():
    def mutating(recipe):
        recipe["T"] = -1.0
        return {"thickness": 100.0}

    gate = ConfirmationBatchGate(
        mutating, TARGETS, n_runs=N_FOR_90, min_in_spec_rate=0.90, provenance_source="real_tool"
    )
    recipe = {"T": 1200.0}
    gate.certify(recipe)
    assert recipe == {"T": 1200.0}


# ---------------------------------------------------------------------------
# §3.5 provenance
# ---------------------------------------------------------------------------


def test_simulator_verifier_passes_but_is_not_tool_qualification():
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="physics_sim",
    )
    record = gate.certify({"T": 1200.0})
    assert record.passed
    assert record.evidence["provenance_source"] == "physics_sim"
    assert record.evidence["headline_eligible"] is False
    assert "NOT tool qualification" in record.evidence["reason"]


def test_real_tool_pass_is_headline_eligible():
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    assert record.evidence["headline_eligible"] is True
    assert "NOT tool qualification" not in record.evidence["reason"]


# ---------------------------------------------------------------------------
# auditability + determinism
# ---------------------------------------------------------------------------


def test_an_auditor_can_recompute_the_verdict_from_the_record_alone():
    gate = ConfirmationBatchGate(
        _noisy_verifier(seed=3, sigma=3.0),
        TARGETS,
        n_runs=40,
        min_in_spec_rate=0.80,
        provenance_source="real_tool",
    )
    record = gate.certify({"T": 1200.0})
    ev = record.evidence

    # Recompute in-spec counts from the raw observations + the logged box. Nothing but
    # the record is consulted -- no gate, no verifier, no model.
    box: dict[str, list[float]] = ev["spec_box"]
    recomputed_flags = [
        all(box[n][0] <= run[n] <= box[n][1] for n in box) for run in ev["observed_values"]
    ]
    assert recomputed_flags == ev["in_spec_flags"]

    n_in_spec = sum(recomputed_flags)
    assert n_in_spec == ev["n_in_spec"]

    bound = clopper_pearson_lower(n_in_spec, ev["n_runs"], ev["confidence"])
    assert bound == pytest.approx(ev["binomial_lower_bound"])
    assert (bound >= ev["min_in_spec_rate"]) is record.passed

    for run_obs, run_marg in zip(ev["observed_values"], ev["per_run_margins"], strict=True):
        for name, value in run_obs.items():
            lo, hi = box[name]
            assert run_marg[name] == pytest.approx(min(hi - value, value - lo))


def test_evidence_carries_every_field_an_audit_needs():
    gate = ConfirmationBatchGate(
        _always_in_spec(),
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    ev = gate.certify({"T": 1200.0}).evidence
    required = {
        "recipe",
        "n_runs",
        "n_in_spec",
        "observed_values",
        "in_spec_flags",
        "per_run_margins",
        "worst_margin_per_output",
        "spec_box",
        "binomial_lower_bound",
        "min_in_spec_rate",
        "confidence",
        "underpowered",
        "max_achievable_lower_bound",
        "min_runs_for_claim",
        "provenance_source",
        "headline_eligible",
        "rule",
        "reason",
        "verifier_independence",
    }
    assert required <= set(ev)
    assert ev["recipe"] == {"T": 1200.0}
    assert len(ev["observed_values"]) == ev["n_runs"] == len(ev["in_spec_flags"])


def test_certification_is_reproducible_under_a_seeded_verifier():
    def build() -> Any:
        return ConfirmationBatchGate(
            _noisy_verifier(seed=7, sigma=4.0),
            TARGETS,
            n_runs=40,
            min_in_spec_rate=0.80,
            provenance_source="real_tool",
        )

    first = build().certify({"T": 1200.0})
    second = build().certify({"T": 1200.0})
    assert first.passed == second.passed
    assert first.evidence == second.evidence
