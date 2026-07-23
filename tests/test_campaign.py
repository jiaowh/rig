"""Tests for the confirmation-campaign orchestrator (rig.active.campaign, audit F2).

F2 (root ``audit.md``): ``ConfirmationBatchGate`` existed but nothing in production
ever called it -- "a successful solve is a recommendation, not a validated recipe."
These tests exercise the remediation: every candidate from a solver's output is
actually run through a confirmation batch, every run is logged, and a gate
rejection actually blocks promotion (never quietly waved through).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from rig.active.campaign import (
    CampaignResult,
    CandidateCertification,
    ConfirmationCampaign,
    NothingToQualify,
)
from rig.interfaces import Infeasible, RecipeCandidate
from rig.qualification import ConfirmationBatchGate, min_runs_for_claim
from rig_adapters.mbe import simlink

_SIM_AVAILABLE = simlink.sim_available()

TARGETS = {"thickness": {"target": 100.0, "tol": 5.0}}
# A 0.90 claim at 95% confidence needs 29 flawless runs (mirrors test_qualification.py).
N_FOR_90 = min_runs_for_claim(0.90, 0.95)


def _candidate(recipe: dict[str, float], **overrides: Any) -> RecipeCandidate:
    """Build a RecipeCandidate via keyword args of its five current fields.

    Deliberately keyword-only and exhaustive over (recipe, confidence,
    predicted_outcome_interval, feasibility_flag, support_score): another agent
    is concurrently adding a new field WITH A DEFAULT to RecipeCandidate in this
    same session, and constructing this way stays robust to that (a new
    defaulted field needs no entry here).
    """
    fields: dict[str, Any] = {
        "recipe": recipe,
        "confidence": 0.9,
        "predicted_outcome_interval": None,
        "feasibility_flag": True,
        "support_score": 0.0,
    }
    fields.update(overrides)
    return RecipeCandidate(**fields)


def _true_function_machine(true_fn, calls: list[dict[str, float]] | None = None):
    """A deterministic single-output machine: recipe -> {"thickness": true_fn(recipe)}.

    No RNG anywhere -- this is the pure-function machine used by the determinism
    and gate-arithmetic tests. ``calls`` (if given) records every recipe the
    machine was actually invoked with, in order, for call-counting assertions.
    """

    def machine(recipe: dict[str, float]) -> dict[str, float]:
        if calls is not None:
            calls.append(dict(recipe))
        return {"thickness": true_fn(recipe)}

    return machine


def _gate_params(**overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "targets": TARGETS,
        "n_runs": N_FOR_90,
        "min_in_spec_rate": 0.90,
        "provenance_source": "real_tool",
    }
    params.update(overrides)
    return params


# ---------------------------------------------------------------------------
# Infeasible input: zero machine calls, typed empty result, never a raise
# ---------------------------------------------------------------------------


def test_infeasible_input_fires_zero_machine_calls_and_returns_nothing_to_qualify():
    calls: list[dict[str, float]] = []
    machine = _true_function_machine(lambda r: 100.0, calls)
    campaign = ConfirmationCampaign(
        process_id="mbe_gaas",
        tool_id="chamber_A",
        seed=0,
        machine=machine,
        gate_params=_gate_params(),
    )
    infeasible = Infeasible(
        nearest_achievable={"x": 0.3}, distance_to_feasible=1.5, reason="too far from spec"
    )
    outcome = campaign.run(infeasible)
    assert isinstance(outcome, NothingToQualify)
    assert outcome.infeasible is infeasible
    assert calls == []  # not one machine run fired


def test_empty_candidate_list_is_not_infeasible_but_fires_nothing_either():
    # InverseResult permits an empty list distinct from Infeasible; handle it
    # without crashing (e.g. a Bonferroni divide-by-n_candidates) and without
    # fabricating an Infeasible verdict that solve() never produced.
    calls: list[dict[str, float]] = []
    machine = _true_function_machine(lambda r: 100.0, calls)
    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(),
        bonferroni=True,
    )
    outcome = campaign.run([])
    assert isinstance(outcome, CampaignResult)
    assert outcome.n_candidates == 0
    assert outcome.certified == ()
    assert outcome.rejected == ()
    assert outcome.all_run_records == ()
    assert calls == []


# ---------------------------------------------------------------------------
# THE BLOCK: a true hitter is certified, a true misser is rejected -- and
# stays rejected (see the red-proof note in the final report for how this was
# adversarially verified against a hand-broken split).
# ---------------------------------------------------------------------------


def test_true_hitter_certified_true_misser_rejected_with_meaningful_stats():
    calls: list[dict[str, float]] = []

    def true_fn(recipe: dict[str, float]) -> float:
        return 100.0 if recipe["x"] < 0.5 else 200.0  # 200 is far outside [95, 105]

    machine = _true_function_machine(true_fn, calls)
    campaign = ConfirmationCampaign(
        process_id="mbe_gaas",
        tool_id="chamber_A",
        seed=0,
        machine=machine,
        gate_params=_gate_params(),
    )
    good = _candidate({"x": 0.0})
    bad = _candidate({"x": 1.0})
    outcome = campaign.run([good, bad])

    assert isinstance(outcome, CampaignResult)
    assert outcome.n_candidates == 2
    assert [c.candidate for c in outcome.certified] == [good]
    assert [c.candidate for c in outcome.rejected] == [bad]
    # the bad candidate is NOT smuggled into certified under any guise
    assert bad not in [c.candidate for c in outcome.certified]
    assert good not in [c.candidate for c in outcome.rejected]

    cert = outcome.certified[0]
    assert cert.passed is True
    assert cert.evidence["n_in_spec"] == N_FOR_90
    assert cert.evidence["binomial_lower_bound"] >= 0.90

    rej = outcome.rejected[0]
    assert rej.passed is False
    assert rej.evidence["n_in_spec"] == 0
    assert rej.evidence["binomial_lower_bound"] == 0.0
    assert "NOT CERTIFIED" in rej.evidence["reason"]

    # exactly n_runs machine calls per candidate were fired, and every one logged
    assert len(calls) == 2 * N_FOR_90
    assert len(cert.run_records) == N_FOR_90
    assert len(rej.run_records) == N_FOR_90
    assert outcome.n_machine_calls == 2 * N_FOR_90
    assert outcome.n_certified == 1
    assert outcome.n_rejected == 1


def test_partition_preserves_original_candidate_order_within_each_bucket():
    def true_fn(recipe: dict[str, float]) -> float:
        return 100.0 if int(recipe["x"]) % 2 == 0 else 200.0

    machine = _true_function_machine(true_fn)
    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=8, min_in_spec_rate=0.5, confidence=0.8),
    )
    candidates = [_candidate({"x": float(i)}) for i in range(6)]
    outcome = campaign.run(candidates)
    assert [c.candidate_index for c in outcome.certified] == [0, 2, 4]
    assert [c.candidate_index for c in outcome.rejected] == [1, 3, 5]
    # never re-ranked/modified: identity and recipe payload both survive untouched
    for idx, cert in zip([0, 2, 4], outcome.certified, strict=True):
        assert cert.candidate is candidates[idx]


def test_static_prebuilt_gate_is_reused_across_all_candidates():
    calls: list[dict[str, float]] = []

    def verifier(recipe: dict[str, float]) -> dict[str, float]:
        calls.append(dict(recipe))
        return {"thickness": 100.0 if recipe["x"] < 0.5 else 200.0}

    gate = ConfirmationBatchGate(
        verifier, TARGETS, n_runs=N_FOR_90, min_in_spec_rate=0.90, provenance_source="real_tool"
    )
    campaign = ConfirmationCampaign(process_id="p", tool_id="t", seed=0, gate=gate)
    outcome = campaign.run([_candidate({"x": 0.0}), _candidate({"x": 1.0})])
    assert isinstance(outcome, CampaignResult)
    assert outcome.n_certified == 1
    assert outcome.n_rejected == 1
    assert len(calls) == 2 * N_FOR_90


# ---------------------------------------------------------------------------
# Provenance (§3.5): every emitted RunRecord matches the DECLARED source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["physics_sim", "real_tool"])
def test_every_run_record_provenance_matches_the_declared_source(source):
    machine = _true_function_machine(lambda r: 100.0)
    campaign = ConfirmationCampaign(
        process_id="mbe_gaas",
        tool_id="chamber_A",
        seed=1,
        machine=machine,
        gate_params=_gate_params(
            n_runs=8, min_in_spec_rate=0.5, confidence=0.8, provenance_source=source
        ),
    )
    outcome = campaign.run([_candidate({"x": 0.0})])
    assert isinstance(outcome, CampaignResult)
    assert outcome.all_run_records  # non-empty
    for rec in outcome.all_run_records:
        assert rec.provenance.source == source
        assert rec.process_id == "mbe_gaas"
        assert rec.tool_id == "chamber_A"
    assert outcome.provenance_source == source
    assert outcome.headline_eligible == (source == "real_tool")
    caveat_blob = " ".join(outcome.caveats)
    if source == "real_tool":
        assert "REHEARSAL" not in caveat_blob
    else:
        assert "REHEARSAL" in caveat_blob
        assert "physics_sim" in caveat_blob


def test_run_records_reconstruct_exactly_the_gates_own_observed_values():
    machine = _true_function_machine(lambda r: 100.0)
    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=8, min_in_spec_rate=0.5, confidence=0.8),
    )
    outcome = campaign.run([_candidate({"x": 0.0})])
    cert = outcome.certified[0]
    observed = cert.evidence["observed_values"]
    assert len(cert.run_records) == len(observed) == 8
    for rec, obs in zip(cert.run_records, observed, strict=True):
        by_name = {o.name: o.value.magnitude for o in rec.outcomes}
        for name, value in obs.items():
            assert by_name[name] == pytest.approx(value)
        assert rec.provenance.source == cert.evidence["provenance_source"]
        assert rec.extra["gate_reason"] == cert.evidence["reason"]


# ---------------------------------------------------------------------------
# Determinism (implementation-plan §13.4): identical (candidates, machine, seed)
# -> identical CampaignResult, RunRecord ids/timestamps included.
# ---------------------------------------------------------------------------


def test_determinism_two_runs_produce_identical_serialized_results():
    def build() -> ConfirmationCampaign:
        return ConfirmationCampaign(
            process_id="mbe_gaas",
            tool_id="chamber_A",
            seed=42,
            machine=_true_function_machine(lambda r: 100.0 if r["x"] < 0.5 else 200.0),
            gate_params=_gate_params(n_runs=12, min_in_spec_rate=0.5, confidence=0.8),
        )

    candidates = [_candidate({"x": 0.0}), _candidate({"x": 1.0})]
    first = build().run(list(candidates))
    second = build().run(list(candidates))

    assert isinstance(first, CampaignResult)
    assert first == second
    first_json = [r.model_dump_json() for r in first.all_run_records]
    second_json = [r.model_dump_json() for r in second.all_run_records]
    assert first_json == second_json
    # sanity: ids are actually deterministic hashes, not incidentally equal
    assert len({r.run_id for r in first.all_run_records}) == len(first.all_run_records)


# ---------------------------------------------------------------------------
# Multiplicity: the Bonferroni knob tightens per-candidate confidence by alpha/q
# ---------------------------------------------------------------------------


def test_bonferroni_knob_tightens_confidence_by_alpha_over_q():
    base_confidence = 0.90
    machine = _true_function_machine(lambda r: 100.0)
    candidates = [_candidate({"x": float(i)}) for i in range(4)]

    off = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=10, min_in_spec_rate=0.5, confidence=base_confidence),
        bonferroni=False,
    )
    on = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=10, min_in_spec_rate=0.5, confidence=base_confidence),
        bonferroni=True,
    )

    result_off = off.run(candidates)
    result_on = on.run(candidates)
    assert isinstance(result_off, CampaignResult)
    assert isinstance(result_on, CampaignResult)

    assert result_off.bonferroni_applied is False
    assert result_off.confidence_per_candidate == pytest.approx(base_confidence)

    assert result_on.bonferroni_applied is True
    expected = 1.0 - (1.0 - base_confidence) / 4
    assert result_on.confidence_per_candidate == pytest.approx(expected)
    assert result_on.confidence_per_candidate > result_off.confidence_per_candidate

    off_caveats = " ".join(result_off.caveats)
    on_caveats = " ".join(result_on.caveats)
    assert "NO correction was applied" in off_caveats
    assert "bonferroni=True applied a conservative" in on_caveats


def test_bonferroni_scales_with_candidate_count():
    base_confidence = 0.80
    machine = _true_function_machine(lambda r: 100.0)

    def run_with(n: int) -> float:
        campaign = ConfirmationCampaign(
            process_id="p",
            tool_id="t",
            seed=0,
            machine=machine,
            gate_params=_gate_params(n_runs=20, min_in_spec_rate=0.5, confidence=base_confidence),
            bonferroni=True,
        )
        candidates = [_candidate({"x": float(i)}) for i in range(n)]
        result = campaign.run(candidates)
        assert isinstance(result, CampaignResult)
        return result.confidence_per_candidate

    c1 = run_with(1)
    c2 = run_with(2)
    c8 = run_with(8)
    assert c1 == pytest.approx(base_confidence)  # n=1: no correction needed
    assert c2 == pytest.approx(1.0 - (1.0 - base_confidence) / 2)
    assert c8 == pytest.approx(1.0 - (1.0 - base_confidence) / 8)
    assert c1 < c2 < c8  # more candidates -> stricter per-candidate bar


# ---------------------------------------------------------------------------
# Fail-loud construction
# ---------------------------------------------------------------------------


def test_bonferroni_is_incompatible_with_a_prebuilt_gate():
    gate = ConfirmationBatchGate(
        lambda r: {"thickness": 100.0},
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    with pytest.raises(ValueError, match="bonferroni"):
        ConfirmationCampaign(process_id="p", tool_id="t", seed=0, gate=gate, bonferroni=True)


def test_requires_exactly_one_of_gate_or_gate_params():
    gate = ConfirmationBatchGate(
        lambda r: {"thickness": 100.0},
        TARGETS,
        n_runs=N_FOR_90,
        min_in_spec_rate=0.90,
        provenance_source="real_tool",
    )
    with pytest.raises(ValueError, match="exactly one"):
        ConfirmationCampaign(
            process_id="p", tool_id="t", seed=0, gate=gate, gate_params=_gate_params()
        )
    with pytest.raises(ValueError, match="must supply either"):
        ConfirmationCampaign(process_id="p", tool_id="t", seed=0)


def test_machine_and_gate_params_verifier_both_given_is_rejected():
    with pytest.raises(ValueError, match="both `machine`"):
        ConfirmationCampaign(
            process_id="p",
            tool_id="t",
            seed=0,
            machine=lambda r: {"thickness": 100.0},
            gate_params=_gate_params(verifier=lambda r: {"thickness": 100.0}),
        )


def test_gate_params_without_verifier_or_machine_is_rejected():
    with pytest.raises(ValueError, match="no machine executor"):
        ConfirmationCampaign(process_id="p", tool_id="t", seed=0, gate_params=_gate_params())


def test_blank_process_id_or_tool_id_is_rejected():
    with pytest.raises(ValueError, match="process_id"):
        ConfirmationCampaign(process_id="", tool_id="t", seed=0, gate_params=_gate_params())
    with pytest.raises(ValueError, match="tool_id"):
        ConfirmationCampaign(process_id="p", tool_id="", seed=0, gate_params=_gate_params())


# ---------------------------------------------------------------------------
# Caveats reference (not restate) the gate's documented limitations
# ---------------------------------------------------------------------------


def test_caveats_reference_serial_correlation_and_cpk_and_the_staged_ladder():
    machine = _true_function_machine(lambda r: 100.0)
    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=8, min_in_spec_rate=0.5, confidence=0.8),
    )
    outcome = campaign.run([_candidate({"x": 0.0})])
    assert isinstance(outcome, CampaignResult)
    blob = " ".join(outcome.caveats)
    assert "serial correlation" in blob
    assert "Cpk" in blob
    assert "staged qualification ladder" in blob
    assert "rig.qualification" in blob  # points back at the source of truth


# ---------------------------------------------------------------------------
# CandidateCertification convenience accessors
# ---------------------------------------------------------------------------


def test_candidate_certification_evidence_and_passed_shortcuts_match_qualification():
    machine = _true_function_machine(lambda r: 100.0)
    campaign = ConfirmationCampaign(
        process_id="p",
        tool_id="t",
        seed=0,
        machine=machine,
        gate_params=_gate_params(n_runs=8, min_in_spec_rate=0.5, confidence=0.8),
    )
    outcome = campaign.run([_candidate({"x": 0.0})])
    cert = outcome.certified[0]
    assert isinstance(cert, CandidateCertification)
    assert cert.passed == cert.qualification.passed
    assert cert.evidence is cert.qualification.evidence


# ---------------------------------------------------------------------------
# Sim-gated integration: the real MBE in-silico machine (skip-gated like the
# rest of the suite's sim tests, e.g. tests/test_active_mbe.py)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _SIM_AVAILABLE, reason=f"mbe_sim not found (set {simlink.MBE_SIM_ENV})")
def test_campaign_certifies_and_rejects_against_the_insilico_mbe_machine():
    from rig.forward import records_to_arrays
    from rig_adapters.mbe.adapter import make_adapter
    from rig_adapters.mbe.machine import InSilicoMachine, PathologyConfig

    input_keys = ["T_heater", "film_thickness"]
    output_keys = ["thickness_grown"]

    adapter = make_adapter()
    sim = InSilicoMachine(config=PathologyConfig(), seed=0, adapter=adapter)

    probe_recipes = adapter.seed_design(24, 7)
    probe_records = [sim.run(p, tool_id="A") for p in probe_recipes]
    _, Yp = records_to_arrays(probe_records, input_keys, output_keys)
    lo, hi = float(np.min(Yp)), float(np.max(Yp))
    target = 0.5 * (lo + hi)
    tol = 0.4 * (hi - lo)  # generous: minimize confirmation-batch flakiness from sim noise

    diffs = np.abs(Yp[:, 0] - target)
    good_recipe = probe_recipes[int(np.argmin(diffs))]
    bad_recipe = probe_recipes[int(np.argmax(diffs))]

    def machine(recipe: dict[str, float]) -> dict[str, float]:
        rec = sim.run(recipe, tool_id="A")
        return {o.name: o.value.magnitude for o in rec.outcomes if o.name in output_keys}

    n_runs = min_runs_for_claim(0.5, 0.8)
    campaign = ConfirmationCampaign(
        process_id=adapter.process_id,
        tool_id="A",
        seed=0,
        machine=machine,
        gate_params=dict(
            targets={"thickness_grown": {"target": target, "tol": tol}},
            n_runs=n_runs,
            min_in_spec_rate=0.5,
            confidence=0.8,
            provenance_source="physics_sim",
        ),
    )
    good = _candidate(dict(good_recipe))
    bad = _candidate(dict(bad_recipe))
    outcome = campaign.run([good, bad])

    assert isinstance(outcome, CampaignResult)
    assert outcome.n_candidates == 2
    assert outcome.provenance_source == "physics_sim"
    assert outcome.headline_eligible is False
    assert len(outcome.all_run_records) == 2 * n_runs
    for rec in outcome.all_run_records:
        assert rec.provenance.source == "physics_sim"
        assert rec.process_id == adapter.process_id
        assert rec.tool_id == "A"
    # the deliberately-farthest-from-target recipe must not be promoted
    assert bad not in [c.candidate for c in outcome.certified]
