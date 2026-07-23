"""Confirmation-batch qualification campaigns (audit F2 remediation).

``rig.qualification.ConfirmationBatchGate`` is a sound single-recipe certifier, but
the 2026-07-21 audit (root ``audit.md``, finding F2) found it wired into NOTHING:
"outside its module it is only referenced by interfaces/tests. The inverse solver,
active loop and Empa runner do not call it... a successful solve is a
recommendation, not a validated recipe." This module is the remediation: it takes
an :class:`~rig.interfaces.InverseSolver`'s raw output, actually RUNS the gate's
confirmation batch for every candidate on an injected machine, logs every single
measured run as a :class:`~rig.schema.RunRecord`, and BLOCKS promotion of any
candidate the gate does not certify.

What this module does NOT do
-----------------------------
It does not re-implement the gate's statistics. The Clopper-Pearson acceptance
rule, the D7 non-circularity tripwire (:func:`~rig.qualification._resolve_verifier`
rejects a ``ForwardModel``-shaped verifier), and the fail-closed measurement
semantics all live in :mod:`rig.qualification` and are reused verbatim: a
:class:`ConfirmationCampaign` builds (or is handed) a real
:class:`~rig.qualification.ConfirmationBatchGate` and calls its
``certify(recipe)`` exactly once per candidate. Every one of the gate's
``n_runs`` individual confirmation measurements is ALREADY present in
``QualificationRecord.evidence["observed_values"]`` (one ``{output: value}``
dict per run, in submission order) -- so this module reconstructs a
:class:`~rig.schema.RunRecord` per entry there rather than re-intercepting the
verifier. No new randomness, no new measurement path, no new acceptance rule.

The block (this is the whole point of F2)
------------------------------------------
``CampaignResult`` separates ``certified`` from ``rejected`` by ONE predicate:
``qualification.passed`` from the gate's own verdict, nothing else. Candidates
are never re-scored, re-ranked, or filtered by their own ``confidence`` /
``support_score`` / ``feasibility_flag`` -- those are the SURROGATE's opinion,
which is precisely what an independent qualifier must not consult (D7,
implementation-plan §11.4, restated in ``rig.qualification``'s module
docstring). A recipe is promoted only if the real-tool (or rehearsal)
confirmation batch actually certified it.

Provenance (§3.5) -- read this before quoting a "pass"
--------------------------------------------------------
Every emitted ``RunRecord`` carries the SAME ``Provenance.source`` the gate
itself used to certify (read back from ``evidence["provenance_source"]``, never
re-declared here, so the two can never drift apart). ``provenance_source`` is
therefore the caller's responsibility exactly as it is in
``ConfirmationBatchGate`` -- there is no default, on purpose. A campaign run
with ``provenance_source="physics_sim"`` is an IN-SILICO REHEARSAL: it can
shake out plumbing bugs and give a cheap pre-filter, but it is **not** tool
qualification, and ``CampaignResult.caveats`` says so explicitly whenever the
source is not ``"real_tool"`` (mirroring ``evidence["headline_eligible"]``).
Production promotion requires a campaign run against a ``real_tool`` verifier.

Multiplicity: q candidates x one gate each (NOT IMPLEMENTED by default)
--------------------------------------------------------------------------
Certifying every one of ``n_candidates`` proposals independently at the SAME
per-candidate confidence is a multiple-comparisons surface: ``rig.qualification``
says outright (honest limit 2) that certifying k recipes at 95% each does not
control the aggregate/family-wise error rate at 5%. This module does not
silently pretend otherwise. ``CampaignResult.confidence_per_candidate`` and
``.n_candidates`` are always reported together so the reader can see the
uncorrected exposure, and the constructor knob ``bonferroni=True`` applies the
classic conservative correction -- test each candidate at
``confidence = 1 - (1 - base_confidence) / n_candidates`` (equivalently
``alpha / q``) -- computed once ``run()`` sees the actual candidate count. This
is NOT the full implementation-plan §14.5(c) Learn-then-Test / RCPS
selection-corrected control the plan calls for (that needs a pre-registered
target sequence and a non-conservative FDR/family-wise procedure); Bonferroni
is the simple, conservative stand-in, and ``CampaignResult.caveats`` says which
one was applied.

Serial correlation and Cpk are the gate's caveats, referenced not restated
----------------------------------------------------------------------------
This module inherits, unchanged, ``rig.qualification``'s honest limits 1
(the Clopper-Pearson bound assumes i.i.d. Bernoulli runs; real tools have
first-wafer effects and drift, so the bound is optimistic under positive serial
correlation) and 3 (no Cpk / process-window / SPC is computed anywhere here --
a pass is an in-spec PROPORTION, a weaker claim than a capability index). It
does not attempt to repair either. ``CampaignResult.caveats`` points at them by
name rather than re-deriving the math, so a reader of a campaign report is one
hop from the real explanation.

Determinism (implementation-plan §13.4)
-----------------------------------------
Given the same ``(candidates, machine, seed)``, two campaign runs produce
identical ``CampaignResult``s, including every ``RunRecord``. This module
introduces no randomness of its own: ``run_id`` is a ``uuid5`` hash of
``(seed, candidate_index, run_index)`` (never ``uuid4``, which is
os-random), and the default ``clock`` is a deterministic synthetic sequence
(epoch + run_index seconds), not wall-clock time. Reproducibility of the
MEASURED values is, as in ``rig.qualification``, the injected machine's
contract, not this module's.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from rig.interfaces import Infeasible, InverseResult, QualificationRecord, RecipeCandidate
from rig.qualification import ConfirmationBatchGate
from rig.schema import OutcomeRecord, Provenance, Quantity, RecipeRecord, RunRecord

__all__ = [
    "CampaignResult",
    "CandidateCertification",
    "CampaignOutcome",
    "ConfirmationCampaign",
    "NothingToQualify",
]

# Same shape as rig.qualification.Verifier -- forwarded verbatim as the gate's
# ``verifier`` (a bare ``recipe -> Mapping[str, float]`` callable, or an object
# with a ``.run(recipe)`` method; a ProcessAdapter-paired machine such as
# ``InSilicoMachine`` is the ".run()" case). Resolution (None-rejection, the D7
# ForwardModel tripwire, callable-vs-.run() dispatch) is NOT reimplemented here
# -- ``ConfirmationBatchGate.__init__`` already does it the moment this is
# forwarded as ``verifier=machine``.
type MachineRunner = Callable[[Mapping[str, Any]], Mapping[str, Any]] | Any
type ProvenanceSource = Literal["physics_sim", "real_tool"]

# uuid5 is a pure hash (deterministic, no OS randomness) over a fixed namespace
# -- derived from a readable name rather than a hardcoded literal so the
# provenance of the constant itself is auditable.
_UUID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "rig.active.campaign")
_SYNTHETIC_EPOCH = datetime(2000, 1, 1, tzinfo=UTC)


def _deterministic_run_id(seed: int, candidate_index: int, run_index: int) -> uuid.UUID:
    """Reproducible ``run_id``: a uuid5 hash of ``(seed, candidate_index, run_index)``.

    ``RunRecord.run_id`` must be a UUID (implementation-plan §3.5), but this
    module's determinism contract forbids the usual ``uuid4()`` (os-random)
    generator -- two ``ConfirmationCampaign.run()`` calls with the same seed
    must emit byte-identical RunRecords, ids included. ``uuid5`` over a fixed
    namespace is a pure function of its inputs: same inputs, same id, always,
    on any machine.
    """
    return uuid.uuid5(_UUID_NAMESPACE, f"{seed}:{candidate_index}:{run_index}")


def _synthetic_clock_at(run_index: int) -> datetime:
    """Deterministic default clock: epoch + ``run_index`` seconds -- never wall-clock.

    A real ``datetime.now()`` default would silently break the determinism
    contract (see the module docstring). Callers who want genuine wall-clock
    provenance timestamps (the right choice for an actual ``real_tool``
    campaign) inject their own via the ``clock`` constructor argument, e.g.
    ``clock=lambda _run_index: datetime.now(UTC)`` -- at which point
    reproducibility of timestamps becomes that caller's contract, exactly as
    machine determinism already is.
    """
    return _SYNTHETIC_EPOCH + timedelta(seconds=run_index)


def _bonferroni_confidence(base_confidence: float, n_candidates: int) -> float:
    """``1 - alpha/q``: the classic conservative family-wise correction (alpha/q)."""
    return 1.0 - (1.0 - base_confidence) / n_candidates


# ---------------------------------------------------------------------------
# Result types (a tagged union, mirroring InverseResult's own list-or-Infeasible shape)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateCertification:
    """One candidate's confirmation-batch outcome: the verdict plus its full run log.

    ``qualification`` is the gate's own :class:`~rig.interfaces.QualificationRecord`
    UNMODIFIED -- ``qualification.evidence`` already carries every per-candidate
    statistic an audit needs (``n_in_spec``, ``binomial_lower_bound``, ``reason``,
    the full Clopper-Pearson rule string, ...); this class does not duplicate or
    re-derive any of it. ``run_records`` is the same batch, reconstructed as
    logged :class:`~rig.schema.RunRecord`\\ s (one per confirmation run).
    """

    candidate: RecipeCandidate
    candidate_index: int
    qualification: QualificationRecord
    run_records: tuple[RunRecord, ...]

    @property
    def passed(self) -> bool:
        """Shortcut for ``self.qualification.passed`` -- the gate's verdict, unmodified."""
        return self.qualification.passed

    @property
    def evidence(self) -> dict[str, Any]:
        """Shortcut for ``self.qualification.evidence``."""
        return self.qualification.evidence


@dataclass(frozen=True)
class NothingToQualify:
    """Typed empty result for an :class:`~rig.interfaces.Infeasible` solver output.

    There is no candidate to run a confirmation batch on, so ``run()`` fires
    ZERO machine calls and returns this instead of raising or fabricating a
    result. Carries the original :class:`~rig.interfaces.Infeasible` verdict
    (nearest achievable point, distance-to-feasible, reason) for context.
    """

    infeasible: Infeasible
    reason: str = (
        "solver returned Infeasible: there is no candidate recipe to run a "
        "confirmation batch on. Zero machine calls were made."
    )


@dataclass(frozen=True)
class CampaignResult:
    """Outcome of certifying every candidate from one ``InverseSolver.solve()`` call.

    ``certified`` / ``rejected`` partition the input candidates by ONE rule --
    ``qualification.passed`` -- and preserve their original relative order
    within each partition (this module never re-ranks by anything else, per
    the module docstring). ``n_candidates`` and ``confidence_per_candidate`` are
    always reported together so the multiple-comparisons exposure is visible
    even when ``bonferroni`` was left off. ``caveats`` references (does not
    restate) the gate's documented serial-correlation / multiplicity / Cpk
    limitations -- see the module docstring.
    """

    certified: tuple[CandidateCertification, ...]
    rejected: tuple[CandidateCertification, ...]
    n_candidates: int
    confidence_per_candidate: float
    bonferroni_applied: bool
    provenance_source: str
    all_run_records: tuple[RunRecord, ...]
    caveats: tuple[str, ...]
    seed: int

    @property
    def n_certified(self) -> int:
        return len(self.certified)

    @property
    def n_rejected(self) -> int:
        return len(self.rejected)

    @property
    def n_machine_calls(self) -> int:
        """Total confirmation runs actually fired across every candidate."""
        return len(self.all_run_records)

    @property
    def headline_eligible(self) -> bool:
        """True iff every logged run's provenance supports a headline claim (§3.5).

        Mirrors ``evidence["headline_eligible"]``: only ``real_tool`` runs are
        eligible. A ``physics_sim`` campaign is a rehearsal, never a headline.
        """
        return self.provenance_source == "real_tool"


type CampaignOutcome = CampaignResult | NothingToQualify
"""Tagged union: a real campaign result, or an explicit "nothing to qualify" verdict."""


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------


class ConfirmationCampaign:
    """Run confirmation-batch qualification over an InverseSolver's candidates (audit F2).

    Turns ``InverseSolver.solve(spec)``'s raw output into a promotion decision:
    every :class:`~rig.interfaces.RecipeCandidate` is run through a
    :class:`~rig.qualification.ConfirmationBatchGate` confirmation batch on the
    injected ``machine``, every individual run is logged as a
    :class:`~rig.schema.RunRecord`, and the candidate is promoted (``certified``)
    only if the gate's own ``certify()`` verdict says so. An
    :class:`~rig.interfaces.Infeasible` input is handled without firing a single
    machine call (see :class:`NothingToQualify`).

    Construct with EXACTLY ONE of:

    ``gate``
        A pre-built :class:`~rig.qualification.ConfirmationBatchGate`, reused
        AS-IS for every candidate (its verifier/targets/n_runs/confidence/
        provenance_source were already fixed by the caller). Incompatible with
        ``bonferroni=True`` (see below).
    ``gate_params``
        A mapping of the keyword arguments :class:`~rig.qualification.
        ConfirmationBatchGate` takes (``targets``, ``n_runs``,
        ``min_in_spec_rate``, ``provenance_source``, and optionally
        ``confidence`` / ``output_keys``) MINUS ``verifier`` -- ``machine``
        supplies that. One gate is built once per :meth:`run` call (after
        ``n_candidates`` is known, so ``bonferroni`` can adjust ``confidence``
        first). All candidates passed to one :meth:`run` call are assumed to
        target the SAME spec (as they would coming from one ``solve(spec)``
        call): "q candidates x one gate each" per the module docstring, not a
        distinct gate per candidate.

    ``machine`` supplies the verifier when using ``gate_params`` (ignored, and
    an error, when a pre-built ``gate`` is given directly -- that gate already
    has its own verifier). It is forwarded verbatim as ``verifier=machine``. No
    machine-shape adaptation happens here: the contract is
    :class:`~rig.qualification.ConfirmationBatchGate`'s own -- a callable or
    ``.run()``-having object returning ``Mapping[str, float]`` in the spec
    box's units. A raw ``InSilicoMachine``-style ``.run(recipe) -> RunRecord``
    must be adapted caller-side exactly as ``rig.qualification``'s own
    docstring shows, e.g. ``lambda r: {o.name: o.value.magnitude for o in
    sim.run(r, tool_id=...).outcomes}`` -- so the unit conversion stays visible
    (same rationale as the gate's own verifier contract).

    ``bonferroni``
        If ``True``, ``confidence`` in ``gate_params`` is treated as the
        DESIRED family-wise confidence across the whole candidate pool and is
        tightened per candidate via :func:`_bonferroni_confidence` (alpha/q)
        once ``n_candidates`` is known. Requires ``gate_params`` (raises at
        construction if combined with a pre-built ``gate``, whose ``confidence``
        was already frozen before this campaign could know ``n_candidates``).

    ``seed``
        Required (no default -- see the module docstring on determinism).
        Used only to derive this module's OWN deterministic ``run_id``\\ s; the
        machine's own reproducibility is its contract, not this module's.

    ``clock``
        Optional ``run_index -> datetime`` override for ``RunRecord.timestamp``
        (default: a deterministic synthetic sequence). Pass e.g.
        ``lambda _i: datetime.now(UTC)`` for real wall-clock provenance in an
        actual real-tool campaign; determinism of timestamps then becomes the
        caller's responsibility, same as for the machine.
    """

    def __init__(
        self,
        *,
        process_id: str,
        tool_id: str,
        seed: int,
        machine: MachineRunner | None = None,
        gate: ConfirmationBatchGate | None = None,
        gate_params: Mapping[str, Any] | None = None,
        bonferroni: bool = False,
        clock: Callable[[int], datetime] | None = None,
    ) -> None:
        if not process_id:
            raise ValueError("process_id must be non-empty (implementation-plan §3.5)")
        if not tool_id:
            raise ValueError("tool_id must be non-empty (implementation-plan §3.5)")
        if gate is not None and gate_params is not None:
            raise ValueError(
                "pass exactly one of `gate` (a pre-built ConfirmationBatchGate) or "
                "`gate_params` (kwargs to build one), not both."
            )
        if gate is None and gate_params is None:
            raise ValueError(
                "must supply either `gate` (a pre-built ConfirmationBatchGate, reused "
                "for every candidate) or `gate_params` (kwargs used to build one per "
                "run() call, with `machine` supplying the verifier)."
            )
        if gate is not None and bonferroni:
            raise ValueError(
                "bonferroni=True needs to adjust the gate's `confidence` using "
                "n_candidates, which is only known once run() sees the candidate list "
                "-- a pre-built `gate` has already frozen its confidence at "
                "construction and cannot be adjusted after the fact. Pass `gate_params` "
                "instead of a pre-built `gate` to use the Bonferroni knob."
            )
        if gate is None:
            params = dict(gate_params)  # type: ignore[arg-type]
            if machine is not None:
                if "verifier" in params:
                    raise ValueError(
                        "both `machine` and `gate_params['verifier']` were supplied; "
                        "pass the executor once via `machine=` and leave `verifier` out "
                        "of `gate_params`."
                    )
                params["verifier"] = machine
            if "verifier" not in params:
                raise ValueError(
                    "no machine executor given: supply `machine=` (adapted to "
                    "ConfirmationBatchGate's verifier contract) or put `verifier` "
                    "directly in `gate_params`."
                )
            self._gate_params: dict[str, Any] | None = params
        else:
            self._gate_params = None
        self._static_gate = gate
        self._bonferroni = bool(bonferroni)
        self._process_id = process_id
        self._tool_id = tool_id
        self._seed = int(seed)
        self._clock = clock or _synthetic_clock_at

    def run(self, result: InverseResult) -> CampaignOutcome:
        """Certify every candidate in ``result`` (or fire nothing for an Infeasible).

        Never raises on an :class:`~rig.interfaces.Infeasible` input -- returns
        :class:`NothingToQualify` instead, having fired zero machine calls.
        Otherwise builds (or reuses) the gate, calls ``certify(candidate.recipe)``
        once per candidate IN ORDER, reconstructs every one of the gate's
        ``n_runs`` confirmation measurements as a logged
        :class:`~rig.schema.RunRecord`, and partitions the candidates into
        ``certified`` / ``rejected`` by the gate's verdict alone.
        """
        if isinstance(result, Infeasible):
            return NothingToQualify(infeasible=result)
        candidates = list(result)
        n_candidates = len(candidates)
        if n_candidates == 0:
            # Degenerate but type-legal (InverseResult allows an empty list): nothing
            # to certify, but distinct from Infeasible, so report it as an empty
            # CampaignResult rather than forcing a NothingToQualify that would claim
            # an Infeasible verdict that was never produced.
            return CampaignResult(
                certified=(),
                rejected=(),
                n_candidates=0,
                confidence_per_candidate=float("nan"),
                bonferroni_applied=self._bonferroni,
                provenance_source="",
                all_run_records=(),
                caveats=(),
                seed=self._seed,
            )

        gate = self._build_gate(n_candidates)
        certifications: list[CandidateCertification] = []
        all_records: list[RunRecord] = []
        run_counter = 0
        for idx, candidate in enumerate(candidates):
            qualification = gate.certify(candidate.recipe)
            evidence = qualification.evidence
            records: list[RunRecord] = []
            for observed in evidence["observed_values"]:
                records.append(
                    self._build_run_record(
                        evidence=evidence,
                        observed=observed,
                        candidate_index=idx,
                        run_index=run_counter,
                    )
                )
                run_counter += 1
            certifications.append(
                CandidateCertification(
                    candidate=candidate,
                    candidate_index=idx,
                    qualification=qualification,
                    run_records=tuple(records),
                )
            )
            all_records.extend(records)

        # All candidates in one run() share one gate, so provenance/confidence are
        # invariant across the loop -- read them back from the evidence (the gate's
        # own record) rather than re-declaring them, so they cannot drift apart.
        first_evidence = certifications[0].qualification.evidence
        provenance_source = str(first_evidence["provenance_source"])
        confidence_used = float(first_evidence["confidence"])

        # THE BLOCK: certified/rejected are partitioned by the gate's verdict alone
        # (qualification.passed), preserving each candidate's original order within
        # its partition. Nothing else -- not confidence, not support_score, not
        # feasibility_flag -- ever moves a candidate between these two tuples.
        certified = tuple(c for c in certifications if c.qualification.passed)
        rejected = tuple(c for c in certifications if not c.qualification.passed)

        return CampaignResult(
            certified=certified,
            rejected=rejected,
            n_candidates=n_candidates,
            confidence_per_candidate=confidence_used,
            bonferroni_applied=self._bonferroni,
            provenance_source=provenance_source,
            all_run_records=tuple(all_records),
            caveats=self._caveats(provenance_source=provenance_source),
            seed=self._seed,
        )

    # -- gate construction --------------------------------------------------

    def _build_gate(self, n_candidates: int) -> ConfirmationBatchGate:
        if self._static_gate is not None:
            return self._static_gate
        params = dict(self._gate_params)  # type: ignore[arg-type]
        if self._bonferroni:
            base_confidence = float(params.get("confidence", 0.95))
            params["confidence"] = _bonferroni_confidence(base_confidence, n_candidates)
        return ConfirmationBatchGate(**params)

    # -- RunRecord reconstruction from the gate's own evidence ---------------

    def _build_run_record(
        self,
        *,
        evidence: Mapping[str, Any],
        observed: Mapping[str, float],
        candidate_index: int,
        run_index: int,
    ) -> RunRecord:
        """Build one logged RunRecord from one entry of ``evidence["observed_values"]``.

        Recipe/outcome magnitudes carry no unit metadata by the time they reach
        here (``RecipeCandidate.recipe`` and the gate's verifier both traffic in
        bare floats already in the spec box's units, per the SI-canonical
        contract elsewhere in this repo) -- rather than GUESS a declared unit
        (the exact mistake ``rig.qualification._as_float`` refuses to make),
        every numeric value is tagged ``dimensionless`` and the untouched raw
        mapping is preserved verbatim in ``extra`` (mirroring the established
        ``extra["machine_config"]`` idiom elsewhere in this repo) so no
        information is lost even though the typed fields are unit-naive.
        """
        provenance = Provenance(source=evidence["provenance_source"])
        recipe_values: dict[str, Any] = {}
        untyped: dict[str, Any] = {}
        for name, value in evidence["recipe"].items():
            try:
                recipe_values[name] = Quantity(magnitude=float(value), unit="dimensionless")
            except (TypeError, ValueError):
                untyped[name] = value
        outcomes = [
            OutcomeRecord(
                name=name,
                modality="scalar_vector",
                value=Quantity(magnitude=float(value), unit="dimensionless"),
            )
            for name, value in observed.items()
        ]
        extra: dict[str, Any] = {
            "candidate_index": candidate_index,
            "confirmation_run_index": run_index,
            "raw_recipe": dict(evidence["recipe"]),
            "raw_outcome": dict(observed),
            "gate_reason": evidence["reason"],
        }
        if untyped:
            extra["untyped_recipe_keys"] = untyped
        return RunRecord(
            run_id=_deterministic_run_id(self._seed, candidate_index, run_index),
            process_id=self._process_id,
            tool_id=self._tool_id,
            timestamp=self._clock(run_index),
            recipe=RecipeRecord(values=recipe_values),
            outcomes=outcomes,
            provenance=provenance,
            extra=extra,
        )

    # -- caveats (reference, don't restate, rig.qualification's honest limits) --

    def _caveats(self, *, provenance_source: str) -> tuple[str, ...]:
        if self._bonferroni:
            multiplicity = (
                "multiplicity: bonferroni=True applied a conservative alpha/n_candidates "
                "correction to the per-candidate confidence (family-wise error rate <= the "
                "base alpha, at the cost of a stricter per-candidate bar). This is NOT the "
                "full implementation-plan §14.5(c) Learn-then-Test/RCPS selection-corrected "
                "control -- Bonferroni is simple and conservative, not optimal (see "
                "rig.qualification module docstring, honest limit 2)."
            )
        else:
            multiplicity = (
                "multiplicity: NO correction was applied across n_candidates candidates each "
                "certified independently at the same per-candidate confidence -- the "
                "aggregate/family-wise error rate across this campaign is NOT controlled "
                "(rig.qualification module docstring, honest limit 2). Set bonferroni=True at "
                "construction for a conservative alpha/q correction."
            )
        caveats = [
            multiplicity,
            "serial correlation: each confirmation batch assumes i.i.d. Bernoulli runs; real "
            "tools violate this (first-wafer effects, within-batch drift), and the "
            "Clopper-Pearson bound is optimistic under positive serial correlation -- not "
            "repaired here (rig.qualification module docstring, honest limit 1).",
            "no Cpk / process-window / SPC is computed anywhere in this campaign; a pass is an "
            "in-spec PROPORTION, a weaker and different claim than a capability index over a "
            "characterized process window (rig.qualification module docstring, honest limit 3).",
            "this campaign is one rung of the implementation-plan §11.4 staged qualification "
            "ladder (in-silico independent-solver gate -> single-wafer confirmation -> small "
            "pilot lot -> qualification lot with Cpk acceptance), not the whole ladder "
            "(rig.qualification module docstring, honest limit 4).",
        ]
        if provenance_source and provenance_source != "real_tool":
            caveats.append(
                f"provenance_source={provenance_source!r}: every certification in this "
                "campaign is an IN-SILICO REHEARSAL, not production tool qualification -- "
                "production promotion requires a campaign run with provenance_source="
                "'real_tool' (rig.qualification module docstring, honest limit 5)."
            )
        return tuple(caveats)
