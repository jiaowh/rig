"""Deployment certification — the §3.4 QualificationGate (implementation-plan §3.4, §11.4, §14.9).

§14.9 is blunt: "no in-distribution coverage, risk bound, or SBC pass substitutes
for independent qualification of accepted recipes on the real tool before
production". This module is the concrete gate that spends that physical query and
turns it into a logged :class:`~rig.interfaces.QualificationRecord`.

Why a CONFIRMATION BATCH and not an independent physics solver
--------------------------------------------------------------
§3.4 offers two verifier options: "independent physics solver or a fixed
confirmation batch on the real tool". The first is **not available to us**: both
shipped adapters honestly return ``independent_verifier=None``, because the MBE
fast Arrhenius path shares physics lineage with the kMC ``ZoneEnsemble``, so
neither can verify the other without violating D7 (§13.3; the different-physics
ROM verifier is still owed in Phase 0). Building a physics verifier here would
be pretending. So this module implements the OTHER option, which needs no
physics at all — only an injected way to actually run the recipe.

The non-circularity property (D7, §11.4) — this is the whole point
------------------------------------------------------------------
**The forward model does not appear in the accept/reject decision at any point.**
The gate holds no ``ForwardModel``, imports nothing from ``rig.forward``, and
never calls ``predict``. The verdict is a function of MEASURED outcomes and the
spec box alone. §11.4: "The independent qualifier shares no parameters with the
generator or its surrogate (else it certifies the same hallucination)."
Certifying a recipe against the opinion of the model that proposed it measures
self-consistency, not correctness — it is the circularity the 2026-07-17 audit
flagged in the M3 gate, and the reason the binding evaluation rule is *score
against ground truth, never against the model that produced the answer*.

The gate is therefore **only as independent as the injected verifier**. That
independence is the CALLER's contract and this module cannot prove it: a closure
can capture anything. :func:`_resolve_verifier` trips on the most common mistake
(handing the gate the surrogate) but that is a tripwire, NOT a proof.

The acceptance rule (statistically honest at small N)
-----------------------------------------------------
Naively, "8 of 8 confirmation runs landed in spec" reads as a 100% yield. It is
nothing of the kind: 8/8 is entirely ordinary for a process whose true in-spec
probability is 80%. The rule here is an **exact one-sided Clopper-Pearson LOWER
confidence bound** on the in-spec probability ``p``, which must clear a declared
threshold::

    passed  <=>  clopper_pearson_lower(n_in_spec, n_runs, confidence) >= min_in_spec_rate

This is a single rule with no escape hatch, and it is what makes an underpowered
batch fail: a perfect batch of ``n`` runs certifies at most ``alpha**(1/n)``, so
claiming ``p >= 0.90`` at 95% confidence takes **29 flawless runs** and no fewer
(:func:`min_runs_for_claim`). A too-small ``n_runs`` therefore cannot pass — not
by a special case, but because the bound never reaches the threshold. The gate
never passes by default.

Honest limits — read before quoting a pass (NOT IMPLEMENTED / assumptions)
--------------------------------------------------------------------------
1. **The binomial bound assumes the N runs are i.i.d. Bernoulli** — exchangeable
   and independent. Real tools violate this: first-wafer effects, within-batch
   drift, and serial correlation are all real (this repo's own
   ``InSilicoMachine`` models a first-wafer offset and seasoning *deliberately*).
   Under positive serial correlation the effective sample size is below ``n`` and
   **this bound is optimistic**. It is not repaired here.
2. **No multiplicity correction — NOT IMPLEMENTED.** §14.5(c) requires
   selection-corrected risk control (Learn-then-Test / RCPS) across a candidate
   pool, plus a family-wise/FDR correction across a pre-registered target
   sequence. Certifying ``k`` recipes with this gate at 95% each does **not**
   control the aggregate error rate at 5%. Do not read ``k`` passes as a
   campaign-level guarantee.
3. **No Cpk / process-window / SPC — NOT IMPLEMENTED.** §11.4's acceptance for a
   qualification lot is ``Cpk >= 1.33`` (>=1.67 automotive). This gate scores an
   in-spec *proportion*, which is a weaker and different claim than a capability
   index over a characterized process window.
4. **This is one rung of the §11.4 staged ladder**, not the ladder: the full path
   is ``in-silico independent-solver gate -> single-wafer confirmation -> small
   pilot lot -> qualification lot with Cpk acceptance``. A pass here is whichever
   rung the injected verifier actually is, and no more.
5. **Provenance is load-bearing (§3.5).** ``provenance_source`` is REQUIRED, with
   no default, because either default would be a lie: ``real_tool`` would let a
   simulator masquerade as tool qualification, ``physics_sim`` would silently
   mislabel a real tool. A pass on ``physics_sim`` is a cheap pre-filter and is
   **not** tool qualification; the record says so via ``headline_eligible``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
from scipy.stats import beta

from rig.interfaces import QualificationRecord

# The spec-box grammar is shared with the §8 solver ON PURPOSE: the caller hands the
# SAME spec['targets'] to solve() and to certify(), so the gate is provably checking
# the box the solver claimed rather than a re-typed near-copy of it.
from rig.inverse.pessimistic import SpecBox, parse_targets

__all__ = [
    "ConfirmationBatchGate",
    "clopper_pearson_lower",
    "min_runs_for_claim",
]

type Verifier = Callable[[Mapping[str, Any]], Mapping[str, Any]] | Any
type ProvenanceSource = Literal["physics_sim", "real_tool"]


# ---------------------------------------------------------------------------
# The binomial acceptance rule
# ---------------------------------------------------------------------------


def clopper_pearson_lower(n_success: int, n_trials: int, confidence: float) -> float:
    """Exact one-sided Clopper-Pearson LOWER confidence bound on a binomial ``p``.

    Returns the largest ``p_lo`` with ``P(X >= n_success | n_trials, p_lo) >=
    1 - confidence``: the true in-spec probability is at least this, at the
    stated confidence. Exact (Beta-quantile inversion), never a normal
    approximation — the Wald interval is badly wrong at the small ``n`` and
    near-1 ``p`` a confirmation batch lives at, and its lower limit collapses to
    the point estimate at ``n_success == n_trials``, which is the precise lie
    this gate exists to prevent.

    ``n_success == 0`` yields 0.0 (no evidence of any in-spec mass).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if not 0 <= n_success <= n_trials:
        raise ValueError(f"n_success must be in [0, {n_trials}], got {n_success}")
    if n_success == 0:
        return 0.0
    alpha = 1.0 - confidence
    return float(beta.ppf(alpha, n_success, n_trials - n_success + 1))


def min_runs_for_claim(min_in_spec_rate: float, confidence: float) -> int:
    """Smallest confirmation-batch size whose PERFECT outcome clears the threshold.

    A flawless batch (``n_success == n_trials == n``) certifies exactly
    ``p >= alpha**(1/n)``, so the claim ``p >= min_in_spec_rate`` at ``confidence``
    needs ``n >= log(alpha) / log(min_in_spec_rate)``. Any smaller batch CANNOT
    pass this gate no matter how clean it comes back — size the batch with this
    before spending wafers on one that provably cannot certify.
    """
    if not 0.0 < min_in_spec_rate < 1.0:
        raise ValueError(f"min_in_spec_rate must be in (0, 1), got {min_in_spec_rate}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    # Exact, not an estimate to be corrected afterwards: with a = log(alpha) < 0 and
    # b = log(rate) < 0, alpha**(1/n) >= rate  <=>  a/n >= b  <=>  n >= a/b, so ceil(a/b)
    # is precisely the minimal n. (A step-until-it-clears loop here would be dead code —
    # swept over 6986 (rate, confidence) pairs, it never once fires.)
    return max(1, math.ceil(math.log(1.0 - confidence) / math.log(min_in_spec_rate)))


# ---------------------------------------------------------------------------
# Verifier plumbing
# ---------------------------------------------------------------------------


def _resolve_verifier(verifier: Verifier) -> Callable[[Mapping[str, Any]], Any]:
    """Normalize an injected verifier to ``recipe -> outcome``. Fails closed."""
    if verifier is None:
        raise ValueError(
            "ConfirmationBatchGate requires a verifier: certification is a MEASUREMENT, "
            "and there is no default one (implementation-plan §3.4, §14.9). Inject a callable "
            "recipe -> Mapping[str, float], or an object with a .run(recipe) method."
        )
    # A ForwardModel is the mistake §11.4 names outright ("shares no parameters with
    # the generator or its surrogate ... else it certifies the same hallucination").
    # It is a tripwire for the obvious case, NOT a proof of independence -- a closure
    # can capture the surrogate and this check will never see it.
    if hasattr(verifier, "predict") and hasattr(verifier, "support_score"):
        raise TypeError(
            f"{type(verifier).__name__} looks like a ForwardModel. Certifying a recipe "
            "against the surrogate that proposed it certifies the same hallucination "
            "(implementation-plan §11.4, D7) -- it measures self-consistency, not correctness. "
            "The verifier must be OUTSIDE the training/inversion loop."
        )
    if callable(verifier):
        return verifier
    run = getattr(verifier, "run", None)
    if callable(run):
        return run
    raise TypeError(
        f"verifier of type {type(verifier).__name__} is neither callable nor has a "
        "callable .run(recipe) method."
    )


def _as_float(value: Any, name: str) -> float:
    """Coerce one measured outcome to a float in the spec box's own units."""
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        raise TypeError(
            f"outcome {name!r} is a pint Quantity ({value!r}). The gate compares raw floats "
            "against the spec box and cannot know which unit the box is in, so it will not "
            "guess: pass magnitudes already canonicalized to the box's units (SI per "
            "implementation-plan §3.5). Pairing SI data with declared-unit bounds is a live "
            "trap in this repo -- see the `continuous_si` decision in docs/BUILD_STATE.md."
        )
    return float(value)


# ---------------------------------------------------------------------------
# §3.4 QualificationGate: the confirmation-batch realization
# ---------------------------------------------------------------------------


class ConfirmationBatchGate:
    """Certify a recipe by a fixed confirmation batch on an injected verifier (§3.4).

    Conforms to the :class:`~rig.interfaces.QualificationGate` protocol:
    ``certify(recipe) -> QualificationRecord(passed, evidence)``.

    Runs ``n_runs`` confirmation runs of ``recipe`` on ``verifier``, counts how
    many land inside the spec box, and passes iff the exact one-sided
    Clopper-Pearson lower bound on the in-spec probability clears
    ``min_in_spec_rate`` at ``confidence``. The forward model never enters the
    decision (see the module docstring on D7 non-circularity), and the evidence
    is sufficient to recompute the verdict without re-running anything.

    Parameters
    ----------
    verifier
        A callable ``recipe -> Mapping[str, float]``, or an object with a
        ``.run(recipe)`` method returning the same. Values must already be in the
        spec box's units (:func:`_as_float` refuses pint Quantities rather than
        guess). A machine returning a :class:`~rig.schema.RunRecord` is adapted
        caller-side, deliberately, so the unit conversion stays visible::

            gate = ConfirmationBatchGate(
                lambda r: {o.name: o.value.magnitude for o in machine.run(r).outcomes},
                targets, n_runs=29, min_in_spec_rate=0.90, provenance_source="physics_sim",
            )

        Independence from the training/inversion loop is the caller's contract.
    targets
        Spec box in the §8 ``spec['targets']`` grammar (see
        :func:`~rig.inverse.pessimistic.parse_targets`) — hand it the same object
        the solver got.
    n_runs
        Confirmation-batch size. Sized below :func:`min_runs_for_claim` it cannot
        pass; the constructor warns, and ``certify`` fails with the arithmetic.
    min_in_spec_rate
        The in-spec probability being CLAIMED. Required: it is the safety claim
        itself and must never be an accident.
    confidence
        Confidence for the lower bound (default 0.95 = the §14.5 ``delta=0.05``
        convention).
    output_keys
        Optional declared-output order to validate target names against; defaults
        to the target names. The load-bearing check is at runtime anyway — a spec
        output the verifier does not return raises rather than counting as a miss.
    provenance_source
        ``"real_tool"`` or ``"physics_sim"`` (§3.5). REQUIRED — see the module
        docstring, honest limit 5.
    """

    def __init__(
        self,
        verifier: Verifier,
        targets: Mapping[str, Any],
        *,
        n_runs: int,
        min_in_spec_rate: float,
        provenance_source: ProvenanceSource,
        confidence: float = 0.95,
        output_keys: Sequence[str] | None = None,
    ) -> None:
        self._verifier = _resolve_verifier(verifier)
        if n_runs < 1:
            raise ValueError(f"n_runs must be >= 1, got {n_runs}")
        if not 0.0 < min_in_spec_rate < 1.0:
            raise ValueError(f"min_in_spec_rate must be in (0, 1), got {min_in_spec_rate}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if provenance_source not in ("physics_sim", "real_tool"):
            raise ValueError(
                f"provenance_source must be 'physics_sim' or 'real_tool', got "
                f"{provenance_source!r} (implementation-plan §3.5)"
            )
        self._box: SpecBox = parse_targets(targets, tuple(output_keys or targets))
        self._n_runs = int(n_runs)
        self._min_in_spec_rate = float(min_in_spec_rate)
        self._confidence = float(confidence)
        self._provenance_source: ProvenanceSource = provenance_source

        self._min_runs = min_runs_for_claim(self._min_in_spec_rate, self._confidence)
        self._max_achievable = clopper_pearson_lower(self._n_runs, self._n_runs, self._confidence)
        if self._n_runs < self._min_runs:
            # Warn, don't raise: the acceptance rule stays the single arbiter and the
            # record documents the impossibility with real numbers. But say it BEFORE
            # the wafers burn -- a batch that provably cannot certify is wasted budget (§11.1).
            warnings.warn(
                f"n_runs={self._n_runs} is UNDERPOWERED for the claim p >= "
                f"{self._min_in_spec_rate} at {self._confidence:.0%} confidence: even a "
                f"flawless batch certifies only p >= {self._max_achievable:.4f}. This gate "
                f"cannot pass. Need n_runs >= {self._min_runs}.",
                stacklevel=2,
            )

    @property
    def underpowered(self) -> bool:
        """True iff no outcome of this batch size could clear the threshold."""
        return self._n_runs < self._min_runs

    def certify(self, recipe: Mapping[str, Any]) -> QualificationRecord:
        """Run the confirmation batch and apply the binomial acceptance rule (§3.4).

        The gate contributes no randomness of its own; reproducibility is the
        verifier's contract (§13.4). A run whose measured value is NaN counts as
        OUT of spec (fail-closed); a run MISSING a spec output raises, because
        that is a broken verifier contract, not a failed wafer.
        """
        names = self._box.output_names
        observed: list[dict[str, float]] = []
        per_run_margins: list[dict[str, float]] = []
        in_spec_flags: list[bool] = []

        for i in range(self._n_runs):
            outcome = self._verifier(dict(recipe))
            values = self._extract(outcome, i)
            margins = np.minimum(self._box.upper - values, values - self._box.lower)
            observed.append({n: float(v) for n, v in zip(names, values, strict=True)})
            per_run_margins.append({n: float(m) for n, m in zip(names, margins, strict=True)})
            in_spec_flags.append(bool(np.all(margins >= 0.0)))

        n_in_spec = int(sum(in_spec_flags))
        lower_bound = clopper_pearson_lower(n_in_spec, self._n_runs, self._confidence)
        passed = bool(lower_bound >= self._min_in_spec_rate)

        margin_matrix = np.array([[m[n] for n in names] for m in per_run_margins], float)
        worst = {n: float(np.min(margin_matrix[:, j])) for j, n in enumerate(names)}

        evidence: dict[str, Any] = {
            "recipe": dict(recipe),
            "n_runs": self._n_runs,
            "n_in_spec": n_in_spec,
            "observed_values": observed,
            "in_spec_flags": in_spec_flags,
            "per_run_margins": per_run_margins,
            "worst_margin_per_output": worst,
            "spec_box": {
                n: [float(lo), float(hi)]
                for n, lo, hi in zip(names, self._box.lower, self._box.upper, strict=True)
            },
            "binomial_lower_bound": lower_bound,
            "min_in_spec_rate": self._min_in_spec_rate,
            "confidence": self._confidence,
            "underpowered": self.underpowered,
            "max_achievable_lower_bound": self._max_achievable,
            "min_runs_for_claim": self._min_runs,
            "provenance_source": self._provenance_source,
            "headline_eligible": self._provenance_source == "real_tool",
            "rule": (
                "passed <=> clopper_pearson_lower(n_in_spec, n_runs, confidence) "
                ">= min_in_spec_rate; one-sided exact binomial (implementation-plan §3.4, §11.4)"
            ),
            "reason": self._reason(passed, n_in_spec, lower_bound),
            "verifier_independence": (
                "asserted by the caller; the gate holds no ForwardModel and the model does "
                "not enter this verdict (D7, implementation-plan §11.4)"
            ),
        }
        return QualificationRecord(passed=passed, evidence=evidence)

    def _extract(self, outcome: Any, run_index: int) -> np.ndarray:
        if not isinstance(outcome, Mapping):
            raise TypeError(
                f"verifier returned {type(outcome).__name__} on confirmation run {run_index}; "
                "the contract is Mapping[str, float] in the spec box's units. A machine "
                "returning a RunRecord must be adapted caller-side (see the class docstring) "
                "so the unit conversion stays explicit."
            )
        values = np.empty(len(self._box.output_names), dtype=float)
        for j, name in enumerate(self._box.output_names):
            if name not in outcome:
                raise KeyError(
                    f"verifier did not return spec output {name!r} on confirmation run "
                    f"{run_index} (returned: {sorted(outcome)}). A missing output is a broken "
                    "verifier contract, not an out-of-spec run -- refusing to score it."
                )
            values[j] = _as_float(outcome[name], name)
        return values

    def _reason(self, passed: bool, n_in_spec: int, lower_bound: float) -> str:
        head = (
            f"{n_in_spec}/{self._n_runs} confirmation runs in spec on a "
            f"{self._provenance_source} verifier; exact one-sided Clopper-Pearson "
            f"{self._confidence:.0%} lower bound on P(in spec) = {lower_bound:.4f}"
        )
        if passed:
            tail = f" >= threshold {self._min_in_spec_rate:.4f} -> CERTIFIED"
            if self._provenance_source != "real_tool":
                tail += (
                    " on a SIMULATOR: this is a §11.4 pre-filter rung, NOT tool qualification, "
                    "and is not headline-eligible (§3.5)."
                )
            return head + tail
        if self.underpowered:
            return (
                head + f" < threshold {self._min_in_spec_rate:.4f} -> NOT CERTIFIED. "
                f"UNDERPOWERED: n_runs={self._n_runs} could not have certified this claim at "
                f"any outcome (a flawless batch reaches only {self._max_achievable:.4f}); "
                f"need n_runs >= {self._min_runs}."
            )
        return head + f" < threshold {self._min_in_spec_rate:.4f} -> NOT CERTIFIED."
