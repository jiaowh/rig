"""Canonical process-agnostic interfaces (implementation-plan §3.1-§3.4).

Everything downstream talks through the four interfaces defined here:
:class:`ProcessAdapter` (§3.1), :class:`ForwardModel` (§3.2),
:class:`InverseSolver` (§3.3) and :class:`QualificationGate` (§3.4).

Canonical names are binding (implementation-plan §3.2, §13.2): ``predict(x) ->
PredictiveDistribution(mean, aleatoric_sigma, epistemic_sigma,
conformal_set)`` - never ``OutcomeDist``, never bare scalars, never
``_var`` tuples.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from scipy.stats import qmc

# ---------------------------------------------------------------------------
# §3.2 canonical predictive distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictiveDistribution:
    """Canonical output of ``ForwardModel.predict`` (implementation-plan §3.2).

    The name, field set, and field ORDER are canonical - use verbatim
    everywhere (enforced by a test). ``aleatoric_sigma`` is irreducible
    observation noise; ``epistemic_sigma`` is model ignorance;
    ``conformal_set`` is the shift-robust conformal interval/set the
    operator sees (§5.6).
    """

    mean: np.ndarray
    aleatoric_sigma: np.ndarray
    epistemic_sigma: np.ndarray
    conformal_set: Any


# ---------------------------------------------------------------------------
# §3.1 typed input variables + change-cost classes
# ---------------------------------------------------------------------------


class ChangeCost(enum.Enum):
    """Change-cost class per variable (implementation-plan §3.1, split-plot structure).

    HARD_TO_CHANGE factors (tool, chamber) are treated as conditioning,
    not free variables, by the inverse (§8.3).
    """

    HARD_TO_CHANGE = "hard_to_change"
    EASY = "easy"


@dataclass(frozen=True)
class ContinuousVariable:
    """Continuous factor with box bounds, in the declared engineering unit."""

    name: str
    lower: float
    upper: float
    unit: str = "dimensionless"
    change_cost: ChangeCost = ChangeCost.EASY

    def __post_init__(self) -> None:
        if not self.lower < self.upper:
            raise ValueError(f"{self.name}: lower ({self.lower}) must be < upper ({self.upper})")


@dataclass(frozen=True)
class CategoricalVariable:
    """Categorical factor with enumerated levels."""

    name: str
    levels: tuple[str, ...]
    change_cost: ChangeCost = ChangeCost.EASY

    def __post_init__(self) -> None:
        if len(self.levels) < 2:
            raise ValueError(f"{self.name}: need >=2 levels")


@dataclass(frozen=True)
class CompositionalVariable:
    """TRUE compositional/simplex factor: components sum to 1 (implementation-plan §3.1).

    NB: independent MFC gas setpoints in sccm are NOT a simplex - they are
    independent box-bounded flows whose total is not fixed. Declare those
    as separate :class:`ContinuousVariable` flows; use this class only for
    genuine compositions (alloy mole fractions, fraction-defined blends).
    """

    name: str
    components: tuple[str, ...]
    change_cost: ChangeCost = ChangeCost.EASY

    def __post_init__(self) -> None:
        if len(self.components) < 2:
            raise ValueError(f"{self.name}: a simplex needs >=2 components")


type VariableSpec = ContinuousVariable | CategoricalVariable | CompositionalVariable

# ---------------------------------------------------------------------------
# §3.1 output schema: modality tag + spec semantics
# ---------------------------------------------------------------------------

type Modality = Literal["scalar_vector", "curve_1d", "field_2d"]


@dataclass(frozen=True)
class OutputSpec:
    """One declared process output with modality tag + tolerance/spec semantics."""

    name: str
    modality: Modality
    unit: str = "dimensionless"
    target: float | None = None
    lower_spec: float | None = None
    upper_spec: float | None = None


# ---------------------------------------------------------------------------
# §3.1 cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Fixed per-batch cost vs variable per-recipe cost (implementation-plan §3.1, §11).

    ``c_batch`` is the fixed cost of firing a batch; ``c_recipe(x)`` the
    variable cost of one recipe. The split drives acquisition (§8) and the
    stop/continue rule (§11).
    """

    c_batch: float
    c_recipe: Callable[[Mapping[str, Any]], float]
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")


# ---------------------------------------------------------------------------
# §3.1 ProcessAdapter
# ---------------------------------------------------------------------------


class AdapterValidationError(ValueError):
    """Raised when a ProcessAdapter violates a structural invariant (e.g. D7)."""


class ProcessAdapter(Protocol):
    """Declares the process; owns ALL process-specific knowledge (implementation-plan §3.1).

    Downstream cores (training, UQ, inverse, evaluation) import only
    ``rig.interfaces`` and ``rig.registry``; they never import an adapter
    module directly.
    """

    # -- identity -----------------------------------------------------------
    @property
    def process_id(self) -> str: ...

    # -- input / output schema ---------------------------------------------
    @property
    def input_schema(self) -> Sequence[VariableSpec]: ...

    @property
    def output_schema(self) -> Sequence[OutputSpec]: ...

    # -- cost model (§11) ----------------------------------------------------
    @property
    def cost_model(self) -> CostModel: ...

    # -- DoE / warm-start hooks (§3.1): expert ranges + space-filling seed ---
    @property
    def expert_ranges(self) -> Mapping[str, tuple[float, float]]:
        """Expert-constrained sub-ranges (in declared units) for seeding."""
        ...

    def seed_design(self, n_runs: int, seed: int) -> list[dict[str, Any]]:
        """Scrambled-Sobol space-filling seed design over the expert ranges."""
        ...

    # -- optional physics plug-in + INDEPENDENT verifier (D7) ----------------
    @property
    def physics_plugin(self) -> Callable[..., Any] | None:
        """Optional f_physics(x) prior. Absent (None) by default."""
        ...

    @property
    def independent_verifier(self) -> Callable[..., Any] | None:
        """Optional verifier OUTSIDE the training/inversion loop (§3.4, D7).

        Must be a DIFFERENT object from ``physics_plugin`` when both exist.
        """
        ...

    # -- encoders (§3.1): modality-appropriate embedding ---------------------
    def encode_recipe(self, recipe: Mapping[str, Any]) -> np.ndarray: ...

    def decode_recipe(self, x: np.ndarray) -> dict[str, Any]: ...


def validate_adapter(adapter: ProcessAdapter) -> None:
    """Structural validation of a ProcessAdapter. Raises AdapterValidationError.

    Enforces D7: the physics plug-in and the independent verifier must be
    DIFFERENT objects when both are present (the physics-fidelity benchmark
    must be non-circular by construction, implementation-plan §13.3).
    """
    physics = adapter.physics_plugin
    verifier = adapter.independent_verifier
    if physics is not None and verifier is not None and physics is verifier:
        raise AdapterValidationError(
            f"adapter {adapter.process_id!r} violates D7: independent_verifier "
            "is the SAME object as physics_plugin - the verifier must be "
            "independent of the physics prior (implementation-plan §3.1, §13.3)."
        )
    if not adapter.process_id:
        raise AdapterValidationError("adapter must declare a non-empty process_id")


def sobol_seed_design(
    ranges: Mapping[str, tuple[float, float]], n_runs: int, seed: int
) -> list[dict[str, float]]:
    """Scrambled Sobol' seed design over named box ranges (implementation-plan §3.1 DoE hook).

    Helper adapters may delegate ``seed_design`` to. Uses scipy.stats.qmc
    scrambled Sobol' (the plan's primary space-filling design; maximin LHS
    is the d<=8 fallback, owned by adapters if needed).
    """
    if n_runs < 1:
        raise ValueError("n_runs must be >= 1")
    names = list(ranges)
    if not names:
        raise ValueError("ranges must be non-empty (at least one named box range)")
    lower = np.array([ranges[n][0] for n in names], dtype=float)
    upper = np.array([ranges[n][1] for n in names], dtype=float)
    sampler = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    # A non-power-of-2 n_runs is a valid, deterministic (scrambled) design; scipy
    # only warns that the Sobol' BALANCE property needs a power of 2. Typical DoE
    # sizes (6, 10, 12) are not powers of 2, so suppress the noise deliberately
    # rather than spam every seed_design call (and trip any warnings-as-errors CI).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        u = sampler.random(n_runs)
    x = qmc.scale(u, lower, upper)
    return [dict(zip(names, row, strict=True)) for row in x.tolist()]


# ---------------------------------------------------------------------------
# §3.2 ForwardModel
# ---------------------------------------------------------------------------


@runtime_checkable
class ForwardModel(Protocol):
    """Learned probabilistic simulator (implementation-plan §3.2, §5)."""

    def predict(self, x: np.ndarray) -> PredictiveDistribution:
        """Canonical: predict(x) -> PredictiveDistribution(mean,
        aleatoric_sigma, epistemic_sigma, conformal_set). Never a scalar."""
        ...

    def support_score(self, x: np.ndarray) -> float:
        """In-distribution density / distance-to-support score (§8.2).

        Makes surrogate exploitation a first-class testable signal (§13.2).
        """
        ...

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """d(mean)/dx for sensitivity reporting (§3.2)."""
        ...

    def update(self, records: Iterable[Any]) -> None:
        """Ingest a Loop-B batch of RunRecords to fine-tune + recalibrate
        (invariant 2d)."""
        ...


# ---------------------------------------------------------------------------
# §3.3 InverseSolver + tagged-union result (never a clipped point)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeCandidate:
    """One ranked inverse candidate (implementation-plan §3.3, §13.4 serving contract).

    ``calibration_status`` records HOW MUCH of the §13.2 calibrated-acceptance
    ladder this candidate cleared, so a caller can tell a raw-σ recommendation
    apart from a conformally-accepted one (F1, audit 2026-07-21). It is the
    honest label the audit's F1 remediation asks for ("...or label it merely
    'model-feasible'"). Documented values:

    - ``"model-feasible"`` (default) — the §8 pessimistic κ·σ margins accept it,
      but NO conformal model was available (``self.model`` is not conformal-wrapped
      AND no ``revalidation_model`` was given). This is a surrogate recommendation,
      **NOT a calibrated guarantee**: the κ margin is only ever as trustworthy as
      the surrogate's own σ, which can be optimistic — the mechanism behind the
      deterministic d=20 false success (docs/dimensionality-2026-07-17.md).
    - ``"conformal-checked"`` — ``self.model`` IS conformal-wrapped and the
      candidate cleared the default-path §13.2 containment C(x) ⊆ Z* on it.
    - ``"revalidated"`` — the candidate cleared §13.2 re-validation on an explicit
      ``revalidation_model`` (the full-ensemble + conformal arbiter, §5.7).

    The default keeps every existing construction working and preserves the five
    canonical §3.3 field names verbatim; this is an additive provenance tag.
    """

    recipe: Mapping[str, Any]
    confidence: float
    predicted_outcome_interval: Any
    feasibility_flag: bool
    support_score: float
    calibration_status: str = "model-feasible"


@dataclass(frozen=True)
class Infeasible:
    """Explicit INFEASIBLE verdict (implementation-plan §3.3) - never a clipped point.

    Carries the nearest achievable Pareto point and its distance-to-feasible.
    """

    nearest_achievable: Mapping[str, Any]
    distance_to_feasible: float
    reason: str = ""


type InverseResult = list[RecipeCandidate] | Infeasible
"""Tagged union: a ranked candidate set, or an explicit Infeasible verdict."""


@runtime_checkable
class InverseSolver(Protocol):
    """Target -> set of recipes with calibrated confidence (implementation-plan §3.3, §8)."""

    def solve(self, spec: Mapping[str, Any]) -> InverseResult:
        """spec: multi-objective / set-of-ranges target + optional constraints
        and cost budget. Returns a ranked set, or an explicit Infeasible."""
        ...


# ---------------------------------------------------------------------------
# §3.4 QualificationGate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualificationRecord:
    """Logged qualification evidence (implementation-plan §3.4, §14). No recipe reaches
    production without one."""

    passed: bool
    evidence: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class QualificationGate(Protocol):
    """Independent deployment certification (implementation-plan §3.4, §11, §14).

    Uses a verifier OUTSIDE the training/inversion loop (independent physics
    solver or a fixed confirmation batch on the real tool) - see D7.
    """

    def certify(self, recipe: Mapping[str, Any]) -> QualificationRecord: ...
