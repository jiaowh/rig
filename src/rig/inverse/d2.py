"""D2 inverse engine — amortized proposal + one per-query pessimistic refinement
(implementation-plan §2.2 D2, §14.3).

D2 splits the inverse into two cleanly-separated stages:

1. an **offline "instant-answer" amortized posterior** (the §14.3 NPE flow generator,
   :class:`~rig.inverse.AmortizedInverseGenerator`) that emits a diverse set of
   feasible-by-construction candidate recipes for a spec in one forward pass, and
2. **one** per-query risk-averse refinement — the pessimistic min-max of §8
   (:class:`~rig.inverse.PessimisticInverseSolver`), the single canonical refiner —
   which polishes those proposals by using them as multi-start seeds.

:class:`AmortizedRefiner` is that composition. It is deliberately process-agnostic
and **torch-free**: the generator is injected and used only through its ``sample``
method (duck-typed), so this module lives in the core and never pulls in the
``[torch]`` extra. Any object exposing ``sample(spec, n) -> list[recipe-dict]`` works
as the proposal service.

**Calibration boundary (D2, verbatim):** calibration attaches to the amortized
proposal (via the §14.6 SBC/TARP gate on the generator) and to the conformally
re-validated selected set (via the solver's ``revalidation_model`` + §13.2 gate) —
**never to the refined output**, which optimizes a deliberately different
risk-reweighted objective. This engine only *routes* proposals into the refiner; it
adds no calibration claim of its own.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from rig.interfaces import InverseResult


class _Proposer(Protocol):
    """The offline amortized proposal service (duck-typed; the §14.3 generator)."""

    def sample(self, spec: Mapping[str, Any], n: int) -> list[dict[str, float]]: ...


class _Refiner(Protocol):
    """The single canonical per-query refiner (the §8 pessimistic solver)."""

    def solve(self, spec: Mapping[str, Any]) -> InverseResult: ...


class AmortizedRefiner:
    """D2 inverse engine: amortized proposal + one per-query pessimistic refinement.

    Implements the :class:`~rig.interfaces.InverseSolver` protocol
    (``solve(spec) -> InverseResult``), so it is a drop-in for the bare per-query
    solver — the difference is that its multi-start is *warm-started* from the
    amortized generator's proposals instead of (only) cold Sobol points.

    Parameters
    ----------
    generator:
        The offline amortized posterior — anything with
        ``sample(spec, n) -> list[recipe-dict]`` (the §14.3
        :class:`~rig.inverse.AmortizedInverseGenerator`). Its recipe dicts MUST use
        the same flat-key layout as ``solver``'s recipe variables.
    solver:
        The per-query :class:`~rig.inverse.PessimisticInverseSolver` (the single §8
        refiner). It must honour ``spec['warm_start_recipes']``.
    n_proposals:
        How many amortized proposals to draw and seed the refinement with per query
        (default 8). More proposals = better mode coverage at higher refinement cost.
    """

    def __init__(self, generator: _Proposer, solver: _Refiner, *, n_proposals: int = 8) -> None:
        if n_proposals < 1:
            raise ValueError(f"n_proposals must be >= 1, got {n_proposals}")
        self.generator = generator
        self.solver = solver
        self.n_proposals = int(n_proposals)

    def propose(self, spec: Mapping[str, Any], n: int | None = None) -> list[dict[str, float]]:
        """Draw the amortized proposals for a spec (the offline instant-answer step),
        without refining — exposed for inspection / the M3 acceptance harness."""
        return self.generator.sample(spec, self.n_proposals if n is None else int(n))

    def solve(self, spec: Mapping[str, Any]) -> InverseResult:
        """Amortized-propose, then refine ONCE with the §8 solver seeded on those
        proposals. Returns the solver's ranked candidate set (re-validated if the
        solver carries a ``revalidation_model``) or its explicit ``Infeasible``."""
        proposals = self.propose(spec)
        warmed = {**spec, "warm_start_recipes": proposals}
        return self.solver.solve(warmed)
