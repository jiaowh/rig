"""Inverse recipe generation (implementation-plan §8, §14.3). Per-query pessimistic
solver — the D2 canonical refiner (numpy/scipy, always available); amortized NPE
generator — the D2 "instant-answer" proposal service (torch/zuko, the ``[torch]``
extra, imported lazily so ``import rig`` stays torch-free)."""

from typing import TYPE_CHECKING

from rig.inverse.d2 import AmortizedRefiner
from rig.inverse.pessimistic import (
    PessimisticInverseSolver,
    SpecBox,
    parse_targets,
)

if TYPE_CHECKING:
    from rig.inverse.amortized import AmortizedInverseGenerator, CalibrationGate
    from rig.inverse.typicality import FlowTypicalityScore

__all__ = [
    "PessimisticInverseSolver",
    "SpecBox",
    "parse_targets",
    "AmortizedRefiner",
    "AmortizedInverseGenerator",
    "CalibrationGate",
    "FlowTypicalityScore",
]


def __getattr__(name: str):
    # Lazy so the torch extra is only required if the amortized generator / flow
    # typicality screen is actually used — `import rig` stays torch-free.
    if name in ("AmortizedInverseGenerator", "CalibrationGate"):
        from rig.inverse import amortized

        return getattr(amortized, name)
    if name == "FlowTypicalityScore":
        from rig.inverse.typicality import FlowTypicalityScore

        return FlowTypicalityScore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
