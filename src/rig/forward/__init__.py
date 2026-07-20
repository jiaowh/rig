"""Forward surrogates (implementation-plan §5). Little-data backbone: exact GP (D3);
tool-aware ICM multi-task GP for chamber matching (§10.4 level (a), WP-I);
large-data backbone: deep-ensemble β-NLL + SNGP (D3 backend B, WP-E).

The deep-ensemble tier pulls in torch, which is an OPTIONAL extra
(``pip install -e ".[torch]"``). It is imported lazily via ``__getattr__`` so
``import rig`` / ``import rig.forward`` stays torch-free for the numpy/scipy
core and CI paths that never install the torch stack.
"""

from typing import TYPE_CHECKING

from rig.forward.data import records_to_arrays, records_to_arrays_with_tools
from rig.forward.gp import GPForwardModel
from rig.forward.multitask import MultiToolGPForwardModel, ToolBoundForwardModel

if TYPE_CHECKING:
    from rig.forward.distill import DistilledForwardModel, distill_ensemble
    from rig.forward.ensemble import DeepEnsembleForwardModel

__all__ = [
    "GPForwardModel",
    "MultiToolGPForwardModel",
    "ToolBoundForwardModel",
    "DeepEnsembleForwardModel",
    "DistilledForwardModel",
    "distill_ensemble",
    "records_to_arrays",
    "records_to_arrays_with_tools",
]


def __getattr__(name: str):
    # Lazy so the torch extra is only required if the ensemble tier is used.
    if name == "DeepEnsembleForwardModel":
        from rig.forward.ensemble import DeepEnsembleForwardModel

        return DeepEnsembleForwardModel
    if name in ("DistilledForwardModel", "distill_ensemble"):
        import rig.forward.distill as _distill

        return getattr(_distill, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
