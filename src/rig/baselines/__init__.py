"""Matched-budget baselines to beat (implementation-plan §9.8 / §12.3).

``WarmStartedBO`` — GP + Expected Improvement, warm-started from the expert
ranges, the numpy-tier fair BO comparator for the M2 "beats warm-started BO"
claim. ``BoTorchBO`` (WP-E) — the production BoTorch upgrade: `SingleTaskGP` +
`qLogExpectedImprovement`/`qLCB` optimized CONTINUOUSLY with `optimize_acqf`
(closes the M2 BF-1b "continuous/constrained BoTorch baseline" owed item). It
pulls torch (the ``[torch]`` extra) so it is imported lazily via ``__getattr__``,
keeping ``import rig`` torch-free. qLogNEHVI / TuRBO / SCBO are follow-on WP-E.
"""

from typing import TYPE_CHECKING

from rig.baselines.warm_bo import WarmStartedBO, expected_improvement

if TYPE_CHECKING:
    from rig.baselines.botorch_bo import BoTorchBO
    from rig.baselines.mfl import ModelFeedbackLearning
    from rig.baselines.trust_region_bo import SCBOBaseline, TuRBOBaseline

__all__ = [
    "WarmStartedBO",
    "BoTorchBO",
    "SCBOBaseline",
    "TuRBOBaseline",
    "ModelFeedbackLearning",
    "expected_improvement",
]


def __getattr__(name: str):
    # Lazy so the torch extra is only required if a torch baseline is used.
    if name == "BoTorchBO":
        from rig.baselines.botorch_bo import BoTorchBO

        return BoTorchBO
    if name in ("SCBOBaseline", "TuRBOBaseline"):
        from rig.baselines import trust_region_bo

        return getattr(trust_region_bo, name)
    if name == "ModelFeedbackLearning":
        from rig.baselines.mfl import ModelFeedbackLearning

        return ModelFeedbackLearning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
