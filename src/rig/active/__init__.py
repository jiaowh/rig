"""Active learning & optimal experimental design (implementation-plan §9) — numpy tier.

The closed-loop experiment selector: cost-cooled BALD/EPIG acquisition (§9.4),
BatchBALD-style diverse batches (§9.5), and the warm-started refit-and-re-solve
loop (§9.2/§9.7). The Phase-II qLogNEHVI exploit hand-off (§9.4) and the offline
amortized-posterior re-distillation (D6) are the torch/BoTorch WP-E.
"""

from rig.active.acquisition import (
    anneal,
    bald,
    cost_cooled_acquisition,
    epig,
    qlognehvi_phase2,
)
from rig.active.batch import select_batch
from rig.active.campaign import (
    CampaignOutcome,
    CampaignResult,
    CandidateCertification,
    ConfirmationCampaign,
    NothingToQualify,
)
from rig.active.loop import ActiveLearningLoop, Trajectory

__all__ = [
    "ActiveLearningLoop",
    "CampaignOutcome",
    "CampaignResult",
    "CandidateCertification",
    "ConfirmationCampaign",
    "NothingToQualify",
    "Trajectory",
    "anneal",
    "bald",
    "cost_cooled_acquisition",
    "epig",
    "qlognehvi_phase2",
    "select_batch",
]
