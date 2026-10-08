from gam_softmax.margins.base import MarginFunction
from gam_softmax.margins.classpair_lowrank import ClassPairLowRankMargin
from gam_softmax.margins.sample_confidence import SampleConfidenceMargin
from gam_softmax.margins.absolute_gap import AbsoluteGapMargin
from gam_softmax.margins.loss_mixture import LossMixtureMargin

__all__ = [
    "MarginFunction",
    "ClassPairLowRankMargin",
    "SampleConfidenceMargin",
    "AbsoluteGapMargin",
    "LossMixtureMargin",
]
