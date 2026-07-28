from gam_softmax.margins.base import MarginFunction
from gam_softmax.margins.classpair_lowrank import ClassPairLowRankMargin
from gam_softmax.margins.sample_confidence import SampleConfidenceMargin

__all__ = ["MarginFunction", "ClassPairLowRankMargin", "SampleConfidenceMargin"]
