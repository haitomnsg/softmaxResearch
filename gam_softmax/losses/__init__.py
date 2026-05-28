from gam_softmax.losses.softmax import SoftmaxLoss
from gam_softmax.losses.as_softmax import ASSoftmaxLoss
from gam_softmax.losses.am_softmax import AMSoftmaxLoss
from gam_softmax.losses.sparsemax import SparsemaxLoss
from gam_softmax.losses.entmax import Entmax15Loss
from gam_softmax.losses.focal import FocalLoss
from gam_softmax.losses.label_smoothing import LabelSmoothingLoss
from gam_softmax.losses.power_softmax import PowerSoftmaxLoss
from gam_softmax.losses.gam_softmax import GAMSoftmaxLoss

__all__ = [
    "SoftmaxLoss",
    "ASSoftmaxLoss",
    "AMSoftmaxLoss",
    "SparsemaxLoss",
    "Entmax15Loss",
    "FocalLoss",
    "LabelSmoothingLoss",
    "PowerSoftmaxLoss",
    "GAMSoftmaxLoss",
]
