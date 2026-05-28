from __future__ import annotations

from gam_softmax.losses.softmax import SoftmaxLoss


class LabelSmoothingLoss(SoftmaxLoss):
    """Cross-entropy with label smoothing (Szegedy et al. 2016).

    Same machinery as SoftmaxLoss but exists as a distinct registry entry so
    the experimental row identity in the paper table is unambiguous.
    """

    def __init__(self, n_classes: int, label_smoothing: float = 0.1):
        if label_smoothing <= 0.0:
            raise ValueError(
                f"label_smoothing must be > 0 for LabelSmoothingLoss (got {label_smoothing}); "
                "use SoftmaxLoss for plain cross-entropy"
            )
        super().__init__(n_classes=n_classes, label_smoothing=label_smoothing)
