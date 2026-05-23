from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class SoftmaxLoss(nn.Module):
    """Vanilla cross-entropy with the dict-return contract shared by all GAM-Softmax losses.

    Returned keys (mask/delta_eff/masked_ratio are None/0 here so diagnostics
    code can treat every loss uniformly).
    """

    def __init__(self, n_classes: int, label_smoothing: float = 0.0):
        super().__init__()
        self.n_classes = n_classes
        self.label_smoothing = label_smoothing

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        loss = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return {
            "loss": loss,
            "mask": None,
            "delta_eff": None,
            "masked_ratio": torch.tensor(0.0, device=logits.device),
        }
