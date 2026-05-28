from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al. 2017): -alpha * (1 - p_t)^gamma * log(p_t).

    With gamma=0 and alpha=1 this reduces to standard cross-entropy.
    """

    def __init__(self, n_classes: int, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.n_classes = n_classes
        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = log_p_t.exp()
        focal_weight = (1.0 - p_t).clamp(min=0.0) ** self.gamma
        loss = -(self.alpha * focal_weight * log_p_t).mean()

        return {
            "loss": loss,
            "mask": None,
            "delta_eff": None,
            "masked_ratio": torch.tensor(0.0, device=logits.device),
        }
