from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class AMSoftmaxLoss(nn.Module):
    """Additive Margin Softmax (Wang et al. 2018).

    Owns its own L2-normalized classification head W (B, d) → cosine logits.
    The model's own `head` output is ignored — only `features` is used.

    For the target class, subtracts an additive margin `m` from cos θ_t before
    applying the temperature `scale`, then standard cross-entropy.
    """

    def __init__(
        self,
        n_classes: int,
        feature_dim: int,
        margin: float = 0.35,
        scale: float = 30.0,
    ):
        super().__init__()
        if not 0.0 <= margin < 1.0:
            raise ValueError(f"margin must be in [0, 1), got {margin}")
        if scale <= 0:
            raise ValueError(f"scale must be > 0, got {scale}")
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(feature_dim, n_classes))
        nn.init.xavier_normal_(self.weight)

    def forward(
        self,
        logits: torch.Tensor,           # unused
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        if features is None:
            raise ValueError("AMSoftmaxLoss requires features; received None")

        x_norm = F.normalize(features, dim=-1)
        w_norm = F.normalize(self.weight, dim=0)
        cos = x_norm @ w_norm                                # (B, n)

        target_oh = F.one_hot(targets, num_classes=self.n_classes).to(cos.dtype)
        cos_with_margin = cos - self.margin * target_oh
        scaled = self.scale * cos_with_margin
        loss = F.cross_entropy(scaled, targets)

        return {
            "loss": loss,
            "mask": None,
            "delta_eff": torch.tensor(self.margin, device=cos.device),
            "masked_ratio": torch.tensor(0.0, device=cos.device),
        }
