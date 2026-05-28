from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class PowerSoftmaxLoss(nn.Module):
    """Power Softmax: p_i ∝ (max(o_i, 0) + eps)^gamma.

    ReLU-style sparsity in the output without exponentials. Per the
    [[gam-softmax-roadmap]] pitfalls, for gamma < 1 the gradient at o_i = 0+
    is unbounded; `eps > 0` inside the power bounds it.
    """

    def __init__(self, n_classes: int, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        self.n_classes = n_classes
        self.gamma = gamma
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        z = F.relu(logits) + self.eps
        z_pow = z.pow(self.gamma)
        denom = z_pow.sum(dim=-1, keepdim=True)
        probs = z_pow / denom
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -torch.log(p_t.clamp(min=self.eps)).mean()

        with torch.no_grad():
            # "masked" = non-target slots where the raw logit was <= 0
            # (so they contribute only eps^gamma to the denom — effectively suppressed)
            B, n = logits.shape
            target_oh = torch.zeros_like(logits, dtype=torch.bool)
            target_oh.scatter_(1, targets.unsqueeze(1), True)
            suppressed = (logits <= 0) & ~target_oh
            masked_ratio = suppressed.sum().float() / max(1, B * (n - 1))

        return {
            "loss": loss,
            "mask": (logits > 0) | target_oh,
            "delta_eff": torch.tensor(self.gamma, device=logits.device),
            "masked_ratio": masked_ratio,
        }
