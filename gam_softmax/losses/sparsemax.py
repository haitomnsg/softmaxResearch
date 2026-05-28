from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SparsemaxLoss(nn.Module):
    """Sparsemax (Martins & Astudillo 2016) — Euclidean projection onto the
    simplex; produces exactly-sparse probabilities. Loss from the entmax pkg.
    """

    def __init__(self, n_classes: int):
        super().__init__()
        self.n_classes = n_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        from entmax import sparsemax, sparsemax_loss

        loss = sparsemax_loss(logits, targets).mean()

        with torch.no_grad():
            probs = sparsemax(logits, dim=-1)
            B, n = logits.shape
            target_oh = torch.zeros_like(probs, dtype=torch.bool)
            target_oh.scatter_(1, targets.unsqueeze(1), True)
            non_target_zero = ((probs == 0) & ~target_oh).sum()
            masked_ratio = non_target_zero.float() / max(1, B * (n - 1))

        return {
            "loss": loss,
            "mask": (probs > 0),
            "delta_eff": None,
            "masked_ratio": masked_ratio,
        }
