from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class ASSoftmaxLoss(nn.Module):
    """Adaptive Sparse Softmax (Lv et al. 2023).

    Mask z_i = 0 for non-target classes whose probability satisfies
    p_t - p_i >= delta; target class always kept. Loss is standard
    cross-entropy on the renormalized distribution over kept classes.

    The mask is computed in PROBABILITY space (after softmax), not logit
    space — computing the condition on logits loses the AS-Softmax
    reduction property. Mask is detached so no gradient flows through
    the threshold.
    """

    def __init__(
        self,
        n_classes: int,
        delta: float = 0.1,
        warmup_frac: float = 0.0,
    ):
        super().__init__()
        if not 0.0 < delta <= 1.0:
            raise ValueError(f"delta must be in (0, 1], got {delta}")
        if not 0.0 <= warmup_frac < 1.0:
            raise ValueError(f"warmup_frac must be in [0, 1), got {warmup_frac}")
        self.n_classes = n_classes
        self.delta = delta
        self.warmup_frac = warmup_frac

    def _delta_eff(self, step_frac: float) -> float:
        if self.warmup_frac <= 0.0 or step_frac >= self.warmup_frac:
            return self.delta
        return self.delta * (step_frac / self.warmup_frac)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        B, n = logits.shape
        delta_eff = self._delta_eff(step_frac)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1))            # (B, 1)
            mask_out = (p_t - probs) >= delta_eff                  # (B, n)
            # target class is never masked
            target_oh = torch.zeros_like(probs, dtype=torch.bool)
            target_oh.scatter_(1, targets.unsqueeze(1), True)
            mask_out = mask_out & ~target_oh
            keep = ~mask_out                                       # (B, n) bool

        # set masked logits to -inf so log_softmax renormalizes over kept classes
        masked_logits = logits.masked_fill(~keep, float("-inf"))
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        loss = nll.mean()

        non_target_total = B * (n - 1)
        n_masked = mask_out.sum()
        masked_ratio = n_masked.float() / max(1, non_target_total)

        return {
            "loss": loss,
            "mask": keep,
            "delta_eff": torch.tensor(delta_eff, device=logits.device),
            "masked_ratio": masked_ratio,
        }
