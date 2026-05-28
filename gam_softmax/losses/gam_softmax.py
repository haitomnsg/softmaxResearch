from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from gam_softmax.margins.base import MarginFunction


class GAMSoftmaxLoss(nn.Module):
    """Generalized Adaptive Margin Softmax (docs §2.1).

    Identical to AS-Softmax except the scalar δ is replaced by a per
    (sample, non-target-class) margin tensor produced by a ``MarginFunction``.
    Mask is computed in PROBABILITY space (``p_t - p_i ≥ δ_{t,i}``) and
    detached — gradient flows through the kept logits only.

    Recovers AS-Softmax exactly when the margin tensor is constant.
    """

    def __init__(self, n_classes: int, margin_fn: MarginFunction):
        super().__init__()
        if not isinstance(margin_fn, MarginFunction):
            raise TypeError(
                f"margin_fn must be a MarginFunction, got {type(margin_fn).__name__}"
            )
        self.n_classes = n_classes
        self.margin_fn = margin_fn

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        B, n = logits.shape
        # margin_fn called outside no_grad so its parameters stay in the graph
        # (no gradient actually flows through the hard mask, but this keeps the
        # door open for an STE/soft-surrogate margin variant later).
        delta = self.margin_fn(logits, targets, features, step_frac)  # (B, n)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1))                # (B, 1)
            mask_out = (p_t - probs) >= delta.detach()                 # (B, n)
            target_oh = torch.zeros_like(probs, dtype=torch.bool)
            target_oh.scatter_(1, targets.unsqueeze(1), True)
            mask_out = mask_out & ~target_oh
            keep = ~mask_out

        masked_logits = logits.masked_fill(~keep, float("-inf"))
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        nll = -log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = nll.mean()

        with torch.no_grad():
            non_target_total = B * (n - 1)
            n_masked = mask_out.sum()
            masked_ratio = n_masked.float() / max(1, non_target_total)
            # mean δ across non-target slots (target column is irrelevant)
            non_target = ~target_oh
            delta_eff = (delta.detach() * non_target.float()).sum() / max(1, non_target.sum().item())

        return {
            "loss": loss,
            "mask": keep,
            "delta_eff": delta_eff,
            "masked_ratio": masked_ratio,
        }
