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

    Learnable margin (``learnable_margin=True``)
    --------------------------------------------
    By default the kept-set is a hard threshold, so no gradient reaches the
    margin's parameters (e.g. M3's u, v) — they stay at their random init and
    the margin is effectively fixed. With ``learnable_margin=True`` the loss
    uses a straight-through estimator: the forward value still uses the hard
    kept-set (so the loss is numerically identical and still recovers
    AS-Softmax), but the backward pass routes gradient into δ via a sigmoid
    surrogate of the keep decision (keep ⇔ ``p_t - p_i < δ``). This lets the
    margin's parameters actually train. ``ste_temp`` is the surrogate
    temperature: smaller → sharper (closer to the hard step), larger → smoother
    gradient.
    """

    def __init__(
        self,
        n_classes: int,
        margin_fn: MarginFunction,
        learnable_margin: bool = False,
        ste_temp: float = 0.1,
    ):
        super().__init__()
        if not isinstance(margin_fn, MarginFunction):
            raise TypeError(
                f"margin_fn must be a MarginFunction, got {type(margin_fn).__name__}"
            )
        if ste_temp <= 0.0:
            raise ValueError(f"ste_temp must be positive, got {ste_temp}")
        self.n_classes = n_classes
        self.margin_fn = margin_fn
        self.learnable_margin = learnable_margin
        self.ste_temp = ste_temp
        # Stateful per-sample margins need to know which training examples are in
        # the batch; the trainer passes `sample_idx` only when this is True.
        self.wants_sample_idx = bool(getattr(margin_fn, "wants_sample_idx", False))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
        sample_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        B, n = logits.shape
        # margin_fn called outside no_grad so its parameters stay in the graph;
        # the learnable-margin STE path below is what actually sends gradient to them.
        if self.wants_sample_idx:
            delta = self.margin_fn(logits, targets, features, step_frac, sample_idx=sample_idx)
        else:
            delta = self.margin_fn(logits, targets, features, step_frac)  # (B, n)

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1))                # (B, 1)
            gap = p_t - probs                                          # (B, n), detached
            mask_out = gap >= delta.detach()                           # (B, n)
            target_oh = torch.zeros_like(probs, dtype=torch.bool)
            target_oh.scatter_(1, targets.unsqueeze(1), True)
            mask_out = mask_out & ~target_oh
            keep = ~mask_out

        if self.learnable_margin:
            # Straight-through soft mask. keep_ste forward-equals the hard {0,1}
            # mask (gap is detached, so logit gradients match the default path),
            # but its backward flows through the sigmoid surrogate into δ → params.
            keep_soft = torch.sigmoid((delta - gap) / self.ste_temp)   # (B, n)
            keep_soft = keep_soft.masked_fill(target_oh, 1.0)          # target fully kept
            keep_ste = keep.float() + (keep_soft - keep_soft.detach())
            m = logits.max(dim=-1, keepdim=True).values.detach()       # (B, 1) stabilizer
            weighted = torch.exp(logits - m) * keep_ste                # dropped slots = 0 in fwd
            log_denom = m.squeeze(1) + torch.log(weighted.sum(dim=-1).clamp_min(1e-12))
            z_t = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
            nll = -(z_t - log_denom)
        else:
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

        out = {
            "loss": loss,
            "mask": keep,
            "delta_eff": delta_eff,
            "masked_ratio": masked_ratio,
        }
        # Optional diagnostic: a margin that estimates the label-noise rate from
        # the data exposes it as `last_est_noise_rate`; the trainer logs it.
        est = getattr(self.margin_fn, "last_est_noise_rate", None)
        if est is not None:
            out["est_noise_rate"] = torch.as_tensor(float(est), device=logits.device)
        return out
