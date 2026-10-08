"""Standard noisy-label baselines, under the shared dict-return contract.

These are the comparisons a noisy-label claim has to survive. M4
(`SampleConfidenceMargin` with δ < 0) claims to be small-loss sample rejection
expressed as a margin, so `SmallLossLoss` is its most direct competitor: the
same idea with an explicit, oracle-sized rejection budget.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class GCELoss(nn.Module):
    """Generalized Cross-Entropy (Zhang & Sabuncu 2018): (1 - p_t^q) / q.

    q → 0 recovers CE, q = 1 is MAE (noise-robust but slow to fit).
    """

    def __init__(self, n_classes: int, q: float = 0.7):
        super().__init__()
        if not 0.0 < q <= 1.0:
            raise ValueError(f"q must be in (0, 1], got {q}")
        self.n_classes = n_classes
        self.q = q

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        p_t = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = ((1.0 - p_t.clamp(min=1e-7) ** self.q) / self.q).mean()
        return {
            "loss": loss,
            "mask": None,
            "delta_eff": None,
            "masked_ratio": torch.tensor(0.0, device=logits.device),
        }


class SCELoss(nn.Module):
    """Symmetric Cross-Entropy (Wang et al. 2019): alpha * CE + beta * RCE.

    RCE = -Σ_k p_k log y_k with log 0 clamped to `log_zero` (A = -4 in the
    paper), which simplifies to -A * (1 - p_t) for one-hot labels.
    """

    def __init__(self, n_classes: int, alpha: float = 0.1, beta: float = 1.0,
                 log_zero: float = -4.0):
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta == 0:
            raise ValueError(f"need alpha, beta >= 0 and not both 0, got {alpha}, {beta}")
        if log_zero >= 0:
            raise ValueError(f"log_zero must be negative, got {log_zero}")
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.log_zero = log_zero

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        ce = F.cross_entropy(logits, targets)
        p_t = F.softmax(logits, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        rce = (-self.log_zero * (1.0 - p_t)).mean()
        return {
            "loss": self.alpha * ce + self.beta * rce,
            "mask": None,
            "delta_eff": None,
            "masked_ratio": torch.tensor(0.0, device=logits.device),
        }


class SmallLossLoss(nn.Module):
    """Small-loss sample selection (the selection rule of Co-teaching, Han et al. 2018),
    single network: train on the (1 - r(t)) fraction of each batch with the lowest CE.

    r(t) = forget_rate * min(step_frac / ramp_frac, 1). Co-teaching sets
    forget_rate to the (oracle) noise rate. The per-sample selection is detached:
    it's a hard choice of which samples count, like the AS-Softmax mask.

    `masked_ratio` reports the rejected fraction of the batch. A rejected sample
    has every non-target slot dropped, so this equals the fraction of
    non-target slots masked, as the contract asks.
    """

    def __init__(self, n_classes: int, forget_rate: float = 0.4, ramp_frac: float = 0.3):
        super().__init__()
        if not 0.0 <= forget_rate < 1.0:
            raise ValueError(f"forget_rate must be in [0, 1), got {forget_rate}")
        if not 0.0 <= ramp_frac <= 1.0:
            raise ValueError(f"ramp_frac must be in [0, 1], got {ramp_frac}")
        self.n_classes = n_classes
        self.forget_rate = forget_rate
        self.ramp_frac = ramp_frac

    def current_rate(self, step_frac: float) -> float:
        if self.ramp_frac == 0.0:
            return self.forget_rate
        return self.forget_rate * min(step_frac / self.ramp_frac, 1.0)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        step_frac: float = 0.0,
    ) -> dict:
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        b = per_sample.shape[0]
        n_keep = max(1, int(round(b * (1.0 - self.current_rate(step_frac)))))
        with torch.no_grad():
            keep_idx = torch.argsort(per_sample)[:n_keep]
        loss = per_sample[keep_idx].mean()
        return {
            "loss": loss,
            "mask": None,
            "delta_eff": None,
            "masked_ratio": torch.tensor(1.0 - n_keep / b, device=logits.device),
        }
