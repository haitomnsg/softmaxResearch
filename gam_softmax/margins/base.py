from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class MarginFunction(nn.Module):
    """Returns the per-(sample, non-target-class) margin tensor used by GAM-Softmax.

    Implementations parameterize the margin along one or more axes
    (class-pair, sample, time). See [[gam-softmax-method]] in docs/02.
    """

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        step_frac: float,
    ) -> torch.Tensor:
        ...
