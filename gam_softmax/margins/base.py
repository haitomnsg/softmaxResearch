from __future__ import annotations

from abc import abstractmethod

import torch
from torch import nn


class MarginFunction(nn.Module):
    """Returns the per-(sample, non-target-class) margin tensor used by GAM-Softmax.

    Implementations parameterize the margin along one or more axes
    (class-pair, sample, time). See [[gam-softmax-method]] in docs/02.

    Stateful per-sample margins
    ---------------------------
    A margin that keeps running statistics per training example sets the class
    attribute ``wants_sample_idx = True`` and accepts an extra keyword
    ``sample_idx`` (LongTensor ``(B,)``, the examples' positions in the training
    Dataset). ``GAMSoftmaxLoss`` forwards it only to margins that ask, and the
    trainer passes it only during training steps: evaluation calls never carry
    it, so such margins must treat ``sample_idx=None`` as "read-only, do not
    update state". Margins that don't set the flag keep the 4-argument
    signature below.
    """

    wants_sample_idx: bool = False

    @abstractmethod
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor,
        step_frac: float,
    ) -> torch.Tensor:
        ...
