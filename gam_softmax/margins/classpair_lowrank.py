from __future__ import annotations

import math

import torch
from torch import nn

from gam_softmax.margins.base import MarginFunction
from gam_softmax.schedules.base import Schedule


class ClassPairLowRankMargin(MarginFunction):
    """Direction-1 (M3): low-rank class-pair margin.

        δ_{t,j} = δ_min + (δ_max(τ) − δ_min) · σ(u_t · v_j / √k)

    Parameters
    ----------
    n_classes : int
    schedule  : Schedule
        Provides the time-varying upper cap ``δ_max(τ)``. Must expose ``delta_min``
        attribute (the floor); we read ``delta_min`` directly off it so the floor
        and cap can't drift apart.
    rank      : int   k ∈ {4, 8, 16}; defaults to 8 per docs §2.3.
    init_std  : float small-std init keeps the structural prior near uniform
                      at step 0 (σ(≈0) = 0.5). Combined with a warmup schedule
                      whose δ_max(0) == δ_min, the effective margin collapses to
                      δ_min at step 0 — the "start loose" recipe in docs §2.3.

    Gradient note
    -------------
    docs §9 specifies stop-gradient through the AS-Softmax mask. The mask is a
    hard indicator, so even if δ has a gradient, nothing flows back to (u, v)
    through the masked log-sum-exp. (u, v) are kept as nn.Parameters for two
    reasons: (a) they're picked up by the optimizer/state_dict — so adding an
    STE / soft-surrogate path later (M3' variant) is a one-line change, not a
    refactor; (b) ablations that *do* update (u, v) via auxiliary losses still
    have somewhere to put grads. For the H1 quick-check, (u, v) effectively
    stays at init — H1 therefore tests whether *any* per-pair structure beats
    scalar AS-Softmax, before we invest in making (u, v) genuinely learnable.
    """

    def __init__(
        self,
        n_classes: int,
        schedule: Schedule,
        rank: int = 8,
        init_std: float = 0.02,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if init_std <= 0:
            raise ValueError(f"init_std must be positive, got {init_std}")
        if not hasattr(schedule, "delta_min"):
            raise ValueError(
                "schedule must expose `delta_min`; pass a schedule that owns the "
                "(δ_min, δ_max) bounds, e.g. LinearSchedule."
            )
        self.n_classes = n_classes
        self.rank = rank
        self.schedule = schedule
        self.u = nn.Parameter(torch.randn(n_classes, rank) * init_std)
        self.v = nn.Parameter(torch.randn(n_classes, rank) * init_std)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor | None,
        step_frac: float,
    ) -> torch.Tensor:
        delta_min = float(self.schedule.delta_min)
        delta_max_now = float(self.schedule(step_frac))
        # (n, n) per-pair sigmoid score
        scores = (self.u @ self.v.t()) / math.sqrt(self.rank)
        raw = torch.sigmoid(scores)
        delta_matrix = delta_min + (delta_max_now - delta_min) * raw
        # gather row per sample → (B, n); the target-column value is irrelevant
        # (the loss never masks the target slot).
        return delta_matrix.index_select(0, targets)
