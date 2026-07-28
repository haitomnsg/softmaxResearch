from __future__ import annotations

import torch

from gam_softmax.margins.base import MarginFunction
from gam_softmax.schedules.base import Schedule


class SampleConfidenceMargin(MarginFunction):
    """Direction-2 (M4): sample-adaptive margin driven by batch-relative confidence.

        δ_i(τ) = clamp( δ_base(τ) − β(τ) · (2·s_i − 1),  δ_floor, δ_ceil )

        s_i = σ( (p̄_t − p_{t,i}) / (std(p_t) · temp) )

    where ``p_{t,i}`` is the (detached) probability the model currently assigns to
    sample *i*'s training label, and ``p̄_t``/``std`` are taken over the batch.
    ``s_i > 0.5`` means "this sample is less confident than its batch peers" —
    i.e. **suspicious**.

    Direction of the effect
    -----------------------
    - **Suspicious sample** (low p_t) → δ is pushed *down* → more non-target
      classes clear the ``p_t − p_j ≥ δ`` bar → they are masked out → the sample
      contributes less gradient.
    - **Confident sample** (high p_t) → δ is pushed *up* → fewer classes are
      masked → the sample keeps training.

    Why this is the right axis for label noise
    ------------------------------------------
    Mislabeled examples are exactly the ones the model keeps assigning low
    probability to. Letting δ fall **below zero** for them (``delta_floor < 0``)
    masks even classes that are *ahead* of the training label, so every
    non-target comparison disappears and the sample's loss collapses to ~0. That
    makes GAM-Softmax subsume the noisy-label literature's "small-loss trick"
    (drop high-loss samples) as a special case of a *margin* — no separate
    sample-selection heuristic, no extra loss term.

    The correction is **mean-centred** (``2·s_i − 1`` averages to ≈0 over the
    batch), so this knob varies δ *across* samples without changing the overall
    masking level. Moving the level is the time axis's job (the ``Schedule``),
    which keeps the two axes separable in ablations.

    Parameters
    ----------
    n_classes : int
    schedule : Schedule
        Supplies the time-varying base margin ``δ_base(τ)`` (the time axis).
        Ignored when ``base_margin`` is given.
    beta : float
        Maximum size of the per-sample deviation from ``δ_base``. ``beta=0``
        recovers the base margin exactly (useful as an ablation control).
    temp : float
        Softness of the suspicion score in units of the batch std. Smaller →
        closer to a hard "below/above average" split.
    reject_warmup_frac : float
        Fraction of training during which ``β(τ) = 0``. Early on the model is
        near-random, so p_t carries no signal about which labels are wrong;
        rejecting samples then would just discard data at random. β ramps
        linearly from 0 to ``beta`` over the remainder of training.
    delta_floor : float
        Lower clamp on δ. Since ``p_t − p_j ∈ [−1, 1]``, ``delta_floor = −1``
        permits full rejection of a sample; ``delta_floor = 0`` forbids masking
        any class that is currently beating the training label.
    delta_ceiling : float
        Upper clamp on δ. Setting this to the schedule's ``delta_max`` makes the
        axis **one-sided**: suspicious samples get a looser margin, but confident
        samples never get a *tighter* one than the AS-Softmax baseline would give
        them. That keeps the comparison against AS-Softmax honest — any
        difference comes from what the method does to suspicious samples, not
        from quietly reverting confident ones to dense cross-entropy.
    base_margin : MarginFunction | None
        Optional inner margin to modulate instead of the scalar schedule. Pass
        a ``ClassPairLowRankMargin`` here to get the **combined** class-pair ×
        sample variant (M6) without writing a new class.
    """

    def __init__(
        self,
        n_classes: int,
        schedule: Schedule,
        beta: float = 0.5,
        temp: float = 1.0,
        reject_warmup_frac: float = 0.3,
        delta_floor: float = -1.0,
        delta_ceiling: float = 1.0,
        base_margin: MarginFunction | None = None,
    ):
        super().__init__()
        if beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if temp <= 0.0:
            raise ValueError(f"temp must be positive, got {temp}")
        if not 0.0 <= reject_warmup_frac <= 1.0:
            raise ValueError(
                f"reject_warmup_frac must be in [0, 1], got {reject_warmup_frac}"
            )
        if not -1.0 <= delta_floor <= 1.0:
            raise ValueError(f"delta_floor must be in [-1, 1], got {delta_floor}")
        if not -1.0 <= delta_ceiling <= 1.0:
            raise ValueError(f"delta_ceiling must be in [-1, 1], got {delta_ceiling}")
        if delta_floor > delta_ceiling:
            raise ValueError(
                f"delta_floor must be <= delta_ceiling, got {delta_floor} > {delta_ceiling}"
            )
        if base_margin is not None and not isinstance(base_margin, MarginFunction):
            raise TypeError(
                f"base_margin must be a MarginFunction, got {type(base_margin).__name__}"
            )
        self.n_classes = n_classes
        self.schedule = schedule
        self.beta = beta
        self.temp = temp
        self.reject_warmup_frac = reject_warmup_frac
        self.delta_floor = delta_floor
        self.delta_ceiling = delta_ceiling
        self.base_margin = base_margin

    def beta_at(self, step_frac: float) -> float:
        """β(τ): 0 through the warmup, then a linear ramp to ``beta``."""
        w = self.reject_warmup_frac
        if w >= 1.0:
            return 0.0
        ramp = (step_frac - w) / (1.0 - w)
        return self.beta * min(1.0, max(0.0, ramp))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        features: torch.Tensor | None,
        step_frac: float,
    ) -> torch.Tensor:
        B, n = logits.shape

        if self.base_margin is not None:
            delta_base = self.base_margin(logits, targets, features, step_frac)
        else:
            delta_base = float(self.schedule(step_frac))

        with torch.no_grad():
            # Suspicion is a diagnostic read of the model's current state, never a
            # path for gradient — same stop-gradient discipline as the AS-Softmax mask.
            probs = torch.softmax(logits, dim=-1)
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)
            sd = p_t.std(unbiased=False).clamp_min(1e-6)
            z = (p_t.mean() - p_t) / sd                                 # >0 ⇒ below average
            s = torch.sigmoid(z / self.temp)                            # (B,) ∈ (0, 1)
            signed = (2.0 * s - 1.0).unsqueeze(1)                       # (B, 1), batch-mean ≈ 0

        delta = delta_base - self.beta_at(step_frac) * signed
        delta = delta.clamp(self.delta_floor, self.delta_ceiling)
        return delta.expand(B, n) if delta.shape[1] == 1 else delta
