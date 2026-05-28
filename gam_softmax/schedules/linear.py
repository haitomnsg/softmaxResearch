from __future__ import annotations

from gam_softmax.schedules.base import Schedule


class LinearSchedule(Schedule):
    """Linear ramp from ``delta_min`` to ``delta_max`` over [0, warmup_frac],
    then constant at ``delta_max``.

    Returns the *cap* used to scale a margin function whose raw value lives
    in [0, 1]. The margin function is responsible for blending δ_min into
    its own output (see ClassPairLowRankMargin).
    """

    def __init__(self, delta_min: float, delta_max: float, warmup_frac: float = 0.3):
        if not 0.0 <= delta_min <= delta_max <= 1.0:
            raise ValueError(
                f"need 0 <= delta_min <= delta_max <= 1; got {delta_min}, {delta_max}"
            )
        if not 0.0 < warmup_frac <= 1.0:
            raise ValueError(f"warmup_frac must be in (0, 1], got {warmup_frac}")
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.warmup_frac = warmup_frac

    def __call__(self, step_frac: float) -> float:
        if step_frac >= self.warmup_frac:
            return self.delta_max
        t = step_frac / self.warmup_frac
        return self.delta_min + (self.delta_max - self.delta_min) * t
