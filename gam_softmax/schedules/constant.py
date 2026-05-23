from __future__ import annotations

from gam_softmax.schedules.base import Schedule


class ConstantSchedule(Schedule):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, step_frac: float) -> float:
        return self.value
