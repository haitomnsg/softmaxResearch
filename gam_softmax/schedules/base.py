from __future__ import annotations

from abc import ABC, abstractmethod


class Schedule(ABC):
    @abstractmethod
    def __call__(self, step_frac: float) -> float:
        ...
