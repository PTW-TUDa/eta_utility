from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    @abstractmethod
    def value(self, progress_remaining: float) -> float:
        raise NotImplementedError("You can only instantiate subclasses of BaseSchedule.")

    def __call__(self, progress_remaining: float) -> float:
        """Take the current progress remaining and return the result of self.value."""
        return self.value(progress_remaining)


class LinearSchedule(BaseSchedule):
    """
    Linear interpolation schedule between initial_p and final_p. The value is calculated based on the
    remaining progress, which is between 1 (start) and 0 (end).

    :param initial_p: Initial output value.
    :param final_p: Final output value.
    """

    def __init__(self, initial_p: float, final_p: float):
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, current_progress_remaining: float) -> float:
        """
        Value of the schedule for a given remaining process.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        return self.final_p + current_progress_remaining * (self.initial_p - self.final_p)
