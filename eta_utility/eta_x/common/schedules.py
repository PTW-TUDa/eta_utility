from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSchedule(ABC):
    """BaseSchedule provides basic functionality for the implementation of new schedules. Each schedule should
    define a value function.
    """

    @abstractmethod
    def value(self, progress_remaining: float) -> float:
        """Calculate the value of the learning rate based on the remaining progess.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        raise NotImplementedError("You can only instantiate subclasses of BaseSchedule.")

    def __call__(self, progress_remaining: float) -> float:
        """Take the current progress remaining and return the result of self.value."""
        return self.value(progress_remaining)

    def __repr__(self) -> str:
        """Representation of the Schedule

        :return: String representation.
        """
        return f"{self.__class__.__name__}({', '.join([f'{name}={value}' for name, value in self.__dict__.items()])})"


class LinearSchedule(BaseSchedule):
    """
    Linear interpolation schedule adjusts the learning rate between initial_p and final_p.
    The value is calculated based on the remaining progress, which is between 1 (start) and 0 (end).

    :param initial_p: Initial output value.
    :param final_p: Final output value.
    """

    def __init__(self, initial_p: float, final_p: float):
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, progress_remaining: float) -> float:
        """Calculate the value of the learning rate based on the remaining progess.

        :param progress_remaining: Remaing progress, which is calculcated in the base class: 1 (start), 0 (end).
        :return: Output value.
        """
        return self.final_p + progress_remaining * (self.initial_p - self.final_p)
