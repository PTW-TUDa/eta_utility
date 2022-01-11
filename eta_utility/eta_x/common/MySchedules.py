"""
My own schedules, because I could not find it in SB3

"""


class MyLinearSchedule:
    """
    Linear interpolation between initial_p and final_p. The value is calculated based on the current_progess_remaining,
    which is between 1 (start) and 0 (end). This value is calculated in the base class.

    :param initial_p: initial output value
    :param final_p: final output value
    """

    def __init__(self, initial_p: float, final_p: float):
        self.initial_p = initial_p
        self.final_p = final_p

    def value(self, current_progress_remaining: float) -> float:
        return self.final_p + current_progress_remaining * (self.initial_p - self.final_p)
