from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class DampedOscillatorControl(RuleBased):
    """
    Simple controller for input signal of damped oscillator model.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param p: Proportional factor for the PID controller
    :param i: Integral factor for the PID controller
    :param d: Derivative factor for the PID controller
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(
        self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, *, p: float, i: float, d: float, **kwargs: Any
    ):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        #: Proportional factor for the PID controller.
        self.p = p
        #: Integral factor for the PID controller.
        self.i = i
        #: Derivative factor for the PID controller.
        self.d = d
        #: Integral error value.
        self.integral = 0

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. This implements a simple PID controller

        :param observation: Observation from the environment.
        :returns: Resulting action from the PID controller.
        """
        s, v, a = observation
        self.integral += s

        return np.array([-(self.p * s + self.i * self.integral + self.d * v)])
