from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

np.random.seed(123)


class DampedOscillatorControl(RuleBased):
    """
    Simple controller for input signal of damped oscillator model.
    """

    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1):
        super().__init__(policy=policy, env=env, verbose=1, policy_base=None)

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. For this case, function does not use observation.

        :param observation: observation of the environment given one action
        :returns: uniform random value [-1, 1) as action (u) for the model
        """
        return np.random.uniform(low=-1, high=1, size=1)
