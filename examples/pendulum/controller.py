from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.agents import RuleBased
from eta_utility.eta_x.common import is_vectorized_env

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class PendulumControl(RuleBased):
    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, **kwargs: Any) -> None:
        """
        Simple controller to test environments.

        :param policy: The policy model to use (not relevant here).
        :param env: The environment to learn from.
        :param verbose: Logging verbosity.
        :param kwargs: Additional arguments as specified in
            :py:class:`stable_baselins3.commom.base_class.BaseAlgorithm`.
        """

        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)
        assert self.action_space is not None, "action_space not initialized correctly."
        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """This function is abstract and should be used to implement control rules which determine actions from
        the received observations.

        :param observation: Observations as provided by a single, non vectorized environment.
        :return: Action values, as determined by the control rules.
        """
        # Handle vectorized environments
        # correct type ensured by is_vectorized_env
        if is_vectorized_env(self.env) and self.env.num_envs > 1:  # type: ignore
            raise ValueError("The PendulumController can only work on a single environment at once")

        # Calculate actions according to the observations (manually extracting observation values
        cos_th = observation[0]
        # observation[1] (sin_th) not needed here
        th_dot = observation[2]

        # Control rules, can you find a better one? :)
        if cos_th < abs(0.3):
            torque = 0.3 * abs(cos_th) * th_dot
        else:
            torque = -1.5 * abs(cos_th) * th_dot

        action = np.fromiter([torque], dtype=np.floating)

        return action
