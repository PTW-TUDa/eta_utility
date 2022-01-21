import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from eta_utility.eta_x.agents import RuleBased
from eta_utility.eta_x.agents.common.policies import NoPolicy


class SimpleControl(RuleBased):
    def __init__(self, policy: NoPolicy, env: VecEnv, verbose: int = 1, **kwargs) -> None:
        """
        Simple controller to test environments

        :param policy: The policy model to use (not relevant here)
        :param env: The environment to learn from
        :param verbose: Logging verbosity
        :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class
        """
        if "policy_base" not in kwargs:
            kwargs["policy_base"] = None

        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)
        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """This function is abstract and should be used to implement control rules which determine actions from
        the received observations.

        :param observation: Observations as provided by a single, non vectorized environment.
        :return: Action values, as determined by the control rules
        """
        # make sure observation is a numpy array
        observation = np.array(observation)

        # # handle multiple parallel environments or not
        observations = []
        vectorized_env = True if hasattr(self.env, "num_envs") else False
        if not vectorized_env:
            observations[0] = observation
        else:
            observations = observation

        # reset actions
        actions = []

        # calculate actions according to the observations
        # manually unpack values from observation
        cos_th = observation[0]
        # observation[1] (sin_th) not needed here
        th_dot = observation[2]

        # control rules, can you find a better one? :)
        if cos_th < abs(0.3):
            torque = 0.3 * abs(cos_th) * th_dot
        else:
            torque = -1.5 * abs(cos_th) * th_dot

        # prepare action vector
        actions.append(torque)  # here the list of actions contains only one entry, the torque
        actions = np.array(actions, dtype=np.float32)  # turn it into a numpy array

        # check if environment is vectorized
        if not vectorized_env:
            actions = actions[0]

        return actions
