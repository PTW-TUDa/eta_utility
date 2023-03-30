from typing import Any

import numpy as np
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv

from eta_utility.eta_x.agents import RuleBased


class RuleBasedController(RuleBased):
    """A simple rule based agent for testing purposes. The agent works with the CartPole-v1 environment, the control
    rules method decides if the cart should move left or right.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: VecEnv,
        verbose: int = 0,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        if _init_setup_model:
            self._setup_model()

    def control_rules(self, observation: np.ndarray) -> int:
        angle = observation[2]
        if angle > 0:
            return 1  # push cart to the right
        else:
            return 0  # push cart to the left
