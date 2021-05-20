import abc
from typing import Callable, List, Tuple

import numpy as np
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import VecEnv

from eta_utility import get_logger

from ..common.policies import NoPolicy

log = get_logger("eta_x.agents.rule_based")


class RuleBased(BaseRLModel, abc.ABC):
    """The rule based agent base class provides the facilities to easily build a complete rule based agent. To achieve
    this, only the predict function must be implemented. It should take an observation from the environment as input
    and provide actions as an output.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param gym.Env env: Environment to be controlled
    :param int verbose: Logging verbosity
    :param kwargs: Additional arguments as specified in stable_baselins.BaseRLModel
    """

    def __init__(self, policy: NoPolicy, env: VecEnv, verbose: int = 4, **kwargs):
        # Ensure that arguments required by super class are always present
        if "requires_vec_env" not in kwargs:
            kwargs["requires_vec_env"] = False
        if "policy_base" not in kwargs:
            kwargs["policy_base"] = None

        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        #: Last / initial State of the agent.
        self.state = np.zeros(self.action_space.shape)

    @abc.abstractmethod
    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """This function is abstract and should be used to implement control rules which determine actions from
        the received observations.

        :param observation: Observations as provided by a single, non vectorized environment.
        :return: Action values, as determined by the control rules
        """

    def predict(
        self,
        observation: np.ndarray,
        state: np.ndarray = None,
        mask: np.ndarray = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """Perform controller operations and return actions. It will take care of vectorization of environments.
        This will call the control_rules method which should implement the control rules for a single environment.

        :param observation: the input observation
        :param state: The last states (not used here)
        :param mask: The last masks (not used here)
        :param deterministic: Whether or not to return deterministic actions. This agent always returns
                              deterministic actions
        :return: Tuple of the model's action and the next state (state is typically None in this agent)
        """
        if self._vectorize_action:
            action = []
            for obs in observation:
                action.append(self.control_rules(obs))

        else:
            action = [self.control_rules(observation)]

        return np.array(action), None

    @classmethod
    def load(cls, load_path: str, env: VecEnv = None, **kwargs):
        """Load model. This is not implemented for the rule based agent.

        :param load_path: Path to load from
        :param env: Environment for training or prediction
        :param kwargs: Other arguments
        :return: None
        """
        log.info("Rule based agents cannot load data. Loading will be ignored.")

    def save(self, save_path: str, **kwargs):
        """Save model after training. Not implemented for the rule based agent.

        :param save_path: Path to save to
        :param kwargs: Other arguments
        :return: None
        """
        log.info("Rule based agents cannot save data. Loading will be ignored.")

    def _get_pretrain_placeholders(self):
        """Getting tensorflow pretrain placeholders is not implemented for the rule based agent"""
        raise NotImplementedError("The rule based agent cannot provide tensorflow pretrain placeholders.")

    def get_parameter_list(self):
        """Getting tensorflow parameters is not implemented for the rule based agent"""
        raise NotImplementedError("The rule pased agent cannot provide a tensorflow parameter list.")

    def learn(
        self,
        total_timesteps: int,
        callback: Callable = None,
        log_interval: int = 10,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ) -> "RuleBased":
        """Learning is not implemented for the rule based agent."""
        raise NotImplementedError("The rule based agent cannot learn a model.")

    def setup_model(self):
        """Setup model is not required for the rule based agent."""
        pass

    def action_probability(
        self,
        observation: np.ndarray,
        state: np.ndarray = None,
        mask: np.ndarray = None,
        actions: np.ndarray = None,
        **kwargs,
    ):
        """Get the model's action probability distribution from an observation. This is not implemented for this agent."""
        raise NotImplementedError("The rule based agent cannot calculate action probabilities.")
