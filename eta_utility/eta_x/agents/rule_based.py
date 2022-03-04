from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Any, Callable

    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


log = get_logger("eta_x.agents.rule_based")


class RuleBased(BaseAlgorithm, abc.ABC):
    """The rule based agent base class provides the facilities to easily build a complete rule based agent. To achieve
    this, only the predict function must be implemented. It should take an observation from the environment as input
    and provide actions as an output.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled
    :param verbose: Logging verbosity
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class
    """

    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 4, **kwargs: Any) -> None:
        # Ensure that arguments required by super class are always present
        if "policy_base" not in kwargs:
            kwargs["policy_base"] = None

        super().__init__(policy=policy, env=env, verbose=verbose, learning_rate=0, **kwargs)

        #: Last / initial State of the agent.
        self.state: np.ndarray | None = np.zeros(self.action_space.shape) if self.action_space is not None else None

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
        state: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        """Perform controller operations and return actions. It will take care of vectorization of environments.
        This will call the control_rules method which should implement the control rules for a single environment.

        :param observation: the input observation
        :param state: The last states (not used here)
        :param mask: The last masks (not used here)
        :param deterministic: Whether or not to return deterministic actions. This agent always returns
                              deterministic actions
        :return: Tuple of the model's action and the next state (state is typically None in this agent)
        """
        action = []
        for obs in observation:
            action.append(np.array(self.control_rules(obs)))

        return np.array(action), None

    @classmethod
    def load(cls, load_path: str, env: VecEnv = None, **kwargs) -> RuleBased:
        """Load model. This is not implemented for the rule based agent.

        :param load_path: Path to load from
        :param env: Environment for training or prediction
        :param kwargs: Other arguments
        :return: None
        """
        log.info("Rule based agents cannot load data. Loading will be ignored.")

    def save(self, save_path: str, **kwargs) -> None:
        """Save model after training. Not implemented for the rule based agent.

        :param save_path: Path to save to
        :param kwargs: Other arguments
        :return: None
        """
        log.info("Rule based agents cannot save data. Loading will be ignored.")

    def _get_pretrain_placeholders(self) -> None:
        """Getting tensorflow pretrain placeholders is not implemented for the rule based agent"""
        raise NotImplementedError("The rule based agent cannot provide tensorflow pretrain placeholders.")

    def get_parameter_list(self) -> None:
        """Getting tensorflow parameters is not implemented for the rule based agent"""
        raise NotImplementedError("The rule pased agent cannot provide a tensorflow parameter list.")

    def learn(
        self,
        total_timesteps: int,
        callback: Callable = None,
        log_interval: int = 10,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ) -> RuleBased:
        """Learning is not implemented for the rule based agent."""
        raise NotImplementedError("The rule based agent cannot learn a model.")

    def _setup_model(self) -> None:
        """Setup model is not required for the rule based agent."""
        pass

    def action_probability(
        self,
        observation: np.ndarray,
        state: np.ndarray = None,
        mask: np.ndarray = None,
        actions: np.ndarray = None,
        **kwargs,
    ) -> None:
        """Get the model's action probability distribution from an observation. This is not implemented for this
        agent."""
        raise NotImplementedError("The rule based agent cannot calculate action probabilities.")
