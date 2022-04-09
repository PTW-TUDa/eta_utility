from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from eta_utility import get_logger
from eta_utility.eta_x.common import NoPolicy

if TYPE_CHECKING:
    import io
    import pathlib
    from typing import Any, Iterable

    import torch as th
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv


log = get_logger("eta_x.agents")


class RuleBased(BaseAlgorithm, abc.ABC):
    """The rule based agent base class provides the facilities to easily build a complete rule based agent. To achieve
    this, only the *control_rules* function must be implemented. It should take an observation from the environment
    as input and provide actions as an output.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
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
        :return: Action values, as determined by the control rules.
        """

    def predict(
        self,
        observation: np.ndarray,
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """Perform controller operations and return actions. It will take care of vectorization of environments.
        This will call the control_rules method which should implement the control rules for a single environment.

        :param observation: the input observation.
        :param state: The last states (not used here).
        :param episode_start: The last masks (not used here).
        :param deterministic: Whether to return deterministic actions. This agent always returns
                              deterministic actions.
        :return: Tuple of the model's action and the next state (state is typically None in this agent).
        """
        action = []
        for idx, obs in enumerate(observation):
            action.append(np.array(self.control_rules(obs)))
            log.debug(f"Action vector for environment {idx}: {action[-1]}")

        return np.array(action), None

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> RuleBased:
        """Load model. This is not implemented for the rule based agent.

        :param path: path to the file (or a file-like) where to
            load the agent from.
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment.
        :param device: Device on which the code should run..
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force a call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading.
        """
        log.info("Rule based agents cannot load data. Loading will be ignored - using standard initialization.")
        if env is None:
            raise ValueError("Parameter env must be specified.")

        return cls(policy=NoPolicy, env=env)

    def save(
        self,
        path: str | pathlib.Path | io.BufferedIOBase,
        exclude: Iterable[str] | None = None,
        include: Iterable[str] | None = None,
    ) -> None:
        """Save model after training. Not implemented for the rule based agent.

        :param path: Path to the file where the rl agent should be saved.
        :param exclude: Name of parameters that should be excluded in addition to the default ones.
        :param include: Name of parameters that might be excluded but should be included anyway.
        """
        log.info("Rule based agents cannot save data. Saving will be ignored.")

    def _get_pretrain_placeholders(self) -> None:
        """Getting tensorflow pretrain placeholders is not implemented for the rule based agent."""
        raise NotImplementedError("The rule based agent cannot provide tensorflow pretrain placeholders.")

    def get_parameter_list(self) -> None:
        """Getting tensorflow parameters is not implemented for the rule based agent."""
        raise NotImplementedError("The rule pased agent cannot provide a tensorflow parameter list.")

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        eval_env: GymEnv | None = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: str | None = None,
        reset_num_timesteps: bool = True,
    ) -> RuleBased:
        """Return a trained model. Learning is not implemented for the rule based agent.

        :param total_timesteps: The total number of samples (env steps) to train on.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: The name of the run for TensorBoard logging.
        :param eval_env: Environment that will be used to evaluate the agent.
        :param eval_freq: Evaluate the agent every ``eval_freq`` timesteps (this may vary a little).
        :param n_eval_episodes: Number of episode to evaluate the agent.
        :param eval_log_path: Path to a folder where the evaluations will be saved.
        :param reset_num_timesteps: Wether or not to reset the current timestep number (used in logging).
        :return: The trained model.
        """
        raise NotImplementedError("The rule based agent cannot learn a model.")

    def _setup_model(self) -> None:
        """Setup model is not required for the rule based agent."""
        pass

    def action_probability(
        self,
        observation: np.ndarray,
        state: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """Get the model's action probability distribution from an observation. This is not implemented for this
        agent."""
        raise NotImplementedError("The rule based agent cannot calculate action probabilities.")
