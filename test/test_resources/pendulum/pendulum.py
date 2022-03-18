from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gym.envs.classic_control.pendulum import angle_normalize

from eta_utility import get_logger
from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnv
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from typing import Any, Callable

    from eta_utility.type_hints import Path, StepResult

log = get_logger("test_etax", 2)


class PendulumEnv(BaseEnv):
    """Pendulum environment from BaseEnv (abstract class), adapted environment from gym.Env.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param run_name: Identification name for the optimization run.
    :param general_settings: Dictionary of general settings.
    :param path_settings: Dictionary of path settings.
    :param env_settings: Dictionary of environment specific settings.
    :param verbose: Verbosity setting for logging.
    :param callback: Callback method will be called after each episode with all data within the
        environment class.
    """

    version = "v1.0"
    description = "OpenAI"
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: dict[str, Any],
        path_settings: dict[str, Path],
        env_settings: dict[str, Any],
        verbose: int = 1,
        callback: Callable = None,
    ):
        super()._init_legacy(env_id, run_name, general_settings, path_settings, env_settings, verbose, callback)

        # load environment dynamics specific settings from JSON
        self.max_speed = env_settings["max_speed"]
        self.max_torque = env_settings["max_torque"]
        self.g = env_settings["g"]
        self.m = env_settings["m"]
        self.length = env_settings["l"]

        # other
        self.viewer = None

        # initialize counters
        self.n_episodes = 0
        self.n_steps = 0

        # set action space and observation space
        self.state_config = [
            {"name": "torque", "is_agent_action": True, "low_value": -self.max_torque, "high_value": self.max_torque},
            {"name": "cos_th", "is_agent_observation": True, "low_value": -1.0, "high_value": 1.0},
            {"name": "sin_th", "is_agent_observation": True, "low_value": -1.0, "high_value": 1.0},
            {"name": "th_dot", "is_agent_observation": True, "low_value": -1.0, "high_value": 1.0},
        ]
        self._init_state_space()
        self.action_space = self.continuous_action_space_from_state()
        self.observation_space = self.continuous_obs_space_from_state()

    def step(self, u: np.ndarray) -> StepResult:
        """See base_env documentation"""
        # Here, u is a scalar value, but it can also be a numpy array for multiple actions.
        # This depends on the action and observation space chosen

        # update counters
        self.n_steps += 1

        u = np.clip(u, -self.max_torque, self.max_torque)[0]  # clip input from agent by max values
        self.last_u = u  # for rendering

        # dynamic model of the pendulum
        th, thdot = self.state  # get current state variables (th := theta)
        newthdot = (
            thdot
            + (-3 * self.g / (2 * self.length) * np.sin(th + np.pi) + 3.0 / (self.m * self.length**2) * u)
            * self.sampling_time
        )
        newth = th + newthdot * self.sampling_time
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        # update state and derive observations from it (observations are given to the agent)
        self.state = np.array([newth, newthdot])  # update state (current state of the system)
        self.state_log.append(self.state)  # update state_log with current state
        observations = np.array(
            [np.cos(newth), np.sin(newth), newthdot]
        )  # observations (derived from state and given to the agent)

        # reward function
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        # check if episode is over or not
        done = self.n_steps >= self.n_episode_steps

        return observations, -costs, done, {}

    def reset(self) -> np.ndarray:
        # update counters
        if self.n_steps > 0:
            log.info("Finished Episode " + str(self.n_episodes) + ".")
            self.callback(self)
        self.n_episodes += 1

        # reset dynamic model of the pendulum
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None

        # reset counter
        self.n_steps = 0

        # load randomly chosen state variables and derive observations from it
        th, thdot = self.state
        reset_observations = np.array([np.cos(th), np.sin(th), thdot])

        return reset_observations

    def render(self, mode: str = "human") -> None:
        # Method replaced for integration test purpose

        data = {"n_iter": self.n_steps, "n_episode": self.n_episodes}
        csv_export(
            episode_results_path(self.config_run.path_series_results, self.run_name, 1, self.env_id),
            data=data,
        )

    def close(self) -> None:
        pass
