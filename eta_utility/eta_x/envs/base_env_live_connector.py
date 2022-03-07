from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from eta_utility import get_logger
from eta_utility.connectors import LiveConnect
from eta_utility.eta_x.envs import BaseEnv

if TYPE_CHECKING:
    from typing import (
        Any,
        Callable,
        MutableSet,
        Sequence,
        SupportsFloat,
    )

from eta_utility.type_hints import Path

log = get_logger("eta_x.envs")


class BaseEnvLiveConnector(BaseEnv, ABC):
    """Base class for Live Connector"""

    @property
    @abstractmethod
    def config_name(self) -> str:
        """Name of the file including file extension"""
        return ""

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: dict[str, Any],
        path_settings: dict[str, Path],
        env_settings: dict[str, Any],
        verbose: int,
        callback: Callable | None = None,
    ) -> None:

        self.req_general_settings: set[Sequence | MutableSet] = set(self.req_general_settings)
        super().__init__(
            env_id,
            run_name,
            general_settings,
            path_settings,
            env_settings,
            verbose,
            callback,
        )

        #: Path for the config file with the settings for the opc ua server.
        self.live_connect_config: Path | None = (
            (self.path_root / (f"config/{self.config_name}")) if isinstance(self.config_name, str) else None
        )
        #: Instance of the Live Connector.
        self.live_connector: LiveConnect
        #: Path or Dict for initialize the live connector.
        self.files: Path | Sequence[Path] | dict
        #: Maximum error count when connections in live connector are aborted
        self.max_error_count: int

    def _init_live_connector(self, files: Path | Sequence[Path] | dict, max_error_count: int = 10) -> None:
        """Initialize the live connector object. Make sure to call _names_from_state before this or to otherwise
        initialize the names array.

        :param files: Path or Dict for initialize the connection directly from json configuration files or a config
            dictionary.
        """
        self.files = files
        self.max_error_count = max_error_count

        if isinstance(self.files, dict):
            self.live_connector = LiveConnect.from_dict(
                step_size=self.sampling_time,
                max_error_count=max_error_count,
                **self.files,
            )
        else:
            self.live_connector = LiveConnect.from_json(
                files=self.files, step_size=self.sampling_time, max_error_count=max_error_count
            )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.floating | SupportsFloat, bool, str | dict]:
        """Perfom one time step and return its results. This is called for every event or for every time step during
        the optimization run. It should utilize the actions as supplied by the agent to determine
        the new state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        This also updates self.state and self.state_log to store current state information.

        .. warning::
            This function always returns 0 reward. Therefore, it must be extended if it is to be used with reinforcement
            learning agents. If you need to manipulate actions (discretization, policy shaping, ...)
            do this before calling this function.
            If you need to manipulate observations and rewards, do this after calling this function.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **done**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.
        """
        if self.names is not None and (
            self.live_connector._observe_vals is None
            or len(self.names["ext_outputs"]) != len(self.live_connector._observe_vals)
        ):
            shape = len(self.live_connector._observe_vals) if self.live_connector._observe_vals is not None else "None"
            raise RuntimeError(
                f"Observe_vals in live connector (shape: {shape})"
                f" does not correspond to shape of environment ext_outputs in self.names."
            )
        self.n_steps += 1

        # Store actions
        self.state = {}
        # Preparation for the setting of the actions, store actions
        node_in = {}

        # Set actions in the opc ua server and read out the observations
        for idx, name in enumerate(self.names["actions"]):
            self.state[name] = action[idx]
            node_in.update({name: action[idx]})

        # Update scenario data, do one time step in the live connector and store the results.
        self.state.update(self.get_scenario_state())

        results = self.live_connector.step(node_in)

        self.state = {name: results[self.map_ext_ids[name]] for name in self.names["ext_outputs"]}
        self.state_log.append(self.state)

        # Check if the episode is over or not
        done = True if self.n_steps >= self.n_episode_steps else False

        observations = np.empty(len(self.names["observations"]))
        for idx, name in enumerate(self.names["observations"]):
            observations[idx] = self.state[name]

        return observations, 0, done, {}

    def reset(self) -> np.ndarray:
        """Return initial observations. This also calls the callback, increments the episode
        counter, resets the episode steps and appends the state_log to the longtime storage.

        If you want to extend this function, write your own code and call super().reset() afterwards to return
        fresh observations. This allows you to ajust timeseries for example. If you need to manipulate the state
        before initializing or if you want to adjust the initialization itself, overwrite the function entirely.

        :return: Initial observation
        """
        if self.names is not None and (
            self.live_connector._observe_vals is None
            or len(self.names["ext_outputs"]) != len(self.live_connector._observe_vals)
        ):
            shape = len(self.live_connector._observe_vals) if self.live_connector._observe_vals is not None else "None"
            raise RuntimeError(
                f"Observe_vals in live connector (shape: {shape})"
                f" does not correspond to shape of environment ext_outputs in self.names."
            )

        # Save episode's stats
        if self.n_steps > 0:
            if self.callback is not None:
                self.callback(self)

            # Store some logging data
            self.n_episodes += 1
            self.state_log_longtime.append(self.state_log)
            self.n_steps_longtime += self.n_steps

            # Reset episode variables
            self.n_steps = 0
            self.state_log: list[dict[str, float]] = []

        if isinstance(self.files, dict):
            self.live_connector = LiveConnect.from_dict(
                **self.files, step_size=self.sampling_time, max_error_count=self.max_error_count
            )
        else:
            self.live_connector = LiveConnect.from_json(
                files=self.files, step_size=self.sampling_time, max_error_count=self.max_error_count
            )

        # Update scenario data, read out the start conditions from opc ua server and store the results
        start_obs = []
        for name in self.names["ext_outputs"]:
            start_obs.append(self.map_ext_ids[name])

        # Read out and store start conditions
        results = self.live_connector.read(*start_obs)
        self.state = {self.rev_ext_ids[name]: results[name] for name in start_obs}
        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        observations = np.empty(len(self.names["observations"]))
        for idx, name in enumerate(self.names["observations"]):
            observations[idx] = self.state[name]

        return observations

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behaviour for the Live_Connector environment is to do nothing.

        :return:
        """
        pass
