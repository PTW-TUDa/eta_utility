from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

from eta_utility import get_logger
from eta_utility.connectors import LiveConnect
from eta_utility.eta_x.envs import BaseEnv

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable, Sequence

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import Path, StepResult, TimeStep

log = get_logger("eta_x.envs")


class BaseEnvLive(BaseEnv, abc.ABC):
    """Base class for Live Connector environments. The class will prepare the initialization of the LiveConnect class
    and provide facilities to automatically read step results and reset the connection.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param seed: Random seed to use for generating random numbers in this environment.
        (default: None / create random seed).
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param max_errors: Maximum number of connection errors before interrupting the optimization process.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def config_name(self) -> str:
        """Name of the live_connect configuration"""
        return ""

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        max_errors: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id,
            config_run,
            seed,
            verbose,
            callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            **kwargs,
        )
        #: Instance of the Live Connector.
        self.live_connector: LiveConnect
        #: Path or Dict to initialize the live connector.
        self.live_connect_config: Path | Sequence[Path] | dict[str, Any] | None = (
            self.path_env / f"{self.config_name}.json"
        )
        #: Maximum error count when connections in live connector are aborted.
        self.max_error_count: int = max_errors

    def _init_live_connector(self, files: Path | Sequence[Path] | dict[str, Any] | None = None) -> None:
        """Initialize the live connector object. Make sure to call _names_from_state before this or to otherwise
        initialize the names array.

        :param files: Path or Dict to initialize the connection directly from JSON configuration files or a config
            dictionary.
        """
        _files = self.live_connect_config if files is None else files
        self.live_connect_config = _files

        assert _files is not None, (
            "Configuration files or a dictionary must be specified before " "the connector can be initialized."
        )

        if isinstance(_files, dict):
            self.live_connector = LiveConnect.from_dict(
                step_size=self.sampling_time,
                max_error_count=self.max_error_count,
                **_files,
            )
        else:
            self.live_connector = LiveConnect.from_json(
                files=_files, step_size=self.sampling_time, max_error_count=self.max_error_count
            )

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
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
        if self.action_space.shape != action.shape:
            raise RuntimeError(
                f"Agent action {action} (shape: {action.shape})"
                f" does not correspond to shape of environment action space (shape: {self.action_space.shape})."
            )

        assert self.state_config is not None, "Set state_config before calling step function."

        self.n_steps += 1

        # Store actions
        self.state = {} if self.additional_state is None else self.additional_state

        # Preparation for the setting of the actions, store actions
        node_in = {}
        # Set actions in the opc ua server and read out the observations
        for idx, name in enumerate(self.state_config.actions):
            self.state[name] = action[idx]
            node_in.update({str(self.state_config.map_ext_ids[name]): action[idx]})

        # Update scenario data, do one time step in the live connector and store the results.
        self.state.update(self.get_scenario_state())

        results = self.live_connector.step(node_in)

        self.state = {name: results[str(self.state_config.map_ext_ids[name])] for name in self.state_config.ext_outputs}
        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        # Check if the episode is over or not
        done = self.n_steps >= self.n_episode_steps

        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations, 0, done, {}

    def reset(self) -> np.ndarray:
        """Return initial observations. This also calls the callback, increments the episode
        counter, resets the episode steps and appends the state_log to the longtime storage.

        If you want to extend this function, write your own code and call super().reset() afterwards to return
        fresh observations. This allows you to adjust timeseries for example. If you need to manipulate the state
        before initializing or if you want to adjust the initialization itself, overwrite the function entirely.

        :return: Initial observation.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."
        self._reset_state()
        self._init_live_connector()

        self.state = {} if self.additional_state is None else self.additional_state
        # Update scenario data, read out the start conditions from opc ua server and store the results
        start_obs = []
        for name in self.state_config.ext_outputs:
            start_obs.append(str(self.state_config.map_ext_ids[name]))

        # Read out and store start conditions
        results = self.live_connector.read(*start_obs)
        self.state.update({self.state_config.rev_ext_ids[name]: results[name] for name in start_obs})
        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the Live_Connector environment is to do nothing.
        """
        pass
