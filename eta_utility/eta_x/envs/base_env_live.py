from __future__ import annotations

import abc
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from eta_utility.connectors import LiveConnect
from eta_utility.eta_x.envs import BaseEnv

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import ObservationType, Path, StepResult, TimeStep

log = getLogger(__name__)


class BaseEnvLive(BaseEnv, abc.ABC):
    """Base class for Live Connector environments. The class will prepare the initialization of the LiveConnect class
    and provide facilities to automatically read step results and reset the connection.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param max_errors: Maximum number of connection errors before interrupting the optimization process.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
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
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        max_errors: int = 10,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
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

        assert (
            _files is not None
        ), "Configuration files or a dictionary must be specified before the connector can be initialized."

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
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a five-tuple of observations, rewards, terminated, truncated,
        info.

        This also updates self.state and self.state_log to store current state information.

        .. note::
            This function always returns 0 reward. Therefore, it must be extended if it is to be used with reinforcement
            learning agents. If you need to manipulate actions (discretization, policy shaping, ...) do this before
            calling this function. If you need to manipulate observations and rewards, do this after calling this
            function.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
              bounds. Can be used to end the episode prematurely before a terminal state is reached.
              If true, the user needs to call the `reset` function.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.
        """
        self._actions_valid(action)

        assert self.state_config is not None, "Set state_config before calling step function."

        self.n_steps += 1
        self._create_new_state(self.additional_state)

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

        # Render the environment at each step
        if self.render_mode is not None:
            self.render()

        return self._observations(), 0, self._done(), False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset. When using the environment in conjunction with
        *stable_baselines3*, the vectorized environment will take care of seeding your custom environment automatically.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. note ::
            Don't forget to store and reset the episode_timer.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be
                reset. If you pass an integer, the PRNG will be reset even if it already exists. (default: None)
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)

        :return: Tuple of observation and info. The observation of the initial state will be an element of
                :attr:`observation_space` (typically a numpy array) and is analogous to the observation returned by
                :meth:`step`. Info is a dictionary containing auxiliary information complementing ``observation``. It
                should be analogous to the ``info`` returned by :meth:`step`.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."

        super().reset(seed=seed, options=options)
        self._init_live_connector()

        # Initialize state
        self.state = {} if self.additional_state is None else self.additional_state

        # Update scenario data, read out the start conditions from opc ua server and store the results
        start_obs = []
        for name in self.state_config.ext_outputs:
            start_obs.append(str(self.state_config.map_ext_ids[name]))

        # Read out and store start conditions
        results = self.live_connector.read(*start_obs)
        self.state.update({self.state_config.rev_ext_ids[name]: results[name] for name in start_obs})
        self.state.update(self.get_scenario_state())

        # Execute optional state modification callback function
        if self.state_modification_callback:
            self.state_modification_callback()

        # Log the initial state
        self.state_log.append(self.state)

        # Render the environment when calling the reset function
        if self.render_mode is not None:
            self.render()

        return self._observations(), {}

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the Live_Connector environment is to do nothing.
        """
        self.live_connector.close()
