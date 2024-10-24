from __future__ import annotations

import abc
import inspect
import pathlib
import time
from datetime import datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import Env

from eta_utility import timeseries
from eta_utility.eta_x.envs.state import StateConfig
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import ObservationType, Path, StepResult, TimeStep


log = getLogger(__name__)


class BaseEnv(Env, abc.ABC):
    """Abstract environment definition, providing some basic functionality for concrete environments to use.
    The class implements and adapts functions from gymnasium.Env. It provides additional functionality as required by
    the ETA-X framework and should be used as the starting point for new environments.

    The initialization of this superclass performs many of the necessary tasks, required to specify a concrete
    environment. Read the documentation carefully to understand, how new environments can be developed, building on
    this starting point.

    There are some attributes that must be set and some methods that must be implemented to satisfy the interface. This
    is required to create concrete environments.
    The required attributes are:

        - **version**: Version number of the environment.
        - **description**: Short description string of the environment.
        - **action_space**: The action space of the environment (see also gymnasium.spaces for options).
        - **observation_space**: The observation space of the environment (see also gymnasium.spaces for options).

    The gymnasium interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback that should be called after each episode.
    :param state_modification_callback: callback that should be called after state setup, before logging the state.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Version of the environment"""
        return ""

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Long description of the environment"""
        return ""

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,
        verbose: int = 2,
        callback: Callable | None = None,
        state_modification_callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        sim_steps_per_sample: int | str = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        #: Verbosity level used for logging.
        self.verbose: int = verbose
        log.setLevel(int(verbose * 10))

        # Set some standard path settings
        #: Information about the optimization run and information about the paths.
        #: For example, it defines path_results and path_scenarios.
        self.config_run: ConfigOptRun = config_run
        #: Path for storing results.
        self.path_results: pathlib.Path = self.config_run.path_series_results
        #: Path for the scenario data.
        self.path_scenarios: pathlib.Path | None = self.config_run.path_scenarios
        #: Path of the environment file.
        self.path_env: pathlib.Path
        for f in inspect.stack():
            if "__class__" in f.frame.f_locals and f.frame.f_locals["__class__"] is self.__class__:
                self.path_env = pathlib.Path(f.filename).parent
        #: Callback can be used for logging and plotting.
        self.callback: Callable | None = callback

        #: Callback can be used for modifying the state at each time step.
        self.state_modification_callback: Callable | None = state_modification_callback

        # Store some important settings
        #: ID of the environment (useful for vectorized environments).
        self.env_id: int = int(env_id)
        #: Name of the current optimization run.
        self.run_name: str = self.config_run.name
        #: Number of completed episodes.
        self.n_episodes: int = 0
        #: Current step of the model (number of completed steps) in the current episode.
        self.n_steps: int = 0
        #: Current step of the model (total over all episodes).
        self.n_steps_longtime: int = 0
        #: Render mode for rendering the environment
        self.render_mode: str | None = render_mode

        # Set some standard environment settings
        #: Duration of one episode in seconds.
        self.episode_duration: float = float(
            episode_duration if not isinstance(episode_duration, timedelta) else episode_duration.total_seconds()
        )
        #: Sampling time (interval between optimization time steps) in seconds.
        self.sampling_time: float = float(
            sampling_time if not isinstance(sampling_time, timedelta) else sampling_time.total_seconds()
        )
        #: Number of time steps (of width sampling_time) in each episode.
        self.n_episode_steps: int = int(self.episode_duration // self.sampling_time)
        #: Duration of the scenario for each episode (for total time imported from csv).
        self.scenario_duration: float = self.episode_duration + self.sampling_time

        #: Beginning time of the scenario.
        self.scenario_time_begin: datetime
        if isinstance(scenario_time_begin, datetime):
            self.scenario_time_begin = scenario_time_begin
        else:
            self.scenario_time_begin = datetime.strptime(scenario_time_begin, "%Y-%m-%d %H:%M")
        #: Ending time of the scenario (should be in the format %Y-%m-%d %H:%M).
        self.scenario_time_end: datetime
        if isinstance(scenario_time_end, datetime):
            self.scenario_time_end = scenario_time_end
        else:
            self.scenario_time_end = datetime.strptime(scenario_time_end, "%Y-%m-%d %H:%M")
        # Check if scenario begin and end times make sense
        if self.scenario_time_begin > self.scenario_time_end:
            raise ValueError("Start time of the scenario should be smaller than or equal to end time.")

        #: The time series DataFrame contains all time series scenario data. It can be filled by the
        #: import_scenario method.
        self.timeseries: pd.DataFrame = pd.DataFrame()
        #: Data frame containing the currently valid range of time series data.
        self.ts_current: pd.DataFrame = pd.DataFrame()

        #: Configuration to describe what the environment state looks like.
        self.state_config: StateConfig | None = None

        # Store data logs and log other information
        #: Episode timer (stores the start time of the episode).
        self.episode_timer: float = time.time()
        #: Current state of the environment.
        self.state: dict[str, float]
        #: Additional state information to append to the state during stepping and reset
        self.additional_state: dict[str, float] | None = None
        #: Log of the environment state.
        self.state_log: list[dict[str, float]] = []
        #: Log of the environment state over multiple episodes.
        self.state_log_longtime: list[list[dict[str, float]]] = []
        #: Some specific current environment settings / other data, apart from state.
        self.data: dict[str, Any]
        #: Log of specific environment settings / other data, apart from state for the episode.
        self.data_log: list[dict[str, Any]] = []
        #: Log of specific environment settings / other data, apart from state, over multiple episodes.
        self.data_log_longtime: list[list[dict[str, Any]]]
        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample: int = int(sim_steps_per_sample)

    def import_scenario(self, *scenario_paths: Mapping[str, Any], prefix_renamed: bool = True) -> pd.DataFrame:
        """Load data from csv into self.timeseries_data by using scenario_from_csv

        :param scenario_paths: One or more scenario configuration dictionaries (or a list of dicts), which each contain
            a path for loading data from a scenario file. The dictionary should have the following structure, with <X>
            denoting the variable value:

            .. note ::
                [{*path*: <X>, *prefix*: <X>, *interpolation_method*: <X>, *resample_method*: <X>,
                *scale_factors*: {col_name: <X>}, *rename_cols*: {col_name: <X>}, *infer_datetime_cols*: <X>,
                *time_conversion_str*: <X>}]

            * **path**: Path to the scenario file (relative to scenario_path).
            * **prefix**: Prefix for all columns in the file, useful if multiple imported files
              have the same column names.
            * **interpolation_method**: A pandas interpolation method, required if the frequency of
              values must be increased in comparison to the files' data. (e.g.: 'linear' or 'pad').
            * **scale_factors**: Scaling factors for specific columns. This can be useful for
              example, if a column contains data in kilowatt and should be imported in watts.
              In this case, the scaling factor for the column would be 1000.
            * **rename_cols**: Mapping of column names from the file to new names for the imported
              data.
            * **infer_datetime_cols**: Number of the column which contains datetime data. If this
              value is not present, the time_conversion_str variable will be used to determine
              the datetime format.
            * **time_conversion_str**: Time conversion string, determining the datetime format
              used in the imported file (default: %Y-%m-%d %H:%M).
        :param prefix_renamed: Determine whether the prefix is also applied to renamed columns.
        :return: Data Frame of the imported and formatted scenario data.
        """
        paths = []
        prefix = []
        int_methods = []
        scale_factors = []
        rename_cols = {}
        infer_datetime_from = []
        time_conversion_str = []

        for path in scenario_paths:
            paths.append(self.path_scenarios / path["path"])
            prefix.append(path.get("prefix", None))
            int_methods.append(path.get("interpolation_method", None))
            scale_factors.append(path.get("scale_factors", None))
            (rename_cols.update(path.get("rename_cols", {})),)
            infer_datetime_from.append(path.get("infer_datetime_cols", "string"))
            time_conversion_str.append(path.get("time_conversion_str", "%Y-%m-%d %H:%M"))

        self.ts_current = timeseries.scenario_from_csv(
            paths=paths,
            resample_time=self.sampling_time,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            total_time=self.scenario_duration,
            random=self.np_random,
            interpolation_method=int_methods,
            scaling_factors=scale_factors,
            rename_cols=rename_cols,
            prefix_renamed=prefix_renamed,
            infer_datetime_from=infer_datetime_from,
            time_conversion_str=time_conversion_str,
        )

        return self.ts_current

    def get_scenario_state(self) -> dict[str, Any]:
        """Get scenario data for the current time step of the environment, as specified in state_config. This assumes
        that scenario data in self.ts_current is available and scaled correctly.

        :return: Scenario data for current time step.
        """
        assert self.state_config is not None, "StateConfig must be specified in the environment."

        scenario_state = {}
        for scen in self.state_config.scenarios:
            scenario_state[scen] = self.ts_current[self.state_config.map_scenario_ids[scen]].iloc[self.n_steps]

        return scenario_state

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a five-tuple of observations, rewards, terminated, truncated,
        info.

        .. note ::
            Do not forget to increment n_steps and n_steps_longtime.

        :param action: Actions taken by the agent.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
              bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user
              needs to call the `reset` function.
            * **info**: Provide some additional info about the state of the environment. The contents of this may be
              used for logging purposes in the future but typically do not currently serve a purpose.

        """
        raise NotImplementedError("Cannot step an abstract Environment.")

    def _actions_valid(self, action: np.ndarray) -> None:
        """Check whether the actions are within the specified action space.

        :param action: Actions taken by the agent.
        :raise: RuntimeError, when the actions are not inside of the action space.
        """
        if self.action_space.shape is not None and self.action_space.shape != action.shape:
            raise RuntimeError(
                f"Agent action {action} (shape: {action.shape})"
                f" does not correspond to shape of environment action space (shape: {self.action_space.shape})."
            )

    def _create_new_state(self, additional_state: dict[str, float] | None) -> None:
        """Take some initial values and create a new environment state object, stored in self.state.

        :param additional_state: Values to initialize the state.
        """
        self.state = {} if additional_state is None else additional_state

    def _actions_to_state(self, actions: np.ndarray) -> None:
        """Gather actions and store them in self.state.

        :param actions: Actions taken by the agent.
        """
        assert self.state_config is not None, "Set state_config before calling _actions_to_state function."
        for idx, act in enumerate(self.state_config.actions):
            self.state[act] = actions[idx]

    def _observations(self) -> np.ndarray:
        """Determine the observations list from environment state. This uses state_config to determine all
        observations.

        :return: Observations for the agent as determined by state_config.
        """
        assert self.state_config is not None, "Set state_config before calling _observations function."
        return np.array([self.state[name] for name in self.state_config.observations], dtype=np.float64)

    def _done(self) -> bool:
        """Check if the episode is over or not using the number of steps (n_steps) and the total number of
        steps in an episode (n_episode_steps).

        :return: boolean showing, whether the epsiode is done.
        """
        return self.n_steps >= self.n_episode_steps

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
            Don't forget to store and reset the episode_timer by calling self._reset_state() if you overwrite this
            function.

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
        self._reset_state()
        return super().reset(seed=seed, options=options)

    def _reduce_state_log(self) -> list[dict[str, float]]:
        """Removes unwanted parameters from state_log before storing in state_log_longtime

        :return: The return value is a list of dictionaries,
         where the parameters that should not be stored were removed
        """

        assert self.state_config is not None, "Set state_config before calling reduced_state_log."

        dataframe = pd.DataFrame(self.state_log)
        return dataframe.drop(columns=list(set(dataframe.keys()) - self.state_config.add_to_state_log)).to_dict(
            "records"
        )

    def _reset_state(self) -> None:
        """Store episode statistics and reset episode counters."""
        if self.n_steps > 0:
            if self.callback is not None:
                self.callback(self)

            # Store some logging data
            self.n_episodes += 1

            # store reduced_state_log in state_log_longtime
            self.state_log_longtime.append(self._reduce_state_log())
            self.n_steps_longtime += self.n_steps

            # Reset episode variables
            self.n_steps = 0
            self.state_log = []

    @abc.abstractmethod
    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.
        """
        raise NotImplementedError("Cannot close an abstract Environment.")

    @abc.abstractmethod
    def render(self) -> None:
        """Render the environment

        The set of supported modes varies per environment. Some environments do not support rendering at
        all. By convention in Farama *gymnasium*, if mode is:

            * human: render to the current display or terminal and return nothing. Usually for human consumption.
            * rgb_array: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
              suitable for turning into a video.
            * ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        """
        raise NotImplementedError("Cannot render an abstract Environment.")

    @classmethod
    def get_info(cls) -> tuple[str, str]:
        """Get info about environment.

        :return: Tuple of version and description.
        """
        return cls.version, cls.description  # type: ignore

    def export_state_log(
        self,
        path: Path,
        names: Sequence[str] | None = None,
        *,
        sep: str = ";",
        decimal: str = ".",
    ) -> None:
        """Extension of csv_export to include timeseries on the data

        :param names: Field names used when data is a Matrix without column names.
        :param sep: Separator to use between the fields.
        :param decimal: Sign to use for decimal points.

        """
        start_time = datetime.fromtimestamp(self.episode_timer)
        step = self.sampling_time / self.sim_steps_per_sample
        timerange = [start_time + timedelta(seconds=(k * step)) for k in range(len(self.state_log))]
        csv_export(path=path, data=self.state_log, index=timerange, names=names, sep=sep, decimal=decimal)
