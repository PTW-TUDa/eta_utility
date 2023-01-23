from __future__ import annotations

import abc
import inspect
import pathlib
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gym import Env, utils

from eta_utility import get_logger, timeseries
from eta_utility.eta_x.envs.state import StateConfig
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, Sequence

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import Path, StepResult, TimeStep

log = get_logger("eta_x.envs")


class BaseEnv(Env, abc.ABC):
    """Abstract environment definition, providing some basic functionality for concrete environments to use.
    The class implements and adapts functions from gym.Env. It provides additional functionality as required by
    the ETA-X framework and should be used as the starting point for new environments.

    The initialization of this superclass performs many of the necessary tasks, required to specify a concrete
    environment. Read the documentation carefully to understand, how new environments can be developed, building on
    this starting point.

    There are some attributes that must be set and some methods that must be implemented to satisfy the interface. This
    is required to create concrete environments.
    The required attributes are:

        - **version**: Version number of the environment.
        - **description**: Short description string of the environment.
        - **action_space**: The action space of the environment (see also gym.spaces for options).
        - **observation_space**: The observation space of the environment (see also gym.spaces for options).

    The gym interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed).
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
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
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
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

        #: Numpy random generator.
        self.np_random: np.random.Generator
        #: Value used for seeding the random generator.
        self._seed: int
        self.np_random, self._seed = self.seed(seed)

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

    def _init_state_space(self) -> StateConfig:
        """Initialize the StateConfig object from a dataframe stored in self.state_config. Also provide a deprecated
        structure of variables, which includes the self.names mapping and multiple other maps.

        .. deprecated:: 2.0.0
            Initialize the StateConfig object directly instead and use the mappings and functions it provides.

        The names array contains the following (ordered) lists of variables in a dictionary:

            * **actions**: Variables that are agent actions.
            * **observations**: Variables that are agent observations.
            * **ext_inputs**: Variables that should be provided to an external source (such as an FMU).
            * **ext_output**: variables that can be received from an external source (such as an FMU).
            * **abort_conditions_min**: Variables that have minimum values for an abort condition.
            * **abort_conditions_max**: Variables that have maximum values for an abort condition.

        *self.ext_scale* is a dictionary of scaling values for external input values (for example from simulations).

        *self.map_ext_ids* is a mapping of internal environment names to external IDs.

        *self.rev_ext_ids* is a mapping of external IDs to internal environment names.

        *self.map_scenario_ids* is a mapping of internal environment names to scenario IDs.
        """
        self.state_config = StateConfig.from_dict(self.state_config)

        self.names = {
            "actions": self.state_config.actions,
            "observations": self.state_config.observations,
            "ext_inputs": self.state_config.ext_inputs,
            "ext_outputs": self.state_config.ext_outputs,
            "scenario": self.state_config.scenarios,
            "abort_conditions_min": self.state_config.abort_conditions_min,
            "abort_conditions_max": self.state_config.abort_conditions_max,
        }
        self.ext_scale = self.state_config.ext_scale
        self.map_ext_ids = self.state_config.map_ext_ids
        self.rev_ext_ids = self.state_config.rev_ext_ids
        self.map_scenario_ids = self.state_config.map_scenario_ids

        self.continuous_action_space_from_state = self.state_config.continuous_action_space
        self.continuous_obs_space_from_state = self.state_config.continuous_obs_space
        self.continuous_spaces_from_state = self.state_config.continuous_spaces

        return self.state_config

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
            rename_cols.update(path.get("rename_cols", {})),
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
        state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        .. note ::
            Do not forget to increment n_steps and n_steps_longtime.

        :param action: Actions taken by the agent.
        :return: The return value represents the state of the environment after the step was performed.

            * observations: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * reward: The value of the reward function. This is just one floating point value.
            * done: Boolean value specifying whether an episode has been completed. If this is set to true, the reset
              function will automatically be called by the agent or by eta_i.
            * info: Provide some additional info about the state of the environment. The contents of this may be used
              for logging purposes in the future but typically do not currently serve a purpose.

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
        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations

    def _done(self) -> bool:
        """Check if the episode is over or not using the number of steps (n_steps) and the total number of
        steps in an episode (n_episode_steps).

        :return: boolean showing, whether the epsiode is done.
        """
        return self.n_steps >= self.n_episode_steps

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        .. note ::
            Don't forget to store and reset the episode_timer.

        :return: The return value represents the observations (state) of the environment before the first
                 step is performed.
        """
        raise NotImplementedError("Cannot reset an abstract Environment.")

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
    def render(self, mode: str = "human") -> None:
        """Render the environment

        The set of supported modes varies per environment. Some environments do not support rendering at
        all. By convention in OpenAI *gym*, if mode is:

            * human: render to the current display or terminal and return nothing. Usually for human consumption.
            * rgb_array: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
              suitable for turning into a video.
            * ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        :param mode: Rendering mode.
        """
        raise NotImplementedError("Cannot render an abstract Environment.")

    def seed(self, seed: int | None = None) -> tuple[np.random.Generator, int]:
        """Set random seed for the random generator of the environment

        :param seed: Seeding value.
        :return: Tuple of the numpy random generator and the set seed value.
        """
        if seed is None:
            iseed = None
            log.info("The environment seed is set to None, a random seed will be set.")
        else:
            iseed = int(seed) + self.env_id

        np_random, _seed = utils.seeding.np_random(iseed)  # noqa
        log.info(f"The environment seed is set to: {_seed}")

        self._seed = _seed
        self.np_random = np_random

        return self.np_random, self._seed

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
