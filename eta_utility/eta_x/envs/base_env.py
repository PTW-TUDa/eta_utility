from __future__ import annotations

import abc
import inspect
import pathlib
import time
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandas as pd
from gym import Env, spaces, utils

from eta_utility import get_logger, timeseries

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, MutableMapping, MutableSet, SupportsFloat

    from eta_utility.type_hints import Path

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

        - **version**: Version number of the environment
        - **description**: Short description string of the environment
        - **action_space**: The action space of the environment (see also gym.spaces for options)
        - **observation_space**: The observation space of the environment (see also gym.spaces for options)

    Some additional attributes can be used to simplify the creation of new environments but they are not required:

        - **req_general_settings**: List/Set of general settings that need to be present
        - **req_path_settings**: List/Set of path settings that need to be present
        - **req_env_settings**: List/Set of environment settings that need to be present
        - **req_env_config**: Mapping of environment settings that must have specific values

    The gym interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**

    :param env_id: Identification for the environment, usefull when creating multiple environments
    :param run_name: Identification name for the optimization run
    :param general_settings: Dictionary of general settings
    :param path_settings: Dictionary of path settings
    :param env_settings: Dictionary of environment specific settings
    :param verbose: Verbosity setting for logging
    :param callback: callback method will be called after each episode with all data within the
        environment class
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

    #: Required settings in the general 'settings' section
    req_general_settings: Sequence | MutableSet = []
    #: Required settings in the 'path' section
    req_path_settings: Sequence | MutableSet = []
    #: Required settings in the 'environment_specific' section
    req_env_settings: Sequence | MutableSet = []
    #: Some environments may required specific parameters in the 'environment_specific' section to have special
    #   values. These parameter, value pairs can be specified in the req_env_config dictionary.
    req_env_config: MutableMapping = {}

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: dict[str, Any],
        path_settings: dict[str, Path],
        env_settings: dict[str, Any],
        verbose: int,
        callback: Callable | None = None,
    ):
        # Set some additional required settings
        self.req_path_settings: set[Sequence | MutableSet] = set(self.req_path_settings)
        self.req_path_settings.update(("path_root", "path_results", "relpath_scenarios"))  # noqa
        self.req_env_settings: set[Sequence | MutableSet] = set(self.req_env_settings)
        self.req_env_settings.update(("scenario_time_begin", "scenario_time_end"))  # noqa
        self.req_general_settings: set[Sequence | MutableSet] = set(self.req_general_settings)
        self.req_general_settings.update(("episode_duration", "sampling_time"))  # noqa

        super().__init__()

        # save verbosity/debug level
        #: Verbosity level used for logging
        self.verbose: int = verbose
        log.setLevel(int(verbose * 10))

        # Check whether all required settings are present
        errors = False
        for name in self.req_general_settings:
            if name not in general_settings:
                log.error(f"Required parameter '{name}' not found in 'settings'.")
                errors = True

        for name in self.req_path_settings:
            if name not in path_settings:
                log.error(f"Required parameter '{name}' not found in 'paths' settings.")
                errors = True

        for name in self.req_env_settings:
            if name not in env_settings:
                log.error(f"Required parameter '{name}' not found in 'environment_specific' settings.")
                errors = True

        for item, value in self.req_env_config.items():
            if item in env_settings and env_settings[item] != value:
                log.error(
                    "Config parameters are incompatible with the parameters required by this environment. "
                    "'{}' in section 'environment_specific' must be equal to '{}'".format(item, value)
                )
                errors = True

        if errors:
            raise ValueError("Not all required config parameters for the environment were found. Exiting.")

        # Copy settings to ensure they cannot be changed from outside the environment
        #: Configuration from the general 'settings' section.
        self.settings: dict[str, Any] = general_settings.copy()

        # Set some standard environment settings
        #: Configuration from the 'environment_specific' section. This contains scenario_time_begin and
        #:   end as datetime values
        self.env_settings: dict[str, Any] = env_settings.copy()
        self.env_settings.update(
            {"scenario_time_begin": datetime.strptime(env_settings["scenario_time_begin"], "%Y-%m-%d %H:%M")}
        )
        self.env_settings.update(
            {"scenario_time_end": datetime.strptime(env_settings["scenario_time_end"], "%Y-%m-%d %H:%M")}
        )
        # Setup for simulation.
        if self.env_settings["scenario_time_begin"] > self.env_settings["scenario_time_end"]:
            raise ValueError("Start time should be smaller than or equal to end time.")

        # Set some standard path settings
        #: Configuration from the 'paths' section in the settings.
        self.path_settings: dict[str, Any] = path_settings.copy()
        #: Root path of the application as generated by the framework
        self.path_root: pathlib.Path = pathlib.Path(self.path_settings["path_root"])
        #: Path for results storage
        self.path_results: pathlib.Path = pathlib.Path(self.path_settings["path_results"])
        #: Path for scenario storage (this does not include the individual scenario files as specified in the
        #:   environment config)
        self.path_scenarios: pathlib.Path = self.path_root / self.path_settings["relpath_scenarios"]
        #: Path of the environment file
        self.path_env: pathlib.Path = pathlib.Path(inspect.stack()[2].filename).parent
        # store callback function in object
        #: Callback function for the environment. This should be called after every step.
        self.callback: Callable | None = callback

        # Store some important settings
        #: Total number of environments
        self.n_environments = int(self.settings["n_environments"])
        #: Id of the environment (useful for vectorized environments)
        self.env_id: int = env_id
        #: Name of the current optimization run
        self.run_name: str = run_name
        #: Number of completed episodes
        self.n_episodes: int = 0
        #: Current step of the model (number of completed steps) in the current episode
        self.n_steps: int = 0
        #: Current step of the model (total over all episodes)
        self.n_steps_longtime: int = 0
        #: Beginning time of the scenario
        self.scenario_time_begin: datetime = self.env_settings["scenario_time_begin"]
        #: Ending time of the scenario
        self.scenario_time_end: datetime = self.env_settings["scenario_time_end"]
        #: Sampling time (interval between optimization time steps) in seconds
        self.sampling_time: float = float(self.settings["sampling_time"])
        #: Number of time steps (of width sampling_time) in each episode
        self.n_episode_steps: int = int(self.settings["episode_duration"] // self.sampling_time)
        #: Duration of one episode in seconds
        self.episode_duration: int = int(self.n_episode_steps * self.sampling_time)
        #: Duration of the scenario for each episode (for total time imported from csv)
        self.scenario_duration: float = self.episode_duration + self.sampling_time

        #: Numpy random generator
        self.np_random: np.random.Generator
        #: Value used for seeding the random generator
        self._seed: int
        self.np_random, self._seed = (
            self.seed(self.env_settings["seed"]) if "seed" in self.env_settings else self.seed(None)
        )

        #: The time series DataFrame contains all time series scenario data. It can be filled by the
        #:   import_scenario method.
        self.timeseries: pd.DataFrame = pd.DataFrame()
        #: Data frame containing the currently valid range of time series data.
        self.ts_current: pd.DataFrame = pd.DataFrame()

        # Columns (and their order) and default values for the state_config DataFrame.
        self.__state_config_cols = OrderedDict(
            [
                ("name", ""),
                ("is_agent_action", False),
                ("is_agent_observation", False),
                ("ext_id", None),
                ("is_ext_input", False),
                ("is_ext_output", False),
                ("ext_scale_add", 0),
                ("ext_scale_mult", 1),
                ("from_scenario", False),
                ("scenario_id", None),
                ("low_value", np.nan),
                ("high_value", np.nan),
                ("abort_condition_min", None),
                ("abort_condition_max", None),
                ("index", 0),
            ]
        )

        #: The configuration for the action and observation spaces. The values are used to control which variables are
        #: part of the action space and observation space. Additionally the parameters can specify abort conditions
        #: and the handling of values from interaction environments or from simulation. Therefore the state_config
        #: is very important for the functionality of ETA X.
        #:
        #: Some functions that operate on state_config:
        #: The function :func:`append_state` can be used to append values to the DataFrame. The function
        #: :func:`_convert_state_config` can be used to convert an incomplete DataFrame or a list of dictionaries into
        #: the full standardized format.
        #:
        #: Possible column names, their types and default values are:
        #:
        #:   * **name**: str, Name of the state variable (This must always be specified (no default)), names column
        #:     becomes index in DataFrame
        #:   * **is_agent_action**: bool, Should the agent specify actions for this variable? (default: False)
        #:   * **is_agent_observation**: bool, Should the agent be allowed to observe the value
        #:     of this variable? (default: False)
        #:   * **ext_id**: str, Name or identifier (order) of the variable in the external interaction model
        #:     (e.g.: environment or FMU) (default: None)
        #:   * **is_ext_input**: bool, Should this variable be passed to the external model as an input?
        #:     (default: False)
        #:   * **is_ext_output**: bool, Should this variable be parsed from the external model output? (default: False)
        #:   * **ext_scale_add**: int or float, Value to add to the output from an external model (default: 0)
        #:   * **ext_scale_mult**: int or float, Value to multiply to the output from an external model (default: 1)
        #:   * **from_scenario**: bool, Should this variable be read from imported timeseries date? (default: False)
        #:   * **scenario_id**: str, Name of the scenario variable, this value should be read from (default: None)
        #:   * **low_value**: int or float, lowest possible value of the state variable (default: None)
        #:   * **high_value**: int or float, highest possible value of the state variable (default: None)
        #:   * **abort_condition_min**: int or float, If value of variable dips below this, the episode
        #:     will be aborted (default: None)
        #:   * **abort_condition_max**: int or float, If value of variable rises above this, the episode
        #:     will be aborted (default: None)
        #:   * **index**: int, Specify, which Index this value should be read from, in case a list of values is
        #:     returned (default: 0)
        #:
        #: *State_config* can also be specified as a list of dictionaries if many default values are set:
        #:
        #: .. note ::
        #:      state_config = pd.DataFrame(
        #:      [{name:___, ext_id:___, ...},
        #:      {name:___, ext_id:___, ...}])
        self.state_config: pd.DataFrame = pd.DataFrame(columns=self.__state_config_cols.keys())
        self.state_config.set_index("name", drop=True, inplace=True)
        #: Array of shorthands to some frequently used variable names from state_config.
        #: See also: :func:`_names_from_state`
        #:
        #: General structure:
        #: 'self.names = {'actions': array([], dtype=object),
        #: 'observations': array([], dtype=object),
        #: 'ext_inputs': array([], dtype=object),
        #: 'ext_outputs': array([], dtype=object),
        #: 'scenario': array([], dtype=object),
        #: 'abort_conditions_min': array(['temp_tank'], dtype=object),
        #: 'abort_conditions_max': array(['temp_tank'], dtype=object)}
        self.names: dict[str, np.ndarray]
        #: Dictionary of scaling values for external input values (for example from simulations).
        #:  The structure of this dictionary is {"name": {"add": value, "multiply": value}}.
        self.ext_scale: dict[str, dict[str, int | float]]
        #: Mapping of internal environment names to external ids.
        self.map_ext_ids: dict[str, str]
        #: Mapping of external ids to internal environment names.
        self.rev_ext_ids: dict[str, str]
        #: Mapping of internal environment names to scenario ids.
        self.map_scenario_ids: dict[str, str]

        # Store data logs and log other information
        #: Episode timer
        self.episode_timer: float = time.time()
        #: Current state of the environment
        self.state: dict[str, float]
        #: Log of the environment state
        self.state_log: list[dict[str, float]] = []
        #: Log of the environment state over multiple episodes
        self.state_log_longtime: list[list[dict[str, float]]] = []
        #: Some specific current environment settings / other data, apart from state
        self.data: dict[str, Any]
        #: Log of specific environment settings / other data, apart from state for the episode
        self.data_log: list[dict[str, Any]] = []
        #: Log of specific environment settings / other data, apart from state, over multiple episodes.
        self.data_log_longtime: list[list[dict[str, Any]]]

    def append_state(self, *, name: str, **kwargs: Any) -> None:
        """Append a state variable to the state configuration of the environment

        :param name: Name of the state variable
        :param kwargs: Column names and values to be inserted into the respective column. For possible columns, types
                       and default values see state_config. See also: :func:`state_config`
        """
        append = {}
        for key, item in self.__state_config_cols.items():
            # Since name is supplied separately, don't append it here
            if key == "name":
                continue

            val = kwargs[key] if key in kwargs else item
            append[key] = val

        append = pd.Series(append, name=name)
        self.state_config = self.state_config.append(append, sort=True)

    def _init_state_space(self) -> None:
        """Convert state config and store state information. This is a shorthand for the function calls:

        * **_convert_state_config()**

            (See also: :func:`_convert_state_config`)
        * **_names_from_state()**

            (See also: :func:`_names_from_state`)
        * **_store_state_info()**

            (See also: :func:`_store_state_info`)

        """
        self._convert_state_config()
        self._names_from_state()
        self._store_state_info()

    def _names_from_state(self) -> None:
        """Intialize the names array from state_config, which stores shorthands to some frequently used variable names.
        Also initialize some useful shorthand mappings that can be used to speed up lookups.

        The names array contains the following (ordered) lists of variables in a dictionary:

            * **actions**: Variables that are agent actions
            * **observations**: Variables that are agent observations
            * **ext_inputs**: Variables that should be provided to an external source (such as an FMU)
            * **ext_output**: variables that can be received from an external source (such as an FMU)
            * **abort_conditions_min**: Variables that have minimum values for an abort condition
            * **abort_conditions_max**: Variables that have maximum values for an abort condition

        *self.ext_scale* is a dictionary of scaling values for external input values (for example from simulations).

        *self.map_ext_ids* is a mapping of internal environment names to external ids.

        *self.rev_ext_ids* is a mapping of external ids to internal environment names.

        *self.map_scenario_ids* is a mapping of internal environment names to scenario ids.
        """
        self.names = {
            "actions": self.state_config.loc[self.state_config.is_agent_action == True].index.values,  # noqa: E712
            "observations": self.state_config.loc[
                self.state_config.is_agent_observation == True  # noqa: E712
            ].index.values,
            "ext_inputs": self.state_config.loc[self.state_config.is_ext_input == True].index.values,  # noqa: E712
            "ext_outputs": self.state_config.loc[self.state_config.is_ext_output == True].index.values,  # noqa: E712
            "scenario": self.state_config.loc[self.state_config.from_scenario == True].index.values,  # noqa: E712
            "abort_conditions_min": self.state_config.loc[self.state_config.abort_condition_min.notnull()].index.values,
            "abort_conditions_max": self.state_config.loc[self.state_config.abort_condition_max.notnull()].index.values,
        }

        self.ext_scale = {}
        for name, values in self.state_config.iterrows():
            self.ext_scale[name] = {"add": values.ext_scale_add, "multiply": values.ext_scale_mult}

        self.map_ext_ids = {}
        for name in set(self.names["ext_inputs"]) | set(self.names["ext_outputs"]):
            self.map_ext_ids[name] = self.state_config.loc[name].ext_id

        self.rev_ext_ids = {}
        for name in set(self.names["ext_inputs"]) | set(self.names["ext_outputs"]):
            self.rev_ext_ids[self.state_config.loc[name].ext_id] = name

        self.map_scenario_ids = {}
        for name in self.names["scenario"]:
            self.map_scenario_ids[name] = self.state_config.loc[name].scenario_id

    def _convert_state_config(self) -> pd.DataFrame:
        """This will convert an incomplete state_config DataFrame or a list of dictionaries to the standardized
        DataFrame format. This will remove any additional columns. If additional columns are required, ensure
        consistency with the required format otherwise.

        :return: Converted, standardized dataframe
        """
        # If state config is a DataFrame already, check whether the columns correspond. If they don't create a new
        # DataFrame with the correct columns and default values for missing columns
        if isinstance(self.state_config, pd.DataFrame):
            new_state = pd.DataFrame(columns=self.__state_config_cols.keys())
            for col, default in self.__state_config_cols.items():
                if col in self.state_config.columns:
                    new_state[col] = self.state_config[col]
                elif col == "name" and col not in self.state_config.columns:
                    new_state["name"] = self.state_config.index
                else:
                    new_state[col] = np.array([default] * len(self.state_config.index))

            # Fill empty cells (only do this for values, where the default is not None)
            new_state.fillna(
                value={key: val for key, val in self.__state_config_cols.items() if val is not None},
                inplace=True,
            )

        # If state config is a list of dictionaries iterate the list and create the DataFrame iteratively
        elif isinstance(self.state_config, Sequence):
            new_state = []
            for row in self.state_config:
                new_row = {}
                for col, default in self.__state_config_cols.items():
                    new_row[col] = row[col] if col in row else default
                new_state.append(new_row)
            new_state = pd.DataFrame(data=new_state, columns=self.__state_config_cols.keys())
        else:
            raise ValueError(
                "state_config is not in the correct format. It should be a DataFrame or a list "
                "of dictionaries. It is currently {}".format(type(self.state_config))
            )

        new_state.set_index("name", inplace=True, verify_integrity=True)

        self.state_config = new_state
        return self.state_config

    def _store_state_info(self) -> None:
        """Save state_config to csv for info (only first environment)"""
        if self.env_id == 1:
            self.state_config.to_csv(
                path_or_buf=self.path_results / (self.run_name + "_state_config.csv"),
                sep=";",
                decimal=",",
            )

    def import_scenario(self, *scenario_paths: dict[str, Any], prefix_renamed: bool = True) -> pd.DataFrame:
        """Load data from csv into self.timeseries_data by using scenario_from_csv

        :param scenario_paths: One or more scenario configuration dictionaries (Or a list of dicts), which each contain
            a path for loading data from a scenario file. The dictionary should have the following structure, with <X>
            denoting the variable value:

            .. note ::
                [{*path*: <X>, *prefix*: <X>, *interpolation_method*: <X>, *resample_method*: <X>,
                *scale_factors*: {col_name: <X>}, *rename_cols*: {col_name: <X>}, *infer_datetime_cols*: <X>,
                *time_conversion_str*: <X>]

            * **path**: Path to the scenario file (relative to scenario_path)
            * **prefix**: Prefix for all columns in the file, useful if multiple imported files
              have the same column names
            * **interpolation_method**: A pandas interpolation method, required if the frequency of
              values must be increased in comparison to the files data. (e.g.: 'linear' or 'pad')
            * **scale_factors**: Scaling factors for specific columns. This can be useful for
              example if a column contains data in kilowatt and should be imported in watts.
              In this case the scaling factor for the column would be 1000.
            * **rename_cols**: Mapping of column names from the file to new names for the imported
              data.
            * **infer_datetime_cols**: Number of the column which contains datetime data. If this
              value is not present, the time_conversion_str variable will be used to determine
              the datetime format.
            * **time_conversion_str**: Time conversion string, determining the datetime format
              used in the imported file (default: %Y-%m-%d %H:%M)
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

    def continuous_action_space_from_state(self) -> spaces.Space:
        """Use the state_config to generate the action space according to the format required by the OpenAI
        specification. This will set the action_space attribute and return the corresponding space object.
        The generated action space is continous.

        :return: Action space
        """
        action_low = self.state_config.loc[self.state_config.is_agent_action == True].low_value.values  # noqa: E712
        action_high = self.state_config.loc[self.state_config.is_agent_action == True].high_value.values  # noqa: E712
        self.action_space = spaces.Box(action_low, action_high, dtype=float)

        return self.action_space

    def continuous_obs_space_from_state(self) -> spaces.Box:
        """Use the state_config to generate the observation space according to the format required by the OpenAI
        specification. This will set the observation_space attribute and return the corresponding space object.
        The generated observation space is continous.

        :return: Observation Space
        """
        state_low = self.state_config.loc[self.state_config.is_agent_observation == True].low_value.values  # noqa: E712
        state_high = self.state_config.loc[
            self.state_config.is_agent_observation == True  # noqa: E712
        ].high_value.values
        self.observation_space = spaces.Box(state_low, state_high, dtype=float)

        return self.observation_space

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:
        """Check whether the given state is within the abort conditions specified by state_config.

        :param state: The state array to check for conformance
        :return: Result of the check (False if the state does not conform to the required conditions)
        """
        valid = all(
            state[key] > val
            for key, val in self.state_config.loc[self.names["abort_conditions_min"]].abort_condition_min.items()
        )
        if valid:
            valid = all(
                state[key] < val
                for key, val in self.state_config.loc[self.names["abort_conditions_max"]].abort_condition_max.items()
            )

        return valid

    def get_scenario_state(self) -> dict[str, Any]:
        """Get scenario data for the current time step of the environment, as specified in state_config. This assumes
        that scenario data in self.ts_current is available and scaled correctly.

        :return: Scenario data for current time step
        """
        scenario_state = {}
        for scen in self.names["scenario"]:
            scenario_state[scen] = self.ts_current[self.map_scenario_ids[scen]].iloc[self.n_steps]

        return scenario_state

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.floating | SupportsFloat, bool, str | Sequence[str]]:
        """Perfom one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        .. note ::
            Do not forge to increment n_steps and n_steps_longtime.

        :param action:
        :return: The return value represents the state of the environment after the step was performed.

            * observations: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * reward: The value of the reward function. This is just one floating point value.
            * done: Boolean value specifying whether an episode has been completed. If this is set to true, the reset
              function will automatically be called by the agent or by eta_i.
            * info: Provide some additional info about the state of the environment. The contents of this may be used
              for logging purposes in the future but typically do not currently server a purpose.

        """
        raise NotImplementedError("Cannot step an abstract Environment.")

    @abc.abstractmethod
    def reset(self) -> np.ndarray:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        .. note ::
            Don't forget to store and reset the episode_timer

        :return: The return value represents the observations (state) of the environment before the first
                 step is performed
        """
        raise NotImplementedError("Cannot reset an abstract Environment.")

    @abc.abstractmethod
    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        :return:
        """
        raise NotImplementedError("Cannot close an abstract Environment.")

    def seed(self, seed: str | int | None = None) -> tuple[np.random.Generator, int]:
        """Set random seed for the random generator of the environment

        :param seed: Seeding value
        :return: Tuple of the numpy bit generator and the set seed value
        """
        if "seed" in self.env_settings and self.env_settings["seed"] == "":
            self.env_settings["seed"] = None

        if not hasattr(self, "_seed") or not hasattr(self, "np_random") or self.seed is None or self.np_random is None:
            # set seeding for pseudorandom numbers
            if seed == "" or seed is None:
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
    def get_info(cls, _: Any = None) -> tuple[str, str]:
        """
        Get info about environment

        :param _: This parameter should not be used in new implementations
        :return: version and description
        """
        return cls.version, cls.description  # type: ignore
