import abc
import inspect
import pathlib
import time
from collections import OrderedDict
from collections.abc import Hashable
from datetime import datetime, timedelta
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    Optional,
    Sequence,
    Set,
    SupportsFloat,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from gym import Env, spaces, utils
from pyomo.core import base as pyo_base
from pyomo.opt import SolverResults

from eta_utility import get_logger, timeseries
from eta_utility.simulators import FMUSimulator

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

        - version: Version number of the environment
        - description: Short description string of the environment
        - action_space: The action space of the environment (see also gym.spaces for options)
        - observation_space: The observation space of the environment (see also gym.spaces for options)

    Some additional attributes can be used to simplify the creation of new environments but they are not required:

        - req_general_settings: List/Set of general settings that need to be present
        - req_path_settings: List/Set of path settings that need to be present
        - req_env_settings: List/Set of environment settings that need to be present
        - req_env_config: Mapping of environment settings that must have specific values

    The gym interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - step
        - reset
        - close

    :param int env_id: Identification for the environment, usefull when creating multiple environments
    :param str run_name: Identification name for the optimization run
    :param Dict[str, Any] general_settings: Dictionary of general settings
    :param Dict[str, Any] path_settings: Dictionary of path settings
    :param Dict[str, Any] env_settings: Dictionary of environment specific settings
    :param int verbose: Verbosity setting for logging
    :param Callable callback: callback method will be called after each episode with all data within the
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
    req_general_settings: Union[Sequence, MutableSet] = []
    #: Required settings in the 'path' section
    req_path_settings: Union[Sequence, MutableSet] = []
    #: Required settings in the 'environment_specific' section
    req_env_settings: Union[Sequence, MutableSet] = []
    #: Some environments may required specific parameters in the 'environment_specific' section to have special
    #   values. These parameter, value pairs can be specified in the req_env_config dictionary.
    req_env_config: MutableMapping = {}

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: Dict[str, Any],
        path_settings: Dict[str, Union[str, pathlib.Path]],
        env_settings: Dict[str, Any],
        verbose: int,
        callback: Callable = None,
    ):
        # Set some additional required settings
        self.req_path_settings = set(self.req_path_settings)
        self.req_path_settings.update(("path_root", "path_results", "relpath_scenarios"))  # noqa
        self.req_env_settings = set(self.req_env_settings)
        self.req_env_settings.update(("scenario_time_begin", "scenario_time_end"))  # noqa
        self.req_general_settings = set(self.req_general_settings)
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
        self.settings: Dict[str, Any] = general_settings.copy()

        # Set some standard environment settings
        #: Configuration from the 'environment_specific' section. This contains scenario_time_begin and
        #   end as datetime values
        self.env_settings: Dict[str, Any] = env_settings.copy()
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
        self.path_settings: Dict[str, Any] = path_settings.copy()
        #: Root path of the application as generated by the framework
        self.path_root: pathlib.Path = pathlib.Path(self.path_settings["path_root"])
        #: Path for results storage
        self.path_results: pathlib.Path = pathlib.Path(self.path_settings["path_results"])
        #: Path for scenario storage (this does not include the individual scenario files as specified in the
        #   environment config)
        self.path_scenarios: pathlib.Path = self.path_root / self.path_settings["relpath_scenarios"]
        #: Path of the environment file
        self.path_env: pathlib.Path = pathlib.Path(inspect.stack()[2].filename).parent
        # store callback function in object
        #: Callback function for the environment. This should be called after every step.
        self.callback: Callable = callback

        # Store some important settings
        #: Total number of environments
        self.n_environments = int(self.settings["n_environments"])
        #: Id of the environment (useful for vectorized environments)
        self.env_id = env_id
        #: Name of the current optimization run
        self.run_name = run_name
        #: Number of completed episodes
        self.n_episodes = 0
        #: Current step of the model (number of completed steps) in the current episode
        self.n_steps: int = 0
        #: Current step of the model (total over all episodes)
        self.n_steps_longtime: int = 0
        #: Beginning time of the scenario
        self.scenario_time_begin: datetime = self.env_settings["scenario_time_begin"]
        #: Ending time of the scenario
        self.scenario_time_end: datetime = self.env_settings["scenario_time_end"]
        #: Sampling time (interval between optimization time steps) in seconds
        self.sampling_time: int = int(self.settings["sampling_time"])
        #: Number of time steps (of width sampling_time) in each episode
        self.n_episode_steps: int = int(self.settings["episode_duration"]) // self.sampling_time
        #: Duration of one episode in seconds
        self.episode_duration: int = int(self.n_episode_steps * self.sampling_time)

        #: Numpy random generator
        self.np_random: np.random.BitGenerator
        #: Value used for seeding the random generator
        self._seed: int
        self.np_random, self._seed = (
            self.seed(self.env_settings["seed"]) if "seed" in self.env_settings else self.seed(None)
        )

        #: The time series DataFrame contains all time series scenario data. It can be filled by the
        #   import_scenario method.
        self.timeseries: pd.DataFrame = pd.DataFrame()

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
                ("low_value", None),
                ("high_value", None),
                ("abort_condition_min", None),
                ("abort_condition_max", None),
            ]
        )

        #: The configuration for the action and observation spaces. The values are used to control which variables are
        #   part of the action space and observation space. Additionally the parameters can specify abort conditions
        #   and the handling of values from interaction environments or from simulation. Therefore the state_config
        #   is very important for the functionality of ETA X.
        #
        #  Some functions that operate on state_config:
        #   The function append_state can be used to append values to the DataFrame. The function
        #   _convert_state_config can be used to convert an incomplete DataFrame or a list of dictionaries into the full
        #   standardized format.
        #
        #  Possible column names, their types and default values are:
        #
        #   * name: str, Name of the state variable (This must always be specified (no default))
        #   * is_agent_action: bool, Should the agent specify actions for this variable? (default: False)
        #   * is_agent_observation: bool, Should the agent be allowed to observe the value
        #                                 of this variable? (default: False)
        #   * ext_id: str, Name or identifier (order) of the variable in the external interaction model
        #                  (e.g.: environment or FMU) (default: None)
        #   * is_ext_input: bool, Should this variable be passed to the external model as an input? (default: False)
        #   * is_ext_output: bool, Should this variable be parsed from the external model output? (default: False)
        #   * ext_scale_add: int or float, Value to add to the output from an external model (default: 0)
        #   * ext_scale_mult: int or float, Value to multiply to the output from an external model (default: 1)
        #   * low_value: int or float, lowest possible value of the state variable (default: None)
        #   * high_value: int or float, highest possible value of the state variable (default: None)
        #   * abort_condition_min: int or float, If value of variable dips below this, the episode
        #                                        will be aborted (default: None)
        #   * abort_condition_max: int or float, If value of variable rises above this, the episode
        #                                        will be aborted (default: None)
        self.state_config: pd.DataFrame = pd.DataFrame(columns=self.__state_config_cols.keys())
        self.state_config.set_index("name", drop=True, inplace=True)
        #: Array of shorthands to some frequently used variable names from state_config. .. seealso :: _names_from_state
        self.names: Dict[np.ndarray]
        #: Dictionary of scaling values for external input values (for example from simulations).
        #  The structure of this dictionary is {"name": {"add": value, "multiply": value}}.
        self.ext_scale: Dict[str, Dict[str, Union[int, float]]]

        # Store data logs and log other information
        #: episode timer
        self.episode_timer = time.time()
        #: Current state of the environment
        self.state: Dict[str, float]
        #: Log of the environment state
        self.state_log: List[Dict[str, float]] = []
        #: Log of the environment state over multiple episodes
        self.state_log_longtime: List[List[Dict[str, float]]] = []
        #: Some specific current environment settings / other data, apart from state
        self.data: Dict[str, Any]
        #: Log of specific environment settings / other data, apart from state for the episode
        self.data_log: List[Dict[str, Any]] = []
        #: Log of specific environment settings / other data, apart from state, over multiple episodes.
        self.data_log_longtime: List[List[Dict[str, Any]]]

    def append_state(self, *, name, **kwargs) -> None:
        """Append a state variable to the state configuration of the environment

        :param name: Name of the state variable
        :param kwargs: Column names and values to be inserted into the respective column. For possible columns, types
                       and default values see state_config.
                       .. seealso :: state_config
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

    def _init_state_space(self):
        """Convert state config and store state information. This is a shorthand for the function calls:

        * _convert_state_config
        * _names_from_state
        * _store_state_info

        """
        self._convert_state_config()
        self._names_from_state()
        self._store_state_info()

    def _names_from_state(self):
        """Intialize the names array from state_config, which stores shorthands to some frequently used variable names.

        The names array contains the following (ordered) lists of variables:
            * actions: Variables that are agent actions
            * observations: Variables that are agent observations
            * ext_inputs: Variables that should be provided to an external source (such as an FMU)
            * ext_output: variables that can be received from an external source (such as an FMU)
            * abort_conditions_min: Variables that have minimum values for an abort condition
            * abort_conditions_max: Variables that have maximum values for an abort condition

        """
        self.names = {
            "actions": self.state_config.loc[self.state_config.is_agent_action == True].index.values,  # noqa: E712
            "observations": self.state_config.loc[self.state_config.is_agent_observation == True].index.values,
            "ext_inputs": self.state_config.loc[self.state_config.is_ext_input == True].index.values,
            "ext_outputs": self.state_config.loc[self.state_config.is_ext_output == True].index.values,
            "abort_conditions_min": self.state_config.loc[self.state_config.abort_condition_min.notnull()].index.values,
            "abort_conditions_max": self.state_config.loc[self.state_config.abort_condition_max.notnull()].index.values,
        }

        self.ext_scale = {}
        for name, values in self.state_config.iterrows():
            self.ext_scale[name] = {"add": values.ext_scale_add, "multiply": values.ext_scale_mult}

    def _convert_state_config(self):
        """This will convert an incomplete state_config DataFrame or a list of dictionaries to the standardized
        DataFrame format. This will remove any additional columns. If additional columns are required, ensure
        consistency with the required format otherwise.

        :return: Converted, standardized dataframe
        :rtype: pd.DataFrame
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

    def _store_state_info(self):
        """Save state_config to csv for info (only first environment)"""
        if self.env_id == 1:
            self.state_config.to_csv(
                path_or_buf=self.path_results / (self.run_name + "_state_config.csv"),
                sep=";",
                decimal=",",
            )

    def continous_action_space_from_state(self) -> spaces.Space:
        """Use the state_config to generate the action space according to the format required by the OpenAI
        specification. This will set the action_space attribute and return the corresponding space object.
        The generated action space is continous.

        :return: Action space
        :rtype: Tuple[space.Space, spaces.Space]
        """
        action_low = self.state_config.loc[self.state_config.is_agent_action == True].low_value.values  # noqa: E712
        action_high = self.state_config.loc[self.state_config.is_agent_action == True].high_value.values  # noqa: E712
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float)

        return self.action_space

    def continous_obs_space_from_state(self) -> spaces.Space:
        """Use the state_config to generate the observation space according to the format required by the OpenAI
        specification. This will set the observation_space attribute and return the corresponding space object.
        The generated observation space is continous.

        :return:
        """
        state_low = self.state_config.loc[self.state_config.is_agent_observation == True].low_value.values  # noqa: E712
        state_high = self.state_config.loc[
            self.state_config.is_agent_observation == True
        ].high_value.values  # noqa: E712
        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float)

        return self.observation_space

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:
        """Check whether the given state is within the abort conditions specified by state_config.

        :param state: The state array to check for conformance
        :type state: Mapping[str, float]
        :return: Result of the check (False if the state does not conform to the required conditions)
        :rtype: bool
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

    @abc.abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]:
        """Perfom one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        .. note ::
            Do not forge to increment n_steps and n_steps_longtime.

        :param np.ndarray action:
        :return: The return value represents the state of the environment after the step was performed.

            * observations: A numpy array with new observation values as defined by the observation space.
                            Observations is a np.array() (numpy array) with floating point or integer values.
            * reward: The value of the reward function. This is just one floating point value.
            * done: Boolean value specifying whether an episode has been completed. If this is set to true, the reset
                    function will automatically be called by the agent or by eta_i.
            * info: Provide some additional info about the state of the environment. The contents of this may be used
                    for logging purposes in the future but typically do not currently server a purpose.

        :rtype: Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]
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
        :rtype: Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]
        """
        raise NotImplementedError("Cannot reset an abstract Environment.")

    @abc.abstractmethod
    def close(self):
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        :return:
        """
        raise NotImplementedError("Cannot close an abstract Environment.")

    def seed(self, seed: Union[str, int] = None) -> Tuple[np.random.BitGenerator, int]:
        """Set random seed for the random generator of the environment

        :param seed: Seeding value
        :type seed: str or int
        :return: Tuple of the numpy bit generator and the set seed value
        :rtype: Tuple[np.random.BitGenerator, int]
        """
        if "seed" in self.env_settings and self.env_settings["seed"] == "":
            self.env_settings["seed"] = None

        if not hasattr(self, "_seed") or not hasattr(self, "np_random") or self.seed is None or self.np_random is None:
            # set seeding for pseudorandom numbers
            if seed == "" or seed is None:
                seed = None
                log.info("The environment seed is set to None, a random seed will be set.")
            else:
                seed = int(seed) + self.env_id

            np_random, seed = utils.seeding.np_random(seed)  # noqa
            log.info(f"The environment seed is set to: {seed}")

            self._seed = seed
            self.np_random = np_random

        return self.np_random, self._seed

    def _import_scenario(
        self,
        paths: Union[pathlib.Path, Sequence[pathlib.Path]],
        data_prefixes: Sequence[str] = None,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        total_time: Optional[Union[timedelta, int]] = None,
        random: Optional[bool] = False,
        resample_time: Optional[Union[timedelta, int]] = None,
        resample_method: Optional[str] = None,
        interpolation_method: Optional[Union[Sequence[str], str]] = None,
        rename_cols: Optional[Mapping[str, str]] = None,
        prefix_renamed: Optional[bool] = True,
        infer_datetime_from: Optional[Union[str, Sequence[int]]] = "string",
        time_conversion_str: str = "%Y-%m-%d %H:%M",
    ):
        """Import (possibly multiple) scenario data files from csv files and return them as a single pandas
        data frame. The import function supports column renaming and will slice and resample data as specified.

        :raises ValueError: If start and/or end times are outside of the scope of the imported scenario files.

        .. note::
            The ValueError will only be raised when this is true for all files. If only one file is outside of
            the range, and empty series will be returned for that file. TODO - Implement additional checks.

        :param paths: Path(s) to one or more CSV data files. The paths should be fully qualified.
        :type paths: pathlib.Path or Sequence[pathlib.Path]
        :param data_prefixes: If more than file is imported, a list of data_prefixes must be supplied such that
                              ambiquity of column names between the files can be avoided. There must be one prefix
                              for every imported file, such that a distinct prefix can be prepended to all columns
                              of a file.
        :type data_prefixes: Sequence[str]
        :param start_time: (Keyword only) Starting time for the scenario import. Default value is scenario_time_begin.
        :type start_time: datetime
        :param end_time: (Keyword only) Latest ending time for the scenario import. Default value is scenario_time_end.
        :type end_time: datetime
        :param total_time: (Keyword only) Total duration of the imported scenario. If given as int this will be
                           interpreted as seconds. The default is episode_duration.
        :type total_time: timedelta or int
        :param random: (Keyword only) Set to true if a random starting point (within the interval determined by
                       start_time and end_time should be chosen. This will use the environments random generator.
                       The default is false.
        :type random: bool
        :param resample_time: (Keyword only) Resample the scenario data to the specified interval. If this is specified
                              one of 'upsample_fill' or downsample_method' must be supplied as well to determin how
                              the new data points should be determined. If given as an in, this will be interpreted as
                              seconds. The default is no resampling.
        :type resample_time: timedelta or int
        :param resample_method: (Keyword only) Method for filling in / aggregating data when resampling. Pandas
                                resampling methods are supported. Default is None (no resampling)
        :type resample_method: str
        :param interpolation_method: (Keyword only) Method for interpolating missing data values. Pandas missing data
                                     handling methods are supported. If a list with one value per file is given, the
                                     specified method will be selected according the order of paths.
        :type interpolation_method: str or List[str]
        :param rename_cols: (Keyword only) Rename columns of the imported data. Maps the colunms as they appear in the
                            data files to new names. Format: {old_name: new_name}
        :type rename_cols: Mapping[str, str]
        :param prefix_renamed: (Keyword only) Should prefixes be applied to renamed columns as well? Default: True.
                               When setting this to false make sure that all columns in all loaded scenario files
                               have different names. Otherwise there is a risk of overwriting data.
        :type prefix_renamed: bool
        :param infer_datetime_from: (Keyword only) Specify how datetime values should be converted. 'dates' will use
                                    pandas to automatically determine the format. 'string' uses the conversion string
                                    specified in the 'time_conversion_str' parameter. If a two-tuple of the format
                                    (row, col) is given, data from the specified field in the data files will be used
                                    to determine the date format. The default is 'string'
        :type infer_datetime_from: str or Sequence[int]
        :param time_conversion_str: (Keyword only) Time conversion string. This must be specified if the
                                    infer_datetime_from parameter is set to 'string'. The string should specify the
                                    datetime format in the python strptime format. The default is: '%Y-%m-%d %H:%M'.
        :type time_conversion_str: str
        :return:
        """
        if hasattr(paths, "__len__") and len(paths) > 1 and (data_prefixes is None or len(paths) != len(data_prefixes)):
            raise ValueError(
                "The number of paths and data_prefixes does not correspond to "
                "each other: {}\n{}".format(paths, data_prefixes)
            )
        elif not hasattr(paths, "__len__"):
            paths = (paths,)

        if (
            hasattr(paths, "__len__")
            and len(paths) > 1
            and hasattr(interpolation_method, "__len__")
            and len(paths) != len(interpolation_method)
        ):
            raise ValueError(
                "The number of interpolation methods does not match the number of paths. Specify 0, 1 or"
                "'number of paths' interpolation methods."
            )
        elif not hasattr(interpolation_method, "__len__"):
            interpolation_method = [interpolation_method] * len(paths)

        # Set defaults and convert values where necessary
        total_time = (
            timedelta(seconds=self.episode_duration)
            if total_time is None
            else total_time
            if isinstance(total_time, timedelta)
            else timedelta(seconds=total_time)
        )
        start_time = self.scenario_time_begin if start_time is None else start_time
        end_time = self.scenario_time_end if end_time is None else end_time
        random = self.np_random if random else False
        resample = True if resample_time is not None else False
        resample_time = (
            timedelta(seconds=self.sampling_time)
            if resample_time is None
            else resample_time
            if isinstance(resample_time, timedelta)
            else timedelta(seconds=resample_time)
        )

        slice_begin, slice_end = timeseries.find_time_slice(
            start_time,
            end_time,
            total_time=total_time,
            random=random,
            round_to_interval=resample_time,
        )

        df = pd.DataFrame()
        for i, path in enumerate(paths):
            data = timeseries.df_from_csv(
                path,
                infer_datetime_from=infer_datetime_from,
                time_conversion_str=time_conversion_str,
            )
            if resample:
                data = timeseries.df_resample(
                    data,
                    resample_time,
                    resample_method=resample_method,
                    missing_data=interpolation_method[i],
                )
            data = data[slice_begin:slice_end].copy()

            col_names = {}
            for col in data.columns:
                if not prefix_renamed and rename_cols is not None and col in rename_cols:
                    pre = ""
                    name = str(rename_cols[col])
                elif prefix_renamed and rename_cols is not None and col in rename_cols:
                    pre = "{}_".format(data_prefixes[i]) if data_prefixes is not None else ""
                    name = str(rename_cols[col])
                else:
                    pre = "{}_".format(data_prefixes[i]) if data_prefixes is not None else ""
                    name = str(col)
                col_names[col] = pre + name
            data.rename(columns=col_names, inplace=True)
            df = pd.concat((data, df), 1)

        if (
            len(df) <= 0
            or df.first_valid_index() > self.scenario_time_end
            or df.last_valid_index() < self.scenario_time_begin
        ):
            raise ValueError(
                "The loaded scenario file does not contain enough data for the entire episode. Or the set "
                "scenario times do not correspond to the provided data."
            )

        return df

    @classmethod
    def get_info(cls, _=None):
        """
        get info about environment

        :param _: This parameter should not be used in new implementations
        :return: version and description
        :rtype: Tuple[str, str]
        """
        return cls.version, cls.description


class BaseEnvMPC(BaseEnv, abc.ABC):
    """Base class for MPC models"""

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: Dict[str, Any],
        path_settings: Dict[str, Union[str, pathlib.Path]],
        env_settings: Dict[str, Any],
        verbose: int,
        callback: Callable = None,
    ):
        self.req_env_settings = set(self.req_env_settings)
        self.req_env_settings.update(("model_parameters",))  # noqa
        self.req_env_config.update(
            {
                "discretize_state_space": False,
                "discretize_action_space": False,
                "normalize_state_space": False,
                "normalize_reward": False,
                "reward_shaping": False,
            }
        )
        super().__init__(
            env_id,
            run_name,
            general_settings,
            path_settings,
            env_settings,
            verbose,
            callback,
        )

        # Check configuration for MILP compatibility
        errors = False
        if self.settings["prediction_scope"] % self.settings["sampling_time"] != 0:
            log.error(
                "The sampling_time must fit evenly into the prediction_scope "
                "(prediction_scope % sampling_time must equal 0."
            )
            errors = True

        if errors:
            raise ValueError(
                "Some configuration parameters do not conform to the MPC environment " "requirements (see log)."
            )

        # Make some more settings easily accessible
        #: Total duration of one prediction/optimization run when used with the MPC agent.
        #   This is automatically set to the value of episode_duration if it is not supplied
        #   separately
        self.prediction_scope: int
        if "prediction_scope" not in self.settings:
            log.info("prediction_scope parameter is not present. Setting prediction_scope to episode_duration.")
        self.prediction_scope = int(self.settings.setdefault("prediction_scope", self.episode_duration))
        self.model_parameters: dict  #: Configuration for the MILP model parameters
        self.model_parameters = self.env_settings["model_parameters"]
        self.n_prediction_steps: int  #: Number of steps in the prediction (prediction_scope/sampling_time)
        self.n_prediction_steps = self.settings["prediction_scope"] // self.sampling_time

        # Set additional attributes with model specific information.
        self._concrete_model: Union[pyo.ConcreteModel, None] = None  #: Concrete pyomo model as initialized by _model.

        #: Name of the "time" variable/set in the model (i.e. "T"). This is if the pyomo sets must be re-indexed when
        #   updating the model between time steps. If this is None, it is assumed that no reindexing of the timeseries
        #   data is required during updates - this is the default.
        self.time_var: Optional[str] = None

        #: Updating indexed model parameters can be achieved either by updating only the first value of the actual
        #   parameter itself or by having a separate handover parameter that is used for specifying only the first
        #   value. The separate handover parameter can be denoted with an appended string. For example, if the actual
        #   parameter is x.ON then the handover parameter could be x.ON_first. To use x.ON_first for updates, set the
        #   nonindex_update_append_string to "_first". If the attribute is set to None, the first value of the
        #   actual parameter (x.ON) would be updated instead.
        self.nonindex_update_append_string: Optional[str] = None

        #: Some models may not use the actual time increment (sampling_time). Instead they would translate into model
        #   time increments (each sampling time increment equals a single model time step). This means that indices
        #   of the model components simply count 1,2,3,... instead of 0, sampling_time, 2*sampling_time, ...
        #   Set this to true, if model time increments (1, 2, 3, ...) are used. Otherwise sampling_time will be used
        #   as the time increment. Note: This is only relevant for the first model time increment, later increments
        #   may differ.
        self._use_model_time_increments: bool = False

    @property
    def model(self) -> Tuple[pyo.ConcreteModel, list]:
        """The model property is a tuple of the concrete model and the order of the action space. This is used
        such that the MPC algorithm can re-sort the action output. This sorting cannot be conveyed differently through
        pyomo.

        :return: tuple of the concrete model and the order of the action space
        :type: Tuple[pyo.ConcreteModel, list]
        """
        if self._concrete_model is None:
            self._concrete_model = self._model()

        if self.names["actions"] is None:
            self.names["actions"] = [
                com.name
                for com in self._concrete_model.component_objects(pyo.Var)
                if not isinstance(com, pyo.SimpleVar)
            ]

        return self._concrete_model, self.names["actions"]

    @model.setter
    def model(self, value):
        """The model attribute setter should be used for returning the solved model.

        :param value:
        :return:
        """
        if not isinstance(value, pyo.ConcreteModel):
            raise TypeError("The model attribute can only be set with a pyomo concrete model.")
        self._concrete_model = value

    @abc.abstractmethod
    def _model(self) -> pyo.ConcreteModel:
        """Create the abstract pyomo model. This is where the pyomo model description should be placed.

        :return: Abstract pyomo model
        :rtype: pyo.AbstractModel
        """
        raise NotImplementedError("The abstract MPC environment does not implement a model.")

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]:
        """Perfom one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine
        the new state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        This also updates self.state and self.state_log to store current state information.

        TODO: Add something to handle actions, currently they are just ignored (MPC agent does not need to use actions)

        :param np.ndarray action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.

            * observations: A numpy array with new observation values as defined by the observation space.
                            Observations is a np.array() (numpy array) with floating point or integer values.
            * reward: The value of the reward function. This is just one floating point value.
            * done: Boolean value specifying whether an episode has been completed. If this is set to true,
                the reset function will automatically be called by the agent or by eta_i.
            * info: Provide some additional info about the state of the environment. The contents of this may
                be used for logging purposes in the future but typically do not currently serve a purpose.
        :rtype: Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]
        """
        observations = self.update()

        # update and log current state
        self.state = {}
        for idx, act in enumerate(self.names["actions"]):
            self.state[act] = action[idx]
        for idx, obs in enumerate(self.names["observations"]):
            self.state[obs] = observations[idx]
        self.state_log.append(self.state)

        reward = pyo.value(list(self._concrete_model.component_objects(pyo.Objective))[0])
        done = True if self.n_steps >= self.n_episode_steps else False

        info = {}
        if done:
            info["terminal_observation"] = observations
        return observations, reward, done, info

    def update(self, observations: Optional[Sequence[Sequence[Union[float, int]]]] = None) -> np.ndarray:
        """Update the optimization model with observations from another environment.

        :param observations: Observations from another environment
        :type observations: Optional[Sequence[Sequence[Union[float, int]]]]
        :return: Full array of current observations
        :rtype: np.ndarray
        """
        # update shift counter for rolling MPC approach
        self.n_steps += 1

        # The timeseries data must be updated for the next time step. The index depends on whether time itself is being
        # shifted. If time is being shifted, the respective variable should be set as "time_var".
        step = 1 if self._use_model_time_increments else self.sampling_time
        duration = (
            self.prediction_scope // self.sampling_time + 1
            if self._use_model_time_increments
            else self.prediction_scope
        )

        if self.time_var is not None:
            index = range(self.n_steps * step, duration + (self.n_steps * step), step)
            ts_current = self.pyo_convert_timeseries(
                self.timeseries.iloc[self.n_steps : self.n_prediction_steps + self.n_steps + 1],
                index=tuple(index),
                _add_wrapping_None=False,
            )
            ts_current[self.time_var] = list(index)
            log.debug(
                "Updated time_var ({}) with the set from {} to {} and steps (sampling time) {}.".format(
                    self.time_var, index[0], index[1], self.sampling_time
                )
            )
        else:
            index = range(0, duration, step)
            ts_current = self.pyo_convert_timeseries(
                self.timeseries.iloc[self.n_steps : self.n_prediction_steps + self.n_steps + 1],
                index=tuple(index),
                _add_wrapping_None=False,
            )

        # Log current time shift
        if self.n_steps + self.n_prediction_steps + 1 < len(self.timeseries.index):
            log.info(
                "Current optimization time shift: {} of {} | Current scope: {} to {}".format(
                    self.n_steps,
                    self.n_episode_steps,
                    self.timeseries.index[self.n_steps],
                    self.timeseries.index[self.n_steps + self.n_prediction_steps + 1],
                )
            )
        else:
            log.info(
                "Current optimization time shift: {} of {}. Last optimization step reached.".format(
                    self.n_steps, self.n_episode_steps
                )
            )

        updated_params = ts_current
        return_obs = []  # Array for all current observations
        next_index = 0
        for var_name in self.names["observations"]:
            settings = self.state_config.loc[var_name]
            value = None
            if observations is not None and settings["is_ext_output"] is True:
                value = round(
                    (observations[0][settings["ext_id"]] + settings["ext_scale_add"]) * settings["ext_scale_mult"],
                    5,
                )
                return_obs.append(value)
            else:
                for component in self._concrete_model.component_objects():
                    if component.name == var_name:
                        value = round(pyo.value(component[next_index]), 5)  # Get value for the component
                        return_obs.append(round(pyo.value(component[0]), 5))
                        break
                else:
                    log.error(f"Specified observation value {var_name} could not be found")
            updated_params[var_name] = value

            log.info(f"Observed value {var_name}: {value}")

        self.pyo_update_params(updated_params, self.nonindex_update_append_string)
        return np.array(return_obs)

    def solve_failed(self, model: pyo.ConcreteModel, result: SolverResults):
        """This method will try to render the result in case the model could not be solved. It should automatically
        be called by the agent.

        :param pyo.ConcreteModel model: Current model
        :param pyo.SolverStatus result: Result of the last solution attempt
        :return:
        """
        self.model = model
        try:
            self.render()
        except Exception as e:
            log.error("Rendering partial results failed: {}".format(str(e)))
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the model and return initial observations. This also calls the callback, increments the episode
        counter, resets the episode steps and appends the state_log to the

        :return: Initial observation
        :rtype: np.ndarray
        """
        if self.n_steps > 0:
            if self.callback is not None:
                self.callback(self)

            # Store some logging data
            self.n_episodes += 1
            self.state_log_longtime.append(self.state_log)
            self.n_steps_longtime += self.n_steps

            # Reset episode variables
            self.n_steps = 0
            self.state_log = []
            self._concrete_model = self._model()

        # Initialize state with the initial observation
        self.state = {}
        observations = []
        for var_name in self.names["observations"]:
            for component in self._concrete_model.component_objects():
                if component.name == var_name:
                    if hasattr(component, "__getitem__") and (
                        (hasattr(component[0], "stale") and component[0].stale)
                        or (hasattr(component[0], "active") and not component[0].active)
                    ):
                        obs_val = 0
                    elif not hasattr(component, "__getitem__"):
                        obs_val = round(pyo.value(component), 5)
                    else:
                        obs_val = round(pyo.value(component[0]), 5)
                    observations.append(obs_val)
                    break
            self.state[var_name] = observations[-1]

        # Initialize state with zero actions
        for act in self.names["actions"]:
            self.state[act] = 0
        self.state_log.append(self.state)

        return np.array(observations)

    def pyo_component_params(
        self,
        component_name: Union[None, str],
        ts: Optional[Mapping[Any, Any]] = None,
        index: Optional[Union[Sequence, pyo.Set]] = None,
    ) -> Dict[None, Dict[str, Any]]:
        """Retrieve paramters for the named component and convert the parameters into the pyomo dict-format.
        If required, timeseries can be added to the parameters and timeseries may be reindexed. The
        pyo_convert_timeseries function is used for timeseries handling.

        .. seealso:: pyo_convert_timeseries

        :param str component_name: Name of the component
        :param Mapping[str, Any] ts: Timeseries for the component
        :param index: New index for timeseries data. If this is supplied, all timeseries will be copied and
                              reindexed.
        :type index: Sequence or pyo.Set
        :return: Pyomo parameter dictionary
        :rtype: Dict[None, Dict[str, Any]]
        """
        if component_name is None:
            params = self.model_parameters
        elif component_name in self.model_parameters:
            params = self.model_parameters[component_name]
        else:
            params = {}
            log.warning(f"No parameters specified for requested component {component_name}")

        out = {
            param: {None: float(value) if isinstance(value, Hashable) and value in {"inf", "-inf"} else value}
            for param, value in params.items()
        }

        # If component name was specified only look for relevant time series
        if ts is not None:
            out.update(self.pyo_convert_timeseries(ts, index, component_name, _add_wrapping_None=False))

        return {None: out}

    @staticmethod
    def pyo_convert_timeseries(
        ts: Union[pd.DataFrame, pd.Series, Dict[str, Dict], Sequence],
        index: Optional[Union[pd.Index, Sequence, pyo.Set]] = None,
        component_name: Optional[str] = None,
        *,
        _add_wrapping_None=True,
    ) -> Union[Dict[None, Dict[str, Any]], Dict[str, Any]]:
        """Convert a time series data into a pyomo format. Data will be reindexed if a new index is provided.

        :param Mapping ts: Timeseries to convert
        :type ts: pd.DataFrame or pd.Series or Dict[str, Union[Dict or Sequence]] or Sequence
        :param index: New index for timeseries data. If this is supplied, all timeseries will be copied and
                              reindexed.
        :type index: pd.Index or Sequence or pyo.Set
        :param str component_name: Name of a specific component that the timeseries is used for. This limits which
                                   timeseries are returned.
        :param bool _add_wrapping_None: default is True
        :return: pyomo parameter dictionary
        :rtype: Dict[None, Dict[str, Any]]
        """
        output = {}
        if index is not None:
            index = list(index) if type(index) is not list else index

        # If part of the timeseries was converted before, make sure that everything is on the same level again.
        if None in ts and isinstance(ts[None], Mapping):
            ts = ts.copy()
            ts.update(ts[None])
            del ts[None]

        def convert_index(_ts, _index):
            """Take the timeseries and change the index to correspond to _index.

            :param _ts: Original timeseries object (with or without index does not matter)
            :type _ts: pd.Series or Mapping or Sequence
            :param _index: New index
            :type _index: Sequence[int]
            :return: New timeseries dictionary with the converted index.
            :rytpe: Dict[int, Any]
            """
            values = None
            if isinstance(_ts, pd.Series):
                values = _ts.values
            elif isinstance(_ts, Mapping):
                values = ts.values()  # noqa
            elif isinstance(_ts, Sequence):
                values = _ts

            if _index is not None and values is not None:
                _ts = dict(zip(_index, values))
            elif _index is not None and values is None:
                raise ValueError("Unsupported timeseries type for index conversion.")

            return _ts

        if isinstance(ts, pd.DataFrame) or isinstance(ts, Mapping):
            for key, t in ts.items():
                # Determine whether the timeseries should be returned, based on the timeseries name and the requested
                #  component name.
                if component_name is not None and "." in key and component_name in key.split("."):
                    key = key.split(".")[-1]
                elif component_name is not None and "." in key and component_name not in key.split("."):
                    continue

                # Simple values do not need their index converted...
                if not hasattr(t, "__len__") and np.isreal(t):
                    output[key] = {None: t}
                else:
                    output[key] = convert_index(t, index)

        elif isinstance(ts, pd.Series):
            # Determine whether the timeseries should be returned, based on the timeseries name and the requested
            #  component name.
            if (
                component_name is not None
                and type(ts.name) is str
                and "." in ts.name
                and component_name in ts.name.split(".")
            ):  # noqa
                output[ts.name.split(".")[-1]] = convert_index(ts, index)  # noqa
            elif component_name is None or "." not in ts.name:
                output[ts.name] = convert_index(ts, index)

        else:
            output[None] = convert_index(ts, index)

        return {None: output} if _add_wrapping_None else output

    def pyo_update_params(
        self,
        updated_params: Dict[str, Any],
        nonindex_param_append_string: Optional[str] = None,
    ):
        """Updates model parameters and indexed parameters of a pyomo instance with values given in a dictionary.
        It assumes that the dictionary supplied in updated_params has the correct pyomo format.

        :param Dict[str, Any] updated_params: Dictionary with the updated values
        :param str nonindex_param_append_string: String to be appended to values that are not indexed. This can
                                                  be used if indexed parameters need to be set with values that do
                                                  not have an index.
        :return: Updated model instance
        :rtype: pyo.ConcreteModel
        """
        # append string to non indexed values that are used to set indexed parameters.
        if nonindex_param_append_string is not None:
            original_indices = set(updated_params.keys()).copy()
            for param in original_indices:
                if not isinstance(updated_params[param], Mapping):
                    updated_params[str(param) + nonindex_param_append_string] = updated_params[param]
                    del updated_params[param]

        for parameter in self._concrete_model.component_objects():
            parameter_name = str(parameter)
            if parameter_name in updated_params.keys():
                if isinstance(parameter, pyo_base.param.SimpleParam) or isinstance(parameter, pyo_base.var.SimpleVar):
                    # update all simple parameters (single values)
                    parameter.value = updated_params[parameter_name]
                elif isinstance(parameter, pyo_base.indexed_component.IndexedComponent):
                    # update all indexed parameters (time series)
                    if not isinstance(updated_params[parameter_name], Mapping):
                        parameter[list(parameter)[0]] = updated_params[parameter_name]
                    else:
                        for param_val in list(parameter):
                            parameter[param_val] = updated_params[parameter_name][param_val]

        log.info("Pyomo model parameters updated.")

    def pyo_get_solution(
        self, names: Optional[Set[str]] = None
    ) -> Dict[str, Union[float, int, Dict[int, Union[float, int]]]]:
        """Convert the pyomo solution into a more useable format for plotting.

        :param Set[str] names: Names of the model parameters that are returned
        :return: Dictionary of {parameter name: value} pairs. Value may be a dictionary of {time: value} pairs which
                 contains one value for each optimization time step
        :rtype: Dict[str, Union[float, int, Dict[int, Union[float, int]]]]
        """

        solution = {}

        for com in self._concrete_model.component_objects():
            if com.ctype not in {pyo.Var, pyo.Param, pyo.Objective}:
                continue
            if names is not None and com.name not in names:
                continue  # Only include names that were asked for

            # For simple variables we need just the values, for everything else we want time indexed dictionaries
            if (
                isinstance(com, pyo.SimpleVar)
                or isinstance(com, pyo_base.objective.SimpleObjective)
                or isinstance(com, pyo_base.param.SimpleParam)
            ):
                solution[com.name] = pyo.value(com)
            else:
                solution[com.name] = {}
                if self._use_model_time_increments:
                    for ind, val in com.items():
                        solution[com.name][
                            self.timeseries.index[self.n_steps].to_pydatetime()
                            + timedelta(seconds=ind * self.sampling_time)
                        ] = pyo.value(val)
                else:
                    for ind, val in com.items():
                        solution[com.name][
                            self.timeseries.index[self.n_steps].to_pydatetime() + timedelta(seconds=ind)
                        ] = pyo.value(val)

        return solution


class BaseEnvSim(BaseEnv, abc.ABC):
    @property
    @abc.abstractmethod
    def fmu_name(self) -> str:
        """Name of the FMU file"""
        return ""

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: Dict[str, Any],
        path_settings: Dict[str, Union[str, pathlib.Path]],
        env_settings: Dict[str, Any],
        verbose: int,
        callback: Callable = None,
    ):

        self.req_general_settings = set(self.req_general_settings)
        self.req_general_settings.update(("sim_steps_per_sample",))  # noqa
        super().__init__(
            env_id,
            run_name,
            general_settings,
            path_settings,
            env_settings,
            verbose,
            callback,
        )

        # Check configuration for compatibility
        errors = False
        if self.settings["sampling_time"] % self.settings["sim_steps_per_sample"] != 0:
            log.error(
                "'sim_steps_per_sample' must be an even divisor of 'sampling_time' "
                "(sampling_time % sim_steps_per_sample must equal 0."
            )
            errors = True

        if errors:
            raise ValueError(
                "Some configuration parameters do not conform to the Sim environment " "requirements (see log)."
            )

        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample = int(self.settings["sim_steps_per_sample"])
        #: The FMU is expected to be placed in the same folder as the environment
        self.path_fmu = self.path_env / (self.fmu_name + ".fmu")

        #: Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
        self.simulator: FMUSimulator

    def _init_simulator(self, init_values: Mapping[str, float]):
        """Initialize the simulator object. Make sure to call _names_from_state before this or to otherwise initialize
        the names array.

        This can also be used to reset the simulator after an episode is completed. It will reuse the same simulator
        object and reset it to the given initial values.

        :param init_values: Dictionary of initial values for some of the FMU variables
        :type init_values: Mapping[str, Union[int, float]]
        """

        if isinstance(self.simulator, FMUSimulator):
            self.simulator.reset(init_values)
        else:
            #: Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
            self.simulator = FMUSimulator(
                self.env_id,
                self.path_fmu,
                start_time=0.0,
                stop_time=self.episode_duration,
                step_size=int(self.sampling_time / self.sim_steps_per_sample),
                names_inputs=self.state_config.loc[self.names["ext_inputs"]].ext_id,
                names_outputs=self.state_config.loc[self.names["ext_outputs"]].ext_id,
                init_values=init_values,
            )

    def simulate(self, state: Mapping[str, float]) -> Tuple[Dict[str, float], bool, float]:
        """Perform a simulator step and return data as specified by the is_ext_observation parameter of the
        state_config.

        :param state: state of the environment before the simulation
        :type state: Mapping[str, float]
        :return: output of the simulation, boolean showing whether all simulation steps where successful, time elapsed
                 during simulation
        :rtype: Tuple[Dict[str, float], bool, float]
        """
        # generate FMU input from current state
        step_inputs = []
        for key in self.names["ext_inputs"]:
            step_inputs.append(state[key] / self.ext_scale[key]["multiply"] - self.ext_scale[key]["add"])

        sim_time_start = time.time()

        step_success = True
        for i in range(self.sim_steps_per_sample):  # do multiple FMU steps in one environment-step
            try:
                step_outputs = self.simulator.step(step_inputs)

            except Exception as e:
                step_success = False
                log.error(e)
                pass

        # stop timer for simulation step time debugging
        sim_time_elapsed = time.time() - sim_time_start

        # save step_outputs into data_store
        output = {}
        if step_success:
            for idx, name in enumerate(self.names["ext_outputs"]):
                output[name] = (step_outputs[idx] + self.ext_scale[key]["add"]) * self.ext_scale[key]["multiply"]

        return output, step_success, sim_time_elapsed

    def close(self):
        """Close and clean up the environment"""
        self.simulator.close()  # close the FMU
