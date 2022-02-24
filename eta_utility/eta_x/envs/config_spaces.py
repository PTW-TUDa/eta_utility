from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from gym import spaces


class ConfigSpaces:
    """
    The configuration for the action and observation spaces. The values are used to control which variables are
    part of the action space and observation space. Additionally the parameters can specify abort conditions
    and the handling of values from interaction environments or from simulation. Therefore the config_spaces
    is very important for the functionality of ETA X.
    """

    def __init__(self) -> None:
        # Columns (and their order) and default values for the config_spaces DataFrame.
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
                ("low_value", None),
                ("high_value", None),
                ("abort_condition_min", None),
                ("abort_condition_max", None),
                ("index", 0),
            ]
        )

        # Possible column names, their types and default values are:
        # **name**: str, Name of the state variable (This must always be specified (no default)), names column
        #     becomes index in DataFrame
        # **is_agent_action**: bool, Should the agent specify actions for this variable? (default: False)
        # **is_agent_observation**: bool, Should the agent be allowed to observe the value
        #     of this variable? (default: False)
        # **ext_id**: str, Name or identifier (order) of the variable in the external interaction model
        #     (e.g.: environment or FMU) (default: None)
        # **is_ext_input**: bool, Should this variable be passed to the external model as an input?
        #     (default: False)
        # **is_ext_output**: bool, Should this variable be parsed from the external model output? (default: False)
        # **ext_scale_add**: int or float, Value to add to the output from an external model (default: 0)
        # **ext_scale_mult**: int or float, Value to multiply to the output from an external model (default: 1)
        # **from_scenario**: bool, Should this variable be read from imported timeseries date? (default: False)
        # **scenario_id**: str, Name of the scenario variable, this value should be read from (default: None)
        # **low_value**: int or float, lowest possible value of the state variable (default: None)
        # **high_value**: int or float, highest possible value of the state variable (default: None)
        # **abort_condition_min**: int or float, If value of variable dips below this, the episode
        #     will be aborted (default: None)
        # **abort_condition_max**: int or float, If value of variable rises above this, the episode
        #     will be aborted (default: None)
        # **index**: int, Specify, which Index this value should be read from, in case a list of values is
        #     returned (default: 0)
        # *State_config* can also be specified as a list of dictionaries if many default values are set:
        # .. note ::
        # config_spaces = pd.DataFrame([{name:___, ext_id:___, ...}, {name:___, ext_id:___, ...}])

        self.config_spaces: pd.DataFrame = pd.DataFrame(columns=self.__state_config_cols.keys())
        self.config_spaces.set_index("name", drop=True, inplace=True)
        #: Array of shorthands to some frequently used variable names from config_spaces.
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
        self.names: Dict[str, np.ndarray]
        #: Dictionary of scaling values for external input values (for example from simulations).
        #:  The structure of this dictionary is {"name": {"add": value, "multiply": value}}.
        self.ext_scale: Dict[str, Dict[str, Union[int, float]]]
        #: Mapping of internal environment names to external ids.
        self.map_ext_ids: Dict[str, str]
        #: Mapping of external ids to internal environment names.
        self.rev_ext_ids: Dict[str, str]
        #: Mapping of internal environment names to scenario ids.
        self.map_scenario_ids: Dict[str, str]

        # TODO
        # put stuff here
        # data and set_spaces
        # remove "get_" from methods

    def set_spaces(self, env_id: int, path_results: Path, run_name: str, data: List[Dict]) -> "ConfigSpaces":
        """Set config spaces and store state information."""
        self.config_spaces = data
        self._convert_state_config()
        self._names_from_state()
        self._store_state_info(env_id, path_results, run_name)
        return self

    def get_continuous_spaces(self) -> Tuple[spaces.Space, spaces.Box]:
        return (self._continuous_action_space(), self._continuous_obs_space())

    def _convert_state_config(self) -> pd.DataFrame:
        """This will convert an incomplete config_spaces DataFrame or a list of dictionaries to the standardized
        DataFrame format. This will remove any additional columns. If additional columns are required, ensure
        consistency with the required format otherwise.

        :return: Converted, standardized dataframe
        """
        # If state config is a DataFrame already, check whether the columns correspond. If they don't create a new
        # DataFrame with the correct columns and default values for missing columns
        if isinstance(self.config_spaces, pd.DataFrame):
            new_state = pd.DataFrame(columns=self.__state_config_cols.keys())
            for col, default in self.__state_config_cols.items():
                if col in self.config_spaces.columns:
                    new_state[col] = self.config_spaces[col]
                elif col == "name" and col not in self.config_spaces.columns:
                    new_state["name"] = self.config_spaces.index
                else:
                    new_state[col] = np.array([default] * len(self.config_spaces.index))

            # Fill empty cells (only do this for values, where the default is not None)
            new_state.fillna(
                value={key: val for key, val in self.__state_config_cols.items() if val is not None},
                inplace=True,
            )

        # If state config is a list of dictionaries iterate the list and create the DataFrame iteratively
        elif isinstance(self.config_spaces, Sequence):
            new_state = []
            for row in self.config_spaces:
                new_row = {}
                for col, default in self.__state_config_cols.items():
                    new_row[col] = row[col] if col in row else default
                new_state.append(new_row)
            new_state = pd.DataFrame(data=new_state, columns=self.__state_config_cols.keys())
        else:
            raise ValueError(
                "config_spaces is not in the correct format. It should be a DataFrame or a list "
                "of dictionaries. It is currently {}".format(type(self.config_spaces))
            )

        new_state.set_index("name", inplace=True, verify_integrity=True)

        self.config_spaces = new_state
        return self.config_spaces

    def _names_from_state(self) -> None:
        """Intialize the names array from config_spaces, which stores shorthands to some frequently used variable names.
        Also initialize some useful shorthand mappings that can be used to speed up lookups.

        The names array contains the following (ordered) lists of variables in a dictionary:

        **actions**: Variables that are agent actions
        **observations**: Variables that are agent observations
        **ext_inputs**: Variables that should be provided to an external source (such as an FMU)
        **ext_output**: variables that can be received from an external source (such as an FMU)
        **abort_conditions_min**: Variables that have minimum values for an abort condition
        **abort_conditions_max**: Variables that have maximum values for an abort condition

        *self.ext_scale* is a dictionary of scaling values for external input values (for example from simulations).

        *self.map_ext_ids* is a mapping of internal environment names to external ids.

        *self.rev_ext_ids* is a mapping of external ids to internal environment names.

        *self.map_scenario_ids* is a mapping of internal environment names to scenario ids.
        """
        self.names = {
            "actions": self.config_spaces.loc[self.config_spaces.is_agent_action].index.values,  # noqa: E712
            "observations": self.config_spaces.loc[self.config_spaces.is_agent_observation].index.values,
            "ext_inputs": self.config_spaces.loc[self.config_spaces.is_ext_input].index.values,  # noqa: E712
            "ext_outputs": self.config_spaces.loc[self.config_spaces.is_ext_output].index.values,  # noqa: E712
            "scenario": self.config_spaces.loc[self.config_spaces.from_scenario].index.values,  # noqa: E712
            "abort_conditions_min": self.config_spaces.loc[
                self.config_spaces.abort_condition_min.notnull()
            ].index.values,
            "abort_conditions_max": self.config_spaces.loc[
                self.config_spaces.abort_condition_max.notnull()
            ].index.values,
        }

        self.ext_scale = {}
        for name, values in self.config_spaces.iterrows():
            self.ext_scale[name] = {"add": values.ext_scale_add, "multiply": values.ext_scale_mult}

        self.map_ext_ids = {}
        for name in set(self.names["ext_inputs"]) | set(self.names["ext_outputs"]):
            self.map_ext_ids[name] = self.config_spaces.loc[name].ext_id

        self.rev_ext_ids = {}
        for name in set(self.names["ext_inputs"]) | set(self.names["ext_outputs"]):
            self.rev_ext_ids[self.config_spaces.loc[name].ext_id] = name

        self.map_scenario_ids = {}
        for name in self.names["scenario"]:
            self.map_scenario_ids[name] = self.config_spaces.loc[name].scenario_id

    def _store_state_info(self, env_id: int, path_results: Path, run_name: str) -> None:
        """Save config_spaces to csv for info (only first environment)"""
        if env_id == 1:
            self.config_spaces.to_csv(
                path_or_buf=path_results / (run_name + "_state_config.csv"),
                sep=";",
                decimal=",",
            )

    def _continuous_action_space(self) -> spaces.Space:
        """Use the config_spaces to generate the action space according to the format required by the OpenAI
        specification. This will set the action_space attribute and return the corresponding space object.
        The generated action space is continous.

        :return: Action space
        """
        action_low = self.config_spaces.loc[self.config_spaces.is_agent_action].low_value.values  # noqa: E712
        action_high = self.config_spaces.loc[self.config_spaces.is_agent_action].high_value.values  # noqa: E712
        self.action_space = spaces.Box(action_low, action_high, dtype=np.float)

        return self.action_space

    def _continuous_obs_space(self) -> spaces.Box:
        """Use the config_spaces to generate the observation space according to the format required by the OpenAI
        specification. This will set the observation_space attribute and return the corresponding space object.
        The generated observation space is continous.

        :return: Observation Space
        """
        state_low = self.config_spaces.loc[self.config_spaces.is_agent_observation].low_value.values  # noqa: E712
        state_high = self.config_spaces.loc[
            self.config_spaces.is_agent_observation  # noqa: E712
        ].high_value.values  # noqa: E712
        self.observation_space = spaces.Box(state_low, state_high, dtype=np.float)

        return self.observation_space

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:
        """Check whether the given state is within the abort conditions specified by config_spaces.

        :param state: The state array to check for conformance
        :return: Result of the check (False if the state does not conform to the required conditions)
        """
        valid = all(
            state[key] > val
            for key, val in self.config_spaces.loc[self.names["abort_conditions_min"]].abort_condition_min.items()
        )
        if valid:
            valid = all(
                state[key] < val
                for key, val in self.config_spaces.loc[self.names["abort_conditions_max"]].abort_condition_max.items()
            )

        return valid

    def append_state(self, *, name: Any, **kwargs: Dict[str, Any]) -> None:
        """Append a state variable to the state configuration of the environment

        :param name: Name of the state variable
        :param kwargs: Column names and values to be inserted into the respective column. For possible columns, types
                       and default values see config_spaces. See also: :func:`config_spaces`
        """
        append = {}
        for key, item in self.__state_config_cols.items():
            # Since name is supplied separately, don't append it here
            if key == "name":
                continue

            val = kwargs[key] if key in kwargs else item
            append[key] = val

        append = pd.Series(append, name=name)
        self.config_spaces = self.config_spaces.append(append, sort=True)
