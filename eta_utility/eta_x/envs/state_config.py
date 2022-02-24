from collections import OrderedDict, Sequence
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd


class StateConfig(pd.DataFrame):

    # Columns (and their order) and default values for the state_config DataFrame.
    _state_config_cols: OrderedDict = OrderedDict(
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
    #:   * name: str, Name of the state variable (This must always be specified (no default))
    #:   * is_agent_action: bool, Should the agent specify actions for this variable? (default: False)
    #:   * is_agent_observation: bool, Should the agent be allowed to observe the value
    #:     of this variable? (default: False)
    #:   * ext_id: str, Name or identifier (order) of the variable in the external interaction model
    #:     (e.g.: environment or FMU) (default: None)
    #:   * is_ext_input: bool, Should this variable be passed to the external model as an input? (default: False)
    #:   * is_ext_output: bool, Should this variable be parsed from the external model output? (default: False)
    #:   * ext_scale_add: int or float, Value to add to the output from an external model (default: 0)
    #:   * ext_scale_mult: int or float, Value to multiply to the output from an external model (default: 1)
    #:   * from_scenario: bool, Should this variable be read from imported timeseries date? (default: False)
    #:   * scenario_id: str, Name of the scenario variable, this value should be read from (default: None)
    #:   * low_value: int or float, lowest possible value of the state variable (default: None)
    #:   * high_value: int or float, highest possible value of the state variable (default: None)
    #:   * abort_condition_min: int or float, If value of variable dips below this, the episode
    #:     will be aborted (default: None)
    #:   * abort_condition_max: int or float, If value of variable rises above this, the episode
    #:     will be aborted (default: None)
    #:   * index: int, Specify, which Index this value should be read from, in case a list of values is returned.
    #:     (default: 0)

    def __init__(self, dataframe: pd.DataFrame = None) -> None:
        if dataframe is None:
            super().__init__(columns=self._state_config_cols.keys())
        else:
            super().__init__(data=dataframe)
            self._convert_state_config()
        if "name" in self.columns:
            self.set_index("name", drop=True, inplace=True)

    @property
    def _constructor(self) -> "StateConfig":
        return StateConfig

    @classmethod
    def convert_state_config(cls, state_config) -> "StateConfig":
        """This will convert an incomplete state_config DataFrame or a list of dictionaries to the standardized
        DataFrame format. This will remove any additional columns. If additional columns are required, ensure
        consistency with the required format otherwise.

        :return: Converted, standardized dataframe
        """
        # If state config is a DataFrame already, check whether the columns correspond. If they don't create a new
        # DataFrame with the correct columns and default values for missing columns
        if isinstance(state_config, StateConfig):
            new_state = StateConfig(cls._state_config_cols.keys())
            for col, default in cls._state_config_cols.items():
                if col in state_config.columns:
                    new_state[col] = state_config[col]
                elif col == "name" and col not in state_config.columns:
                    new_state["name"] = state_config.index
                else:
                    new_state[col] = np.array([default] * len(state_config.index))

            # Fill empty cells (only do this for values, where the default is not None)
            new_state.fillna(
                value={key: val for key, val in cls._state_config_cols.items() if val is not None},
                inplace=True,
            )

        # If state config is a list of dictionaries iterate the list and create the DataFrame iteratively
        elif isinstance(state_config, Sequence):
            new_state = []
            for row in state_config:
                new_row = {}
                for col, default in cls._state_config_cols.items():
                    new_row[col] = row[col] if col in row else default
                new_state.append(new_row)
            new_state = pd.DataFrame(data=new_state, columns=cls._state_config_cols.keys())
            new_state = StateConfig(new_state)
        else:
            raise ValueError(
                "state_config is not in the correct format. It should be a DataFrame or a list "
                "of dictionaries. It is currently {}".format(type(state_config))
            )

        state_config = new_state
        return state_config

    def append_state(self, *, name: Any, **kwargs) -> "StateConfig":
        """Append a state variable to the state configuration of the environment

        :param name: Name of the state variable
        :param kwargs: Column names and values to be inserted into the respective column. For possible columns, types
                       and default values see state_config. See also: :func:`state_config`
        """
        append = {}
        for key, item in self._state_config_cols.items():
            # Since name is supplied separately, don't append it here
            if key == "name":
                continue

            val = kwargs[key] if key in kwargs else item
            append[key] = val

        append = pd.Series(append, name=name)
        state_config = self.append(append, sort=True)
        return state_config

    def names_from_state(
        self,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Union[int, float]]], Dict[str, str], Dict[str, str]]:
        """Intialize the names array from state_config, which stores shorthands to some frequently used variable names.
        Also initialize some useful shorthand mappings that can be used to speed up lookups.

        The names array contains the following (ordered) lists of variables:
            * actions: Variables that are agent actions
            * observations: Variables that are agent observations
            * ext_inputs: Variables that should be provided to an external source (such as an FMU)
            * ext_output: variables that can be received from an external source (such as an FMU)
            * abort_conditions_min: Variables that have minimum values for an abort condition
            * abort_conditions_max: Variables that have maximum values for an abort condition

        self.ext_scale is a dictionary of scaling values for external input values (for example from simulations).

        self.map_ext_ids is a mapping of internatl environment names to external ids.

        self.map_scenario_ids is a mapping of interal environment names to scenario ids.
        """
        names = {
            "actions": self.loc[self.is_agent_action == True].index.values,  # noqa: E712
            "observations": self.loc[self.is_agent_observation == True].index.values,
            "ext_inputs": self.loc[self.is_ext_input == True].index.values,
            "ext_outputs": self.loc[self.is_ext_output == True].index.values,
            "scenario": self.loc[self.from_scenario == True].index.values,
            "abort_conditions_min": self.loc[self.abort_condition_min.notnull()].index.values,
            "abort_conditions_max": self.loc[self.abort_condition_max.notnull()].index.values,
        }

        ext_scale = {}
        for name, values in self.iterrows():
            ext_scale[name] = {"add": values.ext_scale_add, "multiply": values.ext_scale_mult}

        map_ext_ids = {}
        for name in set(names["ext_inputs"]) | set(names["ext_outputs"]):
            map_ext_ids[name] = self.loc[name].ext_id

        map_scenario_ids = {}
        for name in names["scenario"]:
            map_scenario_ids[name] = self.loc[name].scenario_id

        return names, ext_scale, map_ext_ids, map_scenario_ids

    def store_state_info(self, env_id, path_results, run_name) -> None:
        """Save state_config to csv for info (only first environment)"""
        if env_id == 1:
            self.to_csv(
                path_or_buf=path_results / (run_name + "_state_config.csv"),
                sep=";",
                decimal=",",
            )
