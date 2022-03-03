from collections import OrderedDict, Sequence
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd

from typing import Callable, Any, Optional
from datetime import timedelta
from pydantic import validator, StrictStr, StrictInt, StrictBool, NonNegativeInt, NegativeInt
from pydantic.dataclasses import dataclass
from ..types import ConstantParameter, Numeric, BackendID


@dataclass
class MetaDataEntry:
    """Parent class with basic attributes and validation for meta data related entries."""

    allowed_names = ()

    name: StrictStr
    unit: Optional[StrictStr] = None
    index: Optional[StrictStr] = None
    default: Optional[ConstantParameter] = None
    backend_id: Optional[BackendID] = None
    csv_column: Optional[StrictStr] = None

    # Controls if data for corresponding parameter is loaded during parameter estimation.
    # If the data is not available but this is True (default), an error is raised.
    fix_for_model_error_or_paramest: StrictBool = True

    @validator("name")
    def check_var_name(cls, value):
        """Validate the given name is allowed. Allowed names are defined as child class attribute."""
        if cls.allowed_names:
            if value not in cls.allowed_names:
                raise ValueError(f"Name needs to be on of: {cls.allowed_names}, was '{value}'")

        return value

    def construct_pyomo_index(self, timesteps: list):
        """construct the index, under which the pyomo parameter can be called
        args:
        """
        if isinstance(self.index, tuple):
            raise NotImplementedError("Tuple indexing is not supported!")

        return timesteps if self.index is None else [(self.index, t) for t in timesteps]


@dataclass
class Measurement(MetaDataEntry):
    """Historical measurement data"""


@dataclass
class Action(MetaDataEntry):
    """Decision variable, the value of which will be set on the real machine."""

    allowed_names = ("P", "P_set", "ON", "Temp_out", "Pump_spray")

    # Timestep of the decision variable to be written to machine.
    # Possible use case: Customer wants his machines to be set with a 1 hour delay.
    timestep: NonNegativeInt = 0

    # Value to be written in case of optimizer failure
    fall_back: Numeric = None

    def construct_pyomo_index(self):
        return super().construct_pyomo_index([self.timestep])


@dataclass
class Observation(MetaDataEntry):
    """An observation is a measurement of the state of one aspect of the energy system
    (e.g.: operating state (ON/OFF), current thermal power output, ...).
    An observations corresponds to a parameter in the optimization model.
    You can also see an observation as a start value.
    """

    allowed_names = ("P_set_start", "ON_start", "Temp_start", "power_peak_start")

    # How many time steps back should the measurement be, which is taken as observation?:
    timestep: NegativeInt = -1
    conversion_rule: Callable = None

    def construct_pyomo_index(self):
        return super().construct_pyomo_index([self.timestep])

    def to_param(self):
        if self.default is None:
            raise ValueError(f"{self.unit}: Observation '{self.name}' needs a default value")

        if self.index:
            return {(self.index, self.timestep): self.default}
        return {self.timestep: self.default}


@dataclass
class Forecast(MetaDataEntry):
    """
    A forecast is a timeseries representing a prediction of a time indexed parameter
    (e.g. environmental temperature, heat demand, elecricity price, ...).
    The forecast corresponds to a parameter in the optimization model.
    """

    allowed_names = (
        "el_cost",
        "cost_fuel",
        "Temp_env",
        "Temp_max",
        "Temp_max_wanted",
        "Temp_min",
        "Temp_min_wanted",
        "demand",
        "Temp_in",
        "Temp_out",
        "massflow",
        "rel_humidity",
    )

    default: ConstantParameter = None
    conversion_rule: Callable[[int], int] = None

    # If shift is None, the series of the forecast is expected to be inside the optimization horizon.
    # To use the most recent point of the series set shift = pd.Timedelta("0 minutes")
    # To take a value further in the past, set to a negative value. e.g: shift = - pd.Timedelta("5 minutes")
    shift: Optional[timedelta] = None

    @validator("shift", pre=True)
    def check_shift(cls, value):
        """Validate that shift is a negative timedelta."""
        if value is None:
            pass
        elif not isinstance(value, timedelta):
            raise TypeError("Use a datetime.timedelta for shift.")
        elif value > timedelta():
            raise ValueError("Shift needs to be negative.")

        return value

    def to_param(self, timesteps: list, data=None):
        """Convert the Forecast to an entry usable in the param dictionary."""
        if data is not None:
            values = [data[t] for t in timesteps]
        elif isinstance(self.default, dict):
            try:
                values = [self.default[t] for t in timesteps]
            except KeyError as err:
                raise ValueError(
                    f"Default for forecast '{self.name}' is not specified for all time steps."
                    "Either pass a dictionary with values for each step in global.T or a scalar value."
                ) from err
        else:
            values = [self.default for t in timesteps]

        return dict(zip(self.construct_pyomo_index(timesteps), values))


@dataclass
class ComponentMetaData:
    """Collection of data exchange related attributes"""

    observations: list[Observation] = None
    actions: list[Action] = None
    forecasts: list[Forecast] = None
    measurements: list[Measurement] = None

    component_schema: Any = None
    element_id: StrictInt = None

    def get(self, name, index=None):
        for entry_type in (self.observations, self.actions, self.forecasts, self.measurements):
            if entry_type is None:
                continue
            for entry in entry_type:
                if entry.name == name and entry.index == index:
                    return entry

        raise KeyError(f"No entry found with name={name}, index={index}")


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
