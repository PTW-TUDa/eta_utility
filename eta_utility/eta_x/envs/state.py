from __future__ import annotations

import pathlib
from csv import DictWriter
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from attrs import asdict, converters, define, field, fields_dict, validators
from gymnasium import spaces

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    from eta_utility.type_hints import Path
from logging import getLogger

log = getLogger(__name__)


def _default_field(default: Any) -> Any:
    """Generate a field that sets the default if the value is None."""
    converter = converters.pipe(converters.default_if_none(default), type(default))
    return field(kw_only=True, default=default, converter=converter)


def _id_field(dtype: type) -> Any:
    validator = validators.optional(validators.instance_of(dtype))
    converter = converters.optional(dtype)
    return field(kw_only=True, default=None, converter=converter, validator=validator)


@define(frozen=True)
class StateVar:
    """A variable in the state of an environment."""

    #: Name of the state variable (This must always be specified).
    name: str = field(validator=validators.instance_of(str))
    #: Should the agent specify actions for this variable? (default: False).
    is_agent_action: bool = _default_field(False)
    #: Should the agent be allowed to observe the value of this variable? (default: False).
    is_agent_observation: bool = _default_field(False)

    #: Should the state log of this episode be added to state_log_longtime? (default: True).
    add_to_state_log: bool = _default_field(True)

    #: Name or identifier (order) of the variable in the external interaction model
    #: (e.g.: environment or FMU) (default: StateVar.name if (is_ext_input or is_ext_output) else None).
    ext_id: str | None = _id_field(str)

    #: Should this variable be passed to the external model as an input? (default: False).
    is_ext_input: bool = _default_field(False)
    #: Should this variable be parsed from the external model output? (default: False).
    is_ext_output: bool = _default_field(False)
    #: Value to add to the output from an external model (default: 0.0).
    ext_scale_add: float = _default_field(0.0)
    #: Value to multiply to the output from an external model (default: 1.0).
    ext_scale_mult: float = _default_field(1.0)

    #: Name or identifier (order) of the variable in an interaction environment (default: None).
    interact_id: int | None = _id_field(int)
    #: Should this variable be read from the interaction environment? (default: False).
    from_interact: bool = _default_field(False)
    #: Value to add to the value read from an interaction (default: 0.0).
    interact_scale_add: float = _default_field(0.0)
    #: Value to multiply to the value read from  an interaction (default: 1.0).
    interact_scale_mult: float = _default_field(1.0)

    #: Name of the scenario variable, this value should be read from (default: None).
    scenario_id: str | None = _id_field(str)
    #: Should this variable be read from imported timeseries date? (default: False).
    from_scenario: bool = _default_field(False)
    #: Value to add to the value read from a scenario file (default: 0.0).
    scenario_scale_add: float = _default_field(0.0)
    #: Value to multiply to the value read from a scenario file (default: 1.0).
    scenario_scale_mult: float = _default_field(1.0)

    #: Lowest possible value of the state variable (default: -np.inf).
    low_value: float = _default_field(-np.inf)
    #: Highest possible value of the state variable (default: np.inf).
    high_value: float = _default_field(np.inf)
    #: If the value of the variable dips below this, the episode should be aborted (default: -np.inf).
    abort_condition_min: float = _default_field(-np.inf)
    #: If the value of the variable rises above this, the episode should be aborted (default: np.inf).
    abort_condition_max: float = _default_field(np.inf)

    #: Determine the index, where to look (useful for mathematical optimization, where multiple time steps could be
    #: returned). In this case, the index values might be different for actions and observations.
    index: int = _default_field(0)

    def __attrs_post_init__(self) -> None:
        if (self.is_ext_input or self.is_ext_output) and self.ext_id is None:
            object.__setattr__(self, "ext_id", self.name)
            log.info(f"Using name as ext_id for variable {self.name}")

        if (not self.from_interact) ^ (self.interact_id is None):
            raise KeyError(f"Variable {self.name} is either missing `interact_id` or `from_interact`.")
        if (not self.from_scenario) ^ (self.scenario_id is None):
            raise KeyError(f"Variable {self.name} is either missing `scenario_id` or `from_scenario`.")

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | pd.Series) -> StateVar:
        """Initialize a state var from a dictionary or pandas Series.

        :param mapping: dictionary or pandas Series to initialize from.
        :return: Initialized StateVar object
        """
        mapping = dict(mapping)
        unrecognized_keys = set(mapping) - set(cls.__annotations__)

        for key in unrecognized_keys:
            log.warning(
                f"Unrecognized key '{key}' with value {mapping.pop(key)} in the"
                "environment state config was not recognized and is ignored."
            )

        return cls(**mapping)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)


class StateConfig:
    """The configuration for the action and observation spaces. The values are used to control which variables are
    part of the action space and observation space. Additionally, the parameters can specify abort conditions
    and the handling of values from interaction environments or from simulation. Therefore, the *StateConfig*
    is very important for the functionality of ETA X.
    """

    def __init__(self, *state_vars: StateVar) -> None:
        #: Mapping of the variables names to their StateVar instance with all associated information.
        self.vars = {var.name: var for var in state_vars}
        self.df_vars: pd.DataFrame = pd.DataFrame([asdict(var) for var in state_vars]).set_index("name")
        if not self.df_vars.index.is_unique:
            log.warning("Duplicate variable names in StateConfig. This may lead to unexpected behavior.")

        #: List of variables that are agent actions.
        self.actions: list[str] = self.df_vars.query("is_agent_action == True").index.tolist()
        #: List of variables that are agent observations.
        self.observations: list[str] = self.df_vars.query("is_agent_observation == True").index.tolist()
        #: Set of variables that should be logged.
        self.add_to_state_log: set[str] = set(self.df_vars.query("add_to_state_log == True").index.tolist())

        #: List of variables that should be provided to an external source (such as an FMU).
        self.ext_inputs: list[str] = self.df_vars.query("is_ext_input == True").index.tolist()
        #: List of variables that can be received from an external source (such as an FMU).
        self.ext_outputs: list[str] = self.df_vars.query("is_ext_output == True").index.tolist()
        #: Mapping of variable names to their external IDs.
        self.map_ext_ids: dict[str, str] = self.df_vars.query("ext_id != None").ext_id.to_dict()
        #: Reverse mapping of external IDs to their corresponding variable names.
        self.rev_ext_ids: dict[str, str] = {v: k for k, v in self.map_ext_ids.items()}

        def scale_dict(_id: Literal["ext", "interact", "scenario"]) -> dict[str, dict[str, float]]:
            # Filter rows by type and select the columns for scaling
            cut_df = self.df_vars.loc[self.df_vars[f"{_id}_id"].notna(), [f"{_id}_scale_add", f"{_id}_scale_mult"]]
            # Rename columns to 'add' and 'mult' and convert to dictionary
            return cut_df.set_axis(["add", "multiply"], axis=1).to_dict(orient="index")

        #: Dictionary of scaling values for external input values (for example from simulations).
        #: Contains fields 'add' and 'multiply'
        self.ext_scale: dict[str, dict[str, float]] = scale_dict("ext")

        #: List of variables that should be read from an interaction environment.
        self.interact_outputs: list[str] = self.df_vars.query("from_interact == True").index.tolist()
        #: Mapping of internal environment names to interact IDs.
        self.map_interact_ids: dict[str, str] = self.df_vars["interact_id"].to_dict()
        #: Dictionary of scaling values for interact values. Contains fields 'add' and 'multiply'.
        self.interact_scale: dict[str, dict[str, float]] = scale_dict("interact")

        #: List of variables which are loaded from scenario files.
        self.scenarios: list[str] = self.df_vars.query("from_scenario == True").index.tolist()
        #: Mapping of internal environment names to scenario IDs.
        self.map_scenario_ids: dict[str, str] = self.df_vars["scenario_id"].to_dict()
        #: Dictionary of scaling values for scenario values. Contains fields 'add' and 'multiply'.
        self.scenario_scale: dict[str, dict[str, float]] = scale_dict("scenario")

        #: List of variables that have minimum values for an abort condition.
        self.abort_conditions_min: list[str] = self.df_vars["abort_condition_min"].dropna().index.tolist()
        #: List of variables that have maximum values for an abort condition.
        self.abort_conditions_max: list[str] = self.df_vars["abort_condition_max"].index.tolist()

    @classmethod
    def from_dict(cls, mapping: Sequence[Mapping[str, Any]] | pd.DataFrame) -> StateConfig:
        """This will convert a potentially incomplete StateConfig DataFrame or a list of dictionaries to the
        standardized StateConfig format. This will ignore any additional columns.

        :param mapping: Mapping to be converted to the StateConfig format.
        :return: StateConfig object.
        """
        _mapping = mapping.to_dict("records") if isinstance(mapping, pd.DataFrame) else mapping

        return cls(*[StateVar.from_dict(col) for col in _mapping])

    def store_file(self, file: Path) -> None:
        """Save the StateConfig to a comma separated file.

        :param file: Path to the file.
        """
        _file = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        _header = fields_dict(StateVar).keys()

        with _file.open("w") as f:
            writer = DictWriter(f, _header, restval="None", delimiter=";")
            writer.writeheader()
            for var in self.vars.values():
                writer.writerow(asdict(var))

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:
        """Check whether the given state is within the abort conditions specified by the StateConfig instance.

        :param state: The state array to check for conformance.
        :return: Result of the check (False if the state does not conform to the required conditions).
        """

        valid_min = all(state[name] >= self.vars[name].abort_condition_min for name in state)
        if not valid_min:
            log.warning("Minimum abort condition exceeded by at least one value.")

        valid_max = all(state[name] <= self.vars[name].abort_condition_max for name in state)
        if not valid_max:
            log.warning("Maximum abort condition exceeded by at least one value.")

        return valid_min and valid_max

    def _generate_continuous_space(self, trait: Literal["is_agent_action", "is_agent_observation"]) -> spaces.Box:
        """Generate a continuous space according to the format required by the OpenAI specification.

        :return: Continuous space.
        """
        low_values = self.df_vars.query(f"{trait} == True").low_value.to_numpy()
        high_values = self.df_vars.query(f"{trait} == True").high_value.to_numpy()

        return spaces.Box(low_values, high_values, dtype=np.float32)

    def continuous_action_space(self) -> spaces.Box:
        """Generate an action space according to the format required by the OpenAI
        specification.

        :return: Action space.
        """
        return self._generate_continuous_space("is_agent_action")

    def continuous_obs_space(self) -> spaces.Box:
        """Generate a continuous observation space according to the format required by the OpenAI
        specification.

        :return: Observation Space.
        """
        return self._generate_continuous_space("is_agent_observation")

    # Alias for continuous_obs_space
    continuous_observation_space = continuous_obs_space

    def continuous_spaces(self) -> tuple[spaces.Box, spaces.Box]:
        """Generate continuous action and observation spaces according to the OpenAI specification.

        :return: Tuple of action space and observation space.
        """
        return self.continuous_action_space(), self.continuous_obs_space()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    @property
    def loc(self) -> pd.api.indexers._LocIndexer:
        """Behave like dataframe (enable indexing via loc) for compatibility."""
        return self.vars
