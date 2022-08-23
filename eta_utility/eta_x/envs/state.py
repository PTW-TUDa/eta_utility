from __future__ import annotations

import pathlib
from csv import DictWriter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from attrs import (  # noqa: I900
    asdict,
    converters,
    define,
    field,
    fields_dict,
    validators,
)
from gym import spaces

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence

    from attrs import Attribute  # noqa: I900

    from eta_utility.type_hints import Path


log = get_logger("eta_x.envs")


def _valid_id(instance: type, attribute: Attribute, value: Any) -> None:
    """Validate an ID attribute.

    :param instance: The attrs class instance to check.
    :param attribute: The attribute to validate.
    :param value: New value of the attribute.
    :raises: ValueError.
    """
    if not isinstance(value, str) and not isinstance(value, int):
        raise ValueError("An ID value must be either of type int or str.")


@define(frozen=True)
class StateVar:
    """A variable in the state of an environment."""

    #: Name of the state variable (This must always be specified).
    name: str = field(validator=validators.instance_of(str))
    #: Should the agent specify actions for this variable? (default: False).
    is_agent_action: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Should the agent be allowed to observe the value of this variable? (default: False).
    is_agent_observation: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )

    #: Name or identifier (order) of the variable in the external interaction model
    #: (e.g.: environment or FMU) (default: None).
    ext_id: str | int | None = field(kw_only=True, default=None, validator=validators.optional(_valid_id))
    #: Should this variable be passed to the external model as an input? (default: False).
    is_ext_input: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Should this variable be parsed from the external model output? (default: False).
    is_ext_output: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Value to add to the output from an external model (default: 0).
    ext_scale_add: float = field(
        kw_only=True,
        default=0,
        converter=converters.pipe(converters.default_if_none(0), float)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Value to multiply to the output from an external model (default: 1).
    ext_scale_mult: float = field(
        kw_only=True,
        default=1,
        converter=converters.pipe(converters.default_if_none(1), float)  # type: ignore
        # mypy does not recognize default_if_none
    )

    #: Name or identifier (order) of the variable in an interaction environment (default: None).
    interact_id: str | int | None = field(kw_only=True, default=None, validator=validators.optional(_valid_id))
    #: Should this variable be read from the interaction environment? (default: False).
    from_interact: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )

    #: Name of the scenario variable, this value should be read from (default: None).
    scenario_id: str | None = field(
        kw_only=True, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Should this variable be read from imported timeseries date? (default: False).
    from_scenario: bool = field(
        kw_only=True,
        default=False,
        converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Value to add to the value read from a scenario file (default: 0).
    scenario_scale_add: float = field(
        kw_only=True,
        default=0,
        converter=converters.pipe(converters.default_if_none(0), float)  # type: ignore
        # mypy does not recognize default_if_none
    )
    #: Value to multiply to the value read from a scenario file (default: 1).
    scenario_scale_mult: float = field(
        kw_only=True,
        default=1,
        converter=converters.pipe(converters.default_if_none(1), float)  # type: ignore
        # mypy does not recognize default_if_none
    )

    #: Lowest possible value of the state variable (default: None).
    low_value: float | None = field(kw_only=True, default=np.nan, converter=converters.optional(float))
    #: Highest possible value of the state variable (default: None).
    high_value: float | None = field(kw_only=True, default=np.nan, converter=converters.optional(float))
    #: If the value of the variable dips below this, the episode should be aborted (default: None).
    abort_condition_min: float | None = field(kw_only=True, default=None, converter=converters.optional(float))
    #: If the value of the variable rises above this, the episode should be aborted (default: None).
    abort_condition_max: float | None = field(kw_only=True, default=None, converter=converters.optional(float))

    #: Determine the index, where to look (useful for mathematical optimization, where multiple time steps could be
    #: returned). In this case, the index values might be different for actions and observations.
    index: int = field(
        kw_only=True,
        default=0,
        converter=converters.pipe(converters.default_if_none(0), int)  # type: ignore
        # mypy does not recognize default_if_none
    )

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any] | pd.Series) -> StateVar:
        """Initialize a state var from a dictionary or pandas Series.

        :param mapping: dictionary or pandas Series to initialize from.
        :return: Initialized StateVar object
        """
        _map = dict(mapping)

        if "name" not in _map:
            raise ValueError("Name is not specified for a variable in the environment state config.")
        name = _map.pop("name", None)
        is_agent_action = _map.pop("is_agent_action", None)
        is_agent_observation = _map.pop("is_agent_observation", None)
        ext_id = _map.pop("ext_id", None)
        is_ext_input = _map.pop("is_ext_input", None)
        is_ext_output = _map.pop("is_ext_output", None)
        ext_scale_add = _map.pop("ext_scale_add", None)
        ext_scale_mult = _map.pop("ext_scale_mult", None)
        scenario_id = _map.pop("scenario_id", None)
        from_scenario = _map.pop("from_scenario", None)
        scenario_scale_add = _map.pop("scenario_scale_add", None)
        scenario_scale_mult = _map.pop("scenario_scale_mult", None)
        low_value = _map.pop("low_value", np.nan)
        high_value = _map.pop("high_value", np.nan)
        abort_condition_min = _map.pop("abort_condition_min", None)
        abort_condition_max = _map.pop("abort_condition_max", None)
        index = _map.pop("index", None)

        for name in _map:
            log.warning(f"Specified value '{name}' in the environment state config was not recognized and is ignored.")

        return cls(
            name,
            is_agent_action=is_agent_action,
            is_agent_observation=is_agent_observation,
            ext_id=ext_id,
            is_ext_input=is_ext_input,
            is_ext_output=is_ext_output,
            ext_scale_add=ext_scale_add,
            ext_scale_mult=ext_scale_mult,
            scenario_id=scenario_id,
            from_scenario=from_scenario,
            scenario_scale_add=scenario_scale_add,
            scenario_scale_mult=scenario_scale_mult,
            low_value=low_value,
            high_value=high_value,
            abort_condition_min=abort_condition_min,
            abort_condition_max=abort_condition_max,
            index=index,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: StateVar) -> None:
        if not hasattr(self, name):
            raise KeyError(f"The key {name} does not exist - it cannot be set.")
        setattr(self, name, value)


class StateConfig:
    """The configuration for the action and observation spaces. The values are used to control which variables are
    part of the action space and observation space. Additionally, the parameters can specify abort conditions
    and the handling of values from interaction environments or from simulation. Therefore, the *StateConfig*
    is very important for the functionality of ETA X.
    """

    def __init__(self, *state_vars: StateVar) -> None:
        #: Mapping of the variables names to their StateVar instance with all associated information.
        self.vars: dict[str, StateVar] = {}

        #: List of variables that are agent actions.
        self.actions: list[str] = []
        #: List of variables that are agent observations.
        self.observations: list[str] = []

        #: List of variables that should be provided to an external source (such as an FMU).
        self.ext_inputs: list[str] = []
        #: List of variables that can be received from an external source (such as an FMU).
        self.ext_outputs: list[str] = []
        #: Mapping of variable names to their external IDs.
        self.map_ext_ids: dict[str, str | int] = {}
        #: Reverse mapping of external IDs to their corresponding variable names.
        self.rev_ext_ids: dict[str | int, str] = {}
        #: Dictionary of scaling values for external input values (for example from simulations).
        #: Contains fields 'add' and 'multiply'
        self.ext_scale: dict[str, dict[str, float]] = {}

        #: List of variables that should be read from an interaction environment.
        self.interact_outputs: list[str] = []

        #: List of variables which are loaded from scenario files.
        self.scenarios: list[str] = []
        #: Mapping of internal environment names to scenario IDs.
        self.map_scenario_ids: dict[str, str] = {}
        #: Dictionary of scaling values for scenario values. Contains fields 'add' and 'multiply'.
        self.scenario_scale: dict[str, dict[str, float]] = {}

        #: List of variables that have minimum values for an abort condition.
        self.abort_conditions_min: list[str] = []
        #: List of variables that have maximum values for an abort condition.
        self.abort_conditions_max: list[str] = []

        # Initialize variable mappings
        for var in state_vars:
            self.append_state(var)

    @classmethod
    def from_dict(cls, mapping: Sequence[Mapping[str, Any]] | pd.DataFrame) -> StateConfig:
        """This will convert a potentially incomplete StateConfig DataFrame or a list of dictionaries to the
        standardized StateConfig format. This will ignore any additional columns.

        :param mapping: Mapping to be converted to the StateConfig format.
        :return: StateConfig object.
        """
        state_vars = []
        _mapping = mapping.to_dict("records") if isinstance(mapping, pd.DataFrame) else mapping

        for col in _mapping:
            state_vars.append(StateVar.from_dict(col))

        return cls(*state_vars)

    def append_state(self, var: StateVar) -> None:
        """Append a state variable to the state configuration.

        :param var: StateVar instance to append to the configuration.
        """
        self.vars[var.name] = var

        if var.is_agent_action:
            self.actions.append(var.name)
        if var.is_agent_observation:
            self.observations.append(var.name)

        if var.abort_condition_min is not None:
            self.abort_conditions_min.append(var.name)
        if var.abort_condition_max is not None:
            self.abort_conditions_max.append(var.name)

        # Mappings for external input and output variables
        if var.is_ext_input or var.is_ext_output:
            if var.is_ext_input:
                self.ext_inputs.append(var.name)
            if var.is_ext_output:
                self.ext_outputs.append(var.name)

            if var.ext_id is None:
                raise ValueError(
                    f"Variable {var.name} is specified as external input or output but does not specify an ext_id."
                )
            self.map_ext_ids[var.name] = var.ext_id
            self.rev_ext_ids[var.ext_id] = var.name
            self.ext_scale[var.name] = {"add": var.ext_scale_add, "multiply": var.ext_scale_mult}

        if var.from_interact:
            self.interact_outputs.append(var.name)

        # Mappings for scenario variables
        if var.from_scenario and var.scenario_id is not None:
            self.scenarios.append(var.name)
            self.map_scenario_ids[var.name] = var.scenario_id
            self.scenario_scale[var.name] = {"add": var.scenario_scale_add, "multiply": var.scenario_scale_mult}

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

        valid_min = all(
            state[name] >= self.vars[name].abort_condition_min for name in self.abort_conditions_min  # type: ignore
        )
        if not valid_min:
            log.warning("Minimum abort condition exceeded by at least one value.")

        valid_max = all(
            state[name] <= self.vars[name].abort_condition_max for name in self.abort_conditions_max  # type: ignore
        )
        if not valid_max:
            log.warning("Maximum abort condition exceeded by at least one value.")

        return valid_min and valid_max

    def continuous_action_space(self) -> spaces.Box:
        """Generate an action space according to the format required by the OpenAI
        specification.

        :return: Action space.
        """
        action_low = np.fromiter(
            (var.low_value for var in self.vars.values() if var.is_agent_action and var.low_value is not None),
            dtype=np.float32,
        )
        action_high = np.fromiter(
            (var.high_value for var in self.vars.values() if var.is_agent_action and var.high_value is not None),
            dtype=np.float32,
        )

        return spaces.Box(action_low, action_high, dtype=np.float32)

    def continuous_obs_space(self) -> spaces.Box:
        """Generate a continuous observation space according to the format required by the OpenAI
        specification.

        :return: Observation Space.
        """
        obs_low = np.fromiter(
            (var.low_value for var in self.vars.values() if var.is_agent_observation and var.low_value is not None),
            dtype=np.float32,
        )

        obs_high = np.fromiter(
            (var.high_value for var in self.vars.values() if var.is_agent_observation and var.high_value is not None),
            dtype=np.float32,
        )

        return spaces.Box(obs_low, obs_high, dtype=np.float32)

    def continuous_spaces(self) -> tuple[spaces.Box, spaces.Box]:
        """Generate continuous action and observation spaces according to the OpenAI specification.

        :return: Tuple of action space and observation space.
        """
        return self.continuous_action_space(), self.continuous_obs_space()

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: StateVar) -> None:
        setattr(self, name, value)

    @property
    def loc(self) -> dict[str, StateVar]:
        """Behave like dataframe (enable indexing via loc) for compatibility."""
        return self.vars
