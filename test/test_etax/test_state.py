import numpy as np
import pytest
from gymnasium.spaces.box import Box

from eta_utility.eta_x.envs.state import StateConfig, StateVar


class TestStateVar:
    @pytest.fixture(scope="class")
    def state_var_default(self):
        return StateVar("foo")

    @pytest.fixture(scope="class")
    def state_var_ext(self):
        return StateVar("foo", is_ext_input=True)

    def test_defaults(self, state_var_default: StateVar):
        assert state_var_default["name"] == "foo"
        assert state_var_default.is_agent_action is False
        assert state_var_default.is_agent_observation is False
        assert state_var_default.is_ext_input is False
        assert state_var_default.low_value == -np.inf
        assert state_var_default.high_value == np.inf
        assert state_var_default.ext_id is None
        assert state_var_default.abort_condition_min == -np.inf
        assert state_var_default.abort_condition_max == np.inf

    @pytest.mark.parametrize(
        ("name"),
        [
            pytest.param("interact", id="interact"),
            pytest.param("scenario", id="scenario"),
        ],
    )
    def test_missing_params(self, name):
        msg = f"Variable {name} is either missing `{name}_id` or `from_{name}`"
        with pytest.raises(KeyError, match=msg):
            StateVar(name, **{f"from_{name}": True})

    @pytest.fixture(scope="class")
    def state_var_scenario(self):
        return StateVar(
            "foo",
            is_agent_action=True,
            low_value=0,
            high_value=1,
            from_scenario=True,
            is_ext_input=True,
            scenario_id="0",
            scenario_scale_add=1,
            scenario_scale_mult=2,
        )

    def test_scenario_var(self, state_var_scenario):
        assert state_var_scenario.name == "foo"
        assert state_var_scenario.is_agent_action is True
        assert state_var_scenario.low_value == 0
        assert state_var_scenario.high_value == 1
        assert state_var_scenario.from_scenario is True
        assert state_var_scenario.is_ext_input is True
        assert state_var_scenario.scenario_id == "0"
        assert state_var_scenario.scenario_scale_add == 1.0
        assert state_var_scenario.scenario_scale_mult == 2.0

    def test_state_var_ext_id_should_be_name_by_default(self, state_var_scenario):
        assert state_var_scenario.ext_id == state_var_scenario.name

    @pytest.fixture(scope="class")
    def state_var_interact(self):
        return StateVar(
            "foo",
            is_agent_action=True,
            low_value=0,
            high_value=1,
            from_interact=True,
            interact_id=0,
            interact_scale_add=1,
            interact_scale_mult=2,
        )

    def test_interact_var(self, state_var_interact):
        assert state_var_interact.name == "foo"
        assert state_var_interact.is_agent_action is True
        assert state_var_interact.low_value == 0
        assert state_var_interact.high_value == 1
        assert state_var_interact.from_interact is True
        assert state_var_interact.interact_id == 0
        assert state_var_interact.interact_scale_add == 1.0
        assert state_var_interact.interact_scale_mult == 2.0

    def test_from_dict(self):
        dikt = {"name": "foo", "is_agent_action": True, "foo": "bar"}
        state_var = StateVar.from_dict(dikt)
        assert state_var.name == "foo"
        assert state_var.is_agent_action is True


class TestStateConfig:
    @pytest.fixture(scope="class")
    def state_config(self):
        return StateConfig(
            StateVar(
                "action1",
                is_agent_action=True,
                low_value=0,
                high_value=1,
                abort_condition_min=0,
                abort_condition_max=0.5,
            ),
            StateVar("observation1", is_agent_observation=True, low_value=2, high_value=3),
            StateVar(
                "action2",
                is_agent_action=True,
                low_value=4,
                high_value=5,
                abort_condition_min=4,
                abort_condition_max=4.5,
            ),
            StateVar("observation2", is_agent_observation=True, low_value=6, high_value=7),
        )

    @pytest.fixture(scope="class")
    def state_config_nan(self):
        return StateConfig(
            StateVar("action1", is_agent_action=True),
            StateVar("observation1", is_agent_observation=True),
        )

    def test_continuous_action_space_should_include_all_and_only_agent_actions(self, state_config):
        # also tests: continuous_action_space_should_span_from_low_to_high_value_for_every_statevar
        foo = state_config.continuous_action_space()
        assert foo == Box(low=np.array([0, 4]), high=np.array([1, 5], dtype=np.float32))

    def test_continuous_action_space_should_use_infinity_if_no_low_and_high_values_are_provided(self, state_config_nan):
        foo = state_config_nan.continuous_action_space()
        assert foo == Box(low=np.array([-np.inf]), high=np.array([np.inf], dtype=np.float32))

    def test_continuous_observation_space_should_include_all_and_only_agent_observations(self, state_config):
        # also tests: continuous_observation_space_should_span_from_low_to_high_value_for_every_statevar
        foo = state_config.continuous_observation_space()
        assert foo == Box(low=np.array([2, 6]), high=np.array([3, 7], dtype=np.float32))

    def test_continuous_observation_space_should_use_infinity_if_no_low_and_high_values_are_provided(
        self, state_config_nan
    ):
        foo = state_config_nan.continuous_observation_space()
        assert foo == Box(low=np.array([-np.inf]), high=np.array([np.inf], dtype=np.float32))

    def test_continuous_spaces(self, state_config):
        action_space, observation_space = state_config.continuous_spaces()
        assert action_space == state_config.continuous_action_space()
        assert observation_space == state_config.continuous_observation_space()

    def test_from_dict(self):
        state_vars = [
            {"name": "action1", "is_agent_action": True, "foo": "bar"},
            {"name": "action2", "is_agent_observation": True, "foo": "bar"},
        ]
        StateConfig.from_dict(state_vars)

    @pytest.mark.parametrize(
        ("vals", "truth"),
        [
            pytest.param({"action1": 0, "action2": 4}, True, id="within"),
            pytest.param({"action1": -1, "action2": 5}, False, id="outside"),
        ],
    )
    def test_abort_conditions(self, state_config: StateConfig, vals, truth: bool):
        assert state_config.within_abort_conditions(vals) is truth

    def test_map_ids(self):
        config = StateConfig(
            StateVar("foo1", is_ext_input=True, ext_id="bar1"),
            StateVar("foo2", is_ext_output=True, ext_id="bar2"),
        )
        for name, ext_id in config.map_ext_ids.items():
            assert name == config.rev_ext_ids[ext_id]

    def test_get_items(self, state_config: StateConfig):
        for name in state_config["actions"] + state_config["observations"]:
            assert state_config.loc[name] == state_config.vars[name]
