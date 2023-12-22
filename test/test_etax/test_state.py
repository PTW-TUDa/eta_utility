import numpy as np
import pytest
from gymnasium.spaces.box import Box

from eta_utility.eta_x.envs.state import StateConfig, StateVar


class TestStateConfig:
    @pytest.fixture(scope="class")
    def state_config(self):
        conf = StateConfig()
        conf.vars = {
            "action1": StateVar("action1", is_agent_action=True, low_value=0, high_value=1),
            "observation1": StateVar("observation1", is_agent_observation=True, low_value=2, high_value=3),
            "action2": StateVar("action2", is_agent_action=True, low_value=4, high_value=5),
            "observation2": StateVar("observation2", is_agent_observation=True, low_value=6, high_value=7),
        }
        return conf

    @pytest.fixture(scope="class")
    def state_config_nan(self):
        conf = StateConfig()
        conf.vars = {
            "action1": StateVar("action1", is_agent_action=True),
            "observation1": StateVar("observation1", is_agent_observation=True),
        }
        return conf

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
