from test.resources.agents.mpc_basic_env import MPCBasicEnv
from test.resources.agents.rule_based import RuleBasedController

import gym
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_utility import get_logger
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.agents.math_solver import MathSolver
from eta_utility.eta_x.common import NoPolicy


class TestRuleBased:
    @pytest.fixture(scope="class")
    def vec_env(self):
        env = DummyVecEnv([lambda: gym.make("CartPole-v1")])
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def rb_agent(self, vec_env):
        # Initialize the agent and create an instance of the policy and assign it to the policy attribute
        return RuleBasedController(policy=NoPolicy, env=vec_env)

    def test_rb_save_load(self, vec_env, rb_agent, temp_dir):
        get_logger(level=3)

        # Save the agent
        path = temp_dir / "test_rule_based_agent.zip"
        rb_agent.save(path)

        # Load the agent from the saved file
        loaded_agent = RuleBasedController.load(path=path, env=vec_env)

        assert isinstance(loaded_agent, RuleBasedController)
        assert isinstance(loaded_agent.policy, NoPolicy)

        # Compare attributes before and after loading
        assert loaded_agent.observation_space == rb_agent.observation_space
        assert loaded_agent.num_timesteps == rb_agent.num_timesteps
        assert loaded_agent.state == rb_agent.state

    def test_rb_learn(self, rb_agent):
        assert rb_agent.learn(total_timesteps=5) is not None
        assert isinstance(rb_agent, RuleBasedController)


class TestMathSolver:
    @pytest.fixture(scope="class")
    def mpc_basic_env(self, temp_dir):
        config_run = ConfigOptRun(
            series="MPC_Basic_test_2023",
            name="test_mpc_basic",
            description="",
            path_root=temp_dir / "root",
            path_results=temp_dir / "reults",
        )

        # Create the environment
        env = MPCBasicEnv(
            env_id=1,
            config_run=config_run,
            prediction_horizon=10,
            scenario_time_begin="2021-12-01 06:00",
            scenario_time_end="2021-12-01 07:00",
            episode_duration=1800,
            sampling_time=1,
            model_parameters={},
        )
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def mpc_agent(self, mpc_basic_env):
        # set up the agent
        return MathSolver(NoPolicy, mpc_basic_env)

    def test_mpc_save_load(self, mpc_basic_env, mpc_agent, temp_dir):
        # save
        path = temp_dir / "test_mpc_basic_agent.zip"
        mpc_agent.save(path)

        # Load the agent from the saved file
        loaded_agent = MathSolver.load(path=path, env=mpc_basic_env)

        assert isinstance(loaded_agent, MathSolver)
        assert isinstance(loaded_agent.policy, NoPolicy)

        # Compare attributes before and after loading
        assert loaded_agent.model == mpc_agent.model
        assert loaded_agent.observation_space == mpc_agent.observation_space
        assert loaded_agent.num_timesteps == mpc_agent.num_timesteps

    def test_mpc_learn(self, mpc_agent):
        assert mpc_agent.learn(total_timesteps=5) is not None
        assert isinstance(mpc_agent, MathSolver)
