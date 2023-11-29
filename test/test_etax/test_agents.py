import pathlib
from test.resources.agents.mpc_basic_env import MPCBasicEnv
from test.resources.agents.rule_based import RuleBasedController

import gymnasium
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from eta_utility import get_logger
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.agents.math_solver import MathSolver
from eta_utility.eta_x.common import NoPolicy
from eta_utility.eta_x.envs import NoVecEnv
from eta_utility.util_julia import julia_extensions_available

if julia_extensions_available():
    from eta_utility.eta_x.agents.nsga2 import Nsga2
    from eta_utility.eta_x.envs.julia_env import JuliaEnv


class TestRuleBased:
    @pytest.fixture(scope="class")
    def vec_env(self):
        env = DummyVecEnv([lambda: gymnasium.make("CartPole-v1")])
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
            path_results=temp_dir / "results",
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


@pytest.mark.skipif(not julia_extensions_available(), reason="PyJulia installation required!")
class TestNSGA2:
    scenario_time_begin = "2017-01-24 00:01"
    scenario_time_end = "2017-01-24 23:59"
    episode_duration = 1800
    sampling_time = 1

    def config_run(self, temp_dir):
        directory = pathlib.Path(temp_dir) if not isinstance(temp_dir, pathlib.Path) else temp_dir

        return ConfigOptRun(
            series="Nsga2_test_2023",
            name="test_nsga2",
            description="",
            path_root=directory / "root",
            path_results=directory / "results",
        )

    def create_stored_agent_file(self, path, temp_dir):
        """Execute this function directly when necessary to refresh the stored NSGA2 agent model file. This function
        creates a new NSGA2 model and saves the result in the given path.

        ..code-block: console

            python -c "import tempfile; from test.test_etax.test_agents import TestNSGA2; cls = TestNSGA2();
            cls.create_stored_agent_file('test/resources/agents/', tempfile.TemporaryDirectory().name)"

        :param path: Path where the julia environment file is located. This path is also used to store the trained
                     model file.
        :param temp_dir: Path to store result files during training of the model.
        """
        directory = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

        env = JuliaEnv(
            env_id=1,
            config_run=self.config_run(temp_dir),
            scenario_time_begin=self.scenario_time_begin,
            scenario_time_end=self.scenario_time_end,
            episode_duration=self.episode_duration,
            sampling_time=self.sampling_time,
            julia_env_file=directory.absolute() / "Nsga2Env.jl",
            is_multithreaded=True,
        )
        agent = Nsga2(
            policy=NoPolicy,
            env=NoVecEnv([lambda: env]),
            population=1000,
            crossovers=0.3,
            n_generations=2,
            max_retries=1000000,
            predict_learn_steps=10,
            seed=2139846,
        )
        agent.learn(10)
        agent.save(directory.absolute() / "test_nsga2_agent.zip")

    @pytest.fixture(scope="class")
    def julia_env(self, config_etax_resources_path, temp_dir):
        env = JuliaEnv(
            env_id=1,
            config_run=self.config_run(temp_dir),
            scenario_time_begin=self.scenario_time_begin,
            scenario_time_end=self.scenario_time_end,
            episode_duration=self.episode_duration,
            sampling_time=self.sampling_time,
            julia_env_file=config_etax_resources_path / "Nsga2Env.jl",
            is_multithreaded=True,
        )
        no_vec_env = NoVecEnv([lambda: env])
        yield no_vec_env
        no_vec_env.close()

    @pytest.fixture(scope="class")
    def loaded_agent(self, config_etax_resources_path, julia_env):
        path = config_etax_resources_path / "test_nsga2_agent.zip"
        agent = Nsga2.load(path=path, env=julia_env)
        return agent

    @pytest.fixture(scope="class")
    def default_agent(self, julia_env):
        agent = Nsga2(env=julia_env, policy=NoPolicy)
        return agent

    def test_load_agent_setup(self, loaded_agent):
        assert isinstance(loaded_agent, Nsga2)
        assert isinstance(loaded_agent.env, DummyVecEnv)

    @pytest.mark.parametrize(
        ("attr", "value"),
        [
            ("population", 1000),
            ("mutations", 0.05),
            ("crossovers", 0.3),
            ("n_generations", 2),
            ("max_cross_len", 1),
            ("max_retries", 1000000),
            ("predict_learn_steps", 10),
            ("seed", 2139846),
            ("tensorboard_log", None),
            ("event_params", 60),
        ],
    )
    def test_load_agent_attributes(self, attr, value, loaded_agent):
        assert getattr(loaded_agent, attr) == value

    def test_predict(self, loaded_agent):
        observations = loaded_agent.env.reset()
        action, state = loaded_agent.predict(observations)

        events, variables = action

        assert len(action["events"] == 60)
        assert len(action["variables"] == 60)
        assert state is None
        assert isinstance(events, np.ndarray)
        assert events.shape == (60,)
        assert isinstance(variables, np.ndarray)
        assert variables.shape == (60,)

    @pytest.mark.parametrize(
        ("attr", "value"),
        [
            ("population", 100),
            ("mutations", 0.05),
            ("crossovers", 0.1),
            ("n_generations", 100),
            ("max_cross_len", 1),
            ("max_retries", 100000),
            ("predict_learn_steps", 5),
            ("seed", 42),
        ],
    )
    def test_default_agent_attributes(self, attr, value, default_agent):
        assert getattr(default_agent, attr) == value

    def test_setup_learn(self, default_agent):
        init_total_timesteps = 10
        total_timesteps, callback = default_agent._setup_learn(
            total_timesteps=init_total_timesteps,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="predict",
            progress_bar=False,
        )
        assert total_timesteps == init_total_timesteps
        assert callback is not None
        assert default_agent.num_timesteps == 0
        assert default_agent.ep_info_buffer is not None
        assert default_agent.start_time is not None
