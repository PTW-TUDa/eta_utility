from __future__ import annotations

from typing import TYPE_CHECKING

from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import TimeStep


class DampedOscillatorEnv(BaseEnvSim):
    """
    Damped oscillator environment class from BaseEnvSim.
    Model settings come from fmu file.

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed)
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: callback which should be called after each episode
    :param scenario_time_begin: Beginning time of the scenario
    :param scenario_time_end: Ending time of the scenario
    :param episode_duration: Duration of the episode in seconds
    :param sampling_time: Duration of a single time sample / time step in seconds
    """

    # Set info
    version = "v0.1"
    description = "Damped oscillator"
    fmu_name = "damped_oscillator"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        sim_steps_per_sample: int,
    ):
        super().__init__(
            env_id,
            config_run,
            seed,
            verbose,
            callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            sim_steps_per_sample=sim_steps_per_sample,
        )

        # Set action space and observation space
        self.state_config = StateConfig(
            StateVar(name="u", ext_id="u", is_agent_action=True, low_value=-1.0, high_value=1.0),
            StateVar(
                name="s", ext_id="s", is_agent_observation=True, low_value=-4.0, high_value=4.0, is_ext_output=True
            ),
            StateVar(
                name="v", ext_id="v", is_agent_observation=True, low_value=-8.0, high_value=8.0, is_ext_output=True
            ),
            StateVar(
                name="a", ext_id="a", is_agent_observation=True, low_value=-20.0, high_value=20.0, is_ext_output=True
            ),
        )
        self.action_space, self.observation_space = self.state_config.continuous_spaces()

        # Initialize the simulator
        self._init_simulator()

    def render(self, mode: str = "human") -> None:
        csv_export(
            path=episode_results_path(self.config_run.path_series_results, self.run_name, self.n_episodes, self.env_id),
            data=self.state_log,
        )
