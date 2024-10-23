from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnvLive, StateConfig, StateVar
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import ObservationType, StepResult, TimeStep
from logging import getLogger

log = getLogger(__name__)


class CleaningMachineConnected(BaseEnvLive):
    """Environment for the connection to the cleaning machine.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    """

    version = "v0.2"
    description = "Environment for connection to a single chamber cleaning machine."
    config_name = "live_connection_local"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        **kwargs: Any,
    ):
        # Instantiate BaseEnvSim
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            **kwargs,
        )

        self.state_config = StateConfig(
            StateVar(name="mode_tankheater", ext_id="CM.mode_tankheater", is_ext_input=True, is_agent_action=True),
            StateVar(name="temp_tank", ext_id="CM.temp_tank", is_ext_output=True, is_agent_observation=True),
            StateVar(name="tankheater", ext_id="CM.tankheater", is_ext_output=True, is_agent_observation=True),
            StateVar(
                name="heating_register", ext_id="CM.heating_register", is_ext_output=True, is_agent_observation=True
            ),
            StateVar(name="motor_nozzles", ext_id="CM.motor_nozzles", is_ext_output=True, is_agent_observation=True),
            StateVar(name="motor_basket", ext_id="CM.motor_basket", is_ext_output=True, is_agent_observation=True),
            StateVar(name="pump", ext_id="CM.pump", is_ext_output=True, is_agent_observation=True),
            StateVar(name="fan", ext_id="CM.fan", is_ext_output=True, is_agent_observation=True),
            StateVar(name="valve", ext_id="CM.valve", is_ext_output=True, is_agent_observation=True),
        )
        self.action_space, self.observation_space = self.state_config.continuous_spaces()

        self._init_live_connector()

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the optimization run. It should utilize the actions as supplied by the agent to determine
        the new state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """
        assert self.state_config is not None, "Set state_config before calling step function."

        observations, rewards, terminated, truncated, info = super().step(action)

        # Convert the temperature value to Kelvin.
        self.state["temp_tank"] += (
            273.15 + 20
        )  # 20 degrees added because tank temperature in the OPC UA server is permanently 0
        observations[self.state_config.observations.index("temp_tank")] = self.state["temp_tank"]  # type: ignore
        log.info(f"Current temperature in tank: {self.state['temp_tank']} K")

        # Store the results every five steps
        if self.n_steps % 5 == 0:
            self.render()

        return observations, rewards, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`) (default: None).
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)
        :return: Tuple of observation and info. Analogous to the ``info`` returned by :meth:`step`.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."

        # Turn off the tank heater before each episode
        self.live_connector.write({str(self.state_config.map_ext_ids["mode_tankheater"]): True})
        observations, infos = super().reset(seed=seed, options=options)

        # Convert the temperature value to Kelvin.
        self.state["temp_tank"] += (
            273.15 + 20
        )  # 20 degrees added because tank temperature in the OPC UA server is permanently 0
        observations[self.state_config.observations.index("temp_tank")] = self.state["temp_tank"]  # type: ignore
        log.info(f"Current temperature in tank: {self.state['temp_tank']} K")

        return observations, infos

    def render(self, mode: str = "human") -> None:
        csv_export(
            path=episode_results_path(self.config_run.path_series_results, self.run_name, self.n_episodes, self.env_id),
            data=self.state_log,
        )
