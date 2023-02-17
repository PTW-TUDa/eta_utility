from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from eta_utility import get_logger
from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
from eta_utility.util import csv_export

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable, Mapping, Sequence

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import TimeStep

log = get_logger("eta_x.envs")


class CleaningMachineSimulation(BaseEnvSim):
    """Environment for the simulation of the cleaning machine.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed).
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param sim_steps_per_sample: Number of simulation steps to perform during every sample.
    :param scenario_electricity_prices: Name of the scenario file for electricity prices.
    """

    version = "v0.1"
    description = "Simulation of a single chamber cleaning machine."
    fmu_name = "CleaningMachine"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,
        seed: int | None = 123,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        scenario_files: Sequence[Mapping[str, Any]],
        **kwargs: Any,
    ):
        # Instantiate BaseEnvSim
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            seed=seed,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            **kwargs,
        )

        self.state_config = StateConfig(
            StateVar(
                name="temp_tank_sim",
                ext_id="MAFAC_KEA.tank.tank.medium.T",
                is_ext_output=True,
                is_agent_observation=True,
                is_agent_action=True,
                low_value=50,
                high_value=60,
                abort_condition_min=0,
                abort_condition_max=90,
            ),
            StateVar(
                name="temp_tank_opcua",
                ext_id="MAFAC_KEA.tank.tank.medium.T",
                is_ext_input=True,
                is_agent_observation=True,
                from_interact=True,
            ),
            StateVar(
                name="tankheater",
                is_agent_observation=True,
                from_interact=True,
                ext_id="u_tank_heater",
                is_ext_input=True,
                low_value=1,
                high_value=0,
            ),
            StateVar(name="heating_register", ext_id="u_heating_register", from_interact=True, is_ext_input=True),
            StateVar(name="motor_nozzles", ext_id="u_motor_nozzles", from_interact=True, is_ext_input=True),
            StateVar(name="motor_basket", ext_id="u_motor_basket", from_interact=True, is_ext_input=True),
            StateVar(name="pump", ext_id="u_pump", from_interact=True, is_ext_input=True),
            StateVar(name="fan", ext_id="u_fan", from_interact=True, is_ext_input=True),
            StateVar(name="valve", ext_id="u_valve", from_interact=True, is_ext_input=True),
            StateVar(name="Pel_tankheater", ext_id="MAFAC_KEA.tankHeater.P_el", is_ext_output=True),
            StateVar(
                name="market_price",
                is_agent_observation=True,
                from_scenario=True,
                scenario_id="electrical_energy_price",
            ),
            StateVar(name="Pel_ges", ext_id="MAFAC_KEA.P_el", is_ext_output=True),
            StateVar(name="Q_flow", ext_id="MAFAC_KEA.tankHeater.HeatPort_b.Q_flow", is_ext_output=True),
        )
        self.action_space, self.observation_space = self.state_config.continuous_spaces()

        # Initialize the simulator object
        self._init_simulator()

        self.import_scenario(*scenario_files)

    def first_update(self, observations: np.ndarray) -> np.ndarray:
        """Perform the first update and set values in simulation model to the observed values.

        :param observations: Observations of another environment.
        :return: Full array of observations.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."
        self._reset_state()

        # Reset the FMU after every episode with new parameters
        self._init_simulator(self.model_parameters)

        # State saves results for the current run
        self.state = {}

        # Store observations from the real environment
        for idx, name in enumerate(self.state_config.interact_outputs):
            self.state[name] = observations[0][idx]

        # Update scenario data, simulate one time step and store the results.
        self.state.update(self.get_scenario_state())

        # Inputs saves the start conditions for setting the simulation model
        inputs = [self.state[name] for name in self.state_config.ext_inputs]

        # Update tank temperature with current state from opc ua server
        self.simulator.set_values(values=inputs)
        # Read the start state without stepping the simulation
        output = self.simulator.read_values()

        for idx, name in enumerate(self.state_config.ext_outputs):
            self.state[name] = (output[idx] + self.state_config.ext_scale[name]["add"]) * self.state_config.ext_scale[
                name
            ]["multiply"]

        log.info(f"Current temperature in tank (simulated): {self.state['temp_tank_sim']} K")
        log.info(f"Current market price: {self.state['market_price']} €")
        self.state_log.append(self.state)

        # Return of the observation np- array to the controller
        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations

    def update(self, observations: np.ndarray) -> np.ndarray:
        """Update the optimization model with observations from another environment.

        :param observations: Observations from another environment
        :return: Full array of current observations
        """
        assert self.state_config is not None, "Set state_config before calling update function."
        # Update current step of the model
        self.n_steps += 1

        # State saves results for the current run
        self.state = {}

        # Store observations from the real environment
        for idx, name in enumerate(self.state_config.interact_outputs):
            self.state[name] = observations[0][idx]

        # Update scenario data, simulate one time step and store the results.
        self.state.update(self.get_scenario_state())
        sim_output, step_success, sim_time_elapsed = self.simulate(self.state)

        self.state.update(sim_output)
        log.info(f"Current temperature in tank (simulated): {self.state['temp_tank_sim']} K")
        log.info(f"Current market price: {self.state['market_price']} €")
        self.state_log.append(self.state)

        # Return of the observation np- array to the controller
        observations = np.empty(len(self.state_config.observations))
        for idx, obs in enumerate(self.state_config.observations):
            observations[idx] = self.state[obs]

        # Save the state_log as csv file every second loops
        if self.n_steps % 5 == 0:
            self.render()

        return observations

    def render(self, mode: str = "human") -> None:
        csv_export(
            path=episode_results_path(self.config_run.path_series_results, self.run_name, self.n_episodes, self.env_id),
            data=self.state_log,
        )
