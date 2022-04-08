from __future__ import annotations

import abc
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np

from eta_utility import get_logger
from eta_utility.eta_x.envs import BaseEnv
from eta_utility.simulators import FMUSimulator

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable, Mapping

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import StepResult, TimeStep

log = get_logger("eta_x.envs")


class BaseEnvSim(BaseEnv, abc.ABC):
    """Base class for FMU Simulation models environments.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed).
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scneario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param sim_steps_per_sample: Number of simulation steps to perform during every sample.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    @property
    @abc.abstractmethod
    def fmu_name(self) -> str:
        """Name of the FMU file"""
        return ""

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
        model_parameters: Mapping[str, Any] | None = None,
        sim_steps_per_sample: int | str = 1,
        **kwargs: Any,
    ) -> None:
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
            **kwargs,
        )

        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample: int = int(sim_steps_per_sample)

        #: The FMU is expected to be placed in the same folder as the environment
        self.path_fmu: pathlib.Path = self.path_env / (self.fmu_name + ".fmu")

        #: Configuration for the FMU model parameters, that need to be set for initialization of the Model.
        self.model_parameters: Mapping[str, int | float] | None = model_parameters

        #: Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
        self.simulator: FMUSimulator

    def _init_simulator(self, init_values: Mapping[str, int | float] | None = None) -> None:
        """Initialize the simulator object. Make sure to call _names_from_state before this or to otherwise initialize
        the names array.

        This can also be used to reset the simulator after an episode is completed. It will reuse the same simulator
        object and reset it to the given initial values.

        :param init_values: Dictionary of initial values for some FMU variables.
        """
        assert self.state_config is not None, "Set state_config before calling _init_simulator function."

        _init_vals = {} if init_values is None else init_values

        if hasattr(self, "simulator") and isinstance(self.simulator, FMUSimulator):
            self.simulator.reset(_init_vals)
        else:
            # Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
            self.simulator = FMUSimulator(
                self.env_id,
                self.path_fmu,
                start_time=0.0,
                stop_time=self.episode_duration,
                step_size=float(self.sampling_time / self.sim_steps_per_sample),
                names_inputs=[str(self.state_config.map_ext_ids[name]) for name in self.state_config.ext_inputs],
                names_outputs=[str(self.state_config.map_ext_ids[name]) for name in self.state_config.ext_outputs],
                init_values=_init_vals,
            )

    def simulate(self, state: Mapping[str, float]) -> tuple[dict[str, float], bool, float]:
        """Perform a simulator step and return data as specified by the is_ext_observation parameter of the
        state_config.

        :param state: State of the environment before the simulation.
        :return: Output of the simulation, boolean showing whether all simulation steps where successful, time elapsed
                 during simulation.
        """
        assert self.state_config is not None, "Set state_config before calling simulate function."

        # generate FMU input from current state
        step_inputs = []
        for key in self.state_config.ext_inputs:
            try:
                step_inputs.append(
                    state[key] / self.state_config.ext_scale[key]["multiply"] - self.state_config.ext_scale[key]["add"]
                )
            except KeyError as e:
                raise KeyError(f"{str(e)} is unavailable in environment state.") from e

        sim_time_start = time.time()

        step_success = True
        try:
            step_output = self.simulator.step(step_inputs)

        except Exception as e:
            step_success = False
            log.error(e)

        # stop timer for simulation step time debugging
        sim_time_elapsed = time.time() - sim_time_start

        # save step_outputs into data_store
        output = {}
        if step_success:
            for idx, name in enumerate(self.state_config.ext_outputs):
                output[name] = (
                    step_output[idx] + self.state_config.ext_scale[name]["add"]
                ) * self.state_config.ext_scale[name]["multiply"]

        return output, step_success, sim_time_elapsed

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine
        the new state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        This also updates self.state and self.state_log to store current state information.

        .. warning::
            This function always returns 0 reward. Therefore, it must be extended if it is to be used with reinforcement
            learning agents. If you need to manipulate actions (discretization, policy shaping, ...)
            do this before calling this function.
            If you need to manipulate observations and rewards, do this after calling this function.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **done**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.
        """
        if self.action_space.shape != action.shape:
            raise RuntimeError(
                f"Agent action {action} (shape: {action.shape})"
                f" does not correspond to shape of environment action space (shape: {self.action_space.shape})."
            )

        assert self.state_config is not None, "Set state_config before calling step function."

        self.n_steps += 1

        # Store actions
        self.state = {} if self.additional_state is None else self.additional_state
        for idx, act in enumerate(self.state_config.actions):
            self.state[act] = action[idx]

        # Update scenario data, simulate one time step and store the results.
        self.state.update(self.get_scenario_state())
        for _ in range(self.sim_steps_per_sample):  # do multiple FMU steps in one environment-step
            sim_result, step_success, sim_time_elapsed = self.simulate(self.state)
            self.state.update(sim_result)
            self.state_log.append(self.state.copy())

        # Check if the episode is over or not
        done = self.n_steps >= self.n_episode_steps or not step_success

        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations, 0, done, {}

    def reset(self) -> np.ndarray:
        """Reset the model and return initial observations. This also calls the callback, increments the episode
        counter, resets the episode steps and appends the state_log to the longtime storage.

        If you want to extend this function, write your own code and call super().reset() afterwards to return
        fresh observations. This allows you to adjust timeseries for example. If you need to manipulate the state
        before initializing or if you want to adjust the initialization itself, overwrite the function entirely.

        :return: Initial observation.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."
        self._reset_state()

        # reset the FMU after every episode with new parameters
        self._init_simulator(self.model_parameters)

        # Update scenario data, read values from the fmu without time step and store the results
        start_obs = []
        for obs in self.state_config.ext_outputs:
            start_obs.append(str(self.state_config.map_ext_ids[obs]))

        output = self.simulator.read_values(start_obs)
        self.state = {} if self.additional_state is None else self.additional_state
        for idx, name in enumerate(self.state_config.ext_outputs):
            self.state[name] = (output[idx] + self.state_config.ext_scale[name]["add"]) * self.state_config.ext_scale[
                name
            ]["multiply"]

        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        return observations

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the Simulation environment is to close the FMU object.
        """
        self.simulator.close()  # close the FMU
