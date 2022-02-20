from __future__ import annotations

import abc
import pathlib
import time
from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.envs import BaseEnv
from eta_utility.eta_x.envs.base_env import log
from eta_utility.simulators import FMUSimulator

if TYPE_CHECKING:
    from typing import Any, Callable, Mapping, MutableSet, Sequence, SupportsFloat

    from eta_utility.type_hints import Path


class BaseEnvSim(BaseEnv, abc.ABC):
    @property
    @abc.abstractmethod
    def fmu_name(self) -> str:
        """Name of the FMU file"""
        return ""

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: dict[str, Any],
        path_settings: dict[str, Path],
        env_settings: dict[str, Any],
        verbose: int,
        callback: Callable | None = None,
    ) -> None:

        self.req_general_settings: set[Sequence | MutableSet] = set(self.req_general_settings)
        self.req_general_settings.update(("sim_steps_per_sample",))  # noqa
        super().__init__(
            env_id,
            run_name,
            general_settings,
            path_settings,
            env_settings,
            verbose,
            callback,
        )

        # Check configuration for compatibility
        errors = False
        if self.settings["sampling_time"] % self.settings["sim_steps_per_sample"] != 0:
            log.error(
                "'sim_steps_per_sample' must be an even divisor of 'sampling_time' "
                "(sampling_time % sim_steps_per_sample must equal 0."
            )
            errors = True

        if errors:
            raise ValueError(
                "Some configuration parameters do not conform to the Sim environment requirements (see log)."
            )

        #: Number of simulation steps to be taken for each sample. This must be a divisor of 'sampling_time'.
        self.sim_steps_per_sample: int = int(self.settings["sim_steps_per_sample"])
        #: The FMU is expected to be placed in the same folder as the environment
        self.path_fmu: pathlib.Path = self.path_env / (self.fmu_name + ".fmu")

        #: Configuration for the FMU model parameters, that need to be set for initialization of the Model.
        self.model_parameters: Mapping[str, int | float] | None = self.env_settings.setdefault("model_parameters", None)

        #: Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
        self.simulator: FMUSimulator

    def _init_simulator(self, init_values: Mapping[str, int | float] | None = None) -> None:
        """Initialize the simulator object. Make sure to call _names_from_state before this or to otherwise initialize
        the names array.

        This can also be used to reset the simulator after an episode is completed. It will reuse the same simulator
        object and reset it to the given initial values.

        :param init_values: Dictionary of initial values for some of the FMU variables
        """

        if hasattr(self, "simulator") and isinstance(self.simulator, FMUSimulator):
            self.simulator.reset(init_values)
        else:
            #: Instance of the FMU. This can be used to directly access the eta_utility.FMUSimulator interface.
            self.simulator = FMUSimulator(
                self.env_id,
                self.path_fmu,
                start_time=0.0,
                stop_time=self.episode_duration,
                step_size=int(self.sampling_time / self.sim_steps_per_sample),
                names_inputs=self.state_config.loc[self.names["ext_inputs"]].ext_id,
                names_outputs=self.state_config.loc[self.names["ext_outputs"]].ext_id,
                init_values=init_values,
            )

    def simulate(self, state: Mapping[str, float]) -> tuple[dict[str, float], bool, float]:
        """Perform a simulator step and return data as specified by the is_ext_observation parameter of the
        state_config.

        :param state: State of the environment before the simulation
        :return: Output of the simulation, boolean showing whether all simulation steps where successful, time elapsed
                 during simulation
        """
        # generate FMU input from current state
        step_inputs = []
        for key in self.names["ext_inputs"]:
            try:
                step_inputs.append(state[key] / self.ext_scale[key]["multiply"] - self.ext_scale[key]["add"])
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
            for idx, name in enumerate(self.names["ext_outputs"]):
                output[name] = (step_output[idx] + self.ext_scale[name]["add"]) * self.ext_scale[name]["multiply"]

        return output, step_success, sim_time_elapsed

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.float | SupportsFloat, bool, str | Sequence[str]]:
        """Perfom one time step and return its results. This is called for every event or for every time step during
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
        elif not self.action_space.contains(action):
            raise RuntimeError(
                f"Action {action} ({type(action)}) is invalid. At least one of the actions is not in action space."
            )

        self.n_steps += 1

        # Store actions
        self.state = {}
        for idx, act in enumerate(self.names["actions"]):
            self.state[act] = action[idx]

        # Update scenario data, simulate one time step and store the results.
        self.state.update(self.get_scenario_state())
        for _ in range(self.sim_steps_per_sample):  # do multiple FMU steps in one environment-step
            sim_result, step_success, sim_time_elapsed = self.simulate(self.state)
            self.state.update(sim_result)
            self.state_log.append(self.state)

        # Check if the episode is over or not
        done = self.n_steps >= self.n_episode_steps or not step_success

        observations = np.empty(len(self.names["observations"]))
        for idx, name in enumerate(self.names["observations"]):
            observations[idx] = self.state[name]

        return observations, 0, done, {}

    def reset(self) -> np.ndarray:
        """Reset the model and return initial observations. This also calls the callback, increments the episode
        counter, resets the episode steps and appends the state_log to the longtime storage.

        If you want to extend this function, write your own code and call super().reset() afterwards to return
        fresh observations. This allows you to ajust timeseries for example. If you need to manipulate the state
        before initializing or if you want to adjust the initialization itself, overwrite the function entirely.

        :return: Initial observation
        """

        # save episode's stats
        if self.n_steps > 0:
            if self.callback is not None:
                self.callback(self)

            # Store some logging data
            self.n_episodes += 1
            self.state_log_longtime.append(self.state_log)
            self.n_steps_longtime += self.n_steps

            # Reset episode variables
            self.n_steps = 0
            self.state_log = []

        # reset the FMU after every episode with new parameters
        self._init_simulator(self.model_parameters)

        # Update scenario data, read values from the fmu without time step and store the results
        start_obs = []
        for obs in self.names["ext_outputs"]:
            start_obs.append(self.map_ext_ids[obs])

        result = self.simulator.read_values(start_obs)
        self.state = {name: result[idx] for idx, name in enumerate(self.names["ext_outputs"])}
        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        observations = np.empty(len(self.names["observations"]))
        for idx, name in enumerate(self.names["observations"]):
            observations[idx] = self.state[name]

        return observations

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behaviour for the Simulation environment is to close the FMU object.

        :return:
        """
        self.simulator.close()  # close the FMU
