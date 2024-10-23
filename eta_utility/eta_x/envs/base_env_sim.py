from __future__ import annotations

import abc
import pathlib
import time
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.envs import BaseEnv
from eta_utility.simulators import FMUSimulator

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import ObservationType, StepResult, TimeStep

log = getLogger(__name__)


class BaseEnvSim(BaseEnv, abc.ABC):
    """Base class for FMU Simulation models environments.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scneario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param sim_steps_per_sample: Number of simulation steps to perform during every sample.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
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
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        episode_duration: TimeStep | str,
        sampling_time: TimeStep | str,
        model_parameters: Mapping[str, Any] | None = None,
        sim_steps_per_sample: int | str = 1,
        render_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
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
                value = state[key]
                if isinstance(value, (int, float)):
                    scale_config = self.state_config.ext_scale[key]
                    value = value / scale_config["multiply"] - scale_config["add"]
                step_inputs.append(value)
            except KeyError as e:
                raise KeyError(f"{e!s} is unavailable in environment state.") from e

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
                value = step_output[idx]
                if isinstance(value, (int, float)):
                    scale_config = self.state_config.ext_scale[name]
                    value = (value + scale_config["add"]) * scale_config["multiply"]
                output[name] = value

        return output, step_success, sim_time_elapsed

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a five-tuple of observations, rewards, terminated, truncated,
        info.

        This also updates self.state and self.state_log to store current state information.

        .. note::
            This function always returns 0 reward. Therefore, it must be extended if it is to be used with reinforcement
            learning agents. If you need to manipulate actions (discretization, policy shaping, ...)
            do this before calling this function.
            If you need to manipulate observations and rewards, do this after calling this function.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
              bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the
              user needs to call the `reset` function.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.
        """
        self._actions_valid(action)

        step_success, sim_time_elapsed = self._update_state(action)
        self.state_log.append(self.state)

        terminated = self._done() or not step_success
        info: dict[str, Any] = {"sim_time_elapsed": sim_time_elapsed}

        # Render the environment at each step
        if self.render_mode is not None:
            self.render()

        return self._observations(), 0, terminated, False, info

    def _update_state(self, action: np.ndarray) -> tuple[bool, float]:
        """Take additional_state, execute simulation and get state information from scenario. This function
        updates self.state and increments the step counter.

        .. warning::
            You have to update self.state_log with the entire state before leaving the step
            to store the state information.

        :param action: Actions to perform in the environment.
        :return: Success of the simulation, time taken for simulation.
        """
        assert self.state_config is not None, "Set state_config before calling _update_state function."
        # Store actions
        new_state = {} if self.additional_state is None else self.additional_state
        for idx, act in enumerate(self.state_config.actions):
            new_state[act] = action[idx]

        step_success, sim_time_elapsed = False, 0.0
        # simulate one time step and store the results.
        for i in range(self.sim_steps_per_sample):  # do multiple FMU steps in one environment-step
            sim_result, step_success, sim_time_elapsed = self.simulate({**self.state, **new_state})
            new_state.update(sim_result)

            # Append intermediate simulation results to the state_log
            if i < self.sim_steps_per_sample - 1:
                self.state_log.append({**self.state, **new_state})

        self.n_steps += 1

        # Update scenario and environment state
        new_state.update(self.get_scenario_state())
        self.state = new_state

        return step_success, sim_time_elapsed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset. When using the environment in conjunction with
        *stable_baselines3*, the vectorized environment will take care of seeding your custom environment automatically.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. note ::
            Don't forget to store and reset the episode_timer.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be
                reset. If you pass an integer, the PRNG will be reset even if it already exists. (default: None)
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)

        :return: Tuple of observation and info. The observation of the initial state will be an element of
                :attr:`observation_space` (typically a numpy array) and is analogous to the observation returned by
                :meth:`step`. Info is a dictionary containing auxiliary information complementing ``observation``. It
                should be analogous to the ``info`` returned by :meth:`step`.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."
        super().reset(seed=seed, options=options)

        # reset the FMU after every episode with new parameters
        self._init_simulator(self.model_parameters)

        # Update scenario data, read values from the fmu without time step and store the results
        start_obs = []
        for obs in self.state_config.ext_outputs:
            start_obs.append(str(self.state_config.map_ext_ids[obs]))

        output = self.simulator.read_values(start_obs)
        self.state = {} if self.additional_state is None else self.additional_state
        for idx, name in enumerate(self.state_config.ext_outputs):
            value = output[idx]
            if isinstance(value, (int, float)):
                scale_config = self.state_config.ext_scale[name]
                value = (value + scale_config["add"]) * scale_config["multiply"]
            self.state[name] = value

        self.state.update(self.get_scenario_state())
        self.state_log.append(self.state)

        # Render the environment when calling the reset function
        if self.render_mode is not None:
            self.render()

        return self._observations(), {}

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.

        Default behavior for the Simulation environment is to close the FMU object.
        """
        self.simulator.close()  # close the FMU
