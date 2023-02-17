from __future__ import annotations

import pathlib
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from eta_utility import get_logger
from eta_utility.eta_x.envs import BaseEnv
from eta_utility.util_julia import check_julia_package

if check_julia_package():
    from julia import Main as Jl  # noqa: I900

    from eta_utility.util_julia import import_jl_file

if TYPE_CHECKING:
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import StepResult, TimeStep

Jl.eval("using PyCall")
jl_setattribute = Jl.eval("pyfunction(setfield!, PyAny, Symbol, PyAny)")

log = get_logger("eta_x.envs")


class JuliaEnv(BaseEnv):
    """Abstract environment definition, providing some basic functionality for concrete environments to use.
    The class implements and adapts functions from gym.Env. It provides additional functionality as required by
    the ETA-X framework and should be used as the starting point for new environments.

    The initialization of this superclass performs many of the necessary tasks, required to specify a concrete
    environment. Read the documentation carefully to understand, how new environments can be developed, building on
    this starting point.

    There are some attributes that must be set and some methods that must be implemented to satisfy the interface. This
    is required to create concrete environments.
    The required attributes are:

        - **version**: Version number of the environment.
        - **description**: Short description string of the environment.
        - **action_space**: The action space of the environment (see also gym.spaces for options).
        - **observation_space**: The observation space of the environment (see also gym.spaces for options).

    The gym interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**

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
    :param kwargs: Other keyword arguments (for subclasses).
    """

    version = "1.0"
    description = "This environment uses a julia file to perform its functions."

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
        julia_env_file: pathlib.Path | str,
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
        )
        # Set arguments as instance parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)

        julia_env_path = julia_env_file if isinstance(julia_env_file, pathlib.Path) else pathlib.Path(julia_env_file)
        if not julia_env_path.is_absolute():
            julia_env_path = config_run.path_root / julia_env_path

        self.__jl = import_jl_file(julia_env_path)

        # Make sure that all required functions are implemented in julia.
        for func in {"Environment", "step!", "reset!", "close!", "render", "seed!"}:
            if not hasattr(self.__jl, func):
                raise NotImplementedError(
                    f"Implementation of abstract method {func} missing from julia implmentation of JuliaEnv."
                )

        self._jlenv = self.__jl.Environment(self)

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a four-tuple of observations, rewards, dones, info.

        .. note ::
            Do not forget to increment n_steps and n_steps_longtime.

        :param action: Actions taken by the agent.
        :return: The return value represents the state of the environment after the step was performed.

            * observations: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * reward: The value of the reward function. This is just one floating point value.
            * done: Boolean value specifying whether an episode has been completed. If this is set to true, the reset
              function will automatically be called by the agent or by eta_i.
            * info: Provide some additional info about the state of the environment. The contents of this may be used
              for logging purposes in the future but typically do not currently serve a purpose.

        """
        self._actions_valid(action)
        self.n_steps += 1

        return self.__jl.step_b(self._jlenv, action)

    def reset(self) -> np.ndarray:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        .. note ::
            Don't forget to store and reset the episode_timer.

        :return: The return value represents the observations (state) of the environment before the first
                 step is performed.
        """
        self._reset_state()
        return self.__jl.reset_b(self._jlenv)

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.
        """
        return self.__jl.close_b(self._jlenv)

    def render(self, mode: str = "human") -> None:
        """Render the environment

        The set of supported modes varies per environment. Some environments do not support rendering at
        all. By convention in OpenAI *gym*, if mode is:

            * human: render to the current display or terminal and return nothing. Usually for human consumption.
            * rgb_array: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
              suitable for turning into a video.
            * ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        :param mode: Rendering mode.
        """
        self.__jl.render(self._jlenv, mode)

    def seed(self, seed: int | None = None) -> tuple[np.random.Generator, int]:
        """Set random seed for the random generator of the environment

        :param seed: Seeding value.
        :return: Tuple of the numpy random generator and the set seed value.
        """
        generator, seed = super().seed(seed)

        # Make sure not to seed the julia environment before it exists.
        if hasattr(self, "_JuliaEnv__jl"):
            self.__jl.seed_b(self._jlenv, seed)

        return generator, seed

    def __getattr__(self, name: str) -> Any:
        # Return the item if it is set on the python object
        try:
            return super().__getattribute__(name)
        except AttributeError as e:
            err = e

        # If the item isn't set on the python object, check whether _jlenv exists and has the item
        if "_jlenv" in self.__dict__:
            try:
                return getattr(self._jlenv, name)
            except Exception:
                raise AttributeError(f"Could not get {name} from python or julia environment.")

        raise err

    def __setattr__(self, name: str, value: Any) -> None:
        # Try to set on _jlenv
        if "_jlenv" in self.__dict__ and hasattr(self._jlenv, name):
            try:
                jl_setattribute(self._jlenv, name, value)
            except BaseException as e:
                raise AttributeError(f"Could not set {name} on julia environment: {e}")

        # Otherwise set on the python environment.
        super().__setattr__(name, value)
