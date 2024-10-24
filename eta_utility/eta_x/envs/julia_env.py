from __future__ import annotations

import pathlib
from datetime import datetime
from functools import partial
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

from eta_utility.eta_x.envs import BaseEnv
from eta_utility.util_julia import check_julia_package

if check_julia_package():
    from julia import Main as Jl

    from eta_utility.util_julia import import_jl_file

if TYPE_CHECKING:
    from types import ModuleType
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import StepResult, TimeStep


Jl.eval("using PyCall")
jl_setattribute = Jl.eval("pyfunction(setfield!, PyAny, Symbol, PyAny)")

log = getLogger(__name__)


class JuliaEnv(BaseEnv):
    """
    TODO: UPDATE DOCUMENTATION!
    Abstract environment definition, providing some basic functionality for concrete environments to use.
    The class implements and adapts functions from gymnasium.Env. It provides additional functionality as required by
    the ETA-X framework and should be used as the starting point for new environments.

    The initialization of this superclass performs many of the necessary tasks, required to specify a concrete
    environment. Read the documentation carefully to understand, how new environments can be developed, building on
    this starting point.

    There are some attributes that must be set and some methods that must be implemented to satisfy the interface. This
    is required to create concrete environments.
    The required attributes are:

        - **version**: Version number of the environment.
        - **description**: Short description string of the environment.
        - **action_space**: The action space of the environment (see also gymnasium.spaces for options).
        - **observation_space**: The observation space of the environment (see also gymnasium.spaces for options).

    The gymnasium interface requires the following methods for the environment to work correctly within the framework.
    Consult the documentation of each method for more detail.

        - **step()**
        - **reset()**
        - **close()**

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param verbose: Verbosity to use for logging.
    :param callback: callback which should be called after each episode.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    version = "1.0"
    description = "This environment uses a julia file to perform its functions."

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
        julia_env_file: pathlib.Path | str,
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
        )
        # Set arguments as instance parameters.
        for key, value in kwargs.items():
            setattr(self, key, value)

        #: Julia file name.
        julia_env_file = julia_env_file if isinstance(julia_env_file, pathlib.Path) else pathlib.Path(julia_env_file)

        #: Root Path to the julia file.
        self.julia_env_path: pathlib.Path = (
            julia_env_file if julia_env_file.is_absolute() else config_run.path_root / julia_env_file
        )

        #: Imported Julia file as a module (written in julia) for further initialization of the environment.
        self.__jl: ModuleType = import_jl_file(self.julia_env_path)

        # Make sure that all required functions are implemented in julia.
        for func in ("Environment", "step!", "reset!", "close!", "render", "first_update!", "update!"):
            if not hasattr(self.__jl, func):
                raise NotImplementedError(
                    f"Implementation of abstract method {func} missing from julia implementation of JuliaEnv."
                )

        #: Initialized julia environment (written in julia).
        self._jlenv = self.__jl.Environment(self)

    def first_update(self, observations: np.ndarray) -> np.ndarray:
        """Perform the first update and set values in simulation model to the observed values.

        :param observations: Observations of another environment.
        :return: Full array of observations.
        """
        return self.__jl.first_update_b(observations)

    def update(self, observations: np.ndarray) -> np.ndarray:
        """Update the optimization model with observations from another environment.

        :param observations: Observations from another environment
        :return: Full array of current observations
        """
        return self.__jl.update_b(observations)

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results. This is called for every event or for every time step during
        the simulation/optimization run. It should utilize the actions as supplied by the agent to determine the new
        state of the environment. The method must return a five-tuple of observations, rewards, terminated, truncated,
        info.

        .. note ::
            Do not forget to increment n_steps and n_steps_longtime.

        :param action: Actions taken by the agent.
        :return: The return value represents the state of the environment after the step was performed.

            * **observations**: A numpy array with new observation values as defined by the observation space.
              Observations is a np.array() (numpy array) with floating point or integer values.
            * **reward**: The value of the reward function. This is just one floating point value.
            * **terminated**: Boolean value specifying whether an episode has been completed. If this is set to true,
              the reset function will automatically be called by the agent or by eta_i.
            * **truncated**: Boolean, whether the truncation condition outside the scope is satisfied.
              Typically, this is a timelimit, but could also be used to indicate an agent physically going out of
              bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user
              needs to call the `reset` function.
            * **info**: Provide some additional info about the state of the environment. The contents of this may
              be used for logging purposes in the future but typically do not currently serve a purpose.

        """
        self._actions_valid(action)
        self.n_steps += 1

        observations, reward, terminated, truncated, info = self.__jl.step_b(self._jlenv, action)
        self.state_log.append(observations)

        # Render the environment at each step
        if self.render_mode is not None:
            self.render()

        return observations, reward, terminated, truncated, info

    def _reduce_state_log(self) -> list[dict[str, float]]:
        """Removes unwanted parameters from state_log before storing in state_log_longtime

        :return: The return value is a list of dictionaries, where the parameters that
                 should not be stored were removed
        """
        return self.state_log

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
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
        super().reset(seed=seed, options=options)

        # Render the environment when calling the reset function
        if self.render_mode is not None:
            self.render()

        return self.__jl.reset_b(self._jlenv, seed, options)

    def close(self) -> None:
        """Close the environment. This should always be called when an entire run is finished. It should be used to
        close any resources (i.e. simulation models) used by the environment.
        """
        return self.__jl.close_b(self._jlenv)

    def render(self, **kwargs: Any) -> None:
        """Render the environment

        The set of supported modes varies per environment. Some environments do not support rendering at
        all. By convention in Farama *gymnasium*, if mode is:

            * human: render to the current display or terminal and return nothing. Usually for human consumption.
            * rgb_array: Return a numpy.ndarray with shape (x, y, 3), representing RGB values for an x-by-y pixel image,
              suitable for turning into a video.
            * ansi: Return a string (str) or StringIO.StringIO containing a terminal-style text representation.
              The text can include newlines and ANSI escape sequences (e.g. for colors).

        """
        self.__jl.render(self._jlenv, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Return the item if it is set on the python object
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        # If the item isn't set on the python object, check whether _jlenv exists and has the item
        if "_jlenv" in self.__dict__:
            try:
                return getattr(self._jlenv, name)
            except Exception:
                pass

            try:
                return partial(getattr(self.__jl, name), self._jlenv)
            except Exception:
                pass

        raise AttributeError(f"Could not get {name} from python or julia environment.")

    def __setattr__(self, name: str, value: Any) -> None:
        # Try to set on _jlenv
        if "_jlenv" in self.__dict__ and hasattr(self._jlenv, name):
            try:
                jl_setattribute(self._jlenv, name, value)
            except BaseException as e:
                raise AttributeError(f"Could not set {name} on julia environment: {e}") from e

        # Otherwise set on the python environment.
        super().__setattr__(name, value)
