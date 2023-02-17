from __future__ import annotations

import time
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
from attrs import define
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import safe_mean, set_random_seed
from stable_baselines3.common.vec_env import VecNormalize

from eta_utility import get_logger
from eta_utility.util_julia import check_julia_package

if check_julia_package():
    from julia import Main as Jl  # noqa: I900
    from julia.ju_extensions.Agents import Nsga2 as ju_NSGA2  # noqa: I900

if TYPE_CHECKING:
    import io
    import pathlib
    from typing import Any, Callable

    import torch as th
    from julia import _jlwrapper  # noqa: I900
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
    from stable_baselines3.common.vec_env import VecEnv

Jl.eval("using PyCall")
Jl.eval("import ju_extensions.Agents.Nsga2")
log = get_logger("eta_x.agents")


@define
class _VariableParameters:
    """VariableParameters define the minimum and maximum values as well as the data type of the variables."""

    #: Data type of the variable (can be 'int' or 'float').
    dtype: str
    #: Minimum value of the variable.
    minimum: float
    #: Maximum value of the variable.
    maximum: float

    @classmethod
    def from_space(cls, space: spaces.Space) -> list[_VariableParameters]:
        """Create _VariableParameters from a gym space object.

        :param space: Gym space description object.
        :return: List of _VariableParameters objects (one object for each variable).
        """
        if isinstance(space, spaces.Box):
            dtype = "int" if space.dtype in {np.int32, np.int16, np.int8, np.int64} else "float"
            return [cls(dtype, minimum=space.low[key], maximum=space.high[key]) for key, _ in enumerate(space.shape)]

        elif isinstance(space, spaces.MultiDiscrete):
            return [cls("int", minimum=0, maximum=int(dim)) for dim in space.nvec]

        elif isinstance(space, spaces.MultiBinary):
            return [cls(dtype="int", minimum=0, maximum=1) for _ in range(space.n)]

        elif isinstance(space, spaces.Discrete):
            return [cls(dtype="int", minimum=0, maximum=int(space.n))]

        else:
            raise ValueError("Unknown type of space for variable parameters.")


class Nsga2(BaseAlgorithm):
    """The NSGA2 class implements the non-dominated sorting genetic algorithm 2

    The agent can work with discrete event systems and with continous or mixed integer problems. Alternatively a
    mixture of the above may be specified.

    The action space can specify both events and variables using spaces.Dict in the form::

        action_space= spaces.Dict({'events': spaces.Discrete(15),
                                   'variables': spaces.MultiDiscrete([15]*3)})

    This specifies 15 events and an additional 3 variables. The variables will be integers and have an upper
    boudary value of 15. Other spaces (except Tuple and Dict) can be defined for the variables. Events only takes
    the Discrete space as an input.

    When events is specified, a list will be returned with ordered values, that should achieve a near optimal
    reward. For variables the values will be adjusted to achieve the highest reward. Upper and lower boundaries as
    well as types will be infered from the space.

    .. note:: This agent does not use the observation space. Instead it only relies on rewards returned by the
        environment. Returned rewards can be tuples, if multi-objective optimization is required. Existing
        Environments do not have to be adjusted, however. The agent will also accept standard rewards and will
        ignore any observation spaces.

    .. note:: The number of environments must be equal to the population for this agent because it needs one
        environment for the evaluation of every solution. This allows for solutions to be evaluated in parallel.

    :param policy: Agent policy. Parameter is not used in this agent.
    :param env: Environment to be optimized.
    :param learning_rate: Reduction factor for the crossover and mutation rates (default: 1).
    :param verbose: Logging verbosity.
    :param population: Maximum number of parallel solutions (>= 2).
    :param mutations: Chance for mutations in existing solutions (between 0 and 1).
    :param crossovers: Chance for crossovers between solutions (between 0 and 1).
    :param n_generations: Number of generations to run the algorithm for.
    :param max_cross_len: Maximum number of genes (as a proportion of total elements) to cross over between
        solutions (between 0 and 1) (default 1).
    :param max_retries: Maximum number of tries to find new values before the algorithm fails and returns.
        Using the default should usually be fine (default: 10000).
    :param sense: Determine whether the algorithm looks for minimal ("minimize") or maximal ("maximize")
        rewards (default: "minimize")
    :param tensorboard_log: the log location for tensorboard (if None, no logging).
    :param seed: Seed for the pseudo random generators.
    :param _init_setup_model: Determine whether model should be initialized during setup
    """

    def __init__(
        self,
        policy: type[BasePolicy],
        env: VecEnv,
        learning_rate: float | Schedule = 1.0,
        verbose: int = 2,
        *,
        population: int | None = None,
        mutations: float | None = None,
        crossovers: float | None = None,
        n_generations: int | None = None,
        max_cross_len: float = 1,
        max_retries: int = 100000,
        sense: str = "minimize",
        seed: int | None = None,
        tensorboard_log: str | None = None,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        # Some types are incorrectly defined in the super class; this fixes it for this class and suppresses warnings
        self.start_time: float | None  # type: ignore
        self.lr_schedule: Callable
        self.policy_class: type[BasePolicy]
        self.policy: BasePolicy  # type: ignore

        # Set default values for superclass arguments
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            support_multi_env=True,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
                spaces.Dict,
            ),
            **kwargs,
        )
        if self.observation_space is not None and self.action_space is not None:
            self.policy = self.policy_class(self.observation_space, self.action_space, **self.policy_kwargs)

        log.setLevel(int(verbose * 10))

        if not 0 <= max_cross_len <= 1:
            raise ValueError("The maximum crossover length must be between 0 and 1 (proportion of total length).")
        if self.env is None:
            raise ValueError("The NSGA2 agent needs a specific environment to work correctly. Cannot use env = None.")
        if isinstance(self.env, VecNormalize):
            raise TypeError("The NSGA2 agent does not allow the use of normalized environments.")

        if sense not in {"minimize", "maximize"}:
            raise ValueError(f"The optimization sense must be one of 'minimize' or 'maximize', got {sense}.")

        #: Maximum number of parallel solutions (>= 2).
        self.population: int | None = population
        #: Chance for mutations in existing solutions (between 0 and 1).
        self.mutations: float | None = mutations
        #: Chance for crossovers between solutions (between 0 and 1).
        self.crossovers: float | None = crossovers
        #: Maximum number of genes (as a proportion of total elements) to cross over between solutions
        #: (between 0 and 1) (default 1).
        self.max_cross_len: float = max_cross_len
        #: Maximum number of tries to find new values before the algorithm fails and returns.
        #: Using the default should usually be fine (default: 10000).
        self.max_retries: int = max_retries
        #: Sense of the optimization (maximize or minimize).
        self.sense: str = sense
        #: Maximum number of generations to run for.
        self.n_generations: int | None = n_generations
        #: Maximum value of the reward (positive or negative infinity, depending on the optimization sense).
        self._max_value = np.inf if sense == "minimize" else -np.inf

        #: Parameters defining, how the events chromosome is generated. This is determined
        #: automatically from the action space.
        self.event_params: int = 0
        #: Parameters defining how the variables chromosome is generated. This is determined
        #: automatically from the action space.
        self.variable_params: list[_VariableParameters] = []

        #: Parent generation of solutions.
        self.generation_parent: _jlwrapper = []
        #: Offspring generation of solutions.
        self.generation_offspr: _jlwrapper = []
        #: Current learning rate of the algorithm.
        self._current_learning_rate: float = 1.0

        #: List of solutions which have been seen before (avoids duplicate evaluation of equivalent solutions.
        self.seen_solutions: int = 0
        #: Total number of retries needed during evolution to generate unique solutions.
        self.total_retries: int = 0
        #: List of current minimal values for all parts of the reward
        self.current_minima: np.ndarray = np.full(1, self._max_value, dtype=np.float64, order="F")

        #: Buffer for actions
        self.ep_actions_buffer: deque = deque(maxlen=100)
        #: Buffer for rewards
        self.ep_reward_buffer: deque = deque(maxlen=100)

        self._setup_lr_schedule()
        self._update_learning_rate()

        # Initialize and parametrize the julia functions.
        self.__jl_agent: _jlwrapper
        self._jl_Algorithm = Jl.eval(
            "pyfunctionret("
            "Nsga2.Algorithm, Any, Int, Float64, Float64, Int, Int, Int, "
            "Nsga2.VariableParameters, Float64, String, UInt64"
            ")"
        )
        self._jl_create_generation = Jl.eval("pyfunctionret(Nsga2.create_generation, Any, PyAny, Bool)")
        self._jl_initialize_rnd = Jl.eval("pyfunctionret(Nsga2.initialize_rnd!, Int, PyAny, PyAny)")
        self._jl_reinitialize_rnd = Jl.eval("pyfunctionret(Nsga2.initialize_rnd!, Int, PyAny, PyAny, Vector{Int})")
        self._jl_evolve = Jl.eval("pyfunctionret(Nsga2.evolve!, Int, PyAny, PyAny, PyAny, Float64)")
        self._jl_evaluate_solutions = Jl.eval(
            "pyfunctionret(" "Nsga2.evaluate!, Tuple{Vector{Float64}, Int}, PyAny, PyAny, PyAny, PyAny" ")"
        )
        self._jl_store_reward = Jl.eval("pyfunctionret(Nsga2.py_store_reward, nothing, PyAny, PyArray)")
        self._jl_get_actions = Jl.eval("pyfunctionret(Nsga2.py_actions, PyObject, PyAny)")

        if _init_setup_model:
            self._setup_model()

        log.info(
            f"Agent initialized with parameters population:{self.population}, mutations: {self.mutations}, "
            f"crossovers: {self.crossovers}."
        )

    def _event_and_variable_params(self) -> tuple[int, list[_VariableParameters]]:
        """Read event parameters and variable parameters from the action space.

        :return: Tuple of the events and variable configurations.
        """
        event_params = 0
        variable_space = None

        # If the type of the action space is spaces.Dict, it could contain events as well as variables.
        # Other spaces only contain variables.
        if isinstance(self.action_space, spaces.Dict):
            # Extract events space
            if "events" in self.action_space.spaces:
                if not isinstance(self.action_space.spaces["events"], spaces.Discrete):
                    raise ValueError(
                        f"Events must be specified as a discrete space. Received {type(self.action_space['events'])}."
                    )

                event_params = self.action_space.spaces["events"].n

            # Extract variables spaces.
            if "variables" in self.action_space.spaces:
                variable_space = self.action_space.spaces["variables"]

        else:
            variable_space = self.action_space

        # Set up the variable parameters by creating VariableParameters objects.
        variable_params = []
        if variable_space is not None:
            variable_params = _VariableParameters.from_space(variable_space)

        log.debug(
            f"Successfully read action space information. "
            f"Length of events: {event_params}, length of variables: {len(variable_params)}."
        )

        return event_params, variable_params

    def _setup_jl_agent(self) -> None:
        self.__jl_agent = self._jl_Algorithm(
            self.population,
            self.mutations,
            self.crossovers,
            self.max_cross_len,
            self.max_retries,
            self.event_params,
            self.variable_params,
            self._max_value,
            self.sense,
            self.seed,
        )

    def _setup_model(self) -> None:
        """Set up the model by taking values from the supplied action space and initializing the first two parent
        generations.
        """
        log.debug("Starting agent initialization.")
        # Set up learning rate and random seeding for all submodules.
        self._setup_lr_schedule()
        # Read the event and variable parameters from the action space.
        self.event_params, self.variable_params = self._event_and_variable_params()

        self._setup_jl_agent()

        self.set_random_seed(self.seed)

        log.debug("Successfully initialized NSGA 2 agent.")

    def _update_learning_rate(self, optmimizers: list[th.optim.Optimizer] | th.optim.Optimizer | None = None) -> None:
        """Update the learning rate as well as mutation and crossover rates. The mutation and crossover rates depend
        on the learning rate. Thus, a learning rate schedule will affect the crossover and mutation probabilities for
        each generation.

        :param optimizers: List of torch optimizers (not used by Nsga2).
        """
        self._current_learning_rate = self.lr_schedule(self._current_progress_remaining)

    def _check_learn_config(self) -> None:
        # Check configuration of the algorithm for compatibility
        if self.population is None or self.population < 2:
            raise ValueError("The population size must be at least two.")
        if self.mutations is None or (not 0 <= self.mutations < 1):
            raise ValueError("The mutation rate must be between 0 and 1.")
        if self.crossovers is None or (not 0 <= self.crossovers < 0.5):
            raise ValueError(
                "The crossover rate must be between 0 and 0.5 (cannot cross more than half of population)."
            )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> Nsga2:
        """
        Return a trained model. The environment which the agent is training on should return an info dictionary when
        a solution is invalid. The info dictionary should contain a 'valid' key which is set to false in that case.
        If there are too many invalid solutions (more than half of the population), the agent will try to
        re-initialize these solutions until there is a sufficient number of valid solutions.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether to reset the current timestep number (used in logging)
        :param progress_bar: Parameter to show progress bar, used by stable_baselines (currently unused!)
        :return: the trained model
        """
        self._check_learn_config()
        assert self.crossovers is not None, "Set the crossover rate before starting to learn."
        assert self.mutations is not None, "Set the mutation rate before starting to learn."

        if self.n_generations is not None and total_timesteps > self.n_generations:
            total_timesteps = self.n_generations

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
        )
        assert self.ep_info_buffer is not None, "Make sure that ep_info_buffer is exists before starting to learn."
        assert self.start_time is not None, "Make sure that start_time is set before starting to learn."
        callback.on_training_start(locals(), globals())

        log.info(
            f"Starting optimization for {total_timesteps} generations with parameters: "
            f"crossover rate: {self.crossovers}, mutation rate: {self.mutations}, population: {self.population}."
        )

        iteration_time = time.time()
        evolve_time = iteration_time
        eval_time = iteration_time
        sorting_time = iteration_time
        retries = 0
        invalid_sol = 0

        # Initialize the parent generation in case it is empty (usually when the algorithm is first initialized)
        if Jl.length(self.generation_parent) == 0:
            log.debug("Initializing parent generation.")
            self.generation_parent = self._jl_create_generation(self.__jl_agent, False)
            retries = self._jl_initialize_rnd(self.__jl_agent, self.generation_parent)
            evolve_time = time.time()
            self.total_retries += retries

            log.debug("Evaluating parent generation.")
            self.generation_parent, invalid_sol = self._evaluate(self.generation_parent)
            eval_time = time.time()
            sorting_time = time.time()  # No sorting during first step.

            log.info(f"Successfully initialized first parent generation with {self.population} solutions.")

        # Enter time step loop (each loop is one generation of solutions)
        while self.num_timesteps < total_timesteps:
            self.num_timesteps += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record(
                        "general/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer])
                    )
                self.logger.record("time/total", time.time() - self.start_time / 1000000000, exclude="tensorboard")
                self.logger.record("time/iteration", time.time() - iteration_time)
                self.logger.record("time/evolve", evolve_time - iteration_time)
                self.logger.record("time/evaluate", eval_time - evolve_time)
                self.logger.record("time/sort", sorting_time - eval_time)
                self.logger.record("train/retries", retries)
                self.logger.record("train/total_retries", self.total_retries)
                self.logger.record("train/learning_rate", self._current_learning_rate)
                self.logger.record("train/mutation_rate", self.mutations * self._current_learning_rate)
                self.logger.record("train/crossover_rate", self.crossovers * self._current_learning_rate)
                self.logger.record("train/seensolutions", self.seen_solutions)
                self.logger.record("evaluate/invalid", invalid_sol)
                for idx, val in enumerate(self.current_minima):
                    self.logger.record(f"evaluate/minimum_{idx}", val)
                self.logger.dump(step=self.num_timesteps)

            iteration_time = time.time()
            self._update_learning_rate()

            # Create empty offspring generation
            log.debug("Initializing offspring generation and performing evolution.")
            self.generation_offspr = self._jl_create_generation(self.__jl_agent, True)
            retries = self._jl_evolve(
                self.__jl_agent, self.generation_offspr, self.generation_parent, self._current_learning_rate
            )
            evolve_time = time.time()
            self.total_retries += retries

            log.debug("Evaluating offspring generation.")
            self.generation_offspr, retries = self._evaluate(self.generation_offspr)
            self.total_retries += retries
            eval_time = time.time()

            log.debug("Performing non-dominated sort with parent and offspring")
            new_generation_parent = self._jl_create_generation(self.__jl_agent, True)
            self.current_minima, self.seen_solutions = self._jl_evaluate_solutions(
                self.__jl_agent, self.generation_offspr, self.generation_parent, new_generation_parent
            )
            self.generation_parent = new_generation_parent
            sorting_time = time.time()

            log.debug(f"Successfully created and evaluated offspring generation with {self.population} solutions.")
            iteration += 1

            if not callback.on_step():
                break

        callback.on_training_end()

        return self

    def _evaluate(self, generation: _jlwrapper) -> tuple[_jlwrapper, int]:
        """Evaluate all solutions in the generation and store rewards

        :param generation: Sequence of solutions to evaluate
        :return: Sequence of evaluated solutions
        """
        assert self.env is not None, "The agent needs to know the environment to evaluate solutions."
        assert self.population is not None, "The agent needs to know the population size to evaluate solutions."

        rewards = None
        retries = 0
        while retries < self.max_retries:
            observations, rewards, dones, infos = self.env.step(self._get_actions(generation))
            self._update_info_buffer(infos, dones)

            # Ensure that there are always multiple rewards for every solution.
            if len(rewards.shape) == 1:
                rewards = np.reshape(rewards, (len(rewards), 1), order="F")

            solution_invalid = []
            for idx, _ in enumerate(rewards):
                if "valid" in infos[idx] and infos[idx]["valid"] is False:
                    rewards[idx] = np.full((len(rewards[idx]),), self._max_value, dtype=np.float64)
                    solution_invalid.append(idx + 1)

            if len(solution_invalid) < self.population / 2:
                break
            else:
                retries += len(solution_invalid)
                retries += self._jl_reinitialize_rnd(self.__jl_agent, generation, solution_invalid)
                log.info(
                    f"Randomized the generation again because "
                    f"there were too many invalid solutions: {len(solution_invalid)}; retries: {retries}"
                )

        assert rewards is not None
        self._store_reward(generation, rewards)
        return generation, retries

    def _get_actions(self, generation: _jlwrapper) -> np.ndarray:
        """Convert and retrieve actions from the julia agent. This also stores the actions in self.ep_actions_buffer.

        :param generation: Generation object to get actions from.
        :return: Numpy array of the actions.
        """
        self.ep_actions_buffer.append(self._jl_get_actions(generation))
        return self.ep_actions_buffer[-1]

    def _store_reward(self, generation: _jlwrapper, rewards: np.ndarray) -> None:
        """Store reward in the generation object and in self.ep_reward_buffer.

        :param generation: Generation object to store rewards in.
        :param rewards: Array of rewards to store.
        """
        self.ep_reward_buffer.append(rewards)
        self._jl_store_reward(generation, rewards)

    def set_random_seed(self, seed: int | None = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, julia)

        .. note::
            action_space is currently not seeded because the gym version stable_baselines3 relies on has broken
            seeding for space.Dict. Seeding is fixed in gym==0.26. This can be updated once stable_baselines3 supports
            newer versions of gym.

        :param seed: Seet for the
        """
        if seed is None:
            return
        set_random_seed(seed)

        ju_NSGA2.seed_b(self.__jl_agent, seed)

        if self.env is not None:
            self.env.seed(seed)

    def _excluded_save_params(self) -> list[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling.

        :return: List of parameters that should be excluded from being saved with pickle.
        """

        excluded_params = super()._excluded_save_params()
        excluded_params.extend(
            [
                "_Nsga2__jl_agent",
                "_jl_Algorithm",
                "_jl_create_generation",
                "_jl_initialize_rnd",
                "_jl_reinitialize_rnd",
                "_jl_evolve",
                "_jl_evaluate_solutions",
                "_jl_store_reward",
                "_jl_get_actions",
                "generation_parent",
                "generation_offspr",
            ]
        )
        return excluded_params

    @classmethod
    def load(
        cls: type[Nsga2],
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> Nsga2:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace upon loading. If a variable is present in
            this dictionary as a key, it will not be deserialized and the corresponding item will be used instead.
        :param print_system_info: Whether to print system info from the saved model and the current system info
            (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training to avoid unexpected behavior.
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """
        model: Nsga2 = super().load(path, env, device, custom_objects, print_system_info, force_reset, **kwargs)

        model._setup_jl_agent()
        model._load_generation()

        return model

    def _load_generation(self) -> None:
        setup_generation = Jl.eval("pyfunctionret(Nsga2.load_generation, Any, PyArray, PyArray, Float64)")
        self.generation_parent = setup_generation(
            self.ep_actions_buffer[-1]["events"], self.ep_actions_buffer[-1]["variables"], self._max_value
        )
