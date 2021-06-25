"""The genetic optimiser implements a genetic algorithm based on the NSGA-II

The NSGA-II was first developed and implemented by Kalyanmoy Deb, Amrit Pratap, Sameer Agarwal and T. Meyarivan in their
collaborative paper 'A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II' in 2002.

This implementation is based on their thoughts and ideas and is adapted to optimising production plans. Some of the
special concepts employed here include the use of two separate chromosomes. The first chromosome (called chromosome)
includes the jobs and the order of the chromosome determines the job operation execution order. The second chromosome
is a normal chromosome that contains simple integer values. The values of the second chromosome (called numbchromosome)
are used to determine the length of inserted pauses.

The nondominated sorting algorithm is also improved to use as few loops as possible by comparing two values at once as
opposed to doing the comparison for each solution separately. Taking n as the number of solutions per generation
this means that the solution comparison loop always needs
sum from (i=1 to n) of (n-i) = (2n^2 - n) iterations. The construction of all offspring fronts takes
sum from (i=1 to n/2) of (n-i) = (3n^2/2 - n/2) iterations. This means that this implementation of the algorithm has an
overall time complexity of O(7/2 n^2) as compared to the complexity of the original NSGA-II of O(8n^2)
"""

import operator
import pickle
import time
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import mmh3
import numpy as np
from gym import spaces
from stable_baselines.common import BaseRLModel
from stable_baselines.common.vec_env import VecEnv

from eta_utility import get_logger
from eta_utility.type_hints.custom_types import Numbers

from ..common import ProcessPool, cpu_count
from ..common.policies import NoPolicy

log = get_logger("eta_x.agents")


class NSGA2(BaseRLModel):
    """The NSGA2 class implements the non-dominated sorting genetic algorithm 2

    The agent can work with discrete event systems and with continous or mixed integer problems. Alternatively a
    mixture of the above may be specified.

    The action space can be specify both events and variables using spaces.Dict in the form::

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

    :param policy: Agent policy. Parameter is not used in this agent
    :param env: Environment to be optimized
    :param verbose: Logging verbosity
    :param population: Maximum number of parallel solutions (>= 2)
    :param mutations: Chance for mutations in existing solutions (between 0 and 1)
    :param crossovers: Chance for crossovers between solutions (between 0 and 1)
    :param max_cross_len: Maximum number of genes (as a proportion of total elements) to cross
                              over between solutions (between 0 and 1)
    :param max_retries: Maximum number of tries to find new values before the algorithm fails and returns.
                            (default: 1000) Using the default should usually be fine.
    :param threads: Use this number of threads to perform calculations (default: 4)
    :param _init_setup_model: Determine whether model should be initialized during setup
    :param kwargs: Additional arguments as specified in stable_baselins.BaseRLModel
    """

    def __init__(
        self,
        policy: NoPolicy,
        env: VecEnv,
        verbose: int = 4,
        *,
        population: int,
        mutations: float,
        crossovers: float,
        max_cross_len: float = 1,
        max_retries: int = 10000,
        threads: int = cpu_count(),
        _init_setup_model=True,
        **kwargs,
    ) -> None:

        if population < 2:
            raise ValueError("The population size must be at least two.")
        if not 0 <= mutations < 1:
            raise ValueError("The mutation rate must be between 0 and 1.")
        if not 0 <= crossovers < 1:
            raise ValueError("The crossover rate must be between 0 and 1.")
        if not 0 <= max_cross_len <= 1:
            raise ValueError("The maximum cross length must be between 0 and 1 (proportion of total length).")

        # Ensure that arguments required by super class are always present
        if "requires_vec_env" not in kwargs:
            kwargs["requires_vec_env"] = True
        if "policy_base" not in kwargs:
            kwargs["policy_base"] = None

        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        log.setLevel(int(verbose * 10))
        self.population: int = population
        self.mutations: float = mutations
        self.crossovers: float = crossovers
        self.max_cross_len: float = max_cross_len

        # Additional parameters that usually do not need to be adjusted from their default values.
        self._max_retries: int = max_retries
        self._threads: int = threads

        # Create a first random number generator.
        # Multiple random number generators are necessary so that different processes don't overlap.
        self._seed_sequence: np.random.SeedSequence = np.random.SeedSequence(self.seed)
        self._rng: np.random.Generator = np.random.default_rng(self._seed_sequence)  # When seed is None,
        # random entropy will be used.
        log.info(f"Initialized random generator with seed {self.seed}, entropy: {self._seed_sequence.entropy}")

        self._generation_parent: List[float] = []
        self._seen_solutions: Set[str] = set()
        self.generation: int = 0
        self._model_trained: bool = False

        if threads > 1:
            self._chunksize: int = self.population // self._threads
            self._processes: Optional[ProcessPool] = ProcessPool(threads, seed_sequence=self._seed_sequence)
            if hasattr(self.env, "set_proc_pool"):
                self.env.set_proc_pool(self._processes)
            log.debug(f"Set up processing pool with {self._chunksize} solutions per process.")
        else:
            self._chunksize = self.population
            self._processes = None

        self._data_store: Dict[str, Dict[str, Any]] = {
            "settings": {
                "seed": self._seed_sequence.entropy,
                "population": self.population,
                "crossovers": self.crossovers,
                "mutations": self.mutations,
                "used_multiprocessing": False if self._processes is None else True,
                "threads": self._threads,
            },
            "stats": {
                "generation": self.generation,
                "total_time_elapsed": 0,
                "last_training_time_elapsed": 0,
                "last_training_proc_time_elapsed": 0,
                "total_retries": 0,
                "last_training_retries": 0,
                "hash_retry_history": np.zeros(1, dtype=np.int),
                "invalid_retry_history": np.zeros(1, dtype=np.int),
            },
            "results": {
                "last_min_values": np.full(1, np.inf, dtype=np.float),
                "min_value_history": [],
                "convergence_history": [],
                "solution_space": [],
            },
        }

        if _init_setup_model:
            self.setup_model()

        log.info(
            f"Agent initialized with parameters population:{self.population}, mutations: {self.mutations}, "
            f"crossovers: {self.crossovers}."
        )

    @classmethod
    def load(cls, load_path: str, env: Optional[VecEnv] = None, **kwargs) -> "NSGA2":
        """
        TODO: Reimplement this function using stable baselines facilities

        :param load_path:
        :param env:
        :param kwargs:
        :return:
        """
        with open(load_path, "rb") as f:
            loaded = pickle.load(f)

        args = {}
        all_args = {"population", "mutations", "crossovers", "max_cross_len", "max_retries", "threads"}
        for arg in all_args:
            load_arg = "_" + arg if arg in {"max_retries", "threads"} else arg
            args[arg] = kwargs[arg] if arg in kwargs else getattr(loaded, load_arg)
        args.update({"_init_setup_model": False, "requires_vec_env": True, "policy_base": None})

        verbose = kwargs["verbose"] if "verbose" in kwargs else 4
        obj = cls(loaded.policy, env, verbose, **args)

        # Restore previous object content
        obj.seed = loaded.seed
        obj._seed_sequence = loaded._seed_sequence
        obj._rng = loaded._rng
        log.info(f"Reinitialized random generator with restored seed {obj.seed}, entropy: {obj._seed_sequence.entropy}")

        obj._generation_parent = loaded._generation_parent
        obj._seen_solutions = loaded._seen_solutions
        obj.generation = loaded.generation

        obj._model_trained = loaded._model_trained
        obj._data_store = loaded._data_store

        return obj

    def setup_model(self) -> None:
        """Setup the model by taking values from the supplied action space and initializind the first two parent
        generations.
        """

        events = None
        vari = None
        params = None

        # Read the action space and determine whether events are present or not
        if isinstance(self.action_space, spaces.Dict) and "events" in self.action_space.spaces:
            if not isinstance(self.action_space.spaces["events"], spaces.Discrete):
                raise ValueError(
                    f"Events must be specified as a discrete space. " f'Received {type(self.action_space["events"])}.'
                )

            events = self.action_space.spaces["events"].n
            if "variables" in self.action_space.spaces:
                var_space = self.action_space.spaces["variables"]
            else:
                var_space = None
        else:
            var_space = self.action_space

        # Interpret the variables from the action space (if present) and infer upper and lower boundaries.
        if var_space is not None:
            vari = var_space.sample()
            params = []

            if isinstance(var_space, spaces.Box):
                dtype = "int" if var_space.dtype in {np.int32, np.int16, np.int8, np.int64} else "float"
                for key, _ in enumerate(var_space.shape):
                    params.append(
                        {
                            "dtype": dtype,
                            "min": var_space.low[key],
                            "max": var_space.high[key],
                        }
                    )

            elif isinstance(var_space, spaces.MultiDiscrete):
                for dim in var_space.nvec:
                    params = [{"dtype": "int", "min": 0, "max": dim}]

            elif isinstance(var_space, spaces.MultiBinary):
                params = [{"dtype": "int", "min": 0, "max": 1}]

            elif isinstance(var_space, spaces.Discrete):
                params = [{"dtype": "int", "min": 0, "max": var_space.n}]

            elif isinstance(var_space, spaces.Dict) or isinstance(var_space, spaces.Tuple):
                raise ValueError(f"Variables cannot be nested further. Received {type(var_space)}.")

        # Generate the first generation
        def empty_generation(this) -> Iterable["GeneticSolution"]:
            return (GeneticSolution() for _ in range(0, this.population))

        if self._threads > 1:
            self._generation_parent = self._processes.map(
                "initialize",
                empty_generation(self),
                args=(events, vari, params),
                method="modify_rng",
            )
        else:
            self._generation_parent = list(
                sol.initialize(events, vari, params, self._rng) for sol in empty_generation(self)
            )

        retries = 0
        for solution in self._generation_parent:
            while not self._check_hash(solution) and retries <= self._max_retries:
                solution.randomize(self._rng, flag=True)
            else:
                if retries >= self._max_retries:
                    raise ValueError("Cannot find any more alternative solutions.")

        self._generation_parent, _, _ = self._evaluate(self._generation_parent)

        log.debug("Successfully initialized NSGA 2 agent.")

    def _get_pretrain_placeholders(self) -> None:
        """Getting pretrain placeholders is not implemented for the genetic algorithm

        :return:
        """
        pass

    def pretrain(self, env_method: str, **pretrain_kwargs) -> None:
        """Pretrain the genetic algorithm by setting up the starting solutions using results created by an environment
        method.

        :param env_method: Environment method that provides pretraining solutions
        :param pretrain_kwargs: Arguments for the environment method
        :return:
        """
        if "count" not in pretrain_kwargs:
            pretrain_kwargs["count"] = self.population

        pretrained = self.env.env_method(env_method, indices=0, **pretrain_kwargs)
        idx = 0
        for pre in pretrained[0]:
            self._generation_parent[idx].echromo = np.asarray(pre, np.int)
        parent, hash_rt, invalid_rt = self._evaluate(self._generation_parent)

        offspr = deepcopy(self._generation_parent)
        for sol in offspr:
            sol.vchromo = np.zeros_like(sol.vchromo)

        offspr, hash_rt, invalid_rt = self._evaluate(offspr)

        fronts = self._sort_nondominated(parent, offspr)

        generation_parent = []
        for front in fronts:
            if len(front) <= self.population - len(generation_parent):
                generation_parent.extend(front)
            else:
                front = self._sort_crowding_distance(front, self.population - len(generation_parent))
                generation_parent.extend(front)
                break

        self._generation_parent = generation_parent

    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Callable] = None,
        log_interval: int = 10,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
    ) -> "NSGA2":
        """Return a trained model.

        .. note:: Parameters tb_log_name and reset_num_timesteps are currently ignored because tensorboard logging is
            not implemented

        :param total_timesteps: The total number of generations to train
        :param callback: boolean function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for tensorboard log
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :return: the trained model
        """
        # prepare statistics
        total_retries = 0
        if self.generation == 0:
            hash_retries = np.zeros(total_timesteps, dtype=np.int)
            invalid_retries = np.zeros(total_timesteps, dtype=np.int)
        else:
            hash_retries = np.concatenate(
                (
                    self._data_store["stats"]["hash_retry_history"],
                    np.zeros(total_timesteps, dtype=np.int),
                )
            )
            invalid_retries = np.concatenate(
                (
                    self._data_store["stats"]["invalid_retry_history"],
                    np.zeros(total_timesteps, dtype=np.int),
                )
            )

        start_time = time.time()
        start_proc_time = time.perf_counter()

        # learn
        log.info(
            f"Starting optimization for {total_timesteps} generations with parameters: "
            f"crossover rate: {self.crossovers}, mutation rate: {self.mutations}, population: {self.population}."
        )

        for i in range(0, total_timesteps):
            self.generation += 1
            offspr = self._crossover(self._mutate(self._generation_parent))

            for solution in offspr:
                if not self._check_hash(solution):
                    for _ in range(0, self._max_retries):
                        solution = solution.mutate(self.mutations, self._rng, flag=True)
                        hash_retries[i] += 1
                        if self._check_hash(solution):
                            break

                    else:
                        # If no new solution could be found through mutation, try creating a new solution by randomizing
                        solution.randomize(flag=True)
                        hash_retries[i] += 1
                        if not self._check_hash(solution):
                            raise ValueError("Cannot find any more alternative solutions.")

            offspr, hash_rt, invalid_rt = self._evaluate(offspr)
            hash_retries[i] += hash_rt
            invalid_retries[i] += invalid_rt
            fronts = self._sort_nondominated(self._generation_parent, offspr)

            generation_parent = []
            for front in fronts:
                if len(front) <= self.population - len(generation_parent):
                    generation_parent.extend(front)
                else:
                    front = self._sort_crowding_distance(front, self.population - len(generation_parent))
                    generation_parent.extend(front)
                    break

            self._generation_parent = generation_parent

            total_retries += hash_retries[i] + invalid_retries[i]

            if i % log_interval == 0:
                log.info(
                    f"Completed {self.generation} generations with {total_retries} retries "
                    f"during mutation and crossover."
                )

        # log interesting stats to _data_store
        time_elapsed = time.time() - start_time
        proc_time_elapsed = time.perf_counter() - start_proc_time

        log.info(
            f"Completed optimization after {self.generation} generations. Stats: "
            f"time elapsed: {time_elapsed}, total retries: {total_retries}."
        )
        self._data_store["stats"].update(
            {
                "total_time_elapsed": self._data_store.get("total_time_elapsed", 0) + time_elapsed,
                "last_training_time_elapsed": time_elapsed,
                "last_training_proc_time_elapsed": proc_time_elapsed,
                "total_retries": self._data_store.get("total_retries", 0) + total_retries,
                "last_training_retries": total_retries,
                "hash_retry_history": hash_retries,
                "invalid_retry_history": invalid_retries,
                "generation": self.generation,
            }
        )

        sol_space = np.empty(
            (len(self._generation_parent), len(self._generation_parent[0].reward)),
            dtype=np.float,
        )
        for num, s in enumerate(self._generation_parent):
            sol_space[num, :] = s.reward.copy()
        self._data_store["results"].update({"solution_space": sol_space})

        self._model_trained = True
        return self

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> None:
        """
        TODO: Implement this

        :param observation:
        :param state:
        :param mask:
        :param deterministic:
        :return:
        """
        pass

    def action_probability(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        actions: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """This function is not implemented for the genetic algorithm because it cannot determine the probability
        of individual actions.
        """
        raise NotImplementedError("The genetic algorithm cannot determine an action probability.")

    def save(self, save_path: str, **kwargs) -> None:
        """Save model after training

        TODO: Implement this using stable baselines facilities

        :param save_path:
        :param kwargs:
        :return:
        """
        if self._threads > 1:
            self._processes.close()
            self._processes.join()
            self._processes = None

        self.env = None

        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    def _mutate(self, generation: Sequence["GeneticSolution"]) -> List["GeneticSolution"]:
        """Mutate the generation

        :param generation: List of solutions to be mutated
        :return: List of mutated solutions
        """
        len_genome = len(generation[0])

        # Number of solutions to be mutated such that each mutated solution has two mutations.
        num_solutions = min(int(self.population * len_genome * self.mutations / 2), self.population)
        adjusted_rate = self.population * len_genome * self.mutations / (num_solutions * len_genome)
        mutate_solutions = self._rng.choice(self.population, num_solutions, replace=False, shuffle=False)

        for i in mutate_solutions:
            generation[i].mutate_flag = True

        if self._threads > 1:
            offspr = self._processes.map("mutate", generation, args=(adjusted_rate,), method="return_rng")
        else:
            offspr = list(sol.mutate(adjusted_rate, self._rng) for sol in generation)
        return offspr

    def _crossover(self, generation: Sequence["GeneticSolution"]) -> List["GeneticSolution"]:
        """Cross some solutions of the generation

        :param generation: Lists of solution to be crossed over
        :return: List of crossed over solutions
        """
        num_crossovers = int(self.population * self.crossovers)
        matches_from = self._rng.choice(self.population, num_crossovers, replace=False)
        matches_to = self._rng.choice(self.population, num_crossovers, replace=False)

        for idx, _from in enumerate(matches_from):
            while matches_to[idx] == _from:
                matches_to[idx] = self._rng.integers(0, self.population)

            generation[_from].cross_with = matches_to[idx]

        if self._threads > 1:
            offspr = self._processes.map(
                "crossover",
                generation,
                args=(generation, self.max_cross_len),
                method="return_rng",
            )
        else:
            offspr = list(sol.crossover(generation, self.max_cross_len, self._rng) for sol in generation)

        return offspr

    def _evaluate(self, generation: MutableSequence["GeneticSolution"]) -> Tuple[List["GeneticSolution"], int, int]:
        """Evaluate all solutions in the generation and store rewards

        :param generation: Sequence of solutions to evaluate
        :return: Sequence of evaluated solutions
        """
        invalid_retries = 0
        hash_retries = 0

        _, rewards, _, infos = self.env.step([(sol.echromo, sol.vchromo) for sol in generation])
        solution_invalid = set()
        for idx, sol in enumerate(generation):
            if "valid" in infos[idx] and infos[idx]["valid"] is False:
                rewards[idx] = np.full((len(rewards[idx])), np.inf, dtype=np.float)
                solution_invalid.add(idx)
            sol.reward = rewards[idx]

        while invalid_retries + hash_retries < self._max_retries and len(solution_invalid) >= 0.25 * len(generation):
            invalid_retries += len(solution_invalid)
            new_sol_invalid = set()
            idx = 0

            if self._threads > 1:
                for sol in solution_invalid:
                    generation[sol].randomize_flag = True
                new_sol = self._processes.map(
                    "randomize",
                    (generation[sol] for sol in solution_invalid),
                    method="modify_rng",
                )
                _, rewards, _, infos = self.env.step([(sol.echromo, sol.vchromo) for sol in new_sol])

                for key in solution_invalid:
                    if "valid" in infos[idx] and infos[idx]["valid"] is False:
                        rewards[idx] = np.full((len(rewards[idx])), np.inf, dtype=np.float)
                        new_sol_invalid.add(key)
                    new_sol[idx].reward = rewards[idx]
                    generation[key] = new_sol[idx]
                    idx += 1

            else:
                s_invalid = sorted(solution_invalid)
                for sol in solution_invalid:
                    generation[sol].randomize(self._rng, flag=True)
                _, rewards, _, infos = self.env.step(
                    [(generation[sol].echromo, generation[sol].vchromo) for sol in s_invalid]
                )

                for sol in s_invalid:
                    if "valid" in infos[idx] and infos[idx]["valid"] is False:
                        rewards[idx] = np.full((len(rewards[idx])), np.inf, dtype=np.float)
                        new_sol_invalid.add(sol)
                    generation[sol].reward = rewards[idx]
                    idx += 1

            for sol in solution_invalid:
                while not self._check_hash(generation[sol]) and invalid_retries + hash_retries < self._max_retries:
                    hash_retries += 1
                    generation[sol].randomize(self._rng)
                    _, rewards, _, infos = self.env.step([(generation[sol].echromo, generation[sol].vchromo)])
                    if "valid" in infos[0] and infos[0]["valid"] is False:
                        rewards[0] = np.full((len(rewards[idx])), np.inf, dtype=np.float)
                        new_sol_invalid.add(sol)
                    generation[sol].reward = rewards[0]

            solution_invalid = new_sol_invalid

        if len(solution_invalid) >= 0.5 * len(generation):
            raise ValueError("Could not find any more valid solutions.")

        return list(generation), hash_retries, invalid_retries

    def _sort_nondominated(
        self, parent: List["GeneticSolution"], offspr: List["GeneticSolution"]
    ) -> List[List["GeneticSolution"]]:
        """Perform non-dominated sort

        Perform the non-dominated sort and return as many fronts as necessary to completey fill the offspring
        generation. Calculating all the other fronts is unnecessary because they are discarded anyway. Also calculate
        some solution quality parameters.

        :param parent: Parent generation
        :param offspr: Offspring generation
        :return: List of pareto fronts, each containing a list of solutions
        """
        # assign local function names for faster lookup
        npall = np.all
        npany = np.any
        npgreater = np.greater
        npless = np.less
        npgeq = np.greater_equal
        npleq = np.less_equal

        if self.generation <= 1:
            self._data_store["results"]["last_min_values"] = parent[0].reward
        current_minima = self._data_store["results"]["last_min_values"].copy()

        population = parent + offspr

        len_population = len(population)
        fronts = [[]]
        # Compare solutions and assign domination values
        for _ in range(0, len_population):
            sol1 = population.pop()

            for sol2 in population:
                if npall(npleq(sol1.reward, sol2.reward)) and npany(npless(sol1.reward, sol2.reward)):
                    sol1.dominates.append(sol2)
                    sol2.dominatedby += 1

                elif npall(npgeq(sol1.reward, sol2.reward)) and npany(npgreater(sol1.reward, sol2.reward)):
                    sol1.dominatedby += 1
                    sol2.dominates.append(sol1)

            if sol1.dominatedby == 0:
                np.putmask(current_minima, npless(sol1.reward, current_minima), sol1.reward)
                fronts[0].append(sol1)
                sol1.rank = 0

        # Sort solutions into fronts
        leftsolutions = len_population - len(fronts[0])
        i = 0
        while leftsolutions > self.population:
            i += 1
            fronts.append([])
            for sol1 in fronts[i - 1]:
                for sol2 in sol1.dominates:
                    sol2.dominatedby -= 1
                    if sol2.dominatedby == 0:
                        sol2.rank = i
                        fronts[i].append(sol2)

            leftsolutions = leftsolutions - len(fronts[i])

        # Calculate some addition solution quality parameters and convergence
        self._data_store["results"]["min_value_history"].append(current_minima)
        self._data_store["results"]["convergence_history"].append(
            self._data_store["results"]["last_min_values"] - current_minima
        )
        self._data_store["results"]["last_min_values"] = current_minima

        return fronts

    @staticmethod
    def _sort_crowding_distance(front: List["GeneticSolution"], length: int) -> List["GeneticSolution"]:
        """Sort a front by crowding distance

        Take the front and sort it by crowding distance for all optimisation objectives. Return as many solutions as
        requested by length.

        TODO: This only works for one objective. NEEDS FIXING!

        :param front: List of solutions in the front to be sorted
        :param length: length of the list that is returned
        :return: List of length solutions with the highest crowding distance
        """

        # Preassign some values for fast access
        frontlength = len(front)
        attrgetter = operator.attrgetter

        # Perform the sort and crowding distance assignment for each objective
        for i in range(0, len(front[0].reward)):
            front.sort(key=lambda a, r=attrgetter("reward"): r(a)[i])
            maxmindist = front[-1].reward[i] - front[0].reward[i]
            if maxmindist <= 0:
                log.debug(f"Crowding distance sort failed because all solutions are equal in dimension {i}.")
                continue

            for idx in range(1, frontlength - 2):
                front[idx].crowding_distance += (front[idx - 1].reward[i] + front[idx + 1].reward[i]) / maxmindist

        # Return solutions with the highest crowding distance
        return sorted(front, key=attrgetter("crowding_distance"), reverse=True)[:length]

    def _check_hash(self, sol: "GeneticSolution") -> bool:
        """Check whether a solution hash has already been encountered

        :param GeneticSolution sol: Solution to be checked
        :return:
        """
        if hash(sol) not in self._seen_solutions:
            self._seen_solutions.add(hash(sol))
            return True
        else:
            return False

    def get_parameter_list(self) -> None:
        """
        TODO: Implement this

        :return:
        """
        pass


class GeneticSolution:
    """Solution object containing methods to simplify operation of the genetic algorithm.
    This should not be accessed directly.

    :param echromo: Events chromosome
    :param vchromo: Chromosome of variables
    :param params: List of Dictionaries of parameters. Dictionaries should be of the format:
                   {'dtype': 'int' or 'float', 'min': int or float, 'max': int or float}
                   If only one dictionary is given, it will be used for every variable. Otherwise one dictionary
                   should be provided for each variable.
    """

    # define some class attributes to improve efficiency of outside function access
    _hash_method = mmh3.hash
    registry = {"generators": {"int": "integers", "float": "uniform"}}

    def __init__(
        self,
        echromo: Optional[np.ndarray] = None,
        vchromo: Optional[np.ndarray] = None,
        params: Optional[List[Dict[str, Any]]] = None,
    ) -> None:

        self.echromo: Optional[np.ndarray] = echromo
        self.vchromo: Optional[np.ndarray] = vchromo
        self.params: List[Dict[str, Any]] = params
        self._hash = None

        self.cross_with = None
        self.mutate_flag: bool = False
        self.randomize_flag: bool = False
        self.reward = None

        self.dominates: List[int] = []
        self.dominatedby: int = 0
        self.crowding_distance: Numbers = np.inf

    def initialize(
        self,
        events: Union[int, List[Any]] = None,
        vari: Union[int, List[Union[int, float]]] = None,
        params: Optional[List[Dict[str, Union[str, int, float]]]] = None,
        rng: np.random.Generator = None,
    ) -> "GeneticSolution":
        """Initialize chromosomes of the solution. The solution can contain only events, only variables or both.
        If variables are used, parameters vari and params must be provided.

        :param events: Number of events or list of events. Contents of this list are ignored, only the length is used.
        :param vari: Number of variables or list of starting values. Random initialization is used if no starting values
                     are provided (integer value given)
        :param params: List of Dictionaries of parameters. .. seealso:: __init__
        :param rng: Random number generator. Defaults to numpy default_rng().
        :return: None
        """
        if rng is None:
            rng = np.random.default_rng()

        if events is not None:
            _len = len(events) if hasattr(events, "__len__") else events
            self.echromo = rng.permutation(_len)

        if vari is not None:
            assert type(params) is list
            if len(params) == 1:
                try:
                    func = getattr(rng, self.registry["generators"][params[0]["dtype"]])
                    vchromo = func(params[0]["min"], params[0]["max"], len(vari))
                except (IndexError, KeyError) as e:
                    raise ValueError(
                        f"Invalid parameter specification in params: {e}. " f"'dtype', 'min' and 'max' needed."
                    ) from e
            else:
                vchromo = np.empty(len(vari), float)
                try:
                    if hasattr(vari, "__len__"):
                        vchromo = np.asarray(vari)
                    else:
                        for i in range(0, vari):
                            func = getattr(rng, self.registry["generators"][params[i]["dtype"]])
                            vchromo[i] = func(params[i]["min"], params[i]["max"])
                except (IndexError, KeyError) as e:
                    raise ValueError(
                        f"Invalid parameter specification in params: {e}. " f"'dtype', 'min' and 'max' needed."
                    ) from e

            self.vchromo = vchromo
            self.params = params

        self._hash = None
        return self

    def randomize(self, rng: np.random.Generator = np.random.default_rng(), flag: bool = False) -> None:
        """Randomize both chromosomes of the solution. This essentially reinitializes the solution

        :param rng: Random number generator. Defaults to numpy default_rng().
        :param flag: Ignore self.randomize_flag
        :return: None
        """

        if self.randomize_flag or flag:
            if self.echromo is not None:
                rng.shuffle(self.echromo)

            if self.vchromo is not None:
                if len(self.params) == 1:
                    func = getattr(rng, self.registry["generators"][self.params[0]["dtype"]])
                    vchromo = func(self.params[0]["min"], self.params[0]["max"], len(self.vchromo))
                else:
                    vchromo = np.empty(len(self.vchromo), float)
                    for i in range(0, len(vchromo)):
                        func = getattr(rng, self.registry["generators"][self.params[i]["dtype"]])
                        vchromo[i] = func(self.params[i]["min"], self.params[i]["max"])

                self.vchromo = vchromo

            self._hash = None

        self.randomize_flag = False

    def mutate(
        self,
        probability: float,
        rng: np.random.Generator = np.random.default_rng(),
        flag: bool = False,
    ) -> "GeneticSolution":
        """Mutate values of the chromosomes. Returns a new solution object and does not modify the current solution.

        :param probability: Probability for mutation of each gene
        :param rng: Random number generator. Defaults to numpy default_rng().
        :param  flag: Ignore self.mutate_flag
        :return: New, mutated solution object
        """
        echromo = None
        vchromo = None

        if self.mutate_flag or flag:
            # If both echromo and vchromo are available the mutation rate must be divided between both.
            # This is espacially problematic when mutation rates are very low.
            events_rate = probability
            vari_rate = probability
            if self.echromo is not None and self.vchromo is not None:
                distribution = rng.random()
                if (len(self.echromo) * distribution * probability / 2) < 1 or (
                    len(self.vchromo) * (1 - distribution) * probability
                ) < 1:
                    if distribution >= 0.5 and (len(self.echromo) * distribution * probability / 2) < 1:
                        distribution = 2 / (len(self.echromo) * probability)
                    elif distribution < 0.5 and (len(self.vchromo) * (1 - distribution) * probability) < 1:
                        distribution = 1 - (1 / (probability * len(self.vchromo)))
                events_rate = max(distribution * probability, 0)
                vari_rate = max((1 - distribution) * probability, 0)

            # Echromo is mutated by interchanging two genes
            if self.echromo is not None:
                # Each sample means that two values will be interchanged.
                n_samples = int(len(self.echromo) * events_rate / 2)
                e_samples = rng.integers(0, len(self.echromo), n_samples * 2)
                echromo = self.echromo.copy()  # Copy chromosome to make sure not to change the original solution

                for i in range(0, n_samples):
                    echromo[e_samples[i]], echromo[e_samples[n_samples + i]] = (
                        echromo[e_samples[n_samples + i]],
                        echromo[e_samples[i]],
                    )

            # Vchromo is mutated by creating a completely new gene and replacing an old one
            if self.vchromo is not None:
                v_samples = rng.integers(0, len(self.vchromo), int(len(self.vchromo) * vari_rate))
                vchromo = self.vchromo.copy()

                for i in v_samples:
                    p_i = 0 if len(self.params) == 1 else i
                    func = getattr(rng, self.registry["generators"][self.params[p_i]["dtype"]])
                    vchromo[i] = func(self.params[p_i]["min"], self.params[p_i]["max"])

        else:
            # If this solution isn't being mutated, a copy of the old solution is returned.
            if self.echromo is not None:
                echromo = self.echromo.copy()
            if self.vchromo is not None:
                vchromo = self.vchromo.copy()

        return GeneticSolution(echromo, vchromo, self.params)

    def crossover(
        self,
        generation: Sequence["GeneticSolution"],
        max_cross_len: float,
        rng: np.random.Generator = np.random.default_rng(),
    ) -> "GeneticSolution":
        """Cross the solution with another solution. Returns a new solution object and does not modify the current
        solution.

        :param generation: List of all (other) solutions in the generation
        :param max_cross_len: Maximum number of elements to cross between solutions.
        :param rng: Random number generator. Defaults to numpy default_rng().
        :return: New solution object
        """
        echromo = None
        vchromo = None

        if self.cross_with is not None:
            length = rng.integers(0, len(self.echromo) * max_cross_len, endpoint=True)
            start = rng.integers(0, len(self.echromo) - length) if length < len(self.echromo) else 0

            if self.echromo is not None:
                newpart = generation[self.cross_with].echromo[start : start + length]
                echromo = np.empty_like(self.echromo)
                echromo[start : start + length] = newpart.copy()

                idx = 0 if start != 0 else start + length  # Insert all elements that are not in newpart
                for elem in self.echromo:
                    if elem not in newpart:
                        echromo[idx] = elem
                        idx += length + 1 if idx == start - 1 else 1

            if self.vchromo is not None:
                vchromo = np.empty_like(self.vchromo)
                vchromo[:start] = self.vchromo[:start].copy()
                vchromo[start : start + length] = generation[self.cross_with].vchromo[start : start + length].copy()
                vchromo[start + length :] = self.vchromo[start + length :].copy()

            return GeneticSolution(echromo, vchromo, self.params)
        else:
            return GeneticSolution(
                self.echromo.copy() if self.echromo is not None else None,
                self.vchromo.copy() if self.vchromo is not None else None,
                self.params,
            )

    def __str__(self) -> str:
        return "".join(
            (
                "GeneticSolution(echromo = ",
                str(self.echromo).strip().replace("\n", ""),
                ",\nvchromo = ",
                str(self.echromo).strip().replace("\n", ""),
                ")",
            )
        )

    def __hash__(self) -> int:
        """Return a hash that uniquely identifies the solution.

        :return: Hash string
        """
        if self._hash is None:
            if self.echromo is not None and self.vchromo is not None:
                self._hash = self._hash_method(b"".join((self.echromo.data, self.vchromo.data)))
            elif self.vchromo is not None:
                self._hash = self._hash_method(self.vchromo.data)
            elif self.echromo is not None:
                self._hash = self._hash_method(self.echromo.data)

        return self._hash

    def __len__(self) -> int:
        return (len(self.echromo) if self.echromo is not None else 0) + (
            len(self.vchromo) if self.vchromo is not None else 0
        )
