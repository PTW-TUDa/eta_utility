from __future__ import annotations

import sys
from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np
import pyomo.environ as pyo
from pyomo import opt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from eta_utility.util import deprecated

if TYPE_CHECKING:
    import io
    import pathlib
    from collections.abc import Sequence
    from typing import Any

    import torch as th
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

log = getLogger(__name__)


class MathSolver(BaseAlgorithm):
    """Simple, Pyomo based MPC agent.

    The agent requires an environment that specifies the 'model' attribute, returning a
    :py:class:`pyomo.ConcreteModel` and a sorted list as the order for the action space. This list is used to
    avoid ambiguity when returning a list of actions. Since the model specifies its own action and observation
    space, this agent does not use the *action_space* and *observation_space* specified by the environment.

    :param policy: Agent policy. Parameter is not used in this agent.
    :param env: Environment to be optimized.
    :param verbose: Logging verbosity.
    :param solver_name: Name of the solver, could be cplex or glpk.
    :param action_index: Index of the solution value to be used as action (if this is 0, the first value in a list
        of solution values will be used).
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class or as provided by solver.
    """

    def __init__(
        self,
        policy: type[BasePolicy],
        env: VecEnv,
        verbose: int = 1,
        *,
        solver_name: str = "cplex",
        action_index: int = 0,
        _init_setup_model: bool = True,
        **kwargs: Any,
    ) -> None:
        # Prepare kwargs to be sent to the super class and to the solver.
        super_args = {}
        solver_args = {}

        # Set default values for superclass arguments
        kwargs.setdefault("learning_rate", 0)

        for key in kwargs:
            # Find arguments which are meant for the BaseAlgorithm class and extract them into super_args
            if key in {
                "policy_base",
                "learning_rate",
                "policy_kwargs",
                "device",
                "support_multi_env",
                "create_eval_env",
                "monitor_wrapper",
                "seed",
                "use_sde",
                "sde_sample_freq",
                "supported_action_spaces",
            }:
                super_args[key] = kwargs.get(key, None)
            elif key in {"tensorboard_log"}:
                log.warning(
                    "The MPC Basic agent does not support logging to tensorboard. "
                    "Ignoring parameter tensorboard_log."
                )
            else:
                solver_args[key] = kwargs[key]

        super().__init__(policy=policy, env=env, verbose=verbose, **super_args)
        log.setLevel(int(verbose * 10))  # Set logging verbosity

        # Check configuration for MILP compatibility
        if self.env is None:
            raise ValueError("The MPC agent needs a specific environment to work correctly. Cannot use env = None.")

        if self.n_envs is not None and self.n_envs > 1:
            raise ValueError(
                "The MPC agent can only use one environment. It cannot work on multiple vectorized environments."
            )
        if isinstance(self.env, VecNormalize):
            raise TypeError("The MPC agent does not allow the use of normalized environments.")

        # Solver parameters
        self.solver_name: str = solver_name
        self.solver_options: dict = {}
        self.solver_options.update(solver_args)

        self.model: pyo.ConcreteModel  #: Pyomo optimization model as specified by the environment.
        self.actions_order: Sequence[str]  #: Specification of the order in which action values should be returned.

        self.policy_class: type[BasePolicy]
        if _init_setup_model:
            self._setup_model()

        #: Index of the solution value to be used as action (if this is 0, the first value in a list
        #: of solution values will be used).
        self.action_index = action_index

    def _setup_model(self) -> None:
        """Load the MILP model from the environment."""
        assert self.env is not None, "The MPC agent needs a specific environment to work. Cannot use env = None."
        self.model, self.actions_order = self.env.get_attr("model", 0)[0]
        if self.policy_class is not None:
            self.policy: type[BasePolicy] = self.policy_class(  # type: ignore
                self.observation_space,
                self.action_space,
            )

    def solve(self) -> pyo.ConcreteModel:
        """Solve the current pyomo model instance with given parameters. This could also be used separately to solve
        normal MILP problems. Since the entire problem instance is returned, result handling can be outsourced.

        :return: Solved pyomo model instance.
        """
        assert self.env is not None, "The MPC agent needs a specific environment to work. Cannot use env = None."

        solver = pyo.SolverFactory(self.solver_name)
        solver.options.update(self.solver_options)  # Adjust solver settings

        _tee: bool = bool(log.level / 10 <= 1)
        result = solver.solve(self.model, symbolic_solver_labels=True, tee=_tee)
        if _tee:
            print("\n")  # noqa: T201 (print is ok here, because cplex prints directly to console).
        log.debug(
            "Problem information: \n"
            "\t+----------------------------------+\n"
            + "\n".join(
                f"\t {item}: {value.value} "
                for item, value in result["Problem"][0].items()
                if not isinstance(value.value, opt.UndefinedData)
            )
            + "\n\t+----------------------------------+"
        )

        # Log status after the optimization
        log.info(
            "Solver information: \n"
            "\t+----------------------------------+\n"
            + "\n".join(
                f"\t {item}: {value.value} "
                for item, value in result["Solver"][0].items()
                if item != "Statistics" and not isinstance(value.value, opt.UndefinedData)
            )
            + "\n\t+----------------------------------+"
        )

        # Log status after the optimization
        if len(result["Solution"]) >= 1:
            log.debug(
                "Solution information: \n"
                "\t+----------------------------------+\n"
                + "\n".join(
                    f"\t {item}: {value.value} "
                    for item, value in result["Solution"][0].items()
                    if not isinstance(value.value, opt.UndefinedData)
                )
                + "\n\t+----------------------------------+"
            )

        # Interrupt execution if no optimal solution could be found
        if (
            result.solver.termination_condition != opt.TerminationCondition.optimal
            or result.solver.status != opt.SolverStatus.ok
        ):
            log.error("Problem can not be solved - aborting.")
            self.env.env_method("solve_failed", self.model, result)
            sys.exit(1)

        return self.model

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        Solve the current pyomo model instance with given parameters and observations and return the optimal actions.

        :param observation: the input observation (not used here).
        :param state: The last states (not used here).
        :param episode_start: The last masks (not used here).
        :param deterministic: Whether to return deterministic actions. This agent always returns
                                   deterministic actions.
        :return: Tuple of the model's action and the next state (not used here).
        """
        assert self.env is not None, "The MPC agent needs a specific environment to work. Cannot use env = None."
        self.model, _ = self.env.get_attr("model", 0)[0]
        self.solve()
        self.env.set_attr("model", self.model, 0)

        # Aggregate the agent actions from pyomo component objects
        solution = {}
        for com in self.model.component_objects(pyo.Var):
            if isinstance(com, pyo.ScalarVar):
                continue
            try:
                solution[com.name] = pyo.value(com[com.index_set().at(self.action_index + 1)])
            except ValueError:
                pass

        # Make sure that actions are returned to the correct order and as a numpy array.
        actions: np.ndarray = np.ndarray((1, len(self.actions_order)))
        for i, action in enumerate(self.actions_order):
            log.debug(f"Action '{action}' value: {solution[action]}")
            actions[0][i] = solution[action]

        return actions, state

    def action_probability(
        self,
        observation: np.ndarray,
        state: np.ndarray | None = None,
        mask: np.ndarray | None = None,
        actions: np.ndarray | None = None,
        logp: bool = False,
    ) -> None:
        """The MPC approach cannot predict probabilities of single actions."""
        raise NotImplementedError("The MPC agent cannot predict probabilities of single actions.")

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 100,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> MathSolver:
        """The MPC approach cannot learn a new model. Specify the model attribute as a pyomo Concrete model instead,
        to use the prediction function of this agent.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: The trained model.
        """

        return self

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path | io.BufferedIOBase,
        env: GymEnv | None = None,
        device: th.device | str = "auto",
        custom_objects: dict[str, Any] | None = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs: Any,
    ) -> MathSolver:
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!

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
        if env is None:
            raise ValueError("Parameter env must be specified.")
        model: MathSolver = super().load(path, env, device, custom_objects, print_system_info, force_reset, **kwargs)

        return model


@deprecated("Use `MathSolver` instead of `MPCBasic`.")
class MPCBasic(MathSolver):
    """Deprecated MPCBasic class. Use `MathSolver` instead."""
