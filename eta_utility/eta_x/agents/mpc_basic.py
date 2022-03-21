from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import numpy as np
import pyomo.environ as pyo
from pyomo import opt
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from eta_utility import get_logger

if TYPE_CHECKING:
    import io
    import pathlib
    from typing import Any, Iterable, Sequence

    import torch as th
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback

log = get_logger("eta_x.agents")


class MPCBasic(BaseAlgorithm):
    """Simple, Pyomo based MPC agent.

    The agent requires an environment that specifies the 'model' attribute, returning a
    :py:class:`pyomo.ConcreteModel` and a sorted list as the order for the action space. This list is used to
    avoid ambiguity when returning a list of actions. Since the model specifies its own action and observation
    space, this agent does not use the *action_space* and *observation_space* specified by the environment.

    :param policy: Agent policy. Parameter is not used in this agent.
    :param env: Environment to be optimized.
    :param verbose: Logging verbosity.
    :param solver_name: Name of the solver, could be cplex or glpk.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class or as provided by solver.
    """

    def __init__(
        self,
        policy: type[BasePolicy],
        env: VecEnv,
        verbose: int = 1,
        *,
        solver_name: str = "cplex",
        **kwargs: Any,
    ) -> None:

        # Prepare kwargs to be sent to the super class and to the solver.
        super_args = {}
        solver_args = {}

        # Set default values for superclass arguments
        kwargs.setdefault("policy_base", None)
        kwargs.setdefault("learning_rate", 0)

        for key in kwargs.keys():
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

        # Stepping parameters
        self._current_shift: int = 0  #: Current shift determines the current optimization step. Starting value 0 should
        # not be changed  # noqa

        self.model: pyo.ConcreteModel  #: Pyomo optimization model as specified by the environment.
        self.actions_order: Sequence[str]  #: Specification of the order in which action values should be returned.
        self._setup_model()

    def _setup_model(self) -> None:
        """Load the MILP model from the environment."""
        assert self.env is not None, "The MPC agent needs a specific environment to work. Cannot use env = None."
        self.model, self.actions_order = self.env.get_attr("model", 0)[0]

    def solve(self) -> pyo.ConcreteModel:
        """Solve the current pyomo model instance with given parameters. This could also be used separately to solve
        normal MILP problems. Since the entire problem instance is returned, result handling can be outsourced.

        :return: Solved pyomo model instance.
        """
        assert self.env is not None, "The MPC agent needs a specific environment to work. Cannot use env = None."

        solver = pyo.SolverFactory(self.solver_name)
        solver.options.update(self.solver_options)  # Adjust solver settings

        _tee = True if log.level / 10 <= 1 else False
        result = solver.solve(self.model, symbolic_solver_labels=True, tee=_tee)
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
        observation: np.ndarray,
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
        solution = {
            com.name: pyo.value(com[next(com.keys())])
            for com in self.model.component_objects(pyo.Var)
            if not isinstance(com, pyo.SimpleVar)
        }

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
        eval_env: GymEnv | None = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        eval_log_path: str | None = None,
        reset_num_timesteps: bool = True,
    ) -> MPCBasic:
        """The MPC approach cannot learn a new model. Specify the model attribute as a pyomo Concrete model instead,
        to use the prediction function of this agent.
        :param kwargs:

        """
        raise NotImplementedError("The MPC_simple approach does not need to learn a model.")

    def save(
        self,
        path: str | pathlib.Path | io.BufferedIOBase,
        exclude: Iterable[str] | None = None,
        include: Iterable[str] | None = None,
    ) -> None:
        """Saving is currently not implemented for the MPC agent."""
        raise NotImplementedError("The MPC approach creates no savable model.")

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
    ) -> MPCBasic:
        """Loading a model is currently not implemented for the MPC agent."""
        raise NotImplementedError("The MPC approach cannot load a model.")

    def get_parameter_list(self) -> list:
        """
        Get tensorflow Variables of model's parameters.

        This includes all variables necessary for continuing training (saving / loading).

        :return: List of tensorflow Variables.
        """
        pass

    def _get_pretrain_placeholders(self) -> None:
        """Pretaining is not implemented for the MPC agent."""
        raise NotImplementedError("The MILP Optimizer does not need to be pre-trained.")
