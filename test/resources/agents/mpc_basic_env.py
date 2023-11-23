import gymnasium
import numpy as np
import pyomo.environ as pyo

from eta_utility.eta_x.envs import BaseEnvMPC


class MPCBasicEnv(BaseEnvMPC):
    """Environment for MPC basic agent, that implements a quadratic optimization problem. The goal is to
    minimize the sum of the squared values of both x and u over the prediction horizon. The environment
    was created for testing purposes.

    :param env_id: Identification for the environment, useful when creating multiple environments.
    :param config_run: Configuration of the optimization run.
    :param prediction_horizon: time horizon over which the optimization problem is solved.
    :param scenario_time_begin: Beginning time of the scenario.
    :param scenario_time_end: Ending time of the scenario.
    :param episode_duration: Duration of the episode in seconds.
    :param sampling_time: Duration of a single time sample / time step in seconds.
    :param model_parameters: Parameters for the mathematical model.
    :param kwargs: Other keyword arguments (for subclasses).
    """

    description = "quadratic optimization problem"
    version = "1.0"

    def __init__(
        self,
        env_id,
        config_run,
        prediction_horizon: int = 10,
        *,
        scenario_time_begin,
        scenario_time_end,
        episode_duration,
        sampling_time,
        model_parameters,
        **kwargs,
    ):
        self.prediction_horizon = prediction_horizon

        # Observation Space and action space are not used in this specific case.'
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(self.prediction_horizon,))
        self.action_space = gymnasium.spaces.Box(low=-100_000, high=100_000, shape=(self.prediction_horizon,))

        self.state = None

        super().__init__(
            env_id=env_id,
            config_run=config_run,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            model_parameters=model_parameters,
            **kwargs,
        )

    def _model(self):
        model = pyo.AbstractModel()

        # Define model sets
        model.T = pyo.RangeSet(0, self.n_prediction_steps - 1)

        # Define model parameters
        model.x = pyo.Var(model.T, initialize=0)
        model.u = pyo.Var(model.T, bounds=(-1, 1), initialize=0)

        def obj_rule(m):
            # sum of the squared values of model.x and model.u at each time step in the prediction horizon
            return sum(m.x[t] ** 2 + m.u[t] ** 2 for t in m.T)

        model.obj = pyo.Objective(rule=obj_rule)

        def constr_rule(m, t):
            if t == 0:
                return pyo.Constraint.Skip
            return m.x[t] == m.x[t - 1] + m.u[t - 1]

        model.constr = pyo.Constraint(model.T, rule=constr_rule)

        return model.create_instance()

    def render(self):
        """The visual representation is not needed for the testing purpose."""
        pass
