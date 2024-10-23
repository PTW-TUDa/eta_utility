from __future__ import annotations

from logging import getLogger
from typing import TYPE_CHECKING

import numpy as np

try:
    import pygame  # noqa: F401
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For the PendulumEnv example, the pygame module is required. Install eta_utility with the "
        "[examples] option to get all packages required for running examples.",
        name="pygame",
    ) from None
else:
    from gymnasium.envs.classic_control.pendulum import (
        PendulumEnv as GymPendulum,
        angle_normalize,
    )

from eta_utility.eta_x.envs import BaseEnv, StateConfig, StateVar

if TYPE_CHECKING:
    from datetime import datetime
    from typing import Any, Callable

    from eta_utility.eta_x import ConfigOptRun
    from eta_utility.type_hints import ObservationType, StepResult, TimeStep

log = getLogger(__name__)


class PendulumEnv(BaseEnv, GymPendulum):
    """Pendulum environment in the *eta_utility* style. This class is adapted from the
    `pendulum example <https://gymnasium.farama.org/environments/classic_control/pendulum/>`_ in
    the Farama gymnasium: :py:class:`gymnasium.envs.classic_control.PendulumEnv`:

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: callback which should be called after each episode
    :param scenario_time_begin: Beginning time of the scenario
    :param scenario_time_end: Ending time of the scenario
    :param episode_duration: Duration of the episode in seconds
    :param sampling_time: Duration of a single time sample / time step in seconds
    :param max_speed: Maximum speed of the pendulum
    :param max_torque: Maximum torque that can be applied by the agent
    :param g: Gravitational acceleration
    :param mass: Mass at the tip of the pendulum
    :param length: Length of the pendulum
    :param do_render: Render the learning/playing process or not? (default: True)
    :param screen_dim: Dimension of the screen to render on in pixels (default: 500)
    :param render_mode: Renders the environments to help visualise what the agent see, examples
        modes are "human", "rgb_array", "ansi" for text.
    """

    version = "v1.0"
    description = "OpenAI"

    def __init__(  # noqa: PLR0913
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
        max_speed: int,
        max_torque: float,
        g: float,
        mass: float,
        length: float,
        do_render: bool = True,
        screen_dim: int = 500,
        render_mode: str = "human",
    ):
        super().__init__(
            env_id,
            config_run,
            verbose,
            callback,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            render_mode=render_mode,
        )

        # Load environment dynamics specific settings
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.g = g
        self.mass = mass
        self.length = length

        # Other
        self.screen_dim = screen_dim
        self.do_render = do_render

        # Initialize counters
        self.n_episodes = 0
        self.n_steps = 0
        self.last_u: float | None = None

        # Setup environment state and action / observation spaces
        self.state_config = StateConfig(
            StateVar(name="torque", is_agent_action=True, low_value=-self.max_torque, high_value=self.max_torque),
            StateVar(name="th", low_value=-np.pi, high_value=np.pi),
            StateVar(name="cos_th", is_agent_observation=True, low_value=-1.0, high_value=1.0),
            StateVar(name="sin_th", is_agent_observation=True, low_value=-1.0, high_value=1.0),
            StateVar(name="th_dot", is_agent_observation=True, low_value=-1.0, high_value=1.0),
        )
        self.action_space, self.observation_space = self.state_config.continuous_spaces()

    def step(self, action: np.ndarray) -> StepResult:
        """See base_env documentation"""
        assert self.state_config is not None, "Set state_config before calling step function."

        # Update counters
        self.n_steps += 1

        # Get previous step values (th := theta)
        th = self.state["th"]
        thdot = self.state["th_dot"]

        # Store actions
        self.state = {}
        for idx, act in enumerate(self.state_config.actions):
            self.state[act] = action[idx]

        # Clip input from agent by max values
        self.state["torque"] = np.clip(self.state["torque"], -self.max_torque, self.max_torque)
        self.last_u = self.state["torque"]  # for rendering

        # Calculate state of the pendulum
        u, g, m, length, dt = self.state["torque"], self.g, self.mass, self.length, self.sampling_time

        newthdot = thdot + (3 * g / (2 * length) * np.sin(th) + 3.0 / (m * length**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state["th_dot"] = newthdot
        self.state["th"] = newth
        self.state["cos_th"] = np.cos(self.state["th"])
        self.state["sin_th"] = np.sin(self.state["th"])

        # reward function
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        # Prepare observations
        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        # Check if the episode is completed
        terminated = self.n_steps >= self.n_episode_steps
        truncated = False

        # Render the environment at each step
        if self.render_mode == "human":
            self.render()

        return observations, -costs, terminated, truncated, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObservationType, dict[str, Any]]:
        """Reset the environment. This is called after each episode is completed and should be used to reset the
        state of the environment such that simulation of a new episode can begin.

        :param seed: The seed that is used to initialize the environment's PRNG (`np_random`) (default: None).
        :param options: Additional information to specify how the environment is reset (optional,
                depending on the specific environment) (default: None)
        :return: Tuple of observation and info. Analogous to the ``info`` returned by :meth:`step`.
        """
        assert self.state_config is not None, "Set state_config before calling reset function."

        # save episode's stats
        super().reset(seed=seed, options=options)

        self.last_u = None
        self.state = {}

        # Reset actions and position
        for name in self.state_config.vars:
            if name in self.state_config.actions:
                self.state[name] = 0
            elif name in {"th", "th_dot"}:
                var = self.state_config.vars[name]
                assert var.low_value is not None, f"low_value for {name} must be set."
                assert var.high_value is not None, f"high_value for {name} must be set."
                self.state[name] = self.np_random.uniform(low=var.low_value, high=var.high_value)
        # Calculate sin and cos
        self.state["cos_th"] = np.cos(self.state["th"])
        self.state["sin_th"] = np.sin(self.state["th"])

        # Log the state
        self.state_log.append(self.state)

        # Get the observations from environment state
        observations = np.empty(len(self.state_config.observations))
        for idx, name in enumerate(self.state_config.observations):
            observations[idx] = self.state[name]

        # Render the environment when calling the reset function
        if self.render_mode == "human":
            self.render()

        return observations, {}

    def render(self) -> None:
        """Use the render function from the Farama gymnasium PendulumEnv environment.
        This requires a little hack because our 'self.state' attribute is different."""
        if self.do_render:
            state = self.state.copy()
            self.state = (
                [self.state["th"], self.state["th_dot"]] if not isinstance(state, np.ndarray) else state  # type: ignore
            )
            GymPendulum.last_u = self.last_u
            GymPendulum.render(self)
            self.state = state

    def close(self) -> None:
        GymPendulum.close(self)
