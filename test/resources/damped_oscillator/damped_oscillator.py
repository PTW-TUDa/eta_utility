from __future__ import annotations

from typing import TYPE_CHECKING

from eta_utility.eta_x.common import episode_results_path
from eta_utility.eta_x.envs import BaseEnvSim

if TYPE_CHECKING:
    from typing import Any, Callable

    from eta_utility.type_hints import Dict, Path


class DampedOscillatorEnv(BaseEnvSim):
    """
    Damped oscillator environment class from BaseEnvSim.
    Model settings come from fmu file.
    """

    # Set info
    version = "v0.1"
    description = "Damped oscillator"
    fmu_name = "damped_oscillator"

    def __init__(
        self,
        env_id: int,
        run_name: str,
        general_settings: Dict[str, Any],
        path_settings: Dict[str, Path],
        env_settings: Dict[str, Any],
        verbose: int = 1,
        callback: Callable = None,
    ):
        super()._init_legacy(env_id, run_name, general_settings, path_settings, env_settings, verbose, callback)

        # set action space and observation space
        self.state_config = [
            {"name": "u", "ext_id": "u", "is_agent_action": True, "low_value": -1.0, "high_value": 1.0},
            {
                "name": "s",
                "ext_id": "s",
                "is_agent_observation": True,
                "low_value": -4.0,
                "high_value": 4.0,
                "is_ext_output": True,
            },
            {
                "name": "v",
                "ext_id": "v",
                "is_agent_observation": True,
                "low_value": -8.0,
                "high_value": 8.0,
                "is_ext_output": True,
            },
            {
                "name": "a",
                "ext_id": "a",
                "is_agent_observation": True,
                "low_value": -20.0,
                "high_value": 20.0,
                "is_ext_output": True,
            },
        ]
        self._init_state_space()
        self.action_space = self.continuous_action_space_from_state()
        self.observation_space = self.continuous_obs_space_from_state()

        # Initialize the simulator object with the fmu- simulator from eta x
        self._init_simulator()

    def render(self):
        self.export_state_log(
            path=episode_results_path(self.config_run.path_series_results, self.run_name, 1, self.env_id)
        )
