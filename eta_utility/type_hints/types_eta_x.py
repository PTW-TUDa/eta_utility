from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv  # noqa:F401
from stable_baselines3.common.type_aliases import MaybeCallback  # noqa:F401
from stable_baselines3.common.type_aliases import GymObs as ObservationType  # noqa:F401
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymResetReturn as ResetResult,
)
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymStepReturn as StepResult,
)

ActionType = np.ndarray
EnvSettings = Dict[str, Any]
AlgoSettings = Dict[str, Any]
PyoParams = Dict[Optional[str], Union[Dict[Optional[str], Any], Any]]
