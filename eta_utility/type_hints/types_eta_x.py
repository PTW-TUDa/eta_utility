from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymEnv,
    GymObs as ObservationType,
    GymResetReturn as ResetResult,
    GymStepReturn as StepResult,
    MaybeCallback,
)

ActionType = np.ndarray
EnvSettings = dict[str, Any]
AlgoSettings = dict[str, Any]
PyoParams = dict[Optional[str], Union[dict[Optional[str], Any], Any]]
