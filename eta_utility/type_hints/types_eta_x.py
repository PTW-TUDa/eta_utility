from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from stable_baselines3.common.type_aliases import (
    GymEnv,  # noqa:F401
    MaybeCallback,  # noqa:F401
)
from stable_baselines3.common.type_aliases import GymObs as ObservationType  # noqa:F401
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymResetReturn as ResetResult,
)
from stable_baselines3.common.type_aliases import (  # noqa:F401
    GymStepReturn as StepResult,
)

ActionType = np.ndarray
EnvSettings = dict[str, Any]
AlgoSettings = dict[str, Any]
PyoParams = dict[Optional[str], Union[dict[Optional[str], Any], Any]]
