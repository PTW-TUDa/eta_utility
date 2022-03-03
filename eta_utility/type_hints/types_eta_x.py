from __future__ import annotations

from typing import Any, Sequence, SupportsFloat, Tuple, Union

import numpy as np


StepResult = Tuple[np.ndarray, Union[SupportsFloat], bool, Union[str, Sequence[str]]]
EnvSettings = dict[str, Any]
