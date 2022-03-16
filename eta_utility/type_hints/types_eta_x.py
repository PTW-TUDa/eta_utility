from __future__ import annotations

from typing import Any, Dict, Optional, SupportsFloat, Tuple, Union

import numpy as np

StepResult = Tuple[np.ndarray, Union[SupportsFloat], bool, Union[str, Dict[str, Any]]]
EnvSettings = Dict[str, Any]
AlgoSettings = Dict[str, Any]
PyoParams = Dict[Optional[str], Union[Dict[Optional[str], Any], Any]]
