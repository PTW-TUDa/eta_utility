from __future__ import annotations

from typing import Dict, Sequence, Set, SupportsFloat, SupportsInt, Tuple, Union

import numpy as np

StepResult = Tuple[np.ndarray, Union[SupportsFloat], bool, Union[str, Sequence[str]]]
DefaultSettings = Dict[str, Dict[str, Union[str, SupportsInt, bool, None]]]
RequiredSettings = Dict[str, Set[str]]
