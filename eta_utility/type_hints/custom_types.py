from __future__ import annotations

import datetime
from os import PathLike
from typing import Union

import numpy as np

# Other custom types:
Path = Union[str, PathLike]
Number = Union[float, int, np.floating, np.signedinteger, np.unsignedinteger]
TimeStep = Union[int, float, datetime.timedelta]
