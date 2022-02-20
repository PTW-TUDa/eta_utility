from __future__ import annotations

import datetime
from os import PathLike
from typing import Union

# Other custom types:
Path = Union[str, PathLike]
Number = Union[float, int]
TimeStep = Union[float, int, datetime.timedelta]
