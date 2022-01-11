import datetime
from os import PathLike
from typing import Union

from nptyping import Number

# Other custom types:
Path = Union[str, PathLike]
Number = Union[float, int, Number]
TimeStep = Union[int, float, datetime.timedelta]
