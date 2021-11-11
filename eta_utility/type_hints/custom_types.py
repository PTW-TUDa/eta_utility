import datetime
import pathlib
from typing import NewType, Union

from nptyping import Number

# Other custom types:
Path = NewType("Path", Union[pathlib.Path, str])  # better to use maybe os.Pathlike
Number = NewType("Number", Union[float, int, Number])
TimeStep = NewType("TimeStep", Union[int, datetime.timedelta])  # can't be used for Union[float, timedelta] in fmu.py
