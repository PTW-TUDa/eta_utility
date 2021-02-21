import pathlib

import pandas as pd

if (pathlib.Path.cwd() / "test/config_local.py").is_file():
    from config_local import *  # noqa
else:
    from config_local_dummy import *  # noqa

CSV_OUTPUT_FILE = pathlib.Path.cwd() / "test/test_output.csv"

SAMPLE_TIMESERIES = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:15:01.7"])
)
