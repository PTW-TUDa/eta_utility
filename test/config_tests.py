import pathlib

import pandas as pd

if (pathlib.Path.cwd() / "test/config_local.py").is_file():
    from test.config_local import *  # noqa
else:
    from test.config_local_dummy import ConfigParams  # noqa

CSV_OUTPUT_FILE = pathlib.Path(__file__).parent / "test_resources/test_output.csv"
FMU_FILE = pathlib.Path(__file__).parent / "test_resources/test_fmu.fmu"
LIVE_CONNECT_CONFIG = pathlib.Path(__file__).parent / "test_resources/test_config_live_connect.json"

SAMPLE_TIMESERIES = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:15:01.7"])
)
