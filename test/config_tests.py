import pathlib

import pandas as pd

if (pathlib.Path.cwd() / "test/logins.py").is_file():
    from logins import *  # noqa
else:
    from logins_empty import *  # noqa

CSV_OUTPUT_FILE = pathlib.Path.cwd() / "test/test_output.csv"

SAMPLE_TIMESERIES = pd.Series(data = [1, 2, 3],
                              index = pd.DatetimeIndex(['2020-11-05 10:00:00',
                                                        '2020-11-05 10:00:01.1',
                                                        '2020-11-05 10:15:01.7']))

