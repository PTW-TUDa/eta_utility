import asyncio
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from eta_utility.connectors import CsvSubHandler, DFSubHandler, Node

sample_series = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:00:01.7"])
)
sample_series2 = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:0.4", "2020-11-05 10:00:01.2", "2020-11-05 10:00:01.5"])
)

sample_series_nan = pd.Series(
    data=[1, np.nan, np.nan],
    index=pd.DatetimeIndex(["2020-11-05 10:00:0.4", "2020-11-05 10:00:01.2", "2020-11-05 10:00:01.5"]),
)


class TestCSVSubHandler:
    async def push_values(self, handler: CsvSubHandler):
        test_node = Node(name="FirstNode", url="", protocol="local")
        test_node2 = Node(name="SecondNode", url="", protocol="local")

        try:
            for num in range(6):
                idx = num // 2 if num > 1 else 0
                if num % 2 == 0:
                    handler.push(test_node, sample_series.values[idx], sample_series.index[idx])
                else:
                    handler.push(test_node2, sample_series2.values[idx], sample_series2.index[idx])
        finally:
            try:
                handler.close()
            except AttributeError:
                pass

    def test_push_timeseries_to_csv(self, temp_dir):
        file = temp_dir / "csv_test_output.csv"
        handler = CsvSubHandler(file, 0.5)

        executor = ThreadPoolExecutor(max_workers=3)
        loop = asyncio.get_event_loop()
        loop.set_default_executor(executor)
        loop.run_until_complete(self.push_values(handler))
        executor.shutdown()

        with pathlib.Path(file).open("r") as f:
            df = pd.read_csv(f)
            df = df.set_index("Timestamp")
            df_check = pd.DataFrame(
                columns=["FirstNode", "SecondNode"],
                index=[
                    "2020-11-05 10:00:00.000000",
                    "2020-11-05 10:00:00.500000",
                    "2020-11-05 10:00:01.500000",
                    "2020-11-05 10:00:02.000000",
                ],
                data=[[1, np.nan], [1, 1], [2, 3], [3, 3]],
            )
            df_check.index.name = "Timestamp"

            assert all(df == df_check)


class TestDFSubHandler:
    @pytest.mark.parametrize(("value", "timestamp"), [(sample_series.values, sample_series.index)])
    def test_push_timeseries_to_df(self, value, timestamp):
        """Test pushing a Series all at once"""
        handler = DFSubHandler(write_interval=1)
        test_node = Node(name="FirstNode", url="", protocol="local")
        handler.push(test_node, value, timestamp)
        data = handler.data

        assert (data["FirstNode"].values == value).all()

    def test_housekeeping(self):
        """Test keeping the internal data of DFSubHandler short"""
        keep_data_rows = 2
        handler = DFSubHandler(write_interval=1, size_limit=keep_data_rows)
        test_node = Node(name="FirstNode", url="", protocol="local")
        handler.push(test_node, sample_series.values, sample_series.index)
        data = handler.data

        assert len(data) <= keep_data_rows

    def test_get_latest(self):
        handler = DFSubHandler(write_interval=1)
        test_node = Node(name="Test-node", url="", protocol="local")
        handler.push(test_node, sample_series.values, sample_series.index)
        data = handler.get_latest()

        assert (data.values == sample_series.values[-1]).all()

    def test_auto_fillna(self):
        # First test default behavior: nans are filled
        handler = DFSubHandler(write_interval=1)
        test_node = Node(name="Test-node", url="", protocol="local")
        handler.push(test_node, sample_series_nan)
        data = handler.data
        assert data.notna().all().all()

        # Next test auto_fillna = False
        handler = DFSubHandler(write_interval=1, auto_fillna=False)
        handler.push(test_node, sample_series_nan)
        data = handler.data
        assert data.isna().any().any()
