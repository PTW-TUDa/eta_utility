import asyncio
import pathlib

import numpy as np
import pandas as pd
from pytest import mark

from eta_utility.connectors import Node
from eta_utility.connectors.sub_handlers import CsvSubHandler, DFSubHandler

from .config_tests import Config

sample_series = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:00:01.7"])
)
sample_series2 = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:0.4", "2020-11-05 10:00:01.2", "2020-11-05 10:00:01.5"])
)


class TestCSVSubHandler:
    async def push_values(self):
        test_node = Node(name="FirstNode", url="", protocol="local")
        test_node2 = Node(name="SecondNode", url="", protocol="local")

        handler = None
        try:
            handler = CsvSubHandler(Config.CSV_OUTPUT_FILE, 0.5)
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

    def test_push_timeseries_to_csv(self):
        try:
            pathlib.Path(Config.CSV_OUTPUT_FILE).unlink()
        except FileNotFoundError:
            pass

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.push_values())

        with pathlib.Path(Config.CSV_OUTPUT_FILE).open("r") as f:
            df = pd.read_csv(f)
            df = df.set_index("Timestamp")
            df_check = pd.DataFrame(
                columns=["FirstNode", "SecondNode"],
                index=[
                    "2020-11-05 10:00:00.000000",
                    "2020-11-05 10:00:00.500000",
                    "2020-11-05 10:00:01.000000",
                    "2020-11-05 10:00:01.500000",
                    "2020-11-05 10:00:02.000000",
                ],
                data=[[1, np.nan], [1, 1], [2, 1], [2, 2], [3, 3]],
            )
            df_check.index.name = "Timestamp"

            assert all(df == df_check)


class TestDFSubHandler:
    @mark.parametrize("value, timestamp", [(sample_series.values, sample_series.index)])
    def test_push_timeseries_to_df(self, value, timestamp):
        """Test pushing a Series all at once"""
        handler = DFSubHandler(write_interval=1)
        test_node = Node(name="Test-node", url="", protocol="local")
        handler.push(test_node, value, timestamp)
        data = handler.data

        assert (data["Test-node"].values == value).all()

    def test_housekeeping(self):
        """Test keeping the internal data of DFSubHandler short"""
        keep_data_rows = 2
        handler = DFSubHandler(write_interval=1, size_limit=keep_data_rows)
        test_node = Node(name="Test-node", url="", protocol="local")
        handler.push(test_node, sample_series.values, sample_series.index)
        data = handler.data

        assert len(data) <= keep_data_rows
