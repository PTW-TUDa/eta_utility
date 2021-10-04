import pandas as pd
from pytest import mark

from eta_utility.connectors import Node
from eta_utility.connectors.sub_handlers import DFSubHandler

sample_series = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:15:01.7"])
)


@mark.parametrize("value, timestamp", [(sample_series.values, sample_series.index)])
def test_push_timeseries_to_df(value, timestamp):
    """Test pushing a Series all at once"""
    handler = DFSubHandler(write_interval=1)
    test_node = Node(name="Test-node", url="", protocol="local")
    handler.push(test_node, value, timestamp)
    data = handler.data

    assert (data["Test-node"].values == value).all()


def test_housekeeping():
    """Test keeping the internal data of DFSubHandler short"""
    keep_data_rows = 2
    handler = DFSubHandler(write_interval=1, keep_data_rows=keep_data_rows)
    test_node = Node(name="Test-node", url="", protocol="local")
    handler.push(test_node, sample_series.values, sample_series.index)
    data = handler.data

    assert len(data) <= keep_data_rows
