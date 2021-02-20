from config_tests import *  # noqa
from pytest import mark

from eta_utility.connectors import Node
from eta_utility.connectors.sub_handlers import DFSubHandler

# Cases
# 1. value = Timeseries, timestamp=None


@mark.parametrize("value, timestamp", [(SAMPLE_TIMESERIES.values, SAMPLE_TIMESERIES.index)])
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
    handler.push(test_node, SAMPLE_TIMESERIES.values, SAMPLE_TIMESERIES.index)
    data = handler.data

    assert len(data) <= keep_data_rows
