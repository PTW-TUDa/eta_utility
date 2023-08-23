import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytest
import requests

from eta_utility.connectors import CumulocityConnection, DFSubHandler, Node

from ..utilities.requests.cumulocity_request import request


@pytest.fixture()
def cumulocity_nodes(config_cumulocity):
    return {
        "node": Node(
            "P1",
            config_cumulocity["url"],
            "cumulocity",
            measurement_id="1234",
            value_fragment_series="P",
        ),
        "node2": Node(
            "P2",
            config_cumulocity["url"],
            "cumulocity",
            measurement_id="1235",
            value_fragment_series="P",
        ),
    }


async def stop_execution(sleep_time):
    await asyncio.sleep(sleep_time)
    raise KeyboardInterrupt


@pytest.fixture()
def _local_requests(monkeypatch):
    monkeypatch.setattr(requests, "request", request)


def test_check_access(_local_requests, config_cumulocity):
    # Check access to see, whether anything responds
    try:
        result = requests.request("GET", config_cumulocity["url"])
    except Exception as e:
        pytest.fail(str(e))

    if result.status_code == 200:
        assert True
    else:
        pytest.fail("Could not access cumulocity server for testing.")


def test_cumulocity_read(_local_requests, config_cumulocity):
    """Test cumulocity read function"""

    node = Node(
        "P1",
        config_cumulocity["url"],
        "cumulocity",
        measurement_id="1234",
        value_fragment_series="P",
    )
    node2 = Node(
        "P2",
        config_cumulocity["url"],
        "cumulocity",
        measurement_id="1235",
        value_fragment_series="P",
    )

    # Test reading a single node
    server = CumulocityConnection(
        node.url, config_cumulocity["user"], config_cumulocity["pw"], tenant=config_cumulocity["tenant"]
    )
    # The interval is arbitrary here. range int[1,10]
    res = server.read_series(
        datetime.now() - timedelta(seconds=10),
        datetime.now(),
        [node, node2],
        timedelta(seconds=1),
    )

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"P1", "P2"}
    assert len(res.index) == 28

    res2 = server.read({node, node2})
    assert isinstance(res2, pd.DataFrame)
    assert set(res2.columns) == {"P1", "P2"}
    assert len(res2.index) == 1


def test_cumulocity_subscribe_multi(_local_requests, config_cumulocity, cumulocity_nodes):
    """Test cumulocity subscribe_series function."""

    # Test subscribing nodes with multiple time steps
    server = CumulocityConnection(
        cumulocity_nodes["node"].url,
        config_cumulocity["user"],
        config_cumulocity["pw"],
        tenant=config_cumulocity["tenant"],
    )
    # changed write_interval from 10 to 1
    handler = DFSubHandler(write_interval=1)
    loop = asyncio.get_event_loop()

    try:
        # changed req_interval from 60 to 10
        server.subscribe_series(
            handler, req_interval=10, data_interval=2, nodes=[cumulocity_nodes["node"], cumulocity_nodes["node2"]]
        )
        loop.run_until_complete(stop_execution(3))
    except KeyboardInterrupt:
        pass
    finally:
        server.close_sub()
        assert isinstance(handler.data, pd.DataFrame)
        assert set(handler.data.columns) == {"P1", "P2"}
        assert len(handler.data.index) > 0
        handler.close()