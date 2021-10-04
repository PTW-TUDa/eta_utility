import asyncio
from datetime import datetime, timedelta

import pandas as pd
import requests
from pytest import fail, fixture

from eta_utility.connectors import DFSubHandler, Node
from eta_utility.connectors.eneffco import EnEffCoConnection

from .config_tests import Config
from .test_utilities.requests.eneffco_request import request

# Results used for local tests
sample_series = pd.Series(
    data=[1, 2, 3], index=pd.DatetimeIndex(["2020-11-05 10:00:00", "2020-11-05 10:00:01.1", "2020-11-05 10:15:01.7"])
)

node = Node(
    "CH1.Elek_U.L1-N",
    Config.ENEFFCO_URL,
    "eneffco",
    eneffco_code="CH1.Elek_U.L1-N",
)

node2 = Node("Pu3.425.ThHy_Q", Config.ENEFFCO_URL, "eneffco", eneffco_code="Pu3.425.ThHy_Q")

node_write = Node("RM.Elek_P.Progn", Config.ENEFFCO_URL, "eneffco", eneffco_code="RM.Elek_P.Progn")


async def stop_execution(sleep_time):
    await asyncio.sleep(sleep_time)
    raise KeyboardInterrupt


@fixture()
def local_requests(monkeypatch):
    monkeypatch.setattr(requests, "request", request)


def test_check_access(local_requests):
    # Check access to see, whether anything responds"
    try:
        result = requests.request("GET", Config.ENEFFCO_URL)
    except Exception as e:
        fail(str(e))

    if result.status_code == 200:
        assert True
    else:
        assert False


def test_eneffco_read(local_requests):
    """Test eneffco read function"""

    node = Node(
        "CH1.Elek_U.L1-N",
        Config.ENEFFCO_URL,
        "eneffco",
        eneffco_code="CH1.Elek_U.L1-N",
    )
    node2 = Node("Pu3.425.ThHy_Q", Config.ENEFFCO_URL, "eneffco", eneffco_code="Pu3.425.ThHy_Q")

    # Test reading a single node
    server = EnEffCoConnection(node.url, Config.ENEFFCO_USER, Config.ENEFFCO_PW, api_token=Config.ENEFFCO_POSTMAN_TOKEN)
    # The interval is arbitrary here. range int[1,10]
    res = server.read_series(
        datetime.now() - timedelta(seconds=10),
        datetime.now(),
        [node, node2],
        timedelta(seconds=1),
    )

    assert isinstance(res, pd.DataFrame)
    assert set(res.columns) == {"CH1.Elek_U.L1-N", "Pu3.425.ThHy_Q"}

    res2 = server.read({node, node2})
    assert isinstance(res2, pd.DataFrame)
    assert set(res2.columns) == {"CH1.Elek_U.L1-N", "Pu3.425.ThHy_Q"}
    assert len(res2.index) == 10


def test_eneffco_read_info(local_requests):
    """Test the read_info() method"""
    server = EnEffCoConnection(node.url, Config.ENEFFCO_USER, Config.ENEFFCO_PW, api_token=Config.ENEFFCO_POSTMAN_TOKEN)

    res = server.read_info([node, node2])

    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0


def test_eneffco_write(local_requests):
    """Test writing a single node"""
    server = EnEffCoConnection(
        node_write.url, Config.ENEFFCO_USER, Config.ENEFFCO_PW, api_token=Config.ENEFFCO_POSTMAN_TOKEN
    )

    server.write({node: sample_series})


def test_eneffco_subscribe_multi(local_requests):
    """Test eneffco subscribe_series function; this needs network access"""

    # Test subscribing nodes with multiple timesteps
    server = EnEffCoConnection(node.url, Config.ENEFFCO_USER, Config.ENEFFCO_PW, api_token=Config.ENEFFCO_POSTMAN_TOKEN)
    # changed write_interval from 10 to 1
    handler = DFSubHandler(write_interval=1)
    loop = asyncio.get_event_loop()

    try:
        # changed req_interval from 60 to 10
        server.subscribe_series(handler, req_interval=10, data_interval=2, nodes=[node, node2])
        loop.run_until_complete(stop_execution(2))
    except KeyboardInterrupt:
        pass
    finally:
        server.close_sub()
        handler.close()
