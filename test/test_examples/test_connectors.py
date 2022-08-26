import pandas as pd
import pytest
import requests
from pyModbusTCP import client as mbclient  # noqa: I900

from eta_utility.servers import OpcUaServer
from examples.connectors.data_recorder import (  # noqa: I900
    execution_loop as ex_data_recorder,
)
from examples.connectors.read_series_eneffco import (  # noqa: I900
    read_series as ex_read_eneffco,
)

from ..utilities.pyModbusTCP.client import ModbusClient as MockModbusClient
from ..utilities.requests.eneffco_request import request


@pytest.fixture()
def _local_requests(monkeypatch):
    monkeypatch.setattr(requests, "request", request)


@pytest.fixture()
def local_server():
    server = OpcUaServer(5, ip="127.0.0.1", port=4840)
    yield server
    server.stop()


@pytest.fixture()
def _mock_client(monkeypatch):
    monkeypatch.setattr(mbclient, "ModbusClient", MockModbusClient)
    monkeypatch.setattr(requests, "request", request)


def test_example_read_eneffco(_local_requests):
    data = ex_read_eneffco()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"CH1.Elek_U.L1-N", "Pu3.425.ThHy_Q"}


def test_example_data_recorder(temp_dir, _local_requests, _mock_client, config_nodes_file, config_eneffco):
    file = temp_dir / "data_recorder_example_output.csv"
    ex_data_recorder(
        config_nodes_file["file"],
        config_nodes_file["sheet"],
        file,
        5,
        1,
        3,
        config_eneffco["user"],
        config_eneffco["pw"],
        config_eneffco["postman_token"],
        3,
    )
