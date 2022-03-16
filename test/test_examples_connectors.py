import pathlib

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

from .config_tests import Config
from .test_utilities.pyModbusTCP.client import ModbusClient as MockModbusClient
from .test_utilities.requests.eneffco_request import request

EXCEL_NODES_FILE = Config.EXCEL_NODES_FILE
EXCEL_NODES_SHEET = Config.EXCEL_NODES_SHEET
ENEFFCO_USER = Config.ENEFFCO_USER
ENEFFCO_PW = Config.ENEFFCO_PW
ENEFFCO_POSTMAN_TOKEN = Config.ENEFFCO_POSTMAN_TOKEN


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


@pytest.fixture()
def output_file():
    file = pathlib.Path(Config.CSV_OUTPUT_FILE)
    yield file
    file.unlink()


def test_example_read_eneffco(_local_requests):
    data = ex_read_eneffco()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"CH1.Elek_U.L1-N", "Pu3.425.ThHy_Q"}


def test_example_data_recorder(output_file, _local_requests, _mock_client):
    ex_data_recorder(
        EXCEL_NODES_FILE, EXCEL_NODES_SHEET, output_file, 5, 1, 3, ENEFFCO_USER, ENEFFCO_PW, ENEFFCO_POSTMAN_TOKEN, 3
    )
