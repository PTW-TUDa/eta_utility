import pandas as pd
import pytest
import requests_cache
from pyModbusTCP import client as mbclient

from eta_utility.connectors.emonio import NodeModbusFactory
from eta_utility.connectors.node import Node
from eta_utility.servers import OpcUaServer
from eta_utility.servers.modbus import ModbusServer
from examples.connectors.read_emonio_live import (
    emonio_manuell,
    live_from_dict,
    modbus_manuell,
)
from examples.connectors.read_series_eneffco import (
    read_series as ex_read_eneffco,
)
from examples.connectors.read_series_forecast_solar import (
    read_series as ex_read_forecast_solar,
)
from examples.connectors.read_series_wetterdienst import (
    read_series as ex_read_wetterdienst,
)

from ..utilities.pyModbusTCP.client import ModbusClient as MockModbusClient
from ..utilities.requests.eneffco_request import request as request_eneffco
from ..utilities.requests.forecast_solar_request import request as request_forecast_solar


@pytest.fixture
def _local_eneffco_requests(monkeypatch):
    monkeypatch.setattr(requests_cache.CachedSession, "request", request_eneffco)


@pytest.fixture
def _local_forecast_solar_requests(monkeypatch):
    monkeypatch.setattr(requests_cache.CachedSession, "request", request_forecast_solar)


@pytest.fixture
def local_server():
    server = OpcUaServer(5, ip="127.0.0.1", port=4840)
    yield server
    server.stop()


@pytest.fixture
def _mock_client(monkeypatch):
    monkeypatch.setattr(mbclient, "ModbusClient", MockModbusClient)
    monkeypatch.setattr(requests_cache.CachedSession, "request", request_eneffco)


@pytest.mark.usefixtures("_local_eneffco_requests")
def test_example_read_eneffco():
    data = ex_read_eneffco()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"CH1.Elek_U.L1-N", "Pu3.425.ThHy_Q"}


@pytest.mark.skip(reason="wetterdienst API for observations is not working properly.")
def test_example_read_wetterdienst():
    data = ex_read_wetterdienst()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {("Temperature_Darmstadt", "00917")}
    assert data.shape == (19, 1)


@pytest.mark.usefixtures("_local_forecast_solar_requests")
def test_example_read_forecast_solar():
    data = ex_read_forecast_solar()

    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"ForecastSolar Node"}
    assert data.shape == (97, 1)


class TestEmonio:
    @pytest.fixture(scope="class")
    def url(self, config_modbus_port, config_host_ip):
        return f"{config_host_ip}:{config_modbus_port}"

    @pytest.fixture(scope="class")
    def nodes(self, url) -> list[Node]:
        factory = NodeModbusFactory(url)
        voltage_node = factory.get_default_node("Spannung", 300)
        current_node = factory.get_default_node("Strom", 2)
        return [voltage_node, current_node]

    @pytest.fixture(scope="class")
    def server(self, config_modbus_port, nodes):
        with ModbusServer(port=config_modbus_port) as server:
            server.write({nodes[0]: self.values[0]})
            server.write({nodes[1]: self.values[1]})
            # Set phase 'a' to connected
            server._server.data_bank.set_discrete_inputs(0, [1])
            yield server

    values = (230, 1)

    def test_live(self, server, url):
        result = live_from_dict(url)
        assert result["emonio.V_RMS"] // 1 == self.values[0]
        assert result["emonio.I_RMS"] // 1 == self.values[1]

    def test_emonio(self, server, url):
        result = emonio_manuell(url).round(3)
        for value in result.iloc[0].to_numpy():
            assert value in self.values

    def test_modbus(self, server, url):
        result = modbus_manuell(url).round(3)
        for value in result.iloc[0].to_numpy():
            assert value in self.values
