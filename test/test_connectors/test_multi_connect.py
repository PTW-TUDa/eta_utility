import asyncio

import pytest
import requests
from pyModbusTCP import client as mbclient  # noqa: I900

from eta_utility.connectors import CsvSubHandler, Node, connections_from_nodes
from eta_utility.servers import OpcUaServer

from ..conftest import stop_execution
from ..utilities.pyModbusTCP.client import ModbusClient as MockModbusClient
from ..utilities.requests.eneffco_request import request

node = Node(
    "Pu3.425.Mech_n",
    "opc.tcp://192.168.115.10:48050",
    "opcua",
    opc_id="ns=6;s=.HLK.System_425.Pumpe_425.Zustand.Drehzahl",
)
ip = "127.95.11.183"  # local ip address
port = 48050


@pytest.fixture()
def local_server():
    server = OpcUaServer(5, ip=ip, port=port)
    yield server
    server.stop()


@pytest.fixture()
def _mock_client(monkeypatch):
    monkeypatch.setattr(mbclient, "ModbusClient", MockModbusClient)
    monkeypatch.setattr(requests, "request", request)


def test_multi_connect(config_nodes_file, config_eneffco, _mock_client, local_server, temp_dir):
    nodes = Node.from_excel(config_nodes_file["file"], config_nodes_file["sheet"])

    connections = connections_from_nodes(
        nodes, config_eneffco["user"], config_eneffco["pw"], eneffco_api_token=config_eneffco["postman_token"]
    )

    subscription_handler = CsvSubHandler(temp_dir / "multi_connect_test_output.csv")
    loop = asyncio.get_event_loop()

    try:
        for host, connection in connections.items():
            connection.subscribe(subscription_handler)

        loop.run_until_complete(stop_execution(10))

    except KeyboardInterrupt:
        pass
    finally:
        for host, connection in connections.items():
            connection.close_sub()

        try:
            subscription_handler.close()
        except Exception:
            pass
