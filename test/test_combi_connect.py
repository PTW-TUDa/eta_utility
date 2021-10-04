import asyncio
import os

import requests
from pyModbusTCP.client import ModbusClient
from pytest import fixture, mark

from eta_utility.connectors import CsvSubHandler, Node, connections_from_nodes
from eta_utility.servers import OpcUaServer

from .config_tests import Config
from .test_utilities.pyModbusTCP.client import ModbusClient as MockModbusClient
from .test_utilities.requests.eneffco_request import request

EXCEL_NODES_FILE = Config.EXCEL_NODES_FILE
EXCEL_NODES_SHEET = Config.EXCEL_NODES_SHEET
ENEFFCO_USER = Config.ENEFFCO_USER
ENEFFCO_PW = Config.ENEFFCO_PW
ENEFFCO_POSTMAN_TOKEN = Config.ENEFFCO_POSTMAN_TOKEN

node = Node(
    "Pu3.425.Mech_n",
    "opc.tcp://192.168.115.10:48050",
    "opcua",
    opc_id="ns=6;s=.HLK.System_425.Pumpe_425.Zustand.Drehzahl",
)
ip = "127.95.11.183"  # local ip address
port = 48050


async def stop_execution(sleep_time):
    await asyncio.sleep(sleep_time)
    raise KeyboardInterrupt


@fixture
def local_server():
    server = OpcUaServer(5, ip=ip, port=port)
    return server


@fixture
def setup_mock_attributes(monkeypatch):
    monkeypatch.setattr(ModbusClient, "open", MockModbusClient.open)
    monkeypatch.setattr(ModbusClient, "is_open", MockModbusClient.is_open)
    monkeypatch.setattr(ModbusClient, "close", MockModbusClient.close)
    monkeypatch.setattr(ModbusClient, "mode", MockModbusClient.mode)
    monkeypatch.setattr(ModbusClient, "unit_id", MockModbusClient.unit_id)
    monkeypatch.setattr(ModbusClient, "read_holding_registers", MockModbusClient.read_holding_registers)
    monkeypatch.setattr(requests, "request", request)


@mark.parametrize(
    "excel_file, excel_sheet, eneffco_usr, eneffco_pw",
    [
        (EXCEL_NODES_FILE, EXCEL_NODES_SHEET, ENEFFCO_USER, ENEFFCO_PW),
    ],
)
def test_multi_connect(excel_file, excel_sheet, eneffco_usr, eneffco_pw, setup_mock_attributes, local_server):
    try:
        nodes = Node.from_excel(excel_file, excel_sheet)
    except AttributeError:
        raise ValueError("Missing Datatype information")

    connections = connections_from_nodes(
        nodes, eneffco_usr=eneffco_usr, eneffco_pw=eneffco_pw, eneffco_api_token=ENEFFCO_POSTMAN_TOKEN
    )

    try:
        os.remove(Config.CSV_OUTPUT_FILE)
    except FileNotFoundError:
        pass

    subscription_handler = CsvSubHandler(Config.CSV_OUTPUT_FILE)
    loop = asyncio.get_event_loop()

    try:
        for host, connection in connections.items():
            connection.subscribe(subscription_handler)

        loop.run_until_complete(stop_execution(20))

    except KeyboardInterrupt:
        pass
    finally:
        for host, connection in connections.items():
            connection.close_sub()

        subscription_handler.close()

        local_server.stop()
