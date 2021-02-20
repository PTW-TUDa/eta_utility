import datetime

import pandas as pd
from config_tests import *
from pytest import raises

from eta_utility.connectors import Node
from eta_utility.connectors.modbus import ModbusConnection


def test_modbus_failures():
    """Test modbus failures"""
    node_fail = Node(
        "Test.Name",
        "modbus.tcp://someurl:502",
        "modbus",
        mb_channel=19026,
        mb_register="Holding",
        mb_slave=32,
    )

    server_fail = ModbusConnection(node_fail.url)

    with raises(ConnectionError):
        server_fail.read(node_fail)


def test_modbus_read(monkeypatch):
    """Test modbus read function without network access"""

    class MockMBClient:
        @staticmethod
        def open():
            return True

        @staticmethod
        def is_open():
            return True

        @staticmethod
        def close():
            pass

        @staticmethod
        def mode(val):
            pass

        @staticmethod
        def unit_id(val):
            pass

        @staticmethod
        def read_holding_registers(val, num):
            return [17171, 34363]

    node = Node(
        "Serv.NodeName",
        "modbus.tcp://10.0.0.1:502",
        "modbus",
        mb_channel=3861,
        mb_register="Holding",
        mb_slave=32,
    )
    node2 = Node(
        "Serv.NodeName2",
        "modbus.tcp://10.0.0.2:502",
        "modbus",
        mb_channel=3866,
        mb_register="Holding",
        mb_slave=32,
    )

    # Test reading a single node
    server = ModbusConnection(node.url)
    mock_mb_client = MockMBClient()
    monkeypatch.setattr(server, "connection", mock_mb_client)
    res = server.read(node)

    check = pd.DataFrame(
        data=[147.5243377685547],
        index=[datetime.datetime.now()],
        columns=["Serv.NodeName"],
    )

    assert check.columns == res.columns
    assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
    assert isinstance(res.index, pd.DatetimeIndex)

    # Test reading multiple nodes
    server = ModbusConnection(node.url, nodes=[node, node2])
    monkeypatch.setattr(server, "connection", MockMBClient)
    res = server.read()

    check = pd.DataFrame(
        data={
            "Serv.NodeName": [147.5243377685547],
            "Serv.NodeName2": [147.5243377685547],
        },
        index=[datetime.datetime.now()],
    )

    assert set(check.columns) == set(res.columns)
    assert check["Serv.NodeName2"].values == res["Serv.NodeName2"].values
    assert isinstance(res.index, pd.DatetimeIndex)
