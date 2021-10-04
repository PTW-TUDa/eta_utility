import datetime

import pandas as pd
from pytest import approx, fixture, raises

from eta_utility.connectors import ModbusConnection, Node

from .test_utilities.pyModbusTCP.client import ModbusClient

node = Node(
    "Serv.NodeName",
    "modbus.tcp://10.0.0.1:502",
    "modbus",
    mb_channel=3861,
    mb_register="Holding",
    mb_slave=32,
    mb_byteorder="big",
)
value = [17171, 34363]
check = pd.DataFrame(
    data=[147.524],
    index=[datetime.datetime.now()],
    columns=["Serv.NodeName"],
)


def test_modbus_connection_fail():
    """Test modbus failures"""
    server_fail = ModbusConnection(node.url)

    with raises(ConnectionError):
        server_fail.read(node)


class TestMBBigEndian:

    node2 = Node(
        "Serv.NodeName2",
        "modbus.tcp://10.0.0.1:502",
        "modbus",
        mb_channel=6345,
        mb_register="Holding",
        mb_slave=32,
        mb_byteorder="big",
    )
    value2 = [17254, 49900]
    check2 = pd.DataFrame(
        data=[230.761],
        index=[datetime.datetime.now()],
        columns=["Serv.NodeName"],
    )

    @fixture()
    def mb_connect(self, monkeypatch):
        server = ModbusConnection(node.url)

        mock_mb_client = ModbusClient()
        mock_mb_client.value = value
        monkeypatch.setattr(server, "connection", mock_mb_client)

        return server

    def test_modbus_read(self, mb_connect):
        """Test modbus read function without network access"""
        res = mb_connect.read(node)

        assert set(res.columns) == set(check.columns)
        assert res["Serv.NodeName"].values == approx(check["Serv.NodeName"].values, 0.001)
        assert isinstance(res.index, pd.DatetimeIndex)

    def test_modbus_multiple_nodes_read(self, mb_connect, monkeypatch):
        # Test reading multiple nodes
        mb_connect.selected_nodes = {node, self.node2}
        monkeypatch.setattr(mb_connect.connection, "value", self.value2)
        res = mb_connect.read()

        check = pd.DataFrame(
            data={
                "Serv.NodeName": [230.761],
                "Serv.NodeName2": [230.761],
            },
            index=[datetime.datetime.now()],
        )

        assert set(res.columns) == set(check.columns)
        assert res["Serv.NodeName"].values == approx(check["Serv.NodeName"].values, 0.001)
        assert res["Serv.NodeName2"].values == approx(check["Serv.NodeName2"].values, 0.001)
        assert isinstance(res.index, pd.DatetimeIndex)


class TestMBLittleEndian:
    node = Node(
        "Serv.NodeName",
        "modbus.tcp://10.0.0.1:502",
        "modbus",
        mb_channel=3861,
        mb_register="Holding",
        mb_slave=32,
        mb_byteorder="little",
    )
    value = [26179, 60610]
    check = pd.DataFrame(
        data=[230.761],
        index=[datetime.datetime.now()],
        columns=["Serv.NodeName"],
    )

    @fixture()
    def mb_connect(self, monkeypatch):
        server = ModbusConnection(self.node.url)

        mock_mb_client = ModbusClient()
        mock_mb_client.value = self.value
        monkeypatch.setattr(server, "connection", mock_mb_client)

        return server

    def test_read_little_endian(self, mb_connect):
        """Test byteorder as big endian"""
        res = mb_connect.read(self.node)

        assert set(res.columns) == set(self.check.columns)
        assert res["Serv.NodeName"].values == approx(self.check["Serv.NodeName"].values, 0.001)
        assert isinstance(res.index, pd.DatetimeIndex)
