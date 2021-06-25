import datetime
from test.config_tests import *  # noqa
from test.test_utilities.pyModbusTCP.client import ModbusClient
from test.test_utilities.pyModbusTCP.nodes import ModbusNodes as nodes

from pytest import raises

from eta_utility.connectors.modbus import ModbusConnection


class TestModbus:
    _node = nodes.node
    _node2 = nodes.node2
    _node_fail = nodes.node_fail

    def test_modbus_failures(self):
        """Test modbus failures"""
        server_fail = ModbusConnection(self._node_fail.url)

        with raises(ConnectionError):
            server_fail.read(self._node_fail)

    def test_modbus_read(self, monkeypatch):
        """Test modbus read function without network access"""
        # Test reading a single node
        server = ModbusConnection(self._node.url)
        mock_mb_client = ModbusClient()
        monkeypatch.setattr(server, "connection", mock_mb_client)
        res = server.read(self._node)

        check = pd.DataFrame(
            data=[147.5243377685547],
            index=[datetime.datetime.now()],
            columns=["Serv.NodeName"],
        )

        assert check.columns == res.columns
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)

    def test_modbus_multiple_nodes_read(self, monkeypatch):
        # Test reading multiple nodes
        server = ModbusConnection(self._node.url, nodes=[self._node, self._node2])
        mock_mb_client = ModbusClient()
        monkeypatch.setattr(server, "connection", mock_mb_client)
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
