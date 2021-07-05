import datetime
import random
import socket
import struct
from test.test_utilities.pyOPCUA.client import Client
from test.test_utilities.pyOPCUA.nodes import OPCUANodes as nodes

import opcua.ua.uaerrors
import pandas as pd
from pytest import fail, raises

from eta_utility.connectors.opcua import OpcUaConnection
from eta_utility.servers.opcua import OpcUaServer

_node = nodes.node
_node2 = nodes.node2
_node_fail = nodes.node_fail
_node_case_sen = nodes.node_case_sen


class TestOPCUA1:
    def test_opcua_failures(self):
        """Test opcua failures"""
        server_fail = OpcUaConnection(_node_fail.url)

        with raises(ConnectionError):
            server_fail.read(_node_fail)

        server = OpcUaConnection(_node.url)

        # Reading without specifying nodes raises Value Error
        with raises(ValueError):
            server.read()

    def test_opcua_read(self, monkeypatch):
        """Test opcua read function without network access"""

        # Test reading a single node
        server = OpcUaConnection(_node.url)
        mock_opcua_client = Client(server.url)
        monkeypatch.setattr(server, "connection", mock_opcua_client)
        res = server.read(_node)

        check = pd.DataFrame(
            data=[2858.00000],
            index=[datetime.datetime.now()],
            columns=["Serv.NodeName"],
        )
        assert check.columns == res.columns
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)

    def test_opcua_read_multiple_nodes(self, monkeypatch):
        # Test reading multiple nodes
        server = OpcUaConnection(_node.url, nodes=[_node, _node2])
        mock_opcua_client = Client(server.url)
        monkeypatch.setattr(server, "connection", mock_opcua_client)
        res = server.read()

        check = pd.DataFrame(
            data={"Serv.NodeName": [2858.00000], "Serv.NodeName2": [2858.00000]},
            index=[datetime.datetime.now()],
        )

        assert set(check.columns) == set(res.columns)
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)


class TestOPCUA2:
    _server = OpcUaServer(6)
    _connection = OpcUaConnection(_server.url, "admin", nodes=_node)

    def test_for_opcua_server_and_client_write(self):
        TestOPCUA2._connection.create_nodes({_node, _node2})
        values = {_node: 16, _node2: 56}
        TestOPCUA2._connection.write(values)
        assert int(TestOPCUA2._connection.read(_node)[_node.name].iloc[0]) == 16

    def test_recreate_already_existing_node(self, caplog):
        # Create Node that already exists
        TestOPCUA2._connection.create_nodes(_node)
        assert f"Node with NodeId : {_node.opc_id} could not be created. It already exists." in caplog.text

    def test_read_non_existent_node(self):
        # Read from a node that does not exist
        with raises(ConnectionError) as e:
            TestOPCUA2._connection.read(_node_fail)
            assert e == opcua.ua.uaerrors.BadNodeIdUnknown

    def test_delete_one_node_check_for_another(self):
        # Delete one of the nodes and check that the other one remains
        TestOPCUA2._connection.delete_nodes(_node)
        with raises(ConnectionError):
            TestOPCUA2._connection.read(_node)
        assert int(TestOPCUA2._connection.read(_node2)[_node2.name].iloc[0]) == 56

    def test_server_recreate_first_node_write_diff_value(self):
        # Use the server to recreate the first node and to write some different values
        TestOPCUA2._server.create_nodes(_node)
        values = {_node: 19, _node2: 46}
        TestOPCUA2._server.write(values)
        assert int(TestOPCUA2._connection.read(_node)[_node.name].iloc[0]) == 19
        assert int(TestOPCUA2._connection.read(_node2)[_node2.name].iloc[0]) == 46

    def test_delete_all_nodes(self):
        # Finally delete all those nodes using the server
        TestOPCUA2._server.delete_nodes({_node, _node2})
        with raises(ConnectionError):
            TestOPCUA2._connection.read(_node)

        TestOPCUA2._server.stop()


class TestOPCUA3:
    def test_opc_id_not_case_sensitive(self):
        server = OpcUaServer(4)
        connection = OpcUaConnection(server.url, "admin", nodes=_node_case_sen)  # denies access if usr = None
        connection.create_nodes(_node_case_sen)
        values = {_node_case_sen: 10}
        connection.write(values)
        assert int(connection.read(_node_case_sen)[_node_case_sen.name].iloc[0]) == 10
        server.stop()

    def test_ip_address_exclusively_stated(self):
        ip = socket.inet_ntoa(struct.pack(">I", random.randint(0x7F000000, 0x7FFFFFFF)))  # random IP on local network
        try:
            server = OpcUaServer(5, ip=ip)
            server.stop()
        except ConnectionError as e:
            fail(str(e))
