import datetime
import random
import socket
import struct

import pandas as pd
import pytest

from eta_utility import get_logger
from eta_utility.connectors import Node, OpcUaConnection
from eta_utility.servers import OpcUaServer

from .test_utilities.opcua import Client as OpcuaClient

node = Node(
    "Serv.NodeName",
    "opc.tcp://10.0.0.1:48050",
    "opcua",
    opc_id="ns=6;s=.Some_Namespace.Node1",
)

node2 = Node(
    "Serv.NodeName2",
    "opc.tcp://10.0.0.1:48050",
    "opcua",
    opc_id="ns=6;s=.Test_Namespace.Node2",
)

node_fail = Node(
    "Serv.NodeName",
    "opc.tcp://someurl:48050",
    "opcua",
    opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
)


@pytest.fixture()
def local_nodes():
    return (
        Node(
            "Serv.NodeName",
            f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:4840",
            "opcua",
            opc_id="ns=6;s=.Some_Namespace.Node1",
        ),
        Node(
            "Serv.NodeName2",
            f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:4840",
            "opcua",
            opc_id="ns=6;s=.Some_Namespace.Node2",
        ),
    )


@pytest.fixture()
def local_node_case_sensitive():
    return Node(
        "Serv.NodeName",
        f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:4840",
        "opcua",
        opc_id="NS=4;S=.Some_Namespace.Node3",  # NS instead of ns. S instead of s.
    )


def test_opcua_failures():
    """Test opcua failures"""
    server_fail = OpcUaConnection(node_fail.url)

    with pytest.raises(ConnectionError):
        server_fail.read(node_fail)

    server = OpcUaConnection(node.url)

    # Reading without specifying nodes raises Value Error
    with pytest.raises(ValueError, match="Some nodes to read from/write to must be specified"):
        server.read()


class TestOpcUABasics:
    @pytest.fixture()
    def connection(self, monkeypatch):
        # Test reading a single node
        connection = OpcUaConnection(node.url)
        mock_opcua_client = OpcuaClient(connection.url)
        monkeypatch.setattr(connection, "connection", mock_opcua_client)
        return connection

    def test_opcua_read(self, connection):
        """Test opcua read function without network access"""

        res = connection.read(node)

        check = pd.DataFrame(
            data=[2858.00000],
            index=[datetime.datetime.now()],
            columns=["Serv.NodeName"],
        )

        assert check.columns == res.columns
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)

    def test_opcua_read_multiple_nodes(self, connection):
        # Test reading multiple nodes
        res = connection.read([node, node2])

        check = pd.DataFrame(
            data={"Serv.NodeName": [2858.00000], "Serv.NodeName2": [2858.00000]},
            index=[datetime.datetime.now()],
        )

        assert set(check.columns) == set(res.columns)
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)

    def test_connection_creation_from_node_ids(self, local_nodes, monkeypatch):
        # creating connection with node and node2 information
        connection = OpcUaConnection.from_ids(
            ids=["ns=6;s=.Some_Namespace.Node1", "ns=6;s=.Test_Namespace.Node2"],
            url=f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:4840",
        )

        mock_opcua_client = OpcuaClient(connection.url)
        monkeypatch.setattr(connection, "connection", mock_opcua_client)

        res = connection.read(local_nodes)
        check = pd.DataFrame(
            data={"Serv.NodeName": [2858.00000], "Serv.NodeName2": [2858.00000]},
            index=[datetime.datetime.now()],
        )

        assert set(check.columns) == set(res.columns)
        assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
        assert isinstance(res.index, pd.DatetimeIndex)


class TestOpcUAServerAndConnection:
    @pytest.fixture()
    def server_and_connection(self, local_nodes):
        server = OpcUaServer(6)
        connection = OpcUaConnection(local_nodes[0].url, "admin", nodes=local_nodes[0])
        yield connection

        server.stop()

    @pytest.fixture()
    def server_and_connection_with_nodes(self, server_and_connection, local_nodes):
        server_and_connection.create_nodes(local_nodes)
        values = {local_nodes[0]: 16, local_nodes[1]: 56}
        server_and_connection.write(values)
        return server_and_connection

    def test_opcua_client_write(self, server_and_connection, local_nodes):
        server_and_connection.create_nodes(local_nodes)
        values = {local_nodes[0]: 16, local_nodes[1]: 56}
        server_and_connection.write(values)
        assert int(server_and_connection.read(local_nodes[0])[local_nodes[0].name].iloc[0]) == 16

    def test_recreate_existing_node(self, server_and_connection_with_nodes, local_nodes, caplog):
        log = get_logger()
        log.propagate = True

        sacwn = server_and_connection_with_nodes
        # Create Node that already exists
        sacwn.create_nodes(local_nodes[0])
        assert f"Node with NodeId : {local_nodes[0].opc_id} could not be created. It already exists." in caplog.text

    def test_read_non_existent_node(self, server_and_connection, local_nodes):
        # Read from a node that does not exist
        with pytest.raises(ConnectionError, match="The node id refers to a node that does not exist"):
            server_and_connection.read(local_nodes[0])

    def test_delete_one_node_check_for_another(self, server_and_connection_with_nodes, local_nodes):
        sacwn = server_and_connection_with_nodes

        # Delete one of the nodes and check that the other one remains
        sacwn.delete_nodes(local_nodes[0])
        with pytest.raises(ConnectionError):
            sacwn.read(local_nodes[0])
        assert int(sacwn.read(local_nodes[1])[local_nodes[1].name].iloc[0]) == 56

    def test_opc_id_not_case_sensitive(self, server_and_connection_with_nodes, local_node_case_sensitive):
        sacwn = server_and_connection_with_nodes
        node = local_node_case_sensitive

        sacwn.create_nodes(node)
        values = {node: 10}
        sacwn.write(values)
        assert int(sacwn.read(node)[node.name].iloc[0]) == 10


class TestOpcUAServer:
    @pytest.fixture()
    def server_with_nodes(self, local_nodes):
        server = OpcUaServer(6)
        server.create_nodes(local_nodes)
        values = {local_nodes[0]: 16, local_nodes[1]: 56}
        server.write(values)
        yield (server)

        server.stop()

    def test_init_server_with_ip(self):
        ip = socket.inet_ntoa(struct.pack(">I", random.randint(0x7F000000, 0x7FFFFFFF)))  # random IP on local network
        try:
            server = OpcUaServer(5, ip=ip)
            server.stop()
        except ConnectionError as e:
            pytest.fail(str(e))

    def test_server_recreate_first_node_write_diff_value(self, server_with_nodes, local_nodes):
        server = server_with_nodes
        connection = OpcUaConnection(local_nodes[0].url)

        # Use the server to recreate the first node and to write some different values
        values = {local_nodes[0]: 19, local_nodes[1]: 46}
        server.write(values)
        assert int(connection.read(local_nodes[0])[local_nodes[0].name].iloc[0]) == 19
        assert int(connection.read(local_nodes[1])[local_nodes[1].name].iloc[0]) == 46

    def test_delete_all_nodes(self, server_with_nodes, local_nodes):
        server = server_with_nodes
        connection = OpcUaConnection(local_nodes[0].url)

        # Finally delete all those nodes using the server
        server.delete_nodes(local_nodes)
        with pytest.raises(ConnectionError):
            connection.read(local_nodes[0])
