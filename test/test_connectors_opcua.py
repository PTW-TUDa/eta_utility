import datetime
import time

import pandas as pd
from pytest import raises

from eta_utility.connectors import Node
from eta_utility.connectors.opcua import OpcUaConnection
from eta_utility.servers.opcua import OpcUaServer


def test_opcua_failures():
    """Test opcua failures"""

    node_fail = Node(
        "Serv.NodeName",
        "opc.tcp://someurl:48050",
        "opcua",
        opc_id="ns=6;s=.Test_Namespace.Node" ".Drehzahl",
    )
    node = Node(
        "Serv.NodeName2",
        "opc.tcp://10.0.0.1:48050",
        "opcua",
        opc_id="ns=6;s=.Test_Namespace.Node2",
    )

    server_fail = OpcUaConnection(node_fail.url)

    with raises(ConnectionError):
        server_fail.read(node_fail)

    server = OpcUaConnection(node.url)

    # Reading without specifying nodes raises Value Error
    with raises(ValueError):
        server.read()


def test_opcua_read(monkeypatch):
    """Test opcua read function without network access"""

    def mock_connect():
        pass

    class MockNode:
        def get_value(self):
            return 2858.0

    def mock_get_node(node):
        return MockNode()

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
        opc_id="ns=6;s=.Some_Namespace.Node2",
    )

    # Test reading a single node
    server = OpcUaConnection(node.url)
    monkeypatch.setattr(server.connection, "connect", mock_connect)
    monkeypatch.setattr(server.connection, "get_node", mock_get_node)
    res = server.read(node)

    check = pd.DataFrame(
        data=[2858.00000],
        index=[datetime.datetime.now()],
        columns=["Serv.NodeName"],
    )
    assert check.columns == res.columns
    assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
    assert isinstance(res.index, pd.DatetimeIndex)

    # Test reading multiple nodes
    server = OpcUaConnection(node.url, nodes=[node, node2])
    monkeypatch.setattr(server.connection, "connect", mock_connect)
    monkeypatch.setattr(server.connection, "get_node", mock_get_node)
    res = server.read()

    check = pd.DataFrame(
        data={"Serv.NodeName": [2858.00000], "Serv.NodeName2": [2858.00000]},
        index=[datetime.datetime.now()],
    )

    assert set(check.columns) == set(res.columns)
    assert check["Serv.NodeName"].values == res["Serv.NodeName"].values
    assert isinstance(res.index, pd.DatetimeIndex)


def test_opcua_server_and_client_write():
    # Create server and nodes
    server = OpcUaServer(6)
    node = Node(
        "Serv.NodeName",
        "opc.tcp://localhost:4840",
        "opcua",
        opc_id="ns=6;s=.Some_Namespace.Name0",
    )
    node2 = Node(
        "Serv.NodeName2",
        "opc.tcp://localhost:4840",
        "opcua",
        opc_id="ns=6;s=.Some_Namespace.Name1",
    )

    # Create a connection and some values
    time.sleep(5)
    connection = OpcUaConnection(server.url, "admin", nodes=node)
    values = {node: 16, node2: 56}

    # Try creating nodes on the server and writing to them
    connection.create_nodes({node, node2})
    connection.write(values)
    assert int(connection.read(node)[node.name].iloc[0]) == 16

    # Delete one of the nodes and check that the other one remains
    connection.delete_nodes(node)
    with raises(ConnectionError):
        connection.read(node)
    assert int(connection.read(node2)[node2.name].iloc[0]) == 56

    # Use the server to recreate the first node and to write some different values
    server.create_nodes(node)
    values = {node: 19, node2: 46}
    server.write(values)
    assert int(connection.read(node)[node.name].iloc[0]) == 19
    assert int(connection.read(node2)[node2.name].iloc[0]) == 46

    # Finally delete all those nodes using the server
    server.delete_nodes({node, node2})
    with raises(ConnectionError):
        connection.read(node)

    server.stop()
