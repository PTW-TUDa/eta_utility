import asyncio
import datetime
import socket

import pandas as pd
import pytest

from eta_utility import get_logger
from eta_utility.connectors import DFSubHandler, Node, OpcUaConnection
from eta_utility.servers import OpcUaServer

from ..conftest import stop_execution

init_tests = (
    (("opc.tcp://someurl:48050", None, None), {}, {"url": "opc.tcp://someurl:48050"}),
    (
        ("opc.tcp://someurl:48050", "someuser", "somepassword"),
        {},
        {"url": "opc.tcp://someurl:48050", "usr": "someuser", "pwd": "somepassword"},
    ),
    (
        ("opc.tcp://usr:pwd@someurl:48050", "someuser", "somepassword"),
        {},
        {"url": "opc.tcp://someurl:48050", "usr": "someuser", "pwd": "somepassword"},
    ),
    (("opc.tcp://usr:pwd@someurl:48050",), {}, {"url": "opc.tcp://someurl:48050", "usr": "usr", "pwd": "pwd"}),
    (
        ("opc.tcp://usr:pwd@someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "opc.tcp://someurl:48050",
                    "opcua",
                    usr="auser",
                    pwd="apassword",
                    opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
                ),
            )
        },
        {"url": "opc.tcp://someurl:48050", "usr": "usr", "pwd": "pwd"},
    ),
    (
        ("opc.tcp://someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "opc.tcp://someurl:48050",
                    "opcua",
                    usr="auser",
                    pwd="apassword",
                    opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
                ),
            )
        },
        {"url": "opc.tcp://someurl:48050", "usr": "auser", "pwd": "apassword"},
    ),
)


@pytest.mark.parametrize(("args", "kwargs", "expected"), init_tests)
def test_init(args, kwargs, expected):
    connection = OpcUaConnection(*args, **kwargs)

    for key, value in expected.items():
        assert getattr(connection, key) == value


init_nodes = (
    (
        Node(
            "Serv.NodeName",
            "opc.tcp://someurl:48050",
            "opcua",
            usr="auser",
            pwd="apassword",
            opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
        ),
        {},
        {"url": "opc.tcp://someurl:48050", "usr": "auser", "pwd": "apassword"},
    ),
    (
        Node(
            "Serv.NodeName",
            "opc.tcp://someurl:48050",
            "opcua",
            usr="auser",
            pwd="apassword",
            opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
        ),
        {"usr": "another", "pwd": "pwd"},
        {"url": "opc.tcp://someurl:48050", "usr": "another", "pwd": "pwd"},
    ),
    (
        Node(
            "Serv.NodeName",
            "opc.tcp://someurl:48050",
            "opcua",
            opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
        ),
        {"usr": "another", "pwd": "pwd"},
        {"url": "opc.tcp://someurl:48050", "usr": "another", "pwd": "pwd"},
    ),
)


@pytest.mark.parametrize(("node", "kwargs", "expected"), init_nodes)
def test_init_fromnodes(node, kwargs, expected):
    connection = OpcUaConnection.from_node(node, **kwargs)

    for key, value in expected.items():
        assert getattr(connection, key) == value


init_ids = (
    (
        (["ns=6;s=.Some_Namespace.Node1", "ns=6;s=.Test_Namespace.Node2"], "opc.tcp://127.0.0.1:4840"),
        {"url": "opc.tcp://127.0.0.1:4840"},
    ),
    (
        (
            ["ns=6;s=.Some_Namespace.Node1", "ns=6;s=.Test_Namespace.Node2"],
            "opc.tcp://127.0.0.1:4840",
            "user",
            "password",
        ),
        {"url": "opc.tcp://127.0.0.1:4840", "usr": "user", "pwd": "password"},
    ),
)


@pytest.mark.parametrize(("args", "expected"), init_ids)
def test_init_fromids(args, expected):
    connection = OpcUaConnection.from_ids(*args)

    for key, value in expected.items():
        assert getattr(connection, key) == value


init_fail = (
    (
        ("opc.tcp://someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "opc.tcp://someotherurl:48050",
                    "opcua",
                    opc_id="ns=6;s=.Test_Namespace.Node.Drehzahl",
                ),
            )
        },
        "Some nodes to read from/write to must be specified",
    ),
    (
        ("someurl:48050",),
        {},
        "Given URL is not a valid OPC url",
    ),
)


@pytest.mark.parametrize(("args", "kwargs", "expected"), init_fail)
def test_init_fail(args, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        OpcUaConnection(*args, **kwargs)


read = (
    (
        (Node("Serv.NodeName", "opc.tcp://127.0.0.1:4840", "opcua", opc_id="ns=6;s=.Some_Namespace.Node1"),),
        pd.DataFrame(data={"Serv.NodeName": 2858.00000}, index=[datetime.datetime.now()]),
    ),
    (
        (
            Node("Serv.NodeName", "opc.tcp://127.0.0.1:4840", "opcua", opc_id="ns=6;s=.Some_Namespace.Node1"),
            Node("Serv.NodeName2", "opc.tcp://127.0.0.1:4840", "opcua", opc_id="ns=6;s=.Some_Namespace.Node1"),
        ),
        pd.DataFrame(data={"Serv.NodeName": 2858.00000, "Serv.NodeName2": 2858.00000}, index=[datetime.datetime.now()]),
    ),
    (
        (
            Node("Serv.NodeName", "opc.tcp://127.0.0.1:4840", "opcua", opc_id="ns=6;s=.Some_Namespace.Node1"),
            Node("Serv.NodeName2", "opc.tcp://10.10.0.1:4840", "opcua", opc_id="ns=6;s=.Some_Namespace.Node1"),
        ),
        pd.DataFrame(data={"Serv.NodeName": 2858.00000}, index=[datetime.datetime.now()]),
    ),
)


nodes = (
    {
        "name": "Serv.NodeName",
        "port": 4840,
        "protocol": "opcua",
        "opc_id": "ns=6;s=.Some_Namespace.NodeFloat",
        "dtype": "float",
    },
    {
        "name": "Serv.NodeName2",
        "port": 4840,
        "protocol": "opcua",
        "opc_id": "ns=6;s=.Some_Namespace.NodeInt",
        "dtype": "int",
    },
    {
        "name": "Serv.NodeName4",
        "port": 4840,
        "protocol": "opcua",
        "opc_id": "ns=6;s=.Some_Namespace.NodeStr",
        "dtype": "str",
    },
)


@pytest.fixture(scope="module")
def local_nodes():
    _nodes = []
    for node in nodes:
        _nodes.extend(Node.from_dict({**node, "ip": socket.gethostbyname(socket.gethostname())}))

    return _nodes


class TestConnectorOperations:
    @pytest.fixture(scope="class", autouse=True)
    def server(self):
        with OpcUaServer(5, ip=socket.gethostbyname(socket.gethostname())) as server:
            yield server

    @pytest.fixture(scope="class")
    def connection(self, local_nodes):
        connection = OpcUaConnection.from_node(local_nodes[0], usr="admin", pwd="0")
        return connection

    def test_create_nodes(self, server, connection, local_nodes):
        connection.create_nodes(local_nodes)

        for node in local_nodes:
            server.read(local_nodes)

    values = ((0, 1.5), (1, 5), (2, "something"))

    @pytest.mark.parametrize(("index", "value"), values)
    def test_write_node(self, server, connection, local_nodes, index, value):
        connection.write({local_nodes[index]: value})

        assert server.read(local_nodes[index]).iloc[0, 0] == value

    @pytest.mark.parametrize(("index", "expected"), values)
    def test_read_node(self, connection, local_nodes, index, expected):
        val = connection.read({local_nodes[index]})

        assert val.iloc[0, 0] == expected
        assert val.columns[0] == local_nodes[index].name

    def test_read_fail(self, connection, local_nodes):
        n = local_nodes[0]
        fail_node = Node(n.name, n.url, n.protocol, usr=n.usr, pwd=n.pwd, opc_id="ns=6;s=AnotherNamespace.DoesNotExist")
        with pytest.raises(ConnectionError, match=".*BadNodeIdUnknown.*"):
            connection.read(fail_node)

    def test_recreate_existing_node(self, connection, local_nodes, caplog):
        log = get_logger()
        log.propagate = True

        # Create Node that already exists
        connection.create_nodes(local_nodes[0])
        assert f"Node with NodeId : {local_nodes[0].opc_id} could not be created. It already exists." in caplog.text

    def test_login_fail_write(self, local_nodes):
        n = local_nodes[0]
        connection = OpcUaConnection.from_node(n, usr="another", pwd="something")
        with pytest.raises(ConnectionError, match=".*BadUserAccessDenied.*"):
            connection.write({n: 123})

    def test_delete_nodes(self, connection, local_nodes):
        connection.delete_nodes(local_nodes)

        with pytest.raises(ConnectionError, match=".*BadNodeIdUnknown.*"):
            connection.read(local_nodes)

    def test_login_fail_read(self, server, local_nodes):
        n = local_nodes[0]
        connection = OpcUaConnection.from_node(n, usr="another", pwd="something")

        # make server reject everything
        def um(s, u, p):
            return False

        server._server.user_manager.user_manager = um

        with pytest.raises(ConnectionError, match=".*BadUserAccessDenied.*"):
            connection.read(n)


class TestConnectorSubscriptions:
    values = {
        "Serv.NodeName": (1.5, 2, 2.5, 1, 1.1, 3.4, 6.5, 7.1),
        "Serv.NodeName2": (5, 3, 4, 2, 3, 6, 3, 2),
        "Serv.NodeName4": ("something", "some1", "some2", "something else", "different", "1", "2", "3"),
    }

    @pytest.fixture(scope="class", autouse=True)
    def server(self, local_nodes):
        with OpcUaServer(5, ip=socket.gethostbyname(socket.gethostname())) as server:
            server.create_nodes(local_nodes)
            yield server

    @pytest.fixture()
    def _write_nodes_normal(self, server, local_nodes):
        async def write_loop(server, local_nodes, values):
            i = 0
            while True:
                server.write({node: values[node.name][i] for node in local_nodes})
                # Index should fall back to one if the number of provided values is exceeded.
                i = i + 1 if i < len(values[local_nodes[0].name]) - 1 else 0
                await asyncio.sleep(1)

        asyncio.get_event_loop().create_task(write_loop(server, local_nodes, self.values))

    def test_subscribe(self, local_nodes, _write_nodes_normal):
        connection = OpcUaConnection.from_node(local_nodes[0], usr="admin", pwd="0")
        handler = DFSubHandler(write_interval=1)
        connection.subscribe(handler, nodes=local_nodes, interval=1)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(stop_execution(5))

        for node, values in self.values.items():
            assert set(handler.data[node]) <= set(values)

        connection.close_sub()

    @pytest.fixture()
    def _write_nodes_interrupt(self, server, local_nodes):
        async def write_loop(server, local_nodes, values):
            i = 0
            while True:
                if i == 3:
                    server.stop()
                elif 3 < i < 6:
                    pass
                elif i == 6:
                    server.start()
                else:
                    server.write(
                        {node: values[node.name][i % len(values[local_nodes[0].name])] for node in local_nodes}
                    )

                # Index should fall back to one if the number of provided values is exceeded.
                i += 1
                await asyncio.sleep(1)

        asyncio.get_event_loop().create_task(write_loop(server, local_nodes, self.values))

    def test_subscribe_interrupted(self, local_nodes, _write_nodes_interrupt, caplog):
        log = get_logger()
        log.propagate = True

        connection = OpcUaConnection.from_node(local_nodes[0], usr="admin", pwd="0")
        handler = DFSubHandler(write_interval=1)
        connection.subscribe(handler, nodes=local_nodes, interval=1)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(stop_execution(25))
        connection.close_sub()

        for node, values in self.values.items():
            assert set(handler.data[node]) <= set(values)

        # Check if connection was actually interrupted during the test.
        messages_found = 0
        for message in caplog.messages:
            if "Error while checking connection" in message or "Retrying connection to opc" in message:
                messages_found += 1

        assert messages_found >= 2, "Error while interrupting the connection, test could not be executed reliably."
