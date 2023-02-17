import asyncio
import socket

import pytest

from eta_utility import get_logger
from eta_utility.connectors import DFSubHandler, ModbusConnection, Node
from eta_utility.servers import ModbusServer

from ..conftest import stop_execution

init_tests = (
    (("modbus.tcp://someurl:48050", None, None), {}, {"url": "modbus.tcp://someurl:48050"}),
    (
        ("modbus.tcp://someurl:48050", "someuser", "somepassword"),
        {},
        {"url": "modbus.tcp://someurl:48050", "usr": "someuser", "pwd": "somepassword"},
    ),
    (
        ("modbus.tcp://usr:pwd@someurl:48050", "someuser", "somepassword"),
        {},
        {"url": "modbus.tcp://someurl:48050", "usr": "someuser", "pwd": "somepassword"},
    ),
    (("modbus.tcp://usr:pwd@someurl:48050",), {}, {"url": "modbus.tcp://someurl:48050", "usr": "usr", "pwd": "pwd"}),
    (
        ("modbus.tcp://usr:pwd@someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "modbus.tcp://someurl:48050",
                    "modbus",
                    usr="auser",
                    pwd="apassword",
                    mb_register="holding",
                    mb_channel=5000,
                    mb_byteorder="big",
                ),
            )
        },
        {"url": "modbus.tcp://someurl:48050", "usr": "usr", "pwd": "pwd"},
    ),
    (
        ("modbus.tcp://someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "modbus.tcp://someurl:48050",
                    "modbus",
                    usr="auser",
                    pwd="apassword",
                    mb_register="holding",
                    mb_channel=5000,
                    mb_byteorder="big",
                ),
            )
        },
        {"url": "modbus.tcp://someurl:48050", "usr": "auser", "pwd": "apassword"},
    ),
)


@pytest.mark.parametrize(("args", "kwargs", "expected"), init_tests)
def test_init(args, kwargs, expected):
    connection = ModbusConnection(*args, **kwargs)

    for key, value in expected.items():
        assert getattr(connection, key) == value


init_nodes = (
    (
        Node(
            "Serv.NodeName",
            "modbus.tcp://someurl:48050",
            "modbus",
            usr="auser",
            pwd="apassword",
            mb_register="holding",
            mb_channel=5000,
            mb_byteorder="big",
        ),
        {},
        {"url": "modbus.tcp://someurl:48050", "usr": "auser", "pwd": "apassword"},
    ),
    (
        Node(
            "Serv.NodeName",
            "modbus.tcp://someurl:48050",
            "modbus",
            usr="auser",
            pwd="apassword",
            mb_register="holding",
            mb_channel=5000,
            mb_byteorder="big",
        ),
        {"usr": "another", "pwd": "pwd"},
        {"url": "modbus.tcp://someurl:48050", "usr": "another", "pwd": "pwd"},
    ),
    (
        Node(
            "Serv.NodeName",
            "modbus.tcp://someurl:48050",
            "modbus",
            mb_register="holding",
            mb_channel=5000,
            mb_byteorder="big",
        ),
        {"usr": "another", "pwd": "pwd"},
        {"url": "modbus.tcp://someurl:48050", "usr": "another", "pwd": "pwd"},
    ),
)


@pytest.mark.parametrize(("node", "kwargs", "expected"), init_nodes)
def test_init_fromnodes(node, kwargs, expected):
    connection = ModbusConnection.from_node(node, **kwargs)

    for key, value in expected.items():
        assert getattr(connection, key) == value


init_fail = (
    (
        ("opc.tcp://someurl:48050",),
        {
            "nodes": (
                Node(
                    "Serv.NodeName",
                    "modbus.tcp://someotherurl:48050",
                    "modbus",
                    mb_channel=3861,
                    mb_register="Holding",
                    mb_slave=32,
                    mb_byteorder="little",
                ),
            )
        },
        "Some nodes to read from/write to must be specified",
    ),
    (
        ("someurl:48050",),
        {},
        "Given URL is not a valid Modbus url",
    ),
)


@pytest.mark.parametrize(("args", "kwargs", "expected"), init_fail)
def test_init_fail(args, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        ModbusConnection(*args, **kwargs)


def test_modbus_connection_fail():
    """Test modbus failures"""
    node = Node(
        "Serv.NodeName",
        "modbus.tcp://10.0.0.1:502",
        "modbus",
        mb_channel=3861,
        mb_register="Holding",
        mb_slave=32,
        mb_byteorder="big",
    )
    server_fail = ModbusConnection(node.url)

    with pytest.raises(ConnectionError, match="Could not establish connection"):
        server_fail.read(node)


nodes = (
    {
        "name": "Serv.NodeName",
        "protocol": "modbus",
        "mb_channel": 3200,
        "mb_register": "Holding",
        "mb_byteorder": "big",
        "dtype": "float",
    },
    {
        "name": "Serv.NodeName2",
        "protocol": "modbus",
        "mb_channel": 3232,
        "mb_register": "Holding",
        "mb_byteorder": "big",
        "dtype": "int",
    },
    {
        "name": "Serv.NodeName4",
        "protocol": "modbus",
        "mb_channel": 3264,
        "mb_register": "Holding",
        "mb_byteorder": "big",
        "mb_bitlength": 80,
        "dtype": "str",
    },
)


@pytest.fixture(scope="class")
def local_nodes():
    _nodes = []
    for node in nodes:
        _nodes.extend(Node.from_dict({**node, "ip": socket.gethostbyname(socket.gethostname())}))

    return _nodes


class TestConnectorOperations:
    @pytest.fixture(scope="class", autouse=True)
    def server(self):
        with ModbusServer(ip=socket.gethostbyname(socket.gethostname())) as server:
            yield server

    @pytest.fixture(scope="class")
    def connection(self, local_nodes):
        connection = ModbusConnection.from_node(local_nodes[0])
        return connection

    values = ((0, 1.5), (1, 5), (2, " something"))

    @pytest.mark.parametrize(("index", "value"), values)
    def test_write_node(self, server, connection, local_nodes, index, value):
        connection.write({local_nodes[index]: value})

        assert server.read(local_nodes[index]).iloc[0, 0] == value

    @pytest.mark.parametrize(("index", "expected"), values)
    def test_read_node(self, connection, local_nodes, index, expected):
        val = connection.read({local_nodes[index]})

        assert val.iloc[0, 0] == expected
        assert val.columns[0] == local_nodes[index].name

    def test_read_fail_reg_addr(self, connection, local_nodes):
        n = local_nodes[0]
        fail_node = Node(
            n.name,
            n.url,
            n.protocol,
            mb_channel=129387192,
            mb_register=n.mb_register,
            mb_byteorder=n.mb_byteorder,
            mb_bit_length=n.mb_bit_length,
        )
        with pytest.raises(ValueError, match="reg_addr out of range"):
            connection.read(fail_node)


class TestConnectorOperationsLittleEndian:
    @pytest.fixture(scope="class")
    def local_nodes(self):
        _nodes = []
        for node in nodes:
            _nodes.extend(
                Node.from_dict({**node, "ip": socket.gethostbyname(socket.gethostname()), "mb_byteorder": "little"})
            )

        return _nodes

    @pytest.fixture(scope="class", autouse=True)
    def server(self):
        with ModbusServer(ip=socket.gethostbyname(socket.gethostname()), big_endian=False) as server:
            yield server

    @pytest.fixture(scope="class")
    def connection(self, local_nodes):
        connection = ModbusConnection.from_node(local_nodes[0])
        return connection

    values = ((0, 1.5), (1, 5), (2, " something"))

    @pytest.mark.parametrize(("index", "value"), values)
    def test_write_node(self, server, connection, local_nodes, index, value):
        connection.write({local_nodes[index]: value})

        assert server.read(local_nodes[index]).iloc[0, 0] == value

    @pytest.mark.parametrize(("index", "expected"), values)
    def test_read_node(self, connection, local_nodes, index, expected):
        val = connection.read({local_nodes[index]})

        assert val.iloc[0, 0] == expected
        assert val.columns[0] == local_nodes[index].name


class TestConnectorSubscriptions:
    values = {
        "Serv.NodeName": (1.5, 2, 2.5, 1, 1.1, 3.4, 6.5, 7.1),
        "Serv.NodeName2": (5, 3, 4, 2, 3, 6, 3, 2),
        "Serv.NodeName4": (
            " something",
            " thething1",
            " another23",
            " someother",
            " different",
            " 112389223",
            " 285746384",
            " 327338574",
        ),
    }

    @pytest.fixture(scope="class", autouse=True)
    def server(self, local_nodes):
        with ModbusServer(ip=socket.gethostbyname(socket.gethostname())) as server:
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
        connection = ModbusConnection.from_node(local_nodes[0], usr="admin", pwd="0")
        handler = DFSubHandler(write_interval=1)
        connection.subscribe(handler, nodes=local_nodes, interval=1)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(stop_execution(5))

        for node, values in self.values.items():
            for idx, val in enumerate(values):
                try:
                    assert handler.data[node][idx] == pytest.approx(val, 0.01)
                except IndexError:
                    break

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

        connection = ModbusConnection.from_node(local_nodes[0], usr="admin", pwd="0")
        handler = DFSubHandler(write_interval=1)
        connection.subscribe(handler, nodes=local_nodes, interval=1)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(stop_execution(25))
        connection.close_sub()

        for node, values in self.values.items():
            # Don't check floating point values in this case because it is hard to deal with precision problems here.
            if handler.data[node].dtype == "float":
                continue
            assert set(handler.data[node]) <= set(values)

        # Check if connection was actually interrupted during the test.
        messages_found = 0
        for message in caplog.messages:
            if "ModbusError 4 at modbus" in message or "Could not establish connection to host" in message:
                messages_found += 1

        assert messages_found >= 2, "Error while interrupting the connection, test could not be executed reliably."
