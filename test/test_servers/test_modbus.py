import socket

import pytest

from eta_utility.connectors import Node
from eta_utility.connectors.util import bitarray_to_registers, encode_bits
from eta_utility.servers import ModbusServer

nodes = (
    {
        "name": "Serv.NodeName",
        "port": 502,
        "protocol": "modbus",
        "mb_channel": "5000",
        "mb_byteorder": "big",
        "mb_register": "holding",
        "mb_bitlength": 32,
        "dtype": "int",
    },
    {
        "name": "Serv.NodeName",
        "port": 502,
        "protocol": "modbus",
        "mb_channel": "5032",
        "mb_byteorder": "big",
        "mb_register": "holding",
        "mb_bitlength": 32,
        "dtype": "float",
    },
    {
        "name": "Serv.NodeName",
        "port": 502,
        "protocol": "modbus",
        "mb_channel": "5064",
        "mb_byteorder": "big",
        "mb_register": "holding",
        "mb_bitlength": 64,
        "dtype": "string",
    },
    {
        "name": "Serv.NodeName",
        "port": 502,
        "protocol": "modbus",
        "mb_channel": "5128",
        "mb_byteorder": "big",
        "mb_register": "coils",
        "mb_bitlength": 1,
        "dtype": "int",
    },
    {
        "name": "Serv.NodeName",
        "port": 502,
        "protocol": "modbus",
        "mb_channel": "5136",
        "mb_byteorder": "big",
        "mb_register": "coils",
        "mb_bitlength": 4,
        "dtype": "int",
    },
)


@pytest.fixture(scope="module")
def local_nodes():
    _nodes = []
    for node in nodes:
        _nodes.extend(Node.from_dict({**node, "ip": socket.gethostbyname(socket.gethostname())}))

    return _nodes


def test_init():
    try:
        server = ModbusServer(ip="127.0.0.1")
        assert server._server.is_run is True
    finally:
        server.stop()

    # Check session is closed
    assert server._server.is_run is False


def test_init_with():
    with ModbusServer(ip="127.0.0.1") as server:
        assert server._server.is_run is True  # Check session can be created

    # Check session is closed
    assert server._server.is_run is False


class TestServerOperations:
    @pytest.fixture(scope="class")
    def server(self):
        with ModbusServer(ip=socket.gethostbyname(socket.gethostname())) as server:
            yield server

    def test_active(self, server):
        assert server.active is True

    def test_write_data_holding(self, server, local_nodes):
        node = Node(
            "Serv.NodeName",
            url=local_nodes[0].url,
            protocol="modbus",
            mb_channel="5000",
            mb_byteorder="big",
            mb_register="holding",
            mb_bit_length=32,
        )
        value = 5
        server.write({node: value})

        result = server._server.data_bank._h_regs[node.mb_channel : node.mb_channel + node.mb_bit_length // 16]
        expected = bitarray_to_registers(encode_bits(value, node.mb_byteorder, node.mb_bit_length))
        assert result == expected

    def test_write_data_coils(self, server, local_nodes):
        node = Node(
            "Serv.NodeName",
            url=local_nodes[0].url,
            protocol="modbus",
            mb_channel="5000",
            mb_byteorder="big",
            mb_register="coils",
            mb_bit_length=32,
        )
        value = 5
        server.write({node: value})

        result = server._server.data_bank._coils[node.mb_channel : node.mb_channel + node.mb_bit_length]
        expected = encode_bits(value, node.mb_byteorder, node.mb_bit_length)
        assert result == expected

    values = ((0, 0), (1, 1.5), (2, "someelse"), (3, [True]), (4, [False, True, True, False]))

    @pytest.mark.parametrize(("index", "value"), values)
    def test_write(self, server, local_nodes, index, value):
        node = local_nodes[index]
        server.write({node: value})

        if node.mb_register == "holding":
            result = [
                int(x)
                for x in server._server.data_bank._h_regs[node.mb_channel : node.mb_channel + node.mb_bit_length // 16]
            ]
            expected = bitarray_to_registers(encode_bits(value, node.mb_byteorder, node.mb_bit_length))
        else:
            result = [
                int(x) for x in server._server.data_bank._coils[node.mb_channel : node.mb_channel + node.mb_bit_length]
            ]
            expected = value
        assert result == expected

    @pytest.mark.parametrize(("index", "value"), values)
    def test_read(self, server, local_nodes, index, value):
        node = local_nodes[index]
        result = server.read(node)

        if isinstance(value, str):
            assert result[node.name][0] == value
        elif node.mb_register == "coils" or node.mb_register == "discrete_input":
            if len(value) > 1:
                for idx, _ in enumerate(value):
                    assert result[f"{node.name}_{idx}"][0] == value[idx]
            else:
                assert result[node.name][0] == value
        else:
            assert result[node.name][0] == pytest.approx(value)
