from pytest import mark, raises

from eta_utility.connectors import Node

from .config_tests import *  # noqa


class TestNodeInit:
    init_nodes = [
        (
            "modbus",
            Node(
                "Serv.NodeName",
                "modbus.tcp://10.0.0.1:502",
                "modbus",
                mb_channel=3861,
                mb_register="Holding",
                mb_slave=32,
                mb_byteorder="big",
            ),
        ),
        (
            "opcua",
            Node(
                "Serv.NodeName",
                "opc.tcp://10.0.0.1:48050",
                "opcua",
                opc_id="ns=6;s=.Some_Namespace.Node1",
            ),
        ),
        (
            "eneffco",
            Node(
                "Serv.NodeName",
                "https://some_url.de/path",
                "eneffco",
                eneffco_code="A_Code",
            ),
        ),
    ]

    @mark.parametrize(("name", "node"), init_nodes)
    def test_node_init_modbus(self, name, node):
        """Check whether a modbus node can be initialized"""
        Node(
            "Serv.NodeName",
            "modbus.tcp://10.0.0.1:502",
            "modbus",
            mb_channel=3861,
            mb_register="Holding",
            mb_slave=32,
            mb_byteorder="big",
        )


class TestNodeInitFail:
    def test_node_wrong_byteorder(self):
        """Node initialization should fail, if wrong value is provided for mb_byteorder"""
        with raises(ValueError):
            Node(
                "Serv.NodeName",
                "modbus.tcp://10.0.0.1:502",
                "modbus",
                mb_channel=3861,
                mb_register="Holding",
                mb_slave=32,
                mb_byteorder="someendian",
            )
