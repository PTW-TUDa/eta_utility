import pytest

from eta_utility.connectors import Node

from .config_tests import Config


class TestNodeInitFail:
    def test_node_wrong_byteorder(self):
        """Node initialization should fail, if wrong value is provided for mb_byteorder"""
        with pytest.raises(ValueError, match="'mb_byteorder' must be in"):
            Node(
                "Serv.NodeName",
                "modbus.tcp://10.0.0.1:502",
                "modbus",
                mb_channel=3861,
                mb_register="Holding",
                mb_slave=32,
                mb_byteorder="someendian",
            )


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

    @pytest.mark.parametrize(("name", "node"), init_nodes)
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

    @pytest.mark.parametrize(("excel_file", "excel_sheet"), [(Config.EXCEL_NODES_FILE, Config.EXCEL_NODES_SHEET)])
    def test_node_from_excel(self, excel_file, excel_sheet):
        """Test reading nodes from Excel files and check some parameters of the resulting node objects"""
        nodes = Node.from_excel(excel_file, excel_sheet)

        assert len(nodes) == 4

        # OPC UA Node
        assert nodes[0].name == "Pu3.425.Mech_n"
        assert nodes[0].protocol == "opcua"
        assert nodes[0].url == "opc.tcp://127.95.11.183:48050"
        assert nodes[0].opc_id == "ns=6;s=.HLK.System_425.Pumpe_425.Zustand.Drehzahl"

        # Modbus Node
        assert nodes[2].mb_channel == 19050
        assert nodes[2].protocol == "modbus"
        assert type(nodes[2].mb_channel) is int
        assert nodes[2].mb_slave == 32
        assert type(nodes[2].mb_slave) is int

        # EnEffCo Node
        assert nodes[3].name == "CH1.Elek_U.L1-N"
        assert nodes[3].protocol == "eneffco"
        assert nodes[3].eneffco_code == "CH1.Elek_U.L1-N"

    def test_get_eneffco_nodes_from_codes(self):
        """Check if get_eneffco_nodes_from_codes works"""
        sample_codes = ["CH1.Elek_U.L1-N", "CH1.Elek_U.L1-N"]
        nodes = Node.get_eneffco_nodes_from_codes(sample_codes, eneffco_url=None)
        assert {nodes[0].eneffco_code, nodes[1].eneffco_code} == set(sample_codes)
