from __future__ import annotations

import pytest

from eta_utility.connectors import Node

from .config_tests import Config

fail_nodes = (
    (
        {
            "name": "Serv.NodeName",
            "url": "modbus.tcp://10.0.0.1:502",
            "protocol": "modbus",
            "mb_channel": 3861,
            "mb_register": "Holding",
            "mb_slave": 32,
            "mb_byteorder": "someendian",
        },
        "'mb_byteorder' must be in",
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "10.0.0.1:502",
            "protocol": "opcua",
        },
        "Specify opc_id or opc_path_str",
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "10.0.0.1:502",
            "protocol": "something",
        },
        "Specified an unsupported protocol",
    ),
)


@pytest.mark.parametrize(("node_data", "error"), fail_nodes)
def test_node_init_failures(node_data: dict[str, str], error: str):
    with pytest.raises(ValueError, match=error):
        Node(**node_data)


@pytest.mark.parametrize(("node_data", "error"), fail_nodes)
def test_node_dict_init_failures(node_data: dict[str, str], error: str):
    with pytest.raises(ValueError, match=error):
        Node.from_dict(node_data)


nodes = (
    (
        {
            "name": "Serv.NodeName",
            "url": "modbus.tcp://10.0.0.1:502",
            "protocol": "modbus",
            "mb_channel": 3861,
            "mb_register": "Holding",
            "mb_slave": 32,
            "mb_byteorder": "BigEndian",
        },
        {
            "name": "Serv.NodeName",
            "url": "modbus.tcp://10.0.0.1:502",
            "protocol": "modbus",
            "mb_channel": 3861,
            "mb_register": "holding",
            "mb_slave": 32,
            "mb_byteorder": "big",
        },
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "opc.tcp://10.0.0.1:48050",
            "protocol": "opcua",
            "opc_id": "ns=6;s=.Some_Namespace.Node1",
        },
        {
            "name": "Serv.NodeName",
            "url": "opc.tcp://10.0.0.1:48050",
            "protocol": "opcua",
            "opc_id": "ns=6;s=.Some_Namespace.Node1",
        },
    ),
    (
        {"name": "Serv.NodeName", "url": "https://some_url.de/path", "protocol": "eneffco", "eneffco_code": "A_Code"},
        {"name": "Serv.NodeName", "url": "https://some_url.de/path", "protocol": "eneffco", "eneffco_code": "A_Code"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "https://some_url.de/path",
            "protocol": "entsoe",
            "endpoint": "A_Code",
            "bidding_zone": "DE-LU-AT",
        },
        {
            "name": "Serv.NodeName",
            "url": "https://some_url.de/path",
            "protocol": "entsoe",
            "endpoint": "A_Code",
            "bidding_zone": "DE-LU-AT",
        },
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "10.0.0.1:502",
            "protocol": "modbus",
            "mb_channel": 3861,
            "mb_register": "Holding",
            "mb_slave": 32,
            "mb_byteorder": "big",
        },
        {"url": "modbus.tcp://10.0.0.1:502"},
    ),
    (
        {"name": "Serv.NodeName", "url": "10.0.0.1", "protocol": "opcua", "opc_id": "ns=6;s=.Some_Namespace.Node1"},
        {"url": "opc.tcp://10.0.0.1:4840"},
    ),
    (
        {"name": "Serv.NodeName", "url": "some_url.de/path", "protocol": "eneffco", "eneffco_code": "A_Code"},
        {"url": "https://some_url.de/path"},
    ),
    ({"name": "Serv.NodeName", "url": None, "protocol": "eneffco", "eneffco_code": "A_Code"}, {"url": ""}),
    (
        {"name": "Serv.NodeName", "url": None, "protocol": "local"},
        {"name": "Serv.NodeName", "url": "", "protocol": "local"},
    ),
    (
        {"name": "Serv.NodeName", "url": "", "protocol": "local"},
        {"name": "Serv.NodeName", "url": "https://", "protocol": "local"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "someone:password@some_url.de/path",
            "protocol": "eneffco",
            "eneffco_code": "A_Code",
        },
        {"url": "https://some_url.de/path", "usr": "someone", "pwd": "password"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "http://someone:password@some_url.de/path",
            "usr": "someoneelse",
            "pwd": "anotherpwd",
            "protocol": "eneffco",
            "eneffco_code": "A_Code",
        },
        {"url": "http://some_url.de/path", "usr": "someoneelse", "pwd": "anotherpwd"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "some_url.de/path",
            "usr": "someperson",
            "pwd": "somepwd",
            "protocol": "eneffco",
            "eneffco_code": "A_Code",
        },
        {"url": "https://some_url.de/path", "usr": "someperson", "pwd": "somepwd"},
    ),
)


@pytest.mark.parametrize(("node_data", "expected"), nodes)
def test_node_init(node_data, expected):
    node = Node(**node_data)

    for key, val in expected.items():
        assert getattr(node, key) == val


nodes_from_dict = (
    *nodes,
    (
        {
            "name": "Serv.NodeName",
            "ip": "10.0.0.1",
            "port": 502,
            "protocol": "modbus",
            "mb_channel": 3861,
            "mb_register": "Holding",
            "mb_slave": 32,
            "mb_byteorder": "big",
        },
        {"url": "modbus.tcp://10.0.0.1:502"},
    ),
    (
        {"name": "Serv.NodeName", "ip": "10.0.0.1", "protocol": "opcua", "opc_id": "ns=6;s=.Some_Namespace.Node1"},
        {"url": "opc.tcp://10.0.0.1:4840"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "ip": "123.123.217.12",
            "port": "321",
            "protocol": "eneffco",
            "eneffco_code": "A_Code",
        },
        {
            "url": "https://123.123.217.12:321",
        },
    ),
)


@pytest.mark.parametrize(("node_data", "expected"), nodes_from_dict)
def test_node_dict_init(node_data, expected):
    node = Node.from_dict(node_data)[0]

    for key, val in expected.items():
        assert getattr(node, key) == val


@pytest.mark.parametrize(("excel_file", "excel_sheet"), [(Config.EXCEL_NODES_FILE, Config.EXCEL_NODES_SHEET)])
def test_node_from_excel(excel_file, excel_sheet):
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


def test_get_eneffco_nodes_from_codes():
    """Check if get_eneffco_nodes_from_codes works"""
    sample_codes = ["CH1.Elek_U.L1-N", "CH1.Elek_U.L1-N"]
    nodes = Node.get_eneffco_nodes_from_codes(sample_codes, eneffco_url=None)
    assert {nodes[0].eneffco_code, nodes[1].eneffco_code} == set(sample_codes)
