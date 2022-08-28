from __future__ import annotations

import pandas as pd
import pytest

from eta_utility.connectors import Node

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
    (
        {
            "name": "Serv.NodeName",
            "url": "10.0.0.1:502",
            "protocol": "opcua",
            "opc_id": "somestring",
        },
        "When specifying opc_id, make sure it follows the format ns=2;s=.path",
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "10.0.0.1:502",
            "protocol": "opcua",
            "opc_id": "ns=2;somestring",
        },
        "When specifying opc_id, make sure it follows the format ns=2;s=.path",
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
        {"name": "Serv.NodeName", "url": "", "protocol": "local"},
    ),
    (
        {
            "name": "Serv.NodeName",
            "url": "someone:password@some_url.de/path1",
            "protocol": "eneffco",
            "eneffco_code": "A_Code",
        },
        {"url": "https://some_url.de/path1", "usr": "someone", "pwd": "password"},
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
    (
        {
            "name": "Serv.NodeName",
            "url": "some_url.de/path",
            "protocol": "opcua",
            "opc_id": "NS=4;S=.Some_Namespace.Node3",
        },
        {"protocol": "opcua", "opc_id": "ns=4;s=.Some_Namespace.Node3"},
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


@pytest.fixture()
def create_dictionary():
    nodes = []
    expected = []
    for item in nodes_from_dict:
        nodes.append(item[0])
        expected.append(item[1])

    return nodes, expected


@pytest.fixture()
def create_excel(create_dictionary, temp_dir):
    nodes, expected = create_dictionary
    path = temp_dir / "excel_nodes.xlsx"
    pd.DataFrame(data=nodes).to_excel(path, sheet_name="Sheet1")
    return path, expected


def test_node_from_excel(create_excel):
    """Test reading nodes from Excel files and check some parameters of the resulting node objects"""
    path, expected = create_excel
    nodes = Node.from_excel(path, "Sheet1")

    assert len(nodes) == len(expected)

    for idx, item in enumerate(expected):
        for key, value in item.items():
            assert getattr(nodes[idx], key) == value, f"assert {nodes[idx]} == {item}"


def test_get_eneffco_nodes_from_codes():
    """Check if get_eneffco_nodes_from_codes works"""
    sample_codes = ["CH1.Elek_U.L1-N", "CH1.Elek_U.L1-N"]
    nodes = Node.get_eneffco_nodes_from_codes(sample_codes, eneffco_url=None)
    assert {nodes[0].eneffco_code, nodes[1].eneffco_code} == set(sample_codes)
