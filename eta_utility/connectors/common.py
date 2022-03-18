""" This module implements some commonly used connector functions that are protocol independent.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sized

from eta_utility.connectors.eneffco import EnEffCoConnection
from eta_utility.connectors.modbus import ModbusConnection
from eta_utility.connectors.opc_ua import OpcUaConnection

if TYPE_CHECKING:
    from typing import Any

    from eta_utility.type_hints import AnyNode, Nodes


def connections_from_nodes(
    nodes: Nodes,
    usr: str | None = None,
    pwd: str | None = None,
    eneffco_api_token: str | None = None,
) -> dict[str, Any]:
    """Take a list of nodes and return a list of connections.

    .. versionchanged:: v2.0.0
        Removed eneffco_usr and eneffco_pwd - specify the parameters in the node instead.

    :param nodes: List of nodes defining servers to connect to.
    :param usr: Username to use in case a Node does not specify any.
    :param pwd: Password to use in case a Node does not specify any.
    :param eneffco_api_token: Token for EnEffCo API authorization.
    :return: Dictionary of connection objects {hostname: connection}.
    """

    connections: dict[str, Any] = {}

    if not isinstance(nodes, Sized):
        nodes = {nodes}

    for node in nodes:
        # Create connection if it does not exist
        if node.url_parsed.hostname is not None and node.url_parsed.hostname not in connections:
            if node.protocol == "modbus":
                connections[node.url_parsed.hostname] = ModbusConnection.from_node(node, usr=usr, pwd=pwd)
            elif node.protocol == "opcua":
                connections[node.url_parsed.hostname] = OpcUaConnection.from_node(node, usr=usr, pwd=pwd)
            elif node.protocol == "eneffco":
                if eneffco_api_token is None:
                    raise ValueError("Specify API token for EnEffco access.")
                connections[node.url_parsed.hostname] = EnEffCoConnection.from_node(
                    node, usr=usr, pwd=pwd, api_token=eneffco_api_token
                )
            else:
                raise ValueError(
                    f"Node {node.name} does not specify a recognized protocol for initializing a connection."
                )
        elif node.url_parsed.hostname is not None:
            # Otherwise, just mark the node as selected
            connections[node.url_parsed.hostname].selected_nodes.add(node)

    return connections


def name_map_from_node_sequence(nodes: Nodes) -> dict[str, AnyNode]:
    """Convert a Sequence/List of Nodes into a dictionary of nodes, identified by their name.

    .. warning ::

        Make sure that each node in nodes has a unique Name, otherwise this function will fail.

    :param nodes: Sequence of Node objects.
    :return: Dictionary of Node objects (format: {node.name: Node}).
    """
    if not isinstance(nodes, Sized):
        nodes = {nodes}

    if len({node.name for node in nodes}) != len([node.name for node in nodes]):
        raise ValueError("Not all node names are unique. Cannot safely convert to named dictionary.")

    return {node.name: node for node in nodes}
