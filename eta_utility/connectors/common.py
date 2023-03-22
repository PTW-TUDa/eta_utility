""" This module implements some commonly used connector functions that are protocol independent.

"""
from __future__ import annotations

from typing import TYPE_CHECKING, Sized

from eta_utility.connectors.base_classes import BaseConnection

if TYPE_CHECKING:
    from typing import Any

    from eta_utility.type_hints import AnyNode, Nodes
    from eta_utility.util import KeyCertPair


def connections_from_nodes(
    nodes: Nodes,
    usr: str | None = None,
    pwd: str | None = None,
    eneffco_api_token: str | None = None,
    key_cert: KeyCertPair | None = None,
) -> dict[str, Any]:
    """The functionality of this function is outdated,
    instead use directly the from_node function of BaseConnection

    Take a list of nodes and return a list of connections.

    :param nodes: List of nodes defining servers to connect to.
    :param usr: Username to use in case a Node does not specify any.
    :param pwd: Password to use in case a Node does not specify any.
    :param eneffco_api_token: Token for EnEffCo API authorization.
    :param key_cert: Key and certificate pair object from eta_utility for authorization with servers.
    :return: Dictionary of connection objects {hostname: connection}.
    """

    connections = BaseConnection.from_node(nodes, usr=usr, pwd=pwd, api_token=eneffco_api_token, key_cert=key_cert)

    if not isinstance(connections, dict):
        node = nodes[0] if isinstance(nodes, list) else nodes
        return {node.url_parsed.hostname: connections}
    else:
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
