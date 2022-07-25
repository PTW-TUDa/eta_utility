from __future__ import annotations

import socket
from datetime import datetime
from typing import TYPE_CHECKING, Sized

import pandas as pd
from opcua import Server, ua
from opcua.ua import uaerrors

from eta_utility import ensure_timezone, get_logger, url_parse
from eta_utility.connectors.node import NodeOpcUa

if TYPE_CHECKING:
    import types
    from typing import Any, Mapping

    from opcua import Node as OpcNode

    from eta_utility.type_hints import AnyNode, Nodes

log = get_logger("servers.opcua")


class OpcUaServer:
    """Provides an OPC UA server with a number of specified nodes. Each node can contain single values or arrays.

    :param namespace: Namespace of the OPC UA Server.
    :param ip: IP Address to listen on (default: None).
    :param port: Port to listen on (default: 4840).
    """

    def __init__(self, namespace: str | int, ip: str | None = None, port: int = 4840) -> None:
        #: URL of the OPC UA Server.
        self.url: str
        if ip is None:
            self.url = f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:{port}"
        else:
            self.url = f"opc.tcp://{ip}:{port}"
        log.info(f"Server Address is {self.url}")

        self._url, _, _ = url_parse(self.url)

        self._server: Server = Server()
        self._server.set_endpoint(self.url)

        self.idx: int = self._server.register_namespace(str(namespace))  #: idx: Namespace of the OPC UA _server
        log.debug(f'Server Namespace set to "{namespace}"')

        self._server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        self._server.set_server_name("ETA Utility OPC UA Server")
        self._server.start()

    def write(self, values: Mapping[AnyNode, Any]) -> None:
        """Write some values directly to the OPC UA server.

        :param values: Dictionary of data to write {node.name: value}.
        """

        nodes = self._validate_nodes(set(values.keys()))

        for node in nodes:
            var = self._server.get_node(node.opc_id)
            opc_type = var.get_data_type_as_variant_type()
            var.set_value(ua.Variant(values[node], opc_type))

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """
        Read some manually selected values directly from the OPC UA server.

        :param nodes: List of nodes to read from.
        :return: pandas.DataFrame containing current values of the OPC UA-variables.
        :raises RuntimeError: When an error occurs during reading.
        """
        _nodes = self._validate_nodes(nodes)

        _dikt = {}
        for node in _nodes:
            try:
                opcua_variable = self._server.get_node(node.opc_id)
                value = opcua_variable.get_value()
                _dikt[node.name] = [value]
            except uaerrors.BadNodeIdUnknown:
                raise RuntimeError(
                    f"The node id ({node.opc_id}) refers to a node that does not exist in the server address space "
                    f"{self.url}. (BadNodeIdUnknown)"
                )

        return pd.DataFrame(_dikt, index=[ensure_timezone(datetime.now())])

    def create_nodes(self, nodes: Nodes) -> None:
        """Create nodes on the server from a list of nodes. This will try to create the entire node path.

        :param nodes: List or set of nodes to create.
        """

        def create_object(parent: OpcNode, child: NodeOpcUa) -> OpcNode:
            for obj in parent.get_children():
                ident = obj.nodeid.Identifier if type(obj.nodeid.Identifier) is str else obj.nodeid.Identifier
                if child.opc_path_str == ident:
                    return obj
            else:
                return parent.add_object(child.opc_id, child.opc_name)

        _nodes = self._validate_nodes(nodes)

        for node in _nodes:
            try:
                last_obj = create_object(self._server.get_objects_node(), node.opc_path[0])

                for key in range(1, len(node.opc_path) + 1):
                    if key < len(node.opc_path):
                        last_obj = create_object(last_obj, node.opc_path[key])
                    else:
                        init_val: Any
                        if not hasattr(node, "dtype"):
                            init_val = 0.0
                        elif node.dtype is int:
                            init_val = 0
                        elif node.dtype is bool:
                            init_val = False
                        elif node.dtype is str:
                            init_val = ""
                        else:
                            init_val = 0.0

                        last_obj.add_variable(node.opc_id, node.opc_name, init_val)
                        log.debug(f"OPC UA Node created: {node.opc_id}")
            except uaerrors.BadNodeIdExists:
                log.warning(f"Node with NodeId : {node.opc_id} could not be created. It already exists.")
            except RuntimeError as e:
                raise ConnectionError(str(e)) from e

    def delete_nodes(self, nodes: Nodes) -> None:
        """Delete the given nodes and their parents (if the parents do not have other children).

        :param nodes: List or set of nodes to be deleted.
        """

        def delete_node_parents(node: OpcNode, depth: int = 20) -> None:
            parents = node.get_references(direction=ua.BrowseDirection.Inverse)
            if not node.get_children():
                node.delete(delete_references=True)
                log.info(f"Deleted Node {node.nodeid} from server {self.url}.")
            else:
                log.info(f"Node {node.nodeid} on server {self.url} has remaining children and was not deleted.")
            for parent in parents:
                if depth > 0:
                    delete_node_parents(self._server.get_node(parent.NodeId), depth=depth - 1)

        nodes = self._validate_nodes(nodes)

        for node in nodes:
            delete_node_parents(self._server.get_node(node.opc_id))

    def stop(self) -> None:
        """This should always be called, when the server is not needed anymore. It stops the server."""
        self._server.stop()

    def allow_remote_admin(self, allow: bool) -> None:
        """Allow remote administration of the server.

        :param allow: Set to true to enable remote administration of the server.
        """
        self._server.allow_remote_admin(allow)

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeOpcUa]:
        """Make sure that nodes are a Set of nodes and that all nodes correspond to the protocol and url
        of the connection.

        :param nodes: Sequence of Node objects to validate.
        :return: Set of valid Node objects for this connection.
        """
        _nodes = None

        if nodes:
            if not isinstance(nodes, Sized):
                nodes = {nodes}

            # If not using preselected nodes from self.selected_nodes, check if nodes correspond to the connection
            _nodes = {
                node for node in nodes if isinstance(node, NodeOpcUa) and node.url_parsed.hostname == self._url.hostname
            }

        # Make sure that some nodes remain after the checks and raise an error if there are none.
        if not _nodes or len(_nodes) == 0:
            raise ValueError(
                f"Some nodes to read from/write to must be specified. If nodes were specified, they do not "
                f"match the connection {self.url}"
            )

        return _nodes

    def __enter__(self) -> OpcUaServer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None
    ) -> None:
        self.stop()
