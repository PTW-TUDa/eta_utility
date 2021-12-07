import socket
from typing import Any, Mapping, Union

from opcua import Node as OpcNode
from opcua import Server, ua
from opcua.ua import uaerrors

from eta_utility import get_logger
from eta_utility.type_hints import Node, Nodes

log = get_logger("servers.opcua")


class OpcUaServer:
    """Provides an OPC UA server with a number of specified nodes. Each node can contain single values or arrays.

    :param namespace: Namespace of the OPC UA Server
    :param ip: IP Address to listen on (default: None)
    :param port: Port to listen on (default: 4840)
    """

    def __init__(self, namespace: Union[str, int], ip: str = None, port: int = 4840) -> None:
        if ip is None:
            #: URL of the OPC UA Server
            self.url: str = f"opc.tcp://{socket.gethostbyname(socket.gethostname())}:{port}"
        else:
            self.url: str = f"opc.tcp://{ip}:{port}"
        log.info(f"Server Address is {self.url}")

        self._server: Server = Server()
        self._server.set_endpoint(self.url)

        self.idx: int = self._server.register_namespace(str(namespace))  #: idx: Namespace of the OPC UA _server
        log.debug(f'Server Namespace set to "{namespace}"')

        self._server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        self._server.set_server_name("ETA Utility OPC UA Server")
        self._server.start()

    def write(self, values: Mapping[Node, Any]) -> None:
        """
        Writes some values directly to the OPCUA server

        :param values: Dictionary of data to write. {node.name: value}
        """

        nodes = self._validate_nodes(values.keys())

        for node in nodes:
            var = self._server.get_node(node.opc_id)
            opc_type = var.get_data_type_as_variant_type()
            var.set_value(ua.Variant(values[node], opc_type))

    def create_nodes(self, nodes: Nodes) -> None:
        """Create nodes on the server from a list of nodes. This will try to create the entire node path.

        :param nodes: List or set of nodes to create
        """

        def create_object(parent: OpcNode, child: Node) -> OpcNode:
            for obj in parent.get_children():
                ident = obj.nodeid.Identifier if type(obj.nodeid.Identifier) is str else obj.nodeid.Identifier
                if child.opc_path_str == ident:
                    return obj
            else:
                return parent.add_object(child.opc_id, child.opc_name)

        nodes = self._validate_nodes(nodes)

        for node in nodes:
            try:
                last_obj = create_object(self._server.get_objects_node(), node.opc_path[0])

                for key in range(1, len(node.opc_path) + 1):
                    if key < len(node.opc_path):
                        last_obj = create_object(last_obj, node.opc_path[key])
                    else:
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

        :param nodes: List or set of nodes to be deleted
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

    def _validate_nodes(self, nodes: Nodes) -> Nodes:
        if not hasattr(nodes, "__len__"):
            nodes = {nodes}
        else:
            if len(nodes) == 0:
                raise ValueError("Some nodes to read from must be specified.")
        return nodes