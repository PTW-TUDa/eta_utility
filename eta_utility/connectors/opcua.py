""" The OPC UA module provides utilities for the flexible creation of OPC UA connections.

"""
import socket
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Mapping, Union

import pandas as pd
import tzlocal
from opcua import Client, ua

from eta_utility import get_logger

from .base_classes import BaseConnection, Node, Nodes, SubscriptionHandler

log = get_logger("connectors.opcua")


class OpcUaConnection(BaseConnection):
    """The OPC UA Connection class allows reading and writing from and to OPC UA servers. Additionally
    it implements a subscription method, which reads continuously in a specified interval.

    :param url: Url of the OPC UA Server
    :param nodes: List of nodes to use for all operations.
    """

    def __init__(self, url: str, usr: str = None, pwd: str = None, *, nodes: Nodes = None) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

        if self._url.scheme != "opc.tcp":
            raise ValueError("Given URL is not a valid OPC url (scheme: opc.tcp)")

        self.connection = Client(self.url)
        self._sub = None

    @classmethod
    def from_node(cls, node: Node, **kwargs: Any) -> "OpcUaConnection":
        """Initialize OPC UA connection object from a node which specifies OPC UA as its protocol.

        :param node: Node to initialize from
        :param kwargs: Other arguments are ignored.
        :return: OpcUa Connection object
        """

        if node.protocol == "opcua":
            return cls(node.url, nodes=[node])

        else:
            raise ValueError(
                "Tried to initialize OpcUaConnection from a node that does not specify opcua as its"
                "protocol: {}.".format(node.name)
            )

    def read(self, nodes: Nodes = None) -> pd.DataFrame:
        """
        Read some manually selected values from OPCUA capable controller

        :param nodes: List of nodes to read from

        :return: DataFrame containing current values of the OPCUA-variables

        :raises ConnectionError: When an error occurs during reading.
        """
        nodes = self._validate_nodes(nodes)

        with self._connection():
            values = {}
            for node in nodes:
                try:
                    opcua_variable = self.connection.get_node(node.opc_id)
                    value = opcua_variable.get_value()
                    values[node.name] = value
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

        return pd.DataFrame(values, index=[tzlocal.get_localzone().localize(datetime.now())])

    def write(self, values: Mapping[Node, Any]) -> None:
        """
        Writes some manually selected values on OPCUA capable controller

        :param values: Dictionary of nodes and data to write. {node: value}

        :raises ConnectionError: When an error occurs during reading.
        """
        nodes = self._validate_nodes(values.keys())

        with self._connection():
            for node in nodes:
                try:
                    opcua_variable = self.connection.get_node(node.opc_id)
                    opcua_variable_type = opcua_variable.get_data_type_as_variant_type()
                    opcua_variable.set_value(ua.Variant(values[node], opcua_variable_type))
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

    def create_nodes(self, nodes: Nodes) -> None:
        """Create nodes on the server from a list of nodes. This will try to create the entire node path.

        :param nodes: List or set of nodes to create

        :raises ConnectionError: When an error occurs during node creation
        """

        def create_object(parent: Node, child: Node) -> Union[Node, Any]:
            for obj in parent.get_children():
                ident = (
                    obj.nodeid.Identifier.strip(" .") if type(obj.nodeid.Identifier) is str else obj.nodeid.Identifier
                )
                if child.opc_path_str == ident:
                    return obj
            else:
                return parent.add_object(child.opc_id, child.opc_name)

        nodes = self._validate_nodes(nodes)

        with self._connection():
            for node in nodes:
                try:
                    last_obj = create_object(self.connection.get_objects_node(), node.opc_path[0])

                    for key in range(1, len(node.opc_path) + 1):
                        if key < len(node.opc_path):
                            last_obj = create_object(last_obj, node.opc_path[key])
                        else:
                            init_val = 0.0
                            if not hasattr(node, "dtype"):
                                pass
                            elif node.dtype is int:
                                init_val = 0
                            elif node.dtype is bool:
                                init_val = False

                            last_obj.add_variable(node.opc_id, node.opc_name, init_val)
                            log.debug(f"OPC UA Node created: {node.opc_id}")
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

    def delete_nodes(self, nodes: Nodes) -> None:
        """Delete the given nodes and their parents (if the parents do not have other children).

        :param nodes: List or set of nodes to be deleted

        :raises ConnectionError: If deletion of nodes fails.
        """

        def delete_node_parents(node: Node, depth: int = 20):
            parents = node.get_references(direction=ua.BrowseDirection.Inverse)
            if not node.get_children():
                node.delete(delete_references=True)
                log.info(f"Deleted Node {node.nodeid} from server {self.url}.")
            else:
                log.info(f"Node {node.nodeid} on server {self.url} has remaining children and was not deleted.")
            for parent in parents:
                if depth > 0:
                    delete_node_parents(self.connection.get_node(parent.NodeId), depth=depth - 1)

        nodes = self._validate_nodes(nodes)

        with self._connection():
            for node in nodes:
                try:
                    delete_node_parents(self.connection.get_node(node.opc_id))
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes = None, interval: Union[int, timedelta] = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This function works asnychonously.
        Subscriptions must always be closed using the close_sub function (use try, finally!)

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param interval: interval for receiving new data. It it interpreted as seconds when given as an integer.
        """
        nodes = self._validate_nodes(nodes)
        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        self._connect()
        handler_obj = _OPCSubHandler(handler)

        try:
            self._sub = self.connection.create_subscription(int(interval.total_seconds() * 1000), handler_obj)
            for node in nodes:
                try:
                    handler_obj.add_node(node.opc_id, node)
                    _ = self._sub.subscribe_data_change(self.connection.get_node(node.opc_id))
                except RuntimeError as e:
                    log.warning("Server {}, Node Id '{}' error: {}".format(self.url, node.name, str(e)))

        except RuntimeError as e:
            log.warning(str(e))

    def close_sub(self) -> None:
        """Close an open subscription."""
        try:
            self._sub.delete()
        except (OSError, RuntimeError) as e:
            log.error(f"Deleting subscription for server {self.url} failed")
            log.debug(f"Server {self.url} returned error: {e}")
        except AttributeError:
            log.error(f"Failed to delete subscription for server {self.url}. It did not exist.")

        self._disconnect()

    def _connect(self) -> None:
        """Connect to server."""
        try:
            if self.usr is not None:
                self.connection.set_user(self.usr)
            if self.pwd is not None:
                self.connection.set_password(self.pwd)
            self.connection.connect()

            log.debug(f"Connected to OPC UA server: {self.url}")
        except (socket.herror, socket.gaierror) as e:
            raise ConnectionError(f"Host not found: {self.url}") from e
        except socket.timeout as e:
            raise ConnectionError(f"Host timeout: {self.url}") from e
        except RuntimeError as e:
            raise ConnectionError(str(e)) from e
        except ConnectionError as e:
            raise e

    def _disconnect(self) -> None:
        """Disconnect from server"""
        try:
            self.connection.disconnect()
        except (OSError, RuntimeError) as e:
            log.error(f"Closing connection to server {self.url} failed")
            log.info(f"Server {self.url} returned error: {e}")
        except AttributeError:
            log.error(f"Connection to server {self.url} already closed.")

    @contextmanager
    def _connection(self) -> None:
        """Connect to the server and return a context manager that automatically disconnects when finished."""
        try:
            self._connect()
            yield None
        except (socket.herror, socket.gaierror) as e:
            raise ConnectionError(f"Host not found: {self.url}") from e
        except socket.timeout as e:
            raise ConnectionError(f"Host timeout: {self.url}") from e
        except ConnectionError as e:
            raise e
        finally:
            self._disconnect()


class _OPCSubHandler:
    """Wrapper for the OPC UA subscription. Enables the subscription to use the standardized eta_utility subscription
    format.

    :param handler: ETA-Utility style subscription handler
    """

    def __init__(self, handler: SubscriptionHandler) -> None:
        self.handler = handler
        self._sub_nodes = {}

    def add_node(self, opc_id: str, node: Node) -> None:
        """Add a node to the subscription. This is necessary to translate between formats."""
        self._sub_nodes[opc_id] = node

    def datachange_notification(self, node: Node, val: Any, data: Any) -> None:
        """
        datachange_notification is called whenever subscribed input data is recieved via OPC UA. This pushes data
        to the actual eta_utility subscription handler.

        :param node: Node Object, which was subscribed to and which has sent an updated value
        :param val: new value of OPC UA node
        :param data: raw data of OPC UA (not used)
        """

        self.handler.push(self._sub_nodes[str(node)], val, tzlocal.get_localzone().localize(datetime.now()))
