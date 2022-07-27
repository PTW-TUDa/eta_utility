""" The OPC UA module provides utilities for the flexible creation of OPC UA connections.

"""
from __future__ import annotations

import asyncio
import concurrent.futures
import socket
from concurrent.futures._base import CancelledError
from concurrent.futures._base import TimeoutError as ConTimeoutError
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from opcua import Client, Subscription, ua
from opcua.ua import uaerrors

from eta_utility import get_logger
from eta_utility.connectors.node import Node, NodeOpcUa

if TYPE_CHECKING:
    from typing import Any, Generator, Mapping, Sequence
    from opcua import Node as OpcNode
    from eta_utility.type_hints import AnyNode, Nodes, TimeStep

from .base_classes import BaseConnection, SubscriptionHandler

log = get_logger("connectors.opcua")


class OpcUaConnection(BaseConnection):
    """The OPC UA Connection class allows reading and writing from and to OPC UA servers. Additionally,
    it implements a subscription method, which reads continuously in a specified interval.

    :param url: URL of the OPC UA Server.
    :param usr: Username in OPC UA for login.
    :param pwd: Password in OPC UA for login.
    :param nodes: List of nodes to use for all operations.
    """

    _PROTOCOL = "opcua"

    def __init__(self, url: str, usr: str | None = None, pwd: str | None = None, *, nodes: Nodes | None = None) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

        if self._url.scheme != "opc.tcp":
            raise ValueError("Given URL is not a valid OPC url (scheme: opc.tcp)")

        self.connection: Client = Client(self.url)
        self._connected = False

        self._sub: Subscription
        self._sub_task: asyncio.Task
        self._subscription_open: bool = False
        self._subscription_nodes: set[NodeOpcUa] = set()

    @classmethod
    def from_node(cls, node: AnyNode, **kwargs: Any) -> OpcUaConnection:
        """Initialize the connection object from an EnEffCo protocol Node object.

        :param node: Node to initialize from.
        :param usr: Username for OPC UA login.
        :param pwd: Password for OPC UA login.
        :param kwargs: Other arguments are ignored.
        :return: OpcUaConnection object.
        """
        usr = node.usr if node.usr is not None else kwargs.get("usr", None)
        pwd = node.pwd if node.pwd is not None else kwargs.get("pwd", None)

        if node.protocol == "opcua" and isinstance(node, NodeOpcUa):
            return cls(node.url, usr=usr, pwd=pwd, nodes=[node])

        else:
            raise ValueError(
                "Tried to initialize OpcUaConnection from a node that does not specify opcua as its"
                "protocol: {}.".format(node.name)
            )

    @classmethod
    def from_ids(
        cls,
        ids: Sequence[str],
        url: str,
        usr: str | None = None,
        pwd: str | None = None,
    ) -> OpcUaConnection:
        """Initialize the connection object from an OPC UA protocol through the node IDs.

        :param ids: Identification of the Node.
        :param url: URL for  connection.
        :param usr: Username in OPC UA for login.
        :param pwd: Password in OPC UA for login.
        :return: OpcUaConnection object.
        """
        nodes = [Node(name=opc_id, usr=usr, pwd=pwd, url=url, protocol="opcua", opc_id=opc_id) for opc_id in ids]
        return cls(nodes[0].url, usr, pwd, nodes=nodes)

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """
        Read some manually selected values from OPC UA capable controller.

        :param nodes: List of nodes to read from.
        :return: pandas.DataFrame containing current values of the OPC UA-variables.
        :raises ConnectionError: When an error occurs during reading.
        """
        _nodes = self._validate_nodes(nodes)

        def read_node(node: NodeOpcUa) -> dict[str, list]:
            try:
                opcua_variable = self.connection.get_node(node.opc_id)
                value = opcua_variable.get_value()
                return {node.name: [value]}
            except uaerrors.BadNodeIdUnknown:
                raise ConnectionError(
                    f"The node id ({node.opc_id}) refers to a node that does not exist in the server address space "
                    f"{self.url}. (BadNodeIdUnknown)"
                )
            except RuntimeError as e:
                raise ConnectionError(str(e)) from e

        values: dict[str, list] = {}
        with self._connection():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(read_node, _nodes)
        for result in results:
            values.update(result)

        return pd.DataFrame(values, index=[self._assert_tz_awareness(datetime.now())])

    def write(self, values: Mapping[AnyNode, Any]) -> None:
        """
        Writes some manually selected values on OPC UA capable controller.

        :param values: Dictionary of nodes and data to write {node: value}.
        :raises ConnectionError: When an error occurs during reading.
        """
        nodes = self._validate_nodes(set(values.keys()))

        with self._connection():
            for node in nodes:
                try:
                    opcua_variable = self.connection.get_node(node.opc_id)
                    opcua_variable_type = opcua_variable.get_data_type_as_variant_type()
                    opcua_variable.set_value(ua.DataValue(ua.Variant(values[node], opcua_variable_type)))
                except uaerrors.BadNodeIdUnknown:
                    raise ConnectionError(
                        f"The node id ({node.opc_id}) refers to a node that does not exist in the server address space "
                        f"{self.url}. (BadNodeIdUnknown)"
                    )
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

    def create_nodes(self, nodes: Nodes) -> None:
        """Create nodes on the server from a list of nodes. This will try to create the entire node path.

        :param nodes: List or set of nodes to create.
        :raises ConnectionError: When an error occurs during node creation.
        """

        def create_object(parent: OpcNode, child: NodeOpcUa) -> OpcNode:
            for obj in parent.get_children():
                ident = obj.nodeid.Identifier if type(obj.nodeid.Identifier) is str else obj.nodeid.Identifier
                if child.opc_path_str == ident:
                    return obj
            else:
                return parent.add_object(child.opc_id, child.opc_name)

        _nodes = self._validate_nodes(nodes)

        with self._connection():
            for node in _nodes:
                try:
                    last_obj = create_object(self.connection.get_objects_node(), node.opc_path[0])

                    for key in range(1, len(node.opc_path) + 1):
                        init_val: Any

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

        :param nodes: List or set of nodes to be deleted.
        :raises ConnectionError: If deletion of nodes fails.
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
                    delete_node_parents(self.connection.get_node(parent.NodeId), depth=depth - 1)

        _nodes = self._validate_nodes(nodes)

        with self._connection():
            for node in _nodes:
                try:
                    delete_node_parents(self.connection.get_node(node.opc_id))
                except uaerrors.BadNodeIdUnknown:
                    raise ConnectionError(
                        f"The node id ({node.opc_id}) refers to a node that does not exist in the server address space "
                        f"{self.url}. (BadNodeIdUnknown)"
                    )
                except RuntimeError as e:
                    raise ConnectionError(str(e)) from e

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This function works asnychonously.
        Subscriptions must always be closed using the close_sub function (use try, finally!).

        :param nodes: Identifiers for the nodes to subscribe to.
        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs.
        :param interval: Interval for receiving new data. It is interpreted as seconds when given as an integer.
        """
        _nodes = self._validate_nodes(nodes)
        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        self._subscription_nodes.update(_nodes)

        if self._subscription_open:
            # Adding nodes to subscription is enough to include them in the query. Do not start an additional loop
            # if one already exists
            return

        self._subscription_open = True

        loop = asyncio.get_event_loop()
        self._sub_task = loop.create_task(
            self._subscription_loop(_OPCSubHandler(handler), float(interval.total_seconds()))
        )

    async def _subscription_loop(self, handler: "_OPCSubHandler", interval: float) -> None:
        """The subscription loop makes sure that the subscription is reset in case the server generates an error.

        :param handler: Handler object with a push function to receive data.
        :param interval: Interval for requesting data in seconds.
        """
        retry_wait = 0
        subscribed = False
        while self._subscription_open:
            try:
                await asyncio.sleep(retry_wait)
                try:
                    if not self._connected:
                        self._connect()
                        retry_wait = 5
                except ConnectionError as e:
                    retry_wait += 5 if retry_wait <= 60 else 0
                    log.error(str(e))
                    continue

                if self._connected and not subscribed:
                    try:
                        self._sub = self.connection.create_subscription(interval * 1000, handler)
                        subscribed = True
                    except RuntimeError as e:
                        log.warning(str(e))
                        retry_wait += 5 if retry_wait <= 60 else 0
                        subscribed = False
                        self._disconnect()
                        continue

                    for node in self._subscription_nodes:
                        try:
                            handler.add_node(node.opc_id, node)  # type: ignore
                            _ = self._sub.subscribe_data_change(self.connection.get_node(node.opc_id))
                        except RuntimeError as e:
                            log.warning(f"Server {self.url}, Node Id '{node.name}' error: {str(e)}")

            except BaseException as e:
                retry_wait += 5 if retry_wait <= 60 else 0
                self._connected = False
                self.exc = e

    def close_sub(self) -> None:
        """Close an open subscription."""
        self._subscription_open = False
        if self.exc:
            raise self.exc

        try:
            self._sub.delete()
            self._sub_task.cancel()
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
            self._connected = True
        except (socket.herror, socket.gaierror) as e:
            raise ConnectionError(f"Host not found: {self.url}") from e
        except (socket.timeout, TimeoutError, ConTimeoutError) as e:
            raise ConnectionError(f"Host timeout: {self.url}") from e
        except CancelledError as e:
            raise ConnectionError(f"Connection cancelled by host: {self.url}") from e
        except (RuntimeError, ConnectionError) as e:
            raise ConnectionError(f"OPC Connection Error: {self.url}, Error: {str(e)}") from e

    def _disconnect(self) -> None:
        """Disconnect from server."""
        try:
            self.connection.disconnect()
            self._connected = False
        except (OSError, RuntimeError) as e:
            log.error(f"Closing connection to server {self.url} failed")
            log.info(f"Connection to {self.url} returned error: {e}")
        except AttributeError:
            log.error(f"Connection to server {self.url} already closed.")

    @contextmanager
    def _connection(self) -> Generator:
        """Connect to the server and return a context manager that automatically disconnects when finished."""
        try:
            self._connect()
            yield None
        except ConnectionError as e:
            raise e
        finally:
            self._disconnect()

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeOpcUa]:  # type: ignore
        vnodes = super()._validate_nodes(nodes)
        _nodes = set()
        for node in vnodes:
            if isinstance(node, NodeOpcUa):
                _nodes.add(node)

        return _nodes


class _OPCSubHandler:
    """Wrapper for the OPC UA subscription. Enables the subscription to use the standardized eta_utility subscription
    format.

    :param handler: *eta_utility* style subscription handler.
    """

    def __init__(self, handler: SubscriptionHandler) -> None:
        self.handler = handler
        self._sub_nodes: dict[str | int, NodeOpcUa] = {}

    def add_node(self, opc_id: str | int, node: NodeOpcUa) -> None:
        """Add a node to the subscription. This is necessary to translate between formats."""
        self._sub_nodes[opc_id] = node

    def datachange_notification(self, node: NodeOpcUa, val: Any, data: Any) -> None:
        """
        datachange_notification is called whenever subscribed input data is recieved via OPC UA. This pushes data
        to the actual eta_utility subscription handler.

        :param node: Node Object, which was subscribed to and which has sent an updated value.
        :param val: New value of OPC UA node.
        :param data: Raw data of OPC UA (not used).
        """

        self.handler.push(self._sub_nodes[str(node)], val, self.handler._assert_tz_awareness(datetime.now()))
