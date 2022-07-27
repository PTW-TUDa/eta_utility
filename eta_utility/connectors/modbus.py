""" Utilities for connecting to modbus servers
"""
from __future__ import annotations

import asyncio
import socket
import struct
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd
from pyModbusTCP.client import ModbusClient  # noqa: I900

from eta_utility import get_logger
from eta_utility.connectors.node import NodeModbus

if TYPE_CHECKING:
    from typing import Any, Generator, Mapping, Sequence
    from eta_utility.type_hints import AnyNode, Nodes, TimeStep

from .base_classes import BaseConnection, SubscriptionHandler

log = get_logger("connectors.modbus")


class ModbusConnection(BaseConnection):
    """The Modbus Connection class allows reading and writing from and to Modbus servers and clients. Additionally,
    it implements a subscription server, which reads continuously in a specified interval.

    :param url: URL of the Modbus Server.
    :param usr: Username in EnEffco for login.
    :param pwd: Password in EnEffco for login.
    :param nodes: List of nodes to use for all operations.
    """

    _PROTOCOL = "modbus"

    def __init__(self, url: str, usr: str | None = None, pwd: str | None = None, *, nodes: Nodes | None = None) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

        if self._url.scheme != "modbus.tcp":
            raise ValueError("Given URL is not a valid Modbus url (scheme: modbus.tcp)")

        self.connection: ModbusClient = ModbusClient(host=self._url.hostname, port=self._url.port, timeout=2)

        self._subscription_open: bool = False
        self._subscription_nodes: set[NodeModbus] = set()
        self._sub: asyncio.Task

    @classmethod
    def from_node(cls, node: AnyNode, **kwargs: Any) -> ModbusConnection:
        """Initialize the connection object from a modbus protocol node object.

        :param node: Node to initialize from.
        :param kwargs: Other arguments are ignored.
        :return: ModbusConnection object.
        """
        usr = node.usr if node.usr is not None else kwargs.get("usr", None)
        pwd = node.pwd if node.pwd is not None else kwargs.get("pwd", None)

        if node.protocol == "modbus" and isinstance(node, NodeModbus):
            return cls(node.url, usr, pwd, nodes=[node])

        else:
            raise ValueError(
                "Tried to initialize ModbusConnection from a node that does not specify modbus as its"
                "protocol: {}.".format(node.name)
            )

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Read some manually selected nodes from Modbus server.

        :param nodes: List of nodes to read from.
        :return: Dictionary containing current values of the Modbus variables.
        """
        _nodes = self._validate_nodes(nodes)

        values = {}

        with self._connection():
            results = {node: self._read_mb_value(node) for node in _nodes}

        for node, result in results.items():
            value = self._decode(result, node.mb_byteorder)
            values[node.name] = value

        return pd.DataFrame(values, index=[self._assert_tz_awareness(datetime.now())])

    def write(self, values: Mapping[AnyNode, Any]) -> None:
        """Write some manually selected values on Modbus capable controller.

        .. warning::
            This is not implemented.

        :param values: Dictionary of nodes and data to write {node: value}.
        """
        raise NotImplementedError

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available.

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
        self._sub = loop.create_task(self._subscription_loop(handler, float(interval.total_seconds())))

    def close_sub(self) -> None:
        """Close the subsription."""
        self._subscription_open = False
        if self.exc:
            raise self.exc

        try:
            self._sub.cancel()
        except Exception:
            pass

        try:
            self.connection.close()
        except Exception:
            pass

    async def _subscription_loop(self, handler: SubscriptionHandler, interval: float) -> None:
        """The subscription loop handles requesting data from the server in the specified interval.

        :param handler: Handler object with a push function to receive data.
        :param interval: Interval for requesting data in seconds.
        """

        try:
            while self._subscription_open:
                try:
                    self._connect()
                except ConnectionError as e:
                    log.warning(str(e))

                for node in self._subscription_nodes:
                    result = None
                    try:
                        result = self._read_mb_value(node)
                    except (ValueError, ConnectionError) as e:
                        log.warning(str(e))

                    if result is not None:
                        _result = self._decode(result, node.mb_byteorder)
                        handler.push(node, _result, self._assert_tz_awareness(datetime.now()))

                await asyncio.sleep(interval)
        except BaseException as e:
            self.exc = e

    @staticmethod
    def _decode(value: Sequence[int], byteorder: str, type_: str = "f") -> Any:
        r"""Method to decode incoming modbus values.

        :param value: Current value to be decoded into float.
        :param byteorder: Byteorder for decoding i.e. 'little' or 'big' endian.
        :param type\_: Type of the output value. See `Python struct format character documentation
                      <https://docs.python.org/3/library/struct.html#format-characters>` for all possible
                      format strings (default: f).
        :return: Decoded value as a python type.
        """
        if byteorder == "little":
            bo = "<"
        elif byteorder == "big":
            bo = ">"
        else:
            raise ValueError(f"Specified an invalid byteorder: '{byteorder}'")

        # Determine the format strings for packing and unpacking the received byte sequences. These format strings
        # depend on the endianness (determined by bo), the length of the value in bytes and
        pack = f"{bo}{len(value):1d}H"

        size_div = {"h": 2, "H": 2, "i": 4, "I": 4, "l": 4, "L": 4, "q": 8, "Q": 8, "f": 4, "d": 8}
        unpack = f">{len(value) * 2 // size_div[type_]:1d}{type_}"

        # pymodbus gives a Sequence of 16bit unsigned integers which must be converted into the correct format
        return struct.unpack(unpack, struct.pack(pack, *value))[0]

    def _read_mb_value(self, node: NodeModbus) -> list[int]:
        """Read raw value from modbus server. This function should not be used directly. It does not
        establish a connection or handle connection errors.
        """
        if not self.connection.is_open:
            raise ConnectionError(f"Could not establish connection to host {self.url}")

        self.connection.unit_id = node.mb_slave

        if node.mb_register == "holding":
            result = self.connection.read_holding_registers(node.mb_channel, 2)
        else:
            raise ValueError(f"The specified register type is not supported: {node.mb_register}")

        if result is None:
            self._handle_mb_error()

        return result

    def _connect(self) -> None:
        """Connect to server."""
        try:
            if not self.connection.is_open:
                if not self.connection.open():
                    raise ConnectionError(f"Could not establish connection to host {self.url}")
        except (socket.herror, socket.gaierror) as e:
            raise ConnectionError(f"Host not found: {self.url}") from e
        except (socket.timeout) as e:
            raise ConnectionError(f"Host timeout: {self.url}") from e
        except (RuntimeError, ConnectionError) as e:
            raise ConnectionError(f"Connection Error: {self.url}, Error: {str(e)}") from e

    def _disconnect(self) -> None:
        """Disconnect from server."""
        try:
            self.connection.close()
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

    def _handle_mb_error(self) -> None:
        error = self.connection.last_error()
        exception = self.connection.last_except()

        if error is not None:
            raise ConnectionError(f"ModbusError {error} at {self.url}: {self.connection.last_error_txt()}")
        elif exception is not None:
            raise ConnectionError(f"ModbusError {exception} at {self.url}: {self.connection.last_except_txt()}")
        else:
            raise ConnectionError(f"Unknown ModbusError at {self.url}")

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeModbus]:  # type: ignore
        vnodes = super()._validate_nodes(nodes)
        _nodes = set()
        for node in vnodes:
            if isinstance(node, NodeModbus):
                _nodes.add(node)

        return _nodes
