""" Utilities for connecting to modbus servers
"""
import asyncio
import socket
import struct
from datetime import datetime, timedelta
from typing import Any, List, Mapping, Optional, Sequence, Set

import pandas as pd
import tzlocal
from pyModbusTCP.client import ModbusClient

from eta_utility.type_hints import Node, Nodes, TimeStep

from .base_classes import BaseConnection, SubscriptionHandler


class ModbusConnection(BaseConnection):
    """The OPC UA Connection class allows reading and writing from and to OPC UA servers and clients. Additionally
    it implements a subscription server, which reads continuously in a specified interval.

    :param url: Url of the OPC UA Server
    :param usr: Username in EnEffco for login
    :param pwd: Password in EnEffco for login
    :param nodes: List of nodes to use for all operations.
    """

    _PROTOCOL = "modbus"

    def __init__(
        self, url: str, usr: Optional[str] = None, pwd: Optional[str] = None, *, nodes: Optional[Nodes] = None
    ) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

        if self._url.scheme != "modbus.tcp":
            raise ValueError("Given URL is not a valid Modbus url (scheme: modbus.tcp)")

        self.connection: ModbusClient = ModbusClient(host=self._url.hostname, port=self._url.port, timeout=2)

        self._subscription_open: bool = False
        self._subscription_nodes: Set[Node] = set()
        self._sub: Optional[asyncio.Task] = None

    @classmethod
    def from_node(cls, node: Node, **kwargs: Any) -> "ModbusConnection":
        """Initialize the connection object from a modbus protocol node object

        :param node: Node to initialize from
        :param kwargs: Other arguments are ignored.
        :return: ModbusConnection object
        """

        if node.protocol == "modbus":
            return cls(node.url, nodes=[node])

        else:
            raise ValueError(
                "Tried to initialize ModbusConnection from a node that does not specify modbus as its"
                "protocol: {}.".format(node.name)
            )

    def read(self, nodes: Optional[Nodes] = None) -> pd.DataFrame:
        """
        Read some manually selected nodes from Modbus server

        :param nodes: List of nodes to read from

        :return: dictionary containing current values of the OPCUA-variables
        """
        nodes = self._validate_nodes(nodes)

        values = {}

        try:
            self.connection.open()
            self.connection.mode(1)

            for node in nodes:
                result = self._read_mb_value(node)
                value = self._decode(result, node.mb_byteorder)

                values[node.name] = value

        except socket.gaierror as e:
            raise ConnectionError(f"Host not found: {self.url}") from e

        finally:
            self.connection.close()

        return pd.DataFrame(values, index=[tzlocal.get_localzone().localize(datetime.now())])

    def write(self, values: Mapping[Node, Any]) -> None:
        """Write some manually selected values on OPCUA capable controller

        :param values: Dictionary of nodes and data to write. {node: value}
        """
        raise NotImplementedError

    def subscribe(self, handler: SubscriptionHandler, nodes: Optional[Nodes] = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param interval: interval for receiving new data. It it interpreted as seconds when given as an integer.
        """
        nodes = self._validate_nodes(nodes)

        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        self._subscription_nodes.update(nodes)

        if self._subscription_open:
            # Adding nodes to subscription is enough to include them in the query. Do not start an additional loop
            # if one already exists
            return

        self._subscription_open = True

        loop = asyncio.get_event_loop()
        self._sub = loop.create_task(self._subscription_loop(handler, int(interval.total_seconds())))

    def close_sub(self) -> None:
        try:
            self._sub.cancel()
            self._subscription_open = False
        except Exception:
            pass

        try:
            self.connection.close()
        except Exception:
            pass

    async def _subscription_loop(self, handler: SubscriptionHandler, interval: TimeStep) -> None:
        """The subscription loop handles requesting data from the server in the specified interval

        :param handler: Handler object with a push function to receive data
        :param interval: Interval for requesting data in seconds
        """

        try:
            while self._subscription_open:
                try:
                    if not self.connection.is_open():
                        if not self.connection.open():
                            raise ConnectionError(f"Could not establish connection to host {self.url}")
                        self.connection.mode(1)
                except socket.gaierror as e:
                    raise ConnectionError(f"Host not found: {self.url}") from e

                for node in self._subscription_nodes:
                    result = self._read_mb_value(node)
                    self._decode(result, node.mb_byteorder)
                    handler.push(node, result, tzlocal.get_localzone().localize(datetime.now()))
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    @staticmethod
    def _decode(value: Sequence[int], byteorder: str, type: str = "f") -> Any:
        """
        Method to decode incoming modbus values

        :param read_val: current value to be decoded into float
        :param byteorder: byteorder for decoding i.e. 'little' or 'big' endian
        :param type: type of the output value. See `Python struct format character documentation
                     <https://docs.python.org/3/library/struct.html#format-characters>` for all possible
                      format strings. (default: f)
        :return: decoded value as a python type
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
        unpack = f">{len(value) * 2 // size_div[type]:1d}{type}"

        # pymodbus gives a Sequence of 16bit unsigned integers which must be converted into the correct format
        return struct.unpack(unpack, struct.pack(pack, *value))[0]

    def _read_mb_value(self, node: Node) -> Optional[List[int]]:
        """Read raw value from modbus server. This function should not be used directly. It does not
        establish a connection or handle connection errors.

        """
        if not self.connection.is_open():
            raise ConnectionError(f"Could not establish connection to host {self.url}")

        self.connection.unit_id(node.mb_slave)

        if node.mb_register == "holding":
            result = self.connection.read_holding_registers(node.mb_channel, 2)
        else:
            raise ValueError(f"The specified register type is not supported: {node.mb_register}")

        if result is None:
            self._handle_mb_error()

        return result

    def _handle_mb_error(self) -> None:
        error = self.connection.last_error()
        print(self.connection.last_except())
        if error == 2:
            raise ConnectionError("ModbusError 2: Illegal Data Address")
        elif error == 4:
            raise ConnectionResetError("ModbusError 4: Slave Device Failure")
        else:
            raise ConnectionError(f"ModbusError {error}: Unknown ModbusError")
