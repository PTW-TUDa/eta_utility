""" This module implements some commonly used connector functions that are protocol independent.

"""
import pathlib
from typing import Any, AnyStr, Dict, List, Optional, Sequence
from urllib.parse import ParseResult, urlparse

import pandas as pd

from eta_utility.type_hints.custom_types import Node, Nodes, Path

from .base_classes import BaseConnection
from .eneffco import EnEffCoConnection
from .modbus import ModbusConnection
from .opcua import OpcUaConnection


class Node:
    """The node objects represents a single variable. Valid keyword arguments depend on the protocol

    The url may contain the username and password (schema://username:password@hostname:port/path). This is handled
    automatically by the connectors.

    :param name: Any name can be used to identify the node. It is used to identify the node, therefore it should
                     be unique.
    :param url: Valid url string according to the standard format. E.g.: opc.tcp://127.0.0.1:4840.
     Eneffco url with scheme (https://)
    :param protocol: Protocol to be used for connection (either opcua, eneffco or modbus)

    :param mb_slave: (Required for Modbus) Modbus slave ID
    :param mb_register: (Required for Modbus) Modbus register name
    :param mb_channel: (Required for Modbus) Modbus Channel
    :param mb_byteorder: (Required for Modbus) Byteorder eg. "little" or "big" endian.

    :param opc_id: (Required for OPC UA) Full OPC node ID, i.e.:
                        ns=6;s=.Heizung_Lueftung_Klima.System_Fussbodentemperierung_425.Pumpe_425.Zustand.Volumenstrom
                        (This must be used without other OPC related parameters)
    :param opc_path: (Alternative to opc_id for OPC UA) OPC UA node path as string or list (this correspondes to the
                     s=... part in the opc_id).
                     (This must be used in conjunction with the opc_ns parameter)
    :param opc_ns: (Alternative to opc_id for OPC UA) OPC UA namespace identifier number
                       (This must be used in conjunction with the opc_path parameter)

    :param eneffco_code: (Required for EnEffCo) EnEffCo Code

    :param type dtype: Data type of the node. This may be needed in some specific cases, for example for the creation
                       of nodes.
    """

    def __init__(self, name: str, url: str, protocol: str, **kwargs: Any) -> None:

        self.name: str = str(name).strip()
        self.protocol: str = protocol.strip().lower()
        self._url: ParseResult = urlparse(url)

        if "dtype" in kwargs:
            self.dtype = kwargs.pop("dtype")

        if self.protocol == "modbus":
            if not {"mb_slave", "mb_register", "mb_channel", "mb_byteorder"} == kwargs.keys():
                raise ValueError("Slave, register, channel and byteorder must be specified for modbus nodes.")
            self._init_modbus(**kwargs)

        elif self.protocol == "opcua":
            self._init_opcua(**kwargs)

        elif self.protocol == "eneffco":
            if not {"eneffco_code"} == kwargs.keys():
                raise ValueError("eneffco_code must be specified for eneffco nodes.")
            self._init_eneffco(**kwargs)

    def _init_modbus(self, mb_slave: int, mb_register: str, mb_channel: int, mb_byteorder: str) -> None:
        """Initialize the node object for modbus protocol nodes."""
        #: Modbus Slave ID
        self.mb_slave: int = int(mb_slave)
        #: Modbus Register name (i.e. "Holding")
        self.mb_register: str = mb_register.strip().lower()
        #: Modbus Channel
        self.mb_channel: int = int(mb_channel)
        #: Byteorder of values returned by modbus
        self.mb_byteorder: str

        # Figure out the correct byteorder, even if "littleendian" or "bigendian" are provided.
        mb_byteorder = mb_byteorder.strip().lower()
        if mb_byteorder in {"little", "big"}:
            self.mb_byteorder = mb_byteorder
        elif mb_byteorder in {"littleendian", "bigendian"}:
            self.mb_byteorder = "little" if mb_byteorder == "littleendian" else "big"
        else:
            raise ValueError(f"Byteorder must be either 'big' or 'little' endian, '{mb_byteorder}' given")

    def _init_opcua(self, **kwargs: Any) -> None:
        """Initialize the node object for opcua protocol nodes"""
        #: Path to the OPC UA node
        self.opc_path_str: str = ""
        #: Path to the OPC UA node in list representation. Nodes in this list can be used to access any
        #: parent objects
        self.opc_path: List[Node] = []

        if {"opc_id"} == kwargs.keys():
            self.opc_id: str = str(kwargs["opc_id"]).strip()
            parts = self.opc_id.split(";")
            for part in parts:
                key, val = part.split("=")
                if key.lower() == "ns":
                    self.opc_ns: int = int(val)
                elif key.lower() == "s":
                    self.opc_path_str: str = val.strip(" .")
            self.opc_id = f"ns={self.opc_ns};s=.{self.opc_path_str}"
        elif {"opc_path", "ns"} == kwargs.keys():
            self.opc_ns = int(kwargs["ns"])
            self.opc_path_str = kwargs["opc_path"].strip(" .")
            self.opc_id = f"ns={self.opc_ns};s=.{self.opc_path_str}"
        else:
            raise ValueError("Specify opc_id or opc_path and ns for OPC UA nodes.")

        split_path = self.opc_path_str.split(".")
        self.opc_name: str = split_path[-1]
        if len(split_path) > 1:
            for key in range(len(split_path) - 1):
                self.opc_path.append(
                    Node(
                        split_path[key].strip(" ."),
                        self.url,
                        "opcua",
                        opc_id="ns={};s=.{}".format(self.opc_ns, ".".join(split_path[: key + 1])),
                    )
                )

    def _init_eneffco(self, eneffco_code: str) -> None:
        """Initialize the node object for the EnEffCo API."""
        self.eneffco_code: str = eneffco_code

    @property
    def url(self) -> AnyStr:
        """Get node URL"""
        return self._url.geturl()

    @property
    def url_parsed(self) -> ParseResult:
        return self._url

    @classmethod
    def from_dict(cls, dikt: Dict[str, Dict[str, str]]) -> List[Node]:
        """Create nodes from a dictionary of node configurations. The configuration must specify the following
        fields for each node:

            * Code (or name), IP, Port, Protocol (modbus or opcua or eneffco).

        For Modbus nodes the following additional fiels are required:

            * ModbusRegisterType (or mb_register), ModbusSlave (or mb_slave), ModbusChannel (or mb_channel)

        For OPC UA nodes the following additional fields are required:

            * Identifier

        For EnEffCo nodes the Code field must be present

        The IP-Address should always be given without scheme (https://)

        :param dikt: Configuration dictionary

        :return: List of Node objects
        """

        nodes = []

        def dict_get_any(dikt, *names):
            for name in names:
                if name in dikt:
                    return dikt[name]
            else:
                raise KeyError(f"None of the requested keys are in the configuration: {names}")

        for node in dikt.values():

            netloc = str(dict_get_any(node, "IP")) + ":" + str(dict_get_any(node, "Port"))
            name = dict_get_any(node, "Code", "name")

            if dict_get_any(node, "Protocol").strip().lower() == "modbus":
                scheme = "modbus.tcp"
                protocol = dict_get_any(node, "Protocol").strip().lower()
                mb_register = dict_get_any(node, "mb_register", "ModbusRegisterType")
                mb_slave = int(dict_get_any(node, "mb_slave", "ModbusSlave"))
                mb_channel = int(dict_get_any(node, "mb_channel", "ModbusChannel"))
                mb_byteorder = dict_get_any(node, "mb_byteorder", "ModbusByteOrder")

                url = scheme + "://" + netloc
                nodes.append(
                    cls(
                        name,
                        url,
                        protocol,
                        mb_register=mb_register,
                        mb_slave=mb_slave,
                        mb_channel=mb_channel,
                        mb_byteorder=mb_byteorder,
                    )
                )

            elif dict_get_any(node, "Protocol").strip().lower() == "opcua":
                scheme = "opc.tcp"
                protocol = dict_get_any(node, "Protocol").strip().lower()
                opc_id = dict_get_any(node, "opc_id", "Identifier", "identifier")

                url = scheme + "://" + netloc
                nodes.append(cls(name, url, protocol, opc_id=opc_id))

            elif dict_get_any(node, "Protocol").strip().lower() == "eneffco":
                scheme = "https"
                protocol = "eneffco"
                code = dict_get_any(node, "Code", "code")

                url = scheme + "://" + netloc
                nodes.append(cls(name, url, protocol, eneffco_code=code))

        return nodes

    @classmethod
    def from_excel(cls, path: Path, sheet_name: str) -> List[Node]:
        """
        Method to read out nodes from an excel document. The document must specify the following fields:

            * Code, IP, Port, Protocol (modbus or opcua or eneffco).

        For Modbus nodes the following additional fiels are required:

            * ModbusRegisterType, ModbusByte, ModbusChannel

        For OPC UA nodes the following additional fields are required:

            * Identifier

        For EnEffCo nodes the Code field must be present

        The IP-Address should always be given without scheme (https://)

        :param path: Path to excel document
        :param sheet_name: name of Excel sheet, which will be read out

        :return: List of Node objects
        """

        file = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        input = pd.read_excel(file, sheet_name=sheet_name)

        return cls.from_dict(input.to_dict("index"))

    @classmethod
    def get_eneffco_nodes_from_codes(cls, code_list: Sequence[str], eneffco_url: Optional[str]) -> List["Node"]:
        """
        Utility function to retrieve Node objects from a list of EnEffCo Codes (Identifiers).

        :param code_list: List of EnEffCo identifiers to create nodes from
        :param eneffco_url: URL to the eneffco system
        :return: List of EnEffCo-nodes
        """
        nodes = []
        for code in code_list:
            nodes.append(cls(name=code, url=eneffco_url, protocol="eneffco", eneffco_code=code))
        return nodes

    def __hash__(self) -> int:
        return hash(self.name)


def connections_from_nodes(
    nodes: Nodes,
    eneffco_usr: Optional[str] = None,
    eneffco_pw: Optional[str] = None,
    eneffco_api_token: Optional[str] = None,
) -> Dict[str, "BaseConnection"]:
    """Take a list of nodes and return a list of connections

    :param nodes: List of nodes defining servers to connect to
    :param eneffco_usr: Optional username for eneffco login
    :param eneffco_pw: Optional password for eneffco login
    :param eneffco_api_token: Token for EnEffCo API authorization
    :return: Dictionary of connection objects {hostname: connection}
    """

    connections = {}

    for node in nodes:
        # Create connection if it does not exist
        if node.url_parsed.hostname not in connections:
            if node.protocol == "modbus":
                connections[node.url_parsed.hostname] = ModbusConnection.from_node(node)
            elif node.protocol == "opcua":
                connections[node.url_parsed.hostname] = OpcUaConnection.from_node(node)
            elif node.protocol == "eneffco":
                if eneffco_usr is None or eneffco_pw is None or eneffco_api_token is None:
                    raise ValueError("Specify username, password and API token for EnEffco access.")
                connections[node.url_parsed.hostname] = EnEffCoConnection.from_node(
                    node, usr=eneffco_usr, pwd=eneffco_pw, api_token=eneffco_api_token
                )
            else:
                raise ValueError(
                    f"Node {node.name} does not specify a recognized protocol for initializing a connection."
                )
        else:
            # Otherwise just mark the node as selected
            connections[node.url_parsed.hostname].selected_nodes.add(node)

    return connections
