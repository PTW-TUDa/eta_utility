""" This module implements some commonly used connector functions that are protocol independent.

"""
import pathlib
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, urlparse, urlunparse

import pandas as pd
from attrs import converters, define, field, validators

from eta_utility import get_logger, url_parse
from eta_utility.type_hints import Connection, Nodes, Path

from .eneffco import EnEffCoConnection
from .modbus import ModbusConnection
from .opc_ua import OpcUaConnection
from .rest import RESTConnection

default_schemes = {
    "modbus": "modbus.tcp",
    "opcua": "opc.tcp",
    "eneffco": "https",
}

log = get_logger("connectors")


def _strip_str(value: str) -> str:
    """Convenience function to convert a string to its stripped version.

    :param value: string to convert
    :return: stripped string
    """
    return value.strip()


def _lower_str(value: str) -> str:
    """Convenience function to convert a string to its stripped and lowercase version.

    :param value: string to convert
    :return: stripped and lowercase string
    """
    return value.strip().lower()


class NodeMeta(type):
    """Metaclass to instantiate the correct type of node for each protocol."""

    def __new__(cls, name: str, bases: Tuple, namespace: Dict[str, Any]) -> "NodeMeta":
        new_cls = super().__new__(cls, name, bases, namespace)
        return define(frozen=True, slots=False)(new_cls)


class Node:
    """The node objects represents a single variable. Valid keyword arguments depend on the protocol

    The url may contain the username and password (schema://username:password@hostname:port/path). This is handled
    automatically by the connectors.

    :param name: Any name can be used to identify the node. It is used to identify the node, therefore it should
                     be unique.
    :param url: Valid url string according to the standard format. E.g.: opc.tcp://127.0.0.1:4840. The scheme must
        be included (e.g.: https://).
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

    :param rest_endpoint: (Required for REST) Endpoint of the node, e.g. '/Node1'

    :param type dtype: Data type of the node. This may be needed in some specific cases, for example for the creation
                       of nodes.
    """

    #: data type conversion functions (i.e. to convert modbus types to python)
    _dtypes = {
        "boolean": bool,
        "bool": bool,
        "int": int,
        "uint32": int,
        "integer": int,
        "sbyte": int,
        "float": float,
        "double": float,
        "short": float,
        "string": str,
        "str": str,
    }

    def __new__(cls, name, url, protocol, *args, **kwargs):
        # Subclass definitions to instantiate based on protocol
        sub_cls = {
            "modbus": NodeModbus,
            "opcua": NodeOpcUa,
            "eneffco": NodeEnEffCo,
            "entsoe": NodeEntsoE,
            "rest": NodeREST,
        }

        # Check to make sure that if dtype is specified as an argument it is also present in the _dtypes map.
        _dtype = None
        if "dtype" in kwargs:
            dtype = _lower_str(kwargs.pop("dtype"))
            try:
                _dtype = cls._dtypes[dtype]
            except KeyError:
                log.warning(
                    f"The specified data type ({dtype}) is currently not available in the datatype map and "
                    f"will not be applied."
                )

        # Return the class instance
        try:
            return sub_cls[_lower_str(protocol)](name, url, protocol, *args, dtype=_dtype, **kwargs)
        except KeyError:
            raise ValueError(f"Specified an unsupported protocol: {protocol}.")

    @classmethod
    def from_dict(cls, dikt: Union[Sequence[Mapping], Mapping[str, Any]]) -> List["Node"]:
        """Create nodes from a dictionary of node configurations. The configuration must specify the following
        fields for each node:

            * Code (or name), URL, Protocol (modbus or opcua or eneffco).
              The URL should be a complete network location identifier. Alternatively it is possible to specify the
              location in two fields: IP and Port. These should only contain the respective parts (as in only an IP
              address and only the port number.
              The IP-Address should always be given without scheme (https://)

        For Modbus nodes the following additional fiels are required:

            * ModbusRegisterType (or mb_register), ModbusSlave (or mb_slave), ModbusChannel (or mb_channel)

        For OPC UA nodes the following additional fields are required:

            * Identifier

        For EnEffCo nodes the Code field must be present

        For REST nodes the REST_Endpoint field must be present

        :param dikt: Configuration dictionary
        :return: List of Node objects
        """

        nodes = []

        def dict_get_any(dikt: Dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
            """Get any of the specified items from dictionary, if any are available. The function will return
            the first value it finds, even if there are multiple matches.

            :param dikt: Dictionary to get values from
            :param names: Item names to look for
            :param fail: Flag to determine, if the function should fail with a KeyError, if none of the items are found.
                         If this is False, the function will return the value specified by 'default'. (default: True)
            :param default: Value to return, if none of the items are found and 'fail' is False. (default: None)
            :return: Value from dictionary
            :raise: KeyError, if none of the requested items are available and fail is True
            """
            for name in names:
                if name in dikt:
                    # Return first value found in dictionary
                    return dikt[name]
            else:
                if fail is True:
                    raise KeyError(f"Did not find one of the required keys in the configuration: {names}")
                else:
                    return default

        iter_ = [dikt] if isinstance(dikt, Mapping) else dikt
        for lnode in iter_:
            node = {k.strip().lower(): v for k, v in lnode.items()}

            # Find url or ip and port
            if "url" in node:
                loc = urlparse(node["url"].strip())
                scheme = None if loc.scheme == "" else loc.scheme
                loc = loc[1:6]
            else:
                loc = urlparse(f"//{dict_get_any(node, 'ip')}:{dict_get_any(node, 'port')}")[1:6]
                scheme = None
            name = dict_get_any(node, "code", "name")

            # Initialize node if protocol is 'modbus'
            if dict_get_any(node, "protocol").strip().lower() == "modbus":
                protocol = "modbus"
                scheme = default_schemes[protocol] if scheme is None else scheme
                url = urlunparse((scheme, *loc))

                mb_register = dict_get_any(node, "mb_register", "modbusregistertype")
                mb_slave = int(dict_get_any(node, "mb_slave", "modbusslave"))
                mb_channel = int(dict_get_any(node, "mb_channel", "modbuschannel"))
                mb_byteorder = dict_get_any(node, "mb_byteorder", "modbusbyteorder")
                dtype = dict_get_any(node, "dtype", "datentyp", fail=False)

                nodes.append(
                    cls(
                        name,
                        url,
                        protocol,
                        mb_register=mb_register,
                        mb_slave=mb_slave,
                        mb_channel=mb_channel,
                        mb_byteorder=mb_byteorder,
                        dtype=dtype,
                    )
                )

            # Initialize node if protocol is 'opcua'
            elif dict_get_any(node, "protocol").strip().lower() == "opcua":
                protocol = "opcua"
                scheme = default_schemes[protocol] if scheme is None else scheme
                url = urlunparse((scheme, *loc))

                opc_id = dict_get_any(node, "opc_id", "identifier", "identifier")
                dtype = dict_get_any(node, "dtype", "datentyp", fail=False)

                nodes.append(cls(name, url, protocol, opc_id=opc_id, dtype=dtype))

            # Initialize node if protocol is 'eneffco'
            elif dict_get_any(node, "protocol").strip().lower() == "eneffco":
                protocol = "eneffco"
                scheme = default_schemes[protocol] if scheme is None else scheme
                url = urlunparse((scheme, *loc))

                code = dict_get_any(node, "code")

                nodes.append(cls(name, url, protocol, eneffco_code=code))

            # Initialize node if protocol is 'REST'
            elif dict_get_any(node, "protocol").strip().lower() == "rest":
                protocol = "rest"
                scheme = default_schemes[protocol] if scheme is None else scheme
                url = urlunparse((scheme, *loc))

                rest_endpoint = dict_get_any(node, "rest_endpoint")

                nodes.append(cls(name, url, protocol, rest_endpoint=rest_endpoint))

        return nodes

    @classmethod
    def from_excel(cls, path: Path, sheet_name: str) -> List["Node"]:
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
        input_ = pd.read_excel(file, sheet_name=sheet_name)

        return cls.from_dict(list(input_.to_dict("index").values()))

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


class _NodeBase(metaclass=NodeMeta):

    #: Name for the node
    name: str = field(converter=_strip_str, eq=True)
    #: Url of the connection
    url: str = field(eq=True, order=True)
    #: Parse result object of the url (in case more post-processing is required)
    url_parsed: ParseResult = field(init=False, repr=False, eq=False, order=False)
    #: Protocol of the connection
    protocol: str = field(repr=False, eq=False, order=False)
    #: Username for login to the connection (default: None)
    usr: str = field(default=None, kw_only=True, repr=False, eq=False, order=False)
    #: Password for login to the connection (default: None)
    pwd: str = field(default=None, kw_only=True, repr=False, eq=False, order=False)
    #: Data type of the node (for value conversion)
    dtype: Optional[Callable] = field(default=None, kw_only=True, repr=False, eq=False, order=False)

    def __attrs_post_init__(self):
        """Add post-processing to the url, username and password information. Username and password specified during
        class init take precedence.
        """
        url, usr, pwd = url_parse(self.url)
        if self.usr is None and usr is not None:
            object.__setattr__(self, "usr", usr)
        if self.pwd is None and pwd is not None:
            object.__setattr__(self, "pwd", pwd)

        object.__setattr__(self, "url", url.geturl())
        object.__setattr__(self, "url_parsed", url)


def _mb_byteorder_converter(value: str) -> str:
    # Convert some values for mb_byteorder
    value = _lower_str(value)
    if value in {"little", "littleendian"}:
        return "little"

    if value in {"big", "bigendian"}:
        return "big"


class NodeModbus(_NodeBase):
    #: Modbus Slave ID
    mb_slave: int = field(kw_only=True)
    #: Modbus Register name (i.e. "Holding")
    mb_register: str = field(
        kw_only=True, converter=_lower_str, validator=validators.in_({"holding", "input", "output"})
    )
    #: Modbus Channel
    mb_channel: int = field(kw_only=True)
    #: Byteorder of values returned by modbus
    mb_byteorder: str = field(
        kw_only=True, converter=_mb_byteorder_converter, validator=validators.in_({"little", "big"})
    )

    def __attrs_post_init__(self):
        """Add default port to the url and convert mb_byteorder values."""
        super().__attrs_post_init__()

        # Set port to default 502 if it was not explicitly specified
        if not isinstance(self.url_parsed.port, int):
            url = self.url_parsed._replace(netloc=f"{self.url_parsed.hostname}:502")
            object.__setattr__(self, "url", url.get_url())
            object.__setattr__(self, "url_parsed", url)


class NodeOpcUa(_NodeBase):
    #: Node ID of the OPC UA Node
    opc_id: str = field(default=None, kw_only=True, converter=converters.optional(_strip_str))
    #: Path to the OPC UA node
    opc_path_str: str = field(
        default=None, kw_only=True, converter=converters.optional(_strip_str), repr=False, eq=False, order=False
    )
    #: Namespace of the OPC UA Node
    opc_ns: int = field(default=None, kw_only=True, converter=converters.optional(_lower_str))

    # Additional fields which will be determined automatically
    #: Type of the OPC UA Node ID Specification
    opc_id_type: str = field(init=False, validator=validators.in_({"i", "s"}), repr=False, eq=False, order=False)
    #: Name of the OPC UA Node
    opc_name: str = field(init=False, repr=False, eq=False, order=False)
    #: Path to the OPC UA node in list representation. Nodes in this list can be used to access any
    #: parent objects
    opc_path: List[Node] = field(init=False, repr=False, eq=False, order=False)

    def __attrs_post_init__(self):
        """Add default port to the url and convert mb_byteorder values."""
        super().__attrs_post_init__()

        # Set port to default 4840 if it was not explicitly specified
        if not isinstance(self.url_parsed.port, int):
            url = self.url_parsed._replace(netloc=f"{self.url_parsed.hostname}:4840")
            object.__setattr__(self, "url", url.get_url())
            object.__setattr__(self, "url_parsed", url)

        # Determine, which values to use for initialization and set values
        if self.opc_id is not None:
            parts = self.opc_id.split(";")
            for part in parts:
                key, val = part.split("=")
                if key.strip().lower() == "ns":
                    object.__setattr__(self, "opc_ns", int(val))
                else:
                    object.__setattr__(self, "opc_id_type", key.strip().lower())
                    object.__setattr__(self, "opc_path_str", val.strip(" "))

            object.__setattr__(self, "opc_id", f"ns={self.opc_ns};{self.opc_id_type}={self.opc_path_str}")

        elif self.opc_path_str is not None and self.opc_ns is not None:
            object.__setattr__(self, "opc_id_type", "s")
            object.__setattr__(self, "opc_id", f"ns={self.opc_ns};s={self.opc_path_str}")
        else:
            raise ValueError("Specify opc_id or opc_path_str and ns for OPC UA nodes.")

        # Determine the name and path of the opc node
        split_path = (
            self.opc_path_str.rsplit(".", maxsplit=len(self.opc_path_str.split(".")) - 2)
            if self.opc_path_str[0] == "."
            else self.opc_path_str.split(".")
        )

        object.__setattr__(self, "opc_name", split_path[-1].split(".")[-1])
        path = []
        if len(split_path) > 1:
            for key in range(len(split_path) - 1):
                path.append(
                    Node(
                        split_path[key].strip(" ."),
                        self.url,
                        "opcua",
                        usr=self.usr,
                        pwd=self.pwd,
                        opc_id="ns={};s={}".format(self.opc_ns, ".".join(split_path[: key + 1])),
                    )
                )
        object.__setattr__(self, "opc_path", path)


class NodeEnEffCo(_NodeBase):
    #: EnEffCo datapoint code / id
    eneffco_code: str = field(kw_only=True)


class NodeREST(_NodeBase):
    #: Rest endpoint
    rest_endpoint: str = field(kw_only=True)


class NodeEntsoE(NodeREST):
    pass


def connections_from_nodes(
    nodes: Nodes,
    eneffco_usr: Optional[str] = None,
    eneffco_pw: Optional[str] = None,
    eneffco_api_token: Optional[str] = None,
) -> Dict[str, Connection]:
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
            elif node.protocol == "rest":
                connections[node.url_parsed.hostname] = RESTConnection.from_node(node)
            else:
                raise ValueError(
                    f"Node {node.name} does not specify a recognized protocol for initializing a connection."
                )
        else:
            # Otherwise just mark the node as selected
            connections[node.url_parsed.hostname].selected_nodes.add(node)

    return connections


def name_map_from_node_sequence(nodes: Nodes) -> Dict[str, Node]:
    """Convert a Sequence/List of Nodes into a dictionary of nodes, identified by their name.

    .. warning ::

        Make sure that each node in nodes has a unique Name, otherwise this function will fail.

    :param nodes: Sequence of Node objects
    :return: Dictionary of Node objects (format: {node.name: Node})
    """
    if len({node.name for node in nodes}) != len([node.name for node in nodes]):
        raise ValueError("Not all node names are unique. Cannot safely convert to named dictionary.")

    return {node.name: node for node in nodes}
