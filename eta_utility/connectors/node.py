""" This module implements the node class, which is used to parametrize connections

"""
from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Mapping
from urllib.parse import ParseResult, urlparse, urlunparse

import pandas as pd
from attrs import converters, define, field, validators  # noqa: I900

from eta_utility import get_logger, url_parse

if TYPE_CHECKING:
    from typing import Any, Callable, Sequence

    from eta_utility.type_hints import Path

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


def _dtype_converter(value: str) -> Callable | None:
    """Specify data type conversion functions (i.e. to convert modbus types to python).

    :param value: data type string to convert to callacle datatype converter
    :return: python datatype (callable)
    """
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
    try:
        dtype = _dtypes[_lower_str(value)]
    except KeyError:
        log.warning(
            f"The specified data type ({value}) is currently not available in the datatype map and "
            f"will not be applied."
        )
        dtype = None

    return dtype


class NodeMeta(type):
    """Metaclass to define all Node classes as frozen attr dataclasses."""

    def __new__(cls, name: str, bases: tuple, namespace: dict[str, Any], **kwargs: Any) -> NodeMeta:
        new_cls = super().__new__(cls, name, bases, namespace, **kwargs)
        return define(frozen=True, slots=False)(new_cls)


class Node(metaclass=NodeMeta):
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

    #: Name for the node
    name: str = field(converter=_strip_str, eq=True)
    #: Url of the connection
    url: str = field(eq=True, order=True)
    #: Parse result object of the url (in case more post-processing is required)
    url_parsed: ParseResult = field(init=False, repr=False, eq=False, order=False)
    #: Protocol of the connection
    protocol: str = field(repr=False, eq=False, order=False)
    #: Username for login to the connection (default: None)
    usr: str | None = field(default=None, kw_only=True, repr=False, eq=False, order=False)
    #: Password for login to the connection (default: None)
    pwd: str | None = field(default=None, kw_only=True, repr=False, eq=False, order=False)
    #: Data type of the node (for value conversion)
    dtype: Callable | None = field(
        default=None, converter=converters.optional(_dtype_converter), kw_only=True, repr=False, eq=False, order=False
    )

    _registry = {}  # type: ignore

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Store subclass definitions to instantiate based on protocol."""
        protocol = kwargs.pop("protocol", None)
        if protocol:
            cls._registry[protocol] = cls

        return super().__init_subclass__(**kwargs)

    def __new__(cls, name: str, url: str, protocol: str, *args: Any, **kwargs: Any) -> Node:
        """Create node object of correct subclass corresponding to protocol."""
        try:
            subclass = cls._registry[protocol]
        except KeyError:
            raise ValueError(f"Specified an unsupported protocol: {protocol}.")

        # Return the correct subclass for the specified protocol
        obj = object.__new__(subclass)
        return obj

    def __attrs_post_init__(self) -> None:
        """Add post-processing to the url, username and password information. Username and password specified during
        class init take precedence.
        """
        url, usr, pwd = url_parse(self.url)
        if self.usr is None:
            object.__setattr__(self, "usr", usr)
        if self.pwd is None:
            object.__setattr__(self, "pwd", pwd)

        object.__setattr__(self, "url", url.geturl())
        object.__setattr__(self, "url_parsed", url)

    @classmethod
    def from_dict(cls, dikt: Sequence[Mapping] | Mapping[str, Any]) -> list[Node]:
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

        def dict_get_any(dikt: dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
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
    def from_excel(cls, path: Path, sheet_name: str) -> list[Node]:
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
    def get_eneffco_nodes_from_codes(cls, code_list: Sequence[str], eneffco_url: str) -> list[Node]:
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


class NodeLocal(Node, protocol="local"):
    def __attrs_post_init__(self) -> None:
        """Ensure username and password are processed correctly."""
        super().__attrs_post_init__()


def _mb_byteorder_converter(value: str) -> str:
    """Convert some values for mb_byteorder
    :param value: value to be converted to mb_byteorder
    :return: mb_byteorder corresponding to correct scheme
    """
    value = _lower_str(value)
    if value in {"little", "littleendian"}:
        return "little"

    if value in {"big", "bigendian"}:
        return "big"

    return ""


class NodeModbus(Node, protocol="modbus"):
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

    def __attrs_post_init__(self) -> None:
        """Add default port to the url and convert mb_byteorder values."""
        super().__attrs_post_init__()

        # Set port to default 502 if it was not explicitly specified
        if not isinstance(self.url_parsed.port, int):
            url = self.url_parsed._replace(netloc=f"{self.url_parsed.hostname}:502")
            object.__setattr__(self, "url", url.geturl())
            object.__setattr__(self, "url_parsed", url)


class NodeOpcUa(Node, protocol="opcua"):
    #: Node ID of the OPC UA Node
    opc_id: str | None = field(default=None, kw_only=True, converter=converters.optional(_strip_str))
    #: Path to the OPC UA node
    opc_path_str: str | None = field(
        default=None, kw_only=True, converter=converters.optional(_strip_str), repr=False, eq=False, order=False
    )
    #: Namespace of the OPC UA Node
    opc_ns: int | None = field(default=None, kw_only=True, converter=converters.optional(_lower_str))

    # Additional fields which will be determined automatically
    #: Type of the OPC UA Node ID Specification
    opc_id_type: str = field(init=False, validator=validators.in_({"i", "s"}), repr=False, eq=False, order=False)
    #: Name of the OPC UA Node
    opc_name: str = field(init=False, repr=False, eq=False, order=False)
    #: Path to the OPC UA node in list representation. Nodes in this list can be used to access any
    #: parent objects
    opc_path: list[NodeOpcUa] = field(init=False, repr=False, eq=False, order=False)

    def __attrs_post_init__(self) -> None:
        """Add default port to the url and convert mb_byteorder values."""
        super().__attrs_post_init__()

        # Set port to default 4840 if it was not explicitly specified
        if not isinstance(self.url_parsed.port, int):
            url = self.url_parsed._replace(netloc=f"{self.url_parsed.hostname}:4840")
            object.__setattr__(self, "url", url.geturl())
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
            self.opc_path_str.rsplit(".", maxsplit=len(self.opc_path_str.split(".")) - 2)  # type: ignore
            if self.opc_path_str[0] == "."  # type: ignore
            else self.opc_path_str.split(".")  # type: ignore
        )

        object.__setattr__(self, "opc_name", split_path[-1].split(".")[-1])
        path = []
        if len(split_path) > 1:
            for k in range(len(split_path) - 1):
                path.append(
                    Node(
                        split_path[k].strip(" ."),
                        self.url,
                        "opcua",
                        usr=self.usr,
                        pwd=self.pwd,
                        opc_id="ns={};s={}".format(self.opc_ns, ".".join(split_path[: k + 1])),
                    )
                )
        object.__setattr__(self, "opc_path", path)


class NodeEnEffCo(Node, protocol="eneffco"):
    #: EnEffCo datapoint code / id
    eneffco_code: str = field(kw_only=True)

    def __attrs_post_init__(self) -> None:
        """Ensure username and password are processed correctly."""
        super().__attrs_post_init__()


class NodeREST(Node, protocol="rest"):
    #: REST endpoint
    rest_endpoint: str = field(kw_only=True)

    def __attrs_post_init__(self) -> None:
        """Ensure username and password are processed correctly."""
        super().__attrs_post_init__()


class NodeEntsoE(Node, protocol="entsoe"):
    #: REST endpoint
    endpoint: str = field(kw_only=True)
    #: Bidding zone
    bidding_zone: str = field(kw_only=True)

    def __attrs_post_init__(self) -> None:
        """Ensure username and password are processed correctly."""
        super().__attrs_post_init__()