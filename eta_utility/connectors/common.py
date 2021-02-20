""" This module implements some commonly used connector functions that are protocol independent.

"""
import pathlib
from urllib.parse import urlparse

import pandas as pd

from .eneffco import EnEffCoConnection
from .modbus import ModbusConnection
from .opcua import OpcUaConnection


def connections_from_nodes(nodes, eneffco_usr=None, eneffco_pw=None):
    """Take a list of nodes and return a list of connections

    :param nodes: List of nodes defining servers to connect to
    :type nodes: List[Node]
    :param str eneffco_usr: Optional username for eneffco login
    :param str eneffco_pw: Optional password for eneffco login
    :return: Dictionary of connection objects {hostname: connection}
    :rtype: Dict[str, BaseConnection]
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
                connections[node.url_parsed.hostname] = EnEffCoConnection.from_node(
                    node, usr=eneffco_usr, pwd=eneffco_pw
                )
            else:
                raise ValueError(
                    f"Node {node.name} does not specify a recognized protocol for initializing a connection."
                )
        else:
            # Otherwise just mark the node as selected
            connections[node.url_parsed.hostname].selected_nodes.add(node)

    return connections


class Node:
    """The node objects represents a single variable. Valid keyword arguments depend on the protocol

    :param str name: Any name can be used to identify the node. It is used to identify the node, therefore it should
                     be unique.
    :param str url: Valid url string according to the standard format. E.g.: opc.tcp://127.0.0.1:4840
    :param str protocol: Protocol to be used for connection (either opcua, eneffco or modbus)

    :param int mb_slave: (Required for Modbus) Modbus slave ID
    :param str mb_register: (Required for Modbus) Modbus register name
    :param int mb_channel: (Required for Modbus) Modbus Channel

    :param str opc_id: (Required for OPC UA) Full OPC node ID, i.e.:
                        ns=6;s=.Heizung_Lueftung_Klima.System_Fussbodentemperierung_425.Pumpe_425.Zustand.Volumenstrom
                        (This must be used without other OPC related parameters)
    :param opc_path: (Alternative to opc_id for OPC UA) OPC UA node path as string or list (this correspondes to the
                     s=... part in the opc_id).
                     (This must be used in conjunction with the opc_ns parameter)
    :type opc_path: str or List
    :param int opc_ns: (Alternative to opc_id for OPC UA) OPC UA namespace identifier number
                       (This must be used in conjunction with the opc_path parameter)

    :param str eneffco_code: (Required for EnEffCo) EnEffCo Code

    :param type dtype: Data type of the node. This may be needed in some specific cases, for example for the creation
                       of nodes.
    """

    def __init__(self, name, url, protocol, **kwargs):

        self.name = str(name).strip()
        self.protocol = protocol.strip().lower()
        self._url = urlparse(url)

        if "dtype" in kwargs:
            self.dtype = kwargs.pop("dtype")

        if self.protocol == "modbus":
            if not {"mb_slave", "mb_register", "mb_channel"} == kwargs.keys():
                raise ValueError("Slave, register and channel must be specified for modbus nodes.")
            self._init_modbus(**kwargs)

        elif self.protocol == "opcua":
            self._init_opcua(**kwargs)

        elif self.protocol == "eneffco":
            if not {"eneffco_code"} == kwargs.keys():
                raise ValueError("eneffco_code must be specified for eneffco nodes.")
            self._init_eneffco(**kwargs)

    def _init_modbus(self, mb_slave, mb_register, mb_channel):
        """Initialize the node object for modbus protocol nodes."""
        self.mb_slave = int(mb_slave)
        self.mb_register = mb_register.strip().lower()
        self.mb_channel = int(mb_channel)

    def _init_opcua(self, **kwargs):
        """Initialize the node object for opcua protocol nodes"""
        #: opc_path_str: Path to the OPC UA node
        self.opc_path_str = ""
        #: opc_path: Path to the OPC UA node in list representation
        self.opc_path = []

        if {"opc_id"} == kwargs.keys():
            self.opc_id = str(kwargs["opc_id"]).strip()
            parts = self.opc_id.split(";")
            for part in parts:
                key, val = part.split("=")
                if key == "ns":
                    self.opc_ns = int(val)
                elif key == "s":
                    self.opc_path_str = val.strip(" .")
        elif {"opc_path", "ns"} == kwargs.keys():
            self.opc_ns = int(kwargs["ns"])
            self.opc_path_str = kwargs["opc_path"].strip(" .")
            self.opc_id = f"ns={self.opc_ns};s=.{self.opc_path_str}"
        else:
            raise ValueError("Specify opc_id or opc_path and ns for OPC UA nodes.")

        split_path = self.opc_path_str.split(".")
        self.opc_name = split_path[-1]
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

    def _init_eneffco(self, eneffco_code):
        """Initialize the node object for the EnEffCo API."""
        self.eneffco_code = eneffco_code

    @property
    def url(self):
        """Get node URL"""
        return self._url.geturl()

    @property
    def url_parsed(self):
        return self._url

    @classmethod
    def from_excel(cls, path, sheet_name):
        """
        Method to read out nodes from an excel document. The document must specify the following fields:

            * Code, IP, Port, Protocol (modbus or opcua or eneffco).

        For Modbus nodes the following additional fiels are required:

            * ModbusRegisterType, ModbusByte, ModbusChannel

        For OPC UA nodes the following additional fields are required:

            * Identifier

        For EnEffCo nodes the Code field must be present



        :param path: Path to excel document
        :type path: pathlib.Path or str
        :param str sheet_name: name of Excel sheet, which will be read out

        :return: List of Node objects
        """

        file = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
        input = pd.read_excel(file, sheet_name=sheet_name)

        nodes = []
        for _, node in input.iterrows():
            netloc = node["IP"] + ":" + str(node["Port"])
            name = node["Code"]

            if node["Protocol"].strip().lower() == "modbus":
                scheme = "modbus.tcp"
                protocol = "modbus"
                mb_register = node["ModbusRegisterType"]
                mb_slave = int(node["ModbusSlave"])
                mb_channel = int(node["ModbusChannel"])

                url = scheme + "://" + netloc
                nodes.append(
                    cls(
                        name,
                        url,
                        protocol,
                        mb_register=mb_register,
                        mb_slave=mb_slave,
                        mb_channel=mb_channel,
                    )
                )

            elif node["Protocol"].strip().lower() == "opcua":
                scheme = "opc.tcp"
                protocol = "opcua"
                opc_id = node["Identifier"]

                url = scheme + "://" + netloc
                nodes.append(cls(name, url, protocol, opc_id=opc_id))

            elif node["Protocol"].strip().lower() == "eneffco":
                scheme = "https"
                protocol = "eneffco"
                code = node["Code"]

                url = scheme + "://" + netloc
                nodes.append(cls(name, url, protocol, eneffco_code=code))

        return nodes

    @classmethod
    def get_eneffco_nodes_from_codes(cls, code_list, eneffco_url):
        """
        Utility function to retrieve Node objects from a list of EnEffCo Codes (Identifiers).
        :param code_list: List of EnEffCo identifiers to create nodes from
        :type code_list: list[str]
        :param eneffco_url: URL to the eneffco system
        :type eneffco_url: str or None
        :return: List of EnEffCo-nodes
        :rtype: list[eta_utility.connectors.Node]
        """
        nodes = []
        for code in code_list:
            nodes.append(cls(name=code, url=eneffco_url, protocol="eneffco", eneffco_code=code))
        return nodes

    def __hash__(self):
        return hash(self.name)
