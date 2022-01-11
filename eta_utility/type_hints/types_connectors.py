from abc import abstractmethod
from typing import (
    AbstractSet,
    Any,
    AnyStr,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)
from urllib.parse import ParseResult

import pandas as pd

from eta_utility.type_hints import TimeStep


class Node:
    """Annotation class for the original Node class in connectors/common.py."""

    name: str
    url: str
    protocol: str

    dtype: Union[Type[int], Type[str], Type[float], Type[bool]]

    mb_slave: int
    mb_register: str
    mb_channel: int
    mb_byteorder: str

    opc_id: Union[str, int]
    opc_path: List["Node"]
    opc_ns: int
    opc_path_str: str
    opc_id_type: str
    opc_name: str

    eneffco_code: str

    rest_endpoint: str

    def __init__(self) -> None:
        raise NotImplementedError

    def _init_modbus(self) -> None:
        """Placeholder method with modbus relevant fields"""
        pass

    def _init_opcua(self) -> None:
        """Placeholder method with opcua relevant fields"""
        pass

    def _init_eneffco(self) -> None:
        """Placeholder method with EnEffCo API relevant fields"""

    @property
    def url_parsed(self) -> ParseResult:
        pass

    @classmethod
    def from_dict(cls) -> List["Node"]:
        """Placeholder method to create list of nodes from a dictionary of node configurations."""
        pass

    @classmethod
    def from_excel(cls) -> List["Node"]:
        """Placeholder method to read out nodes from an excel document"""
        pass

    @classmethod
    def get_eneffco_nodes_from_codes(cls) -> List["Node"]:
        """
        Placeholder method to retrieve Node objects from a list of EnEffCo Codes (Identifiers).
        """
        pass

    def __hash__(self) -> int:
        pass


Nodes = Union[Sequence[Node], Set[Node], AbstractSet[Node], Node]
SubscriptionHandler = NewType("SubscriptionHandler", object)


class Connection:
    """Annotation class for Connection objects"""

    __PROTOCOL = ""

    def __init__(
        self, url: str, usr: Optional[str] = None, pwd: Optional[str] = None, *, nodes: Optional[Nodes] = None
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_node(cls, node: Node, **kwargs: Any) -> "Connection":
        pass

    def read(self, nodes: Optional[Nodes] = None) -> pd.DataFrame:
        pass

    @abstractmethod
    def write(self, values: Mapping[Node, Any]) -> None:
        pass

    @abstractmethod
    def subscribe(
        self,
        handler: SubscriptionHandler,
        nodes: Optional[Nodes] = None,
        interval: TimeStep = 1,
    ) -> None:
        ...

    @abstractmethod
    def close_sub(self) -> None:
        pass

    @property
    def url(self) -> AnyStr:
        pass

    def _validate_nodes(self, nodes: Nodes) -> Nodes:
        pass
