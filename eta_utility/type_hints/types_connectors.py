from abc import abstractmethod
from typing import (
    Any,
    AnyStr,
    List,
    Mapping,
    MutableSequence,
    MutableSet,
    NewType,
    Optional,
    Union,
)
from urllib.parse import ParseResult

import pandas as pd

from eta_utility.type_hints import TimeStep


class Node:
    """Annotation class for the original Node class in connectors/common.py."""

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
    def url(self) -> AnyStr:
        """Get node URL"""
        pass

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


Nodes = NewType("Nodes", Union[MutableSequence[Node], MutableSet[Node], Node])


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
        self, handler: "SubscriptionHandler", nodes: Optional[Nodes] = None, interval: TimeStep = 1  # noqa:F821
    ) -> None:
        pass

    @abstractmethod
    def close_sub(self) -> None:
        pass

    @property
    def url(self) -> AnyStr:
        pass

    def _validate_nodes(self, nodes: Nodes) -> Nodes:
        pass
