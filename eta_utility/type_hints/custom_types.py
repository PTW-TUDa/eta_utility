import datetime
import pathlib
from abc import abstractmethod
from typing import (
    Any,
    AnyStr,
    Dict,
    List,
    Mapping,
    MutableSequence,
    MutableSet,
    NewType,
    Optional,
    SupportsFloat,
    SupportsInt,
    Tuple,
    Union,
)
from urllib.parse import ParseResult

import numpy as np
import pandas as pd
from nptyping import NDArray

# Other custom types:
Path = NewType("Path", Union[pathlib.Path, str])  # better to use maybe os.Pathlike
Numbers = NewType("Numbers", Union[float, int, np.float64, np.float32, np.int64, np.int32])
StepResult = NewType("StepResult", Tuple[NDArray[NDArray[float]], NDArray[float], NDArray[bool], List[Dict]])
TimeStep = NewType("TimeStep", Union[int, datetime.timedelta])  # can't be used for Union[float, timedelta] in fmu.py


class Node:
    """Container class for the original Node class in connectors/common.py.
    Helps in type annotation of node objects.
    """

    def __init__(self) -> None:
        self.name: str
        self.protocol: str
        self._url: ParseResult
        self.usr: str
        self.pwd: str
        #: Unique identifier for the node which can be used for hashing
        self._id: str
        self.dtype: Union[SupportsInt, SupportsFloat, AnyStr]
        raise NotImplementedError

    def _init_modbus(self) -> None:
        """Placeholder method with modbus relevant fields"""
        self.mb_slave: int = None
        self.mb_register: str = None
        self.mb_channel: int = None

        pass

    def _init_opcua(self) -> None:
        """Placeholder method with opcua relevant fields"""
        self.opc_path_str: str = None
        self.opc_path: List[str] = None
        self.opc_id: str = None
        self.opc_ns: int = None
        self.opc_name: str = None

        pass

    def _init_eneffco(self) -> None:
        """Placeholder method with EnEffCo API relevant fields"""
        self.eneffco_code: str = None

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
    ):
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
