import datetime
import pathlib
from typing import (
    AnyStr,
    Dict,
    List,
    MutableSequence,
    MutableSet,
    NewType,
    Tuple,
    Union,
)
from urllib.parse import ParseResult

import numpy as np
from nptyping import NDArray


class Node:
    """Container class for the original Node class in connectors/common.py.
    Helps in type annotation of node objects.
    """

    def __init__(self) -> None:
        """Placeholer constructor"""
        self.name: str = None
        self.protocol: str = None
        self._url: ParseResult = None
        self.dtype: "value for a given key" = None  # noqa: F722

        pass

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


# Other custom types:
Nodes = NewType("Nodes", Union[MutableSequence[Node], MutableSet[Node], Node])
Path = NewType("Path", Union[pathlib.Path, str])  # better to use maybe os.Pathlike
Numbers = NewType("Numbers", Union[float, int, np.float64, np.float32, np.int64, np.int32])  # only 1 occurrence
StepResult = NewType(
    "StepResult", Tuple[NDArray[NDArray[float]], NDArray[float], NDArray[bool], List[Dict]]
)  # only 1 occurrence
TimeStep = NewType("TimeStep", Union[int, datetime.timedelta])  # can't be used for Union[float, timedelta] in fmu.py
