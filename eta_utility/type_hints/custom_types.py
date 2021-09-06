import abc
import datetime
import pathlib
from abc import abstractmethod
from typing import (
    Any,
    AnyStr,
    Dict,
    List,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    NewType,
    Optional,
    Sequence,
    Set,
    SupportsFloat,
    Tuple,
    Union,
)
from urllib.parse import ParseResult

import numpy as np
import pandas as pd
from gym import Env
from gym.vector.utils import spaces
from nptyping import NDArray

# Other custom types:
Path = NewType("Path", Union[pathlib.Path, str])  # better to use maybe os.Pathlike
Numbers = NewType("Numbers", Union[float, int, np.float64, np.float32, np.int64, np.int32])
StepResult = NewType("StepResult", Tuple[NDArray[NDArray[float]], NDArray[float], NDArray[bool], List[Dict]])
TimeStep = NewType("TimeStep", Union[int, datetime.timedelta])  # can't be used for Union[float, timedelta] in fmu.py
DefSettings = NewType("DefSettings", Mapping[str, Mapping[str, Union[str, int, bool, None]]])
ReqSettings = NewType("ReqSettings", Mapping[str, Set])


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


class BaseEnv(Env, abc.ABC):
    """Annotation class for BaseEnv in envs/base_env.py"""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Version of the environment"""
        pass

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Long description of the environment"""
        pass

    #: Required settings in the general 'settings' section
    req_general_settings: Union[Sequence, MutableSet] = []
    #: Required settings in the 'path' section
    req_path_settings: Union[Sequence, MutableSet] = []
    #: Required settings in the 'environment_specific' section
    req_env_settings: Union[Sequence, MutableSet] = []
    #: Some environments may required specific parameters in the 'environment_specific' section to have special
    #   values. These parameter, value pairs can be specified in the req_env_config dictionary.
    req_env_config: MutableMapping = {}

    def __init__(self):
        raise NotImplementedError

    def append_state(self, *, name: Any, **kwargs) -> None:

        pass

    def _init_state_space(self) -> None:

        pass

    def _names_from_state(self) -> None:

        pass

    def _convert_state_config(self) -> pd.DataFrame:

        pass

    def _store_state_info(self) -> None:

        pass

    def import_scenario(self, *scenario_paths: Dict[str, Any], prefix_renamed: Optional[bool] = True) -> pd.DataFrame:

        pass

    def continuous_action_space_from_state(self) -> spaces.Space:

        pass

    def continuous_obs_space_from_state(self) -> spaces.Box:

        pass

    def within_abort_conditions(self, state: Mapping[str, float]) -> bool:

        pass

    def get_scenario_state(self) -> Dict[str, Any]:

        pass

    @abc.abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]:

        pass

    @abc.abstractmethod
    def reset(self) -> Tuple[np.ndarray, Union[np.float, SupportsFloat], bool, Union[str, Sequence[str]]]:

        pass

    @abc.abstractmethod
    def close(self) -> None:

        pass

    def seed(self, seed: Union[str, int] = None) -> Tuple[np.random.BitGenerator, int]:

        pass

    @classmethod
    def get_info(cls, _=None) -> Tuple[str, str]:

        pass
