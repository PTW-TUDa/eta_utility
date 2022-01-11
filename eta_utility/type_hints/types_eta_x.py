import abc
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    MutableSet,
    NewType,
    Optional,
    Sequence,
    Set,
    SupportsFloat,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from gym import Env
from gym.vector.utils import spaces
from nptyping import Float, NDArray

StepResult = NewType("StepResult", Tuple[NDArray[NDArray[float]], NDArray[float], NDArray[bool], List[Dict]])
DefSettings = NewType("DefSettings", Mapping[str, Mapping[str, Union[str, int, bool, None]]])
ReqSettings = NewType("ReqSettings", Mapping[str, Set])


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

    def __init__(self) -> None:
        ...

    def append_state(self, *, name: Any, **kwargs: Dict[str, Any]) -> None:
        ...

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
    def step(self, action: NDArray) -> Tuple[NDArray, Union[Float, SupportsFloat], bool, Union[str, Sequence[str]]]:
        ...

    @abc.abstractmethod
    def reset(self) -> Tuple[NDArray, Union[Float, SupportsFloat], bool, Union[str, Sequence[str]]]:
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    def seed(self, seed: Union[str, int, None] = None) -> Tuple[np.random.BitGenerator, int]:
        ...

    @classmethod
    def get_info(cls, _: Any = None) -> Tuple[str, str]:
        ...
