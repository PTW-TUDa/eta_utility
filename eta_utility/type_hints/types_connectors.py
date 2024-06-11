from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar, Union

from eta_utility.connectors.node import (
    Node,
    NodeCumulocity,
    NodeEmonio,
    NodeEnEffCo,
    NodeEntsoE,
    NodeLocal,
    NodeModbus,
    NodeOpcUa,
    NodeWetterdienstObservation,
    NodeWetterdienstPrediction,
)

# Deprecated Type
AnyNode = Union[
    Node,
    NodeLocal,
    NodeModbus,
    NodeOpcUa,
    NodeEnEffCo,
    NodeEntsoE,
    NodeCumulocity,
    NodeWetterdienstObservation,
    NodeWetterdienstPrediction,
    NodeEmonio,
]
# Generic Template for Nodes, N has to be a subclass of Node
N = TypeVar("N", bound=Node)

Nodes = Union[Sequence[N], set[N], N]
