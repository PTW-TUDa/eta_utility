from __future__ import annotations

from collections.abc import Sequence
from typing import AbstractSet, Union

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
Nodes = Union[Sequence[AnyNode], set[AnyNode], AbstractSet[AnyNode], AnyNode]
