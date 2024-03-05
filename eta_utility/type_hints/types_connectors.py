from __future__ import annotations

from collections.abc import Sequence
from typing import AbstractSet, Union

from eta_utility.connectors.node import (
    Node,
    NodeCumulocity,
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
]
Nodes = Union[Sequence[AnyNode], set[AnyNode], AbstractSet[AnyNode], AnyNode]
