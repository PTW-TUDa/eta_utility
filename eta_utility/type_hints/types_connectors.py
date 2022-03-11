from typing import AbstractSet, NewType, Sequence, Set, Union

from eta_utility.connectors.node import (
    Node,
    NodeEnEffCo,
    NodeEntsoE,
    NodeLocal,
    NodeModbus,
    NodeOpcUa,
    NodeREST,
)

AnyNode = Union[Node, NodeLocal, NodeModbus, NodeOpcUa, NodeEnEffCo, NodeREST, NodeEntsoE]
Nodes = Union[Sequence[AnyNode], Set[AnyNode], AbstractSet[AnyNode], AnyNode]
SubscriptionHandler = NewType("SubscriptionHandler", object)
