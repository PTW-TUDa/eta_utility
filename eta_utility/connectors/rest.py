from datetime import timedelta
from typing import Any, Dict, Optional, Union

import requests

from eta_utility import get_logger

from ..type_hints import Node, Nodes
from .base_classes import BaseConnection, SubscriptionHandler

log = get_logger("connectors.rest")


class RESTConnection(BaseConnection):
    """
    Class for reading and writing existing REST Endpoints with a known URL

    :param url: URL of the REST Server incl. scheme and port
    """

    _PROTOCOL = "rest"

    def __init__(self, url: str, nodes: Nodes = None) -> None:
        super().__init__(url, nodes=nodes)

    # Signature of method 'RESTConnection.read()' does not match signature of the base method in class 'BaseConnection'
    def read(self, node: Node) -> Dict[str, Any]:
        request_url = f"{self.url}/{node.rest_endpoint}/GetJson"
        response = requests.get(request_url)
        return response.json()

    def write(self, node: Node, payload: dict) -> requests.Response:
        request_url = f"{self.url}/{node.rest_endpoint}/PutJson"
        response = requests.put(request_url, json=payload)
        return response

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes = None, interval: Union[int, timedelta] = 1) -> None:
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: function to be called upon receiving new values, must accept attributes: node, val
        :param interval: interval for receiving new data. Interpreted as seconds when given as integer.
        """
        raise NotImplementedError("This function is currently not implemented yet. Issue #67 on gitlab")

    def close_sub(self) -> None:
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        raise NotImplementedError("This function is currently not implemented yet. Issue #67 on gitlab")

    @classmethod
    def from_node(cls, node: Node, **kwargs: Any) -> Optional["RESTConnection"]:
        if node.protocol == "rest":
            cls.selected_nodes = {node}
            return cls(node.url)

        else:
            raise ValueError(
                "Tried to initialize RESTConnection from a node that does not specify rest as its"
                "protocol: {}.".format(node.name)
            )
