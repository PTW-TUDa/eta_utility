from datetime import timedelta
from typing import Any, Union

import requests

from eta_utility import get_logger

from .base_classes import BaseConnection, Node, Nodes, SubscriptionHandler

log = get_logger("connectors.rest")


class RESTConnection(BaseConnection):
    """
    Class for reading and writing existing REST Endpoints with a known URL

    :param url: URL of the REST Server incl. scheme and port
    """

    _PROTOCOL = "rest"

    def __init__(self, url: str, nodes: Nodes = None):
        super().__init__(url, nodes=nodes)

    def read(self, node):
        request_url = "{}/{}/GetJson".format(self.url, node.rest_endpoint)
        response = requests.get(request_url)
        return response.json()

    def write(self, node, payload: dict):
        request_url = "{}/{}/PutJson".format(self.url, node.rest_endpoint)
        response = requests.put(request_url, json=payload)
        return response

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes = None, interval: Union[int, timedelta] = 1):
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: function to be called upon receiving new values, must accept attributes: node, val
        :param interval: interval for receiving new data. Interpreted as seconds when given as integer.
        """
        raise NotImplementedError("This function is currently not implemented yet. Issue #67 on gitlab")

    def close_sub(self):
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        raise NotImplementedError("This function is currently not implemented yet. Issue #67 on gitlab")

    @classmethod
    def from_node(cls, node: Node, **kwargs: Any):
        if node.protocol == "rest":
            cls.selected_nodes = {node}
            return cls(node.url)

        else:
            raise ValueError(
                "Tried to initialize RESTConnection from a node that does not specify rest as its"
                "protocol: {}.".format(node.name)
            )
