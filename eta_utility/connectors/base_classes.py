""" Base classes for the connectors

"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, AnyStr, List, Mapping, NewType, Sequence, Set, Union
from urllib.parse import urlparse

import pandas as pd

Node = NewType("Node", object)
Nodes = Union[Node, Set[Node], Sequence[Node]]


class SubscriptionHandler(ABC):
    """Subscription handlers do stuff to subscribed data points after they are received. Every handler must have a
    push method which can be called when data is received.

    :param write_interval: Interval for writing data to csv file
    """

    def __init__(self, write_interval: Union[float, int] = 1) -> None:
        self._write_interval = write_interval

    def _round_timestamp(self, timestamp: datetime) -> datetime:
        """Helper method for rounding date time objects to the specified write interval

        :param datetime timestamp: Datetime object to be rounded
        :return: Rounded datetime object without timezone information
        """

        intervals = timestamp.timestamp() // self._write_interval
        return datetime.fromtimestamp(intervals * self._write_interval)

    def _convert_series(
        self, value: Union[pd.Series, Sequence[Any]], timestamp: Union[pd.DatetimeIndex, timedelta, int]
    ) -> pd.Series:
        """Helper function to convert a value, timestamp pair in which value is a Series or list to a Series with
        datetime index according to the given timestamp(s).

        :param value: Series of values. There must be corresponding timestamps for each value.
        :param timestamp: DatetimeIndex of the provided values. Alternatively an integer/timedelta can be provided to
                          determine the interval between data points. Use negative numbers to describe past data.
                          Integers are interpreted as seconds. If value is a pd.Series and has a pd.DatetimeIndex,
                          timestamp can be None.
        :return: pd.Series with corresponding DatetimeIndex
        """
        # Check timestamp first
        # timestamp as datetime-index:
        if isinstance(timestamp, pd.DatetimeIndex):
            if len(timestamp) != len(value):
                raise ValueError(
                    f"Length of timestamp ({len(timestamp)}) and value ({len(value)}) must match if "
                    f"timestamp is given as pd.DatetimeIndex."
                )
        # timestamp as int or timedelta:
        elif isinstance(timestamp, int) or isinstance(timestamp, timedelta):
            if isinstance(timestamp, int):
                timestamp = timedelta(seconds=timestamp)
            if timestamp < timedelta(seconds=0):
                _freq = str((-timestamp).seconds) + "S"
                timestamp = pd.date_range(end=datetime.now(), freq=_freq, periods=len(value))
            else:
                _freq = str(timestamp.seconds) + "S"
                timestamp = pd.date_range(start=datetime.now(), freq=_freq, periods=len(value))
            timestamp = timestamp.round(_freq)
        # timestamp None:
        elif timestamp is None and isinstance(value, pd.Series):
            if not isinstance(value.index, pd.DatetimeIndex):
                raise ValueError("If timestamp is None, value must have a pd.DatetimeIndex")
            else:
                timestamp = value.index
        else:
            raise TypeError(
                f"timestamp must be pd.DatetimeIndex, int or timedelta, is {type(timestamp)}. Else, "
                f"value must have a pd.DatetimeIndex."
            )

        # Check value and build pd.Series
        if isinstance(value, pd.Series):
            value.index = timestamp
        else:
            value = pd.Series(data=value, index=timestamp)
            # If value is multi-dimensional, an Exception will be raised by pandas.

        # Round index to self._write_interval
        value.index = value.index.round(str(self._write_interval) + "S")

        return value

    @abstractmethod
    def push(self, node: Node, value: Any, timestamp: datetime = None):
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to
        :param value: Value of the data
        :param timestamp: Timestamp of receiving the data
        """
        pass


class BaseConnection(ABC):
    """Base class with a common interface for all connection objects

    :param url: URL of the server to connect to.
    :param usr: Username for login to server
    :param pwd: Password for login to server
    :param nodes: List of nodes to select as a standard case
    """

    def __init__(self, url: str, usr: str = None, pwd: str = None, *, nodes: Nodes = None) -> None:
        self._url = urlparse(url)

        if type(usr) is not str and usr is not None:
            raise TypeError("Username should be a string value.")
        self.usr = usr

        if type(pwd) is not str and pwd is not None:
            raise TypeError("Password should be a string value.")
        self.pwd = pwd

        if nodes is not None:
            if not hasattr(nodes, "__len__"):
                self.selected_nodes = {nodes}
            else:
                self.selected_nodes = set(nodes)
        else:
            self.selected_nodes = set()

    @classmethod
    @abstractmethod
    def from_node(cls, node: Node, **kwargs: Any) -> "BaseConnection":
        """Initialize the object from a node with corresponding protocol

        :return: Initialized connection object
        """
        pass

    @abstractmethod
    def read(self, nodes: Nodes = None) -> pd.DataFrame:
        """Read data from nodes

        :param nodes: List of nodes to read from
        :return: Pandas DataFrame with resulting values
        """

        pass

    @abstractmethod
    def write(self, values: Mapping[Node, Any]):
        """Write data to a list of nodes

        :param values: Dictionary of nodes and data to write. {node: value}
        """
        pass

    @abstractmethod
    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes = None, interval: Union[int, timedelta] = 1):
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: function to be called upon receiving new values, must accept attributes: node, val
        :param interval: interval for receiving new data. Interpreted as seconds when given as integer.
        """
        pass

    @abstractmethod
    def close_sub(self):
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        pass

    @property
    def url(self) -> AnyStr:
        return self._url.geturl()

    def _validate_nodes(self, nodes: Nodes) -> Nodes:
        if nodes is None:
            nodes = self.selected_nodes

        if not hasattr(nodes, "__len__"):
            nodes = {nodes}
        else:
            if len(nodes) == 0:
                raise ValueError("Some nodes to read from must be specified.")

        return nodes


class BaseSeriesConnection(BaseConnection, ABC):
    """Connection object for protocols with the ability to provide access to timeseries data.

    :param url: URL of the server to connect to
    """

    def __init__(self, url: str, usr: str = None, pwd: str = None, *, nodes: Nodes = None) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

    @abstractmethod
    def read_series(self, from_time: datetime, to_time: datetime, nodes: List = None, **kwargs: Any) -> pd.DataFrame:
        """Read time series data from the connection, within a specified time interval (from_time until to_time).
        :param nodes: List of nodes to read values from
        :param from_time: Starting time to begin reading (included in output)
        :param to_time: To to stop reading at (not included in output)
        :param kwargs: additional argument list, to be defined by subclasses
        :return: Pandas DataFrame containing the data read from the connection
        """
        pass

    def subscribe_series(
        self,
        handler: SubscriptionHandler,
        time_interval: datetime,
        nodes: Nodes = None,
        interval: int = 1,
    ):
        """Continuously read time series data from the connection, starting at current time and going back read
        interval
        """
        pass
