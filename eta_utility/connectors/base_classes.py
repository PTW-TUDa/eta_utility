""" Base classes for the connectors

"""
import math
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional, Sequence, Set, Union
from urllib.parse import ParseResult

import pandas as pd
from dateutil import tz

from eta_utility import url_parse
from eta_utility.type_hints import Node, Nodes, TimeStep


class SubscriptionHandler(ABC):
    """Subscription handlers do stuff to subscribed data points after they are received. Every handler must have a
    push method which can be called when data is received.

    :param write_interval: Interval for writing data to csv file
    """

    def __init__(self, write_interval: Union[float, int] = 1) -> None:
        self._write_interval: Union[float, int] = write_interval
        self._local_tz = tz.tzlocal()

    def _assert_tz_awareness(self, timestamp: datetime) -> datetime:
        """Helper function to check if timestamp has timezone and if not assign local time zone.

        :param datetime timestamp: Datetime object to be rounded
        :return: Rounded datetime object with timezone information"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self._local_tz)
        return timestamp

    def _round_timestamp(self, timestamp: datetime) -> datetime:
        """Helper method for rounding date time objects to the specified write interval.
        The method will also add local timezone information if not already present.

        :param datetime timestamp: Datetime object to be rounded
        :return: Rounded datetime object with timezone information
        """
        ts_store = self._assert_tz_awareness(timestamp)  # store previous information, include timezone information

        # Round timestamp
        intervals = math.ceil(ts_store.timestamp() / self._write_interval) * self._write_interval
        timestamp = datetime.fromtimestamp(intervals)

        # Restore timezone information
        timestamp = timestamp.replace(tzinfo=ts_store.tzinfo).astimezone(self._local_tz)
        return timestamp

    def _convert_series(
        self, value: Union[pd.Series, Sequence[Any]], timestamp: Union[pd.DatetimeIndex, TimeStep]
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
        # todo resample instead of rounding the index, but this could not work if intervals between data points are of
        # different size
        value.index = value.index.round(str(self._write_interval) + "S")

        return value

    @abstractmethod
    def push(self, node: Node, value: Any, timestamp: Optional[datetime] = None) -> None:
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to
        :param value: Value of the data
        :param timestamp: Timestamp of receiving the data
        """
        pass


class BaseConnection(ABC):
    """Base class with a common interface for all connection objects

    The url may contain the username and password (schema://username:password@hostname:port/path). In this case, the
    parameters usr and pwd are not required. The keyword parameters of the function will take precedence over username
    and password configured in the url.

    :param url: URL of the server to connect to.
    :param usr: Username for login to server
    :param pwd: Password for login to server
    :param nodes: List of nodes to select as a standard case
    """

    #: Protocol of the connection. Value can be used to check if nodes correspond to the connection
    _PROTOCOL = ""

    def __init__(
        self, url: str, usr: Optional[str] = None, pwd: Optional[str] = None, *, nodes: Optional[Nodes] = None
    ) -> None:
        #: URL of the server to connect to
        self._url: ParseResult
        #: Username fot login to server
        self.usr: str
        #: Password for login to server
        self.pwd: str
        self._url, self.usr, self.pwd = url_parse(url)

        if nodes is not None:
            #: Preselected nodes which will be used for reading and writing, if no other nodes are specified
            self.selected_nodes = self._validate_nodes(nodes)
        else:
            self.selected_nodes = set()

        # Get username and password either from the arguments, from the parsed url string or from a Node object
        node = next(iter(self.selected_nodes)) if len(self.selected_nodes) > 0 else None
        if type(usr) is not str and usr is not None:
            raise TypeError("Username should be a string value.")
        elif usr is not None:
            self.usr = usr
        elif self.usr is not None:
            pass
        elif node is not None:
            self.usr = node.usr

        if type(pwd) is not str and pwd is not None:
            raise TypeError("Password should be a string value.")
        elif pwd is not None:
            self.pwd = pwd
        elif self.pwd is not None:
            pass
        elif node is not None:
            self.pwd = node.pwd

        #: Store local time zone
        self._local_tz = tz.tzlocal()

        self.exc: Optional[BaseException] = None

    @classmethod
    @abstractmethod
    def from_node(cls, node: Node, **kwargs: Any) -> "BaseConnection":
        """Initialize the object from a node with corresponding protocol

        :return: Initialized connection object
        """
        pass

    @abstractmethod
    def read(self, nodes: Optional[Nodes] = None) -> pd.DataFrame:
        """Read data from nodes

        :param nodes: List of nodes to read from
        :return: Pandas DataFrame with resulting values
        """

        pass

    @abstractmethod
    def write(self, values: Mapping[Node, Any]) -> None:
        """Write data to a list of nodes

        :param values: Dictionary of nodes and data to write. {node: value}
        """
        pass

    @abstractmethod
    def subscribe(self, handler: SubscriptionHandler, nodes: Optional[Nodes] = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :param handler: function to be called upon receiving new values, must accept attributes: node, val
        :param interval: interval for receiving new data. Interpreted as seconds when given as integer.
        """
        pass

    @abstractmethod
    def close_sub(self) -> None:
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        pass

    @property
    def url(self) -> str:
        return self._url.geturl()

    def _validate_nodes(self, nodes: Union[Nodes, None]) -> Set[Node]:
        """Make sure that nodes are a Set of nodes and that all nodes correspond to the protocol and url
        of the connection.

        :param nodes: Sequence of Node objects to validate
        :return: set of valid Node objects for this connection
        """
        if nodes is None:
            _nodes = self.selected_nodes
        else:
            if not hasattr(nodes, "__len__"):
                nodes = {nodes}

            # If not using preselected nodes from self.selected_nodes, check if nodes correspond to the connection
            _nodes = {
                node
                for node in nodes
                if node.protocol == self._PROTOCOL and node.url_parsed.hostname == self._url.hostname
            }

        # Make sure that some nodes remain after the checks and raise an error if there are none.
        if len(_nodes) == 0:
            raise ValueError(
                f"Some nodes to read from/write to must be specified. If nodes were specified, they do not "
                f"match the connection {self.url}"
            )

        return _nodes

    def _assert_tz_awareness(self, timestamp: datetime) -> datetime:
        """Helper function to check if timestamp has timezone and if not assign local time zone.

        :param datetime timestamp: Datetime object to be rounded
        :return: datetime object with timezone information"""
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=self._local_tz)
        return timestamp

    def _round_timestamp(self, timestamp: datetime, interval: float = 1) -> datetime:
        """Helper method for rounding date time objects to specified interval in seconds.
        The method will also add local timezone information if not already present.

        :param datetime timestamp: Datetime object to be rounded
        :param interval: Interval in seconds to be rounded to
        :return: Rounded datetime object with timezone information
        """
        ts_store = self._assert_tz_awareness(timestamp)  # store previous information, include timezone information

        # Round timestamp
        # Round timestamp
        intervals = math.ceil(ts_store.timestamp() / interval) * interval
        timestamp = datetime.fromtimestamp(intervals)

        # Restore timezone information
        timestamp = timestamp.replace(tzinfo=ts_store.tzinfo)
        return timestamp


class BaseSeriesConnection(BaseConnection, ABC):
    """Connection object for protocols with the ability to provide access to timeseries data.

    :param url: URL of the server to connect to
    """

    def __init__(
        self, url: str, usr: Optional[str] = None, pwd: Optional[str] = None, *, nodes: Optional[Nodes] = None
    ) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

    @abstractmethod
    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Optional[Nodes] = None, **kwargs: Any
    ) -> pd.DataFrame:
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
        nodes: Optional[Nodes] = None,
        interval: int = 1,
    ) -> None:
        """Continuously read time series data from the connection, starting at current time and going back read
        interval
        """
        pass
