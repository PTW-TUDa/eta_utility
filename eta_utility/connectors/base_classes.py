""" Base classes for the connectors

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Iterable

import pandas as pd
from dateutil import tz

from eta_utility import url_parse
from eta_utility.util import ensure_timezone, round_timestamp

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence
    from urllib.parse import ParseResult

    from eta_utility.type_hints import AnyNode, Nodes, TimeStep


class SubscriptionHandler(ABC):
    """Subscription handlers do stuff to subscribed data points after they are received. Every handler must have a
    push method which can be called when data is received.

    :param write_interval: Interval for writing data to csv file.
    """

    def __init__(self, write_interval: TimeStep = 1) -> None:
        self._write_interval: float = (
            write_interval.total_seconds() if isinstance(write_interval, timedelta) else write_interval
        )
        self._local_tz = tz.tzlocal()
        # Method to round a datetime timestamp by the SubscriptionHandler._write_interval interval in seconds
        #: :py:func:`eta_utility.util.round_timestamp`
        self._round_timestamp = lambda dt: round_timestamp(dt, self._write_interval)
        #: Method to ensure timezone by assigning local timezone
        #: :py:func:`eta_utility.util.ensure_timezone`
        self._assert_tz_awareness = ensure_timezone

    def _convert_series(self, value: pd.Series | Sequence[Any], timestamp: pd.DatetimeIndex | TimeStep) -> pd.Series:
        """Helper function to convert a value, timestamp pair in which value is a Series or list to a Series with
        datetime index according to the given timestamp(s).

        :param value: Series of values. There must be corresponding timestamps for each value.
        :param timestamp: DatetimeIndex of the provided values. Alternatively an integer/timedelta can be provided to
                          determine the interval between data points. Use negative numbers to describe past data.
                          Integers are interpreted as seconds. If value is a pd.Series and has a pd.DatetimeIndex,
                          timestamp can be None.
        :return: pandas.Series with corresponding DatetimeIndex.
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
            # If value is multidimensional, an Exception will be raised by pandas.

        # Round index to self._write_interval
        # todo resample instead of rounding the index, but this could not work if intervals between data points are of
        # different size
        value.index = value.index.round(str(self._write_interval) + "S")

        return value

    @abstractmethod
    def push(self, node: AnyNode, value: Any, timestamp: datetime | None = None) -> None:
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to.
        :param value: Value of the data.
        :param timestamp: Timestamp of receiving the data.
        """
        pass


class BaseConnection(ABC):
    """Base class with a common interface for all connection objects

    The URL may contain the username and password (schema://username:password@hostname:port/path). In this case, the
    parameters usr and pwd are not required. The keyword parameters of the function will take precedence over username
    and password configured in the url.

    :param url: URL of the server to connect to.
    :param usr: Username for login to server.
    :param pwd: Password for login to server.
    :param nodes: List of nodes to select as a standard case.
    """

    #: Protocol of the connection. Value can be used to check if nodes correspond to the connection
    _PROTOCOL = ""

    def __init__(self, url: str, usr: str | None = None, pwd: str | None = None, *, nodes: Nodes | None = None) -> None:
        #: URL of the server to connect to
        self._url: ParseResult
        #: Username fot login to server
        self.usr: str | None
        #: Password for login to server
        self.pwd: str | None
        self._url, self.usr, self.pwd = url_parse(url)

        if nodes is not None:
            #: Preselected nodes which will be used for reading and writing, if no other nodes are specified
            self.selected_nodes = self._validate_nodes(nodes)
        else:
            self.selected_nodes = set()

        # Get username and password either from the arguments, from the parsed URL string or from a Node object
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
        #: :py:func:`eta_utility.util.round_timestamp`
        self._round_timestamp = round_timestamp
        #: :py:func:`eta_utility.util.ensure_timezone`
        self._assert_tz_awareness = ensure_timezone

        self.exc: BaseException | None = None

    @classmethod
    @abstractmethod
    def from_node(cls, node: AnyNode, **kwargs: Any) -> BaseConnection:
        """Initialize the object from a node with corresponding protocol

        :return: Initialized connection object.
        """
        pass

    @abstractmethod
    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Read data from nodes

        :param nodes: List of nodes to read from.
        :return: Pandas DataFrame with resulting values.
        """

        pass

    @abstractmethod
    def write(self, values: Mapping[AnyNode, Any]) -> None:
        """Write data to a list of nodes

        :param values: Dictionary of nodes and data to write {node: value}.
        """
        pass

    @abstractmethod
    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: Identifiers for the nodes to subscribe to.
        :param handler: Function to be called upon receiving new values, must accept attributes: node, val.
        :param interval: Interval for receiving new data. Interpreted as seconds when given as integer.
        """
        pass

    @abstractmethod
    def close_sub(self) -> None:
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        pass

    @property
    def url(self) -> str:
        return self._url.geturl()

    def _validate_nodes(self, nodes: Nodes | None) -> set[AnyNode]:
        """Make sure that nodes are a Set of nodes and that all nodes correspond to the protocol and url
        of the connection.

        :param nodes: Sequence of Node objects to validate.
        :return: Set of valid Node objects for this connection.
        """
        if nodes is None:
            _nodes = self.selected_nodes
        else:
            if not isinstance(nodes, Iterable):
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


class BaseSeriesConnection(BaseConnection, ABC):
    """Connection object for protocols with the ability to provide access to timeseries data.

    :param url: URL of the server to connect to.
    """

    def __init__(self, url: str, usr: str | None = None, pwd: str | None = None, *, nodes: Nodes | None = None) -> None:
        super().__init__(url, usr, pwd, nodes=nodes)

    @abstractmethod
    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Nodes | None = None, interval: TimeStep = 1, **kwargs: Any
    ) -> pd.DataFrame:
        """Read time series data from the connection, within a specified time interval (from_time until to_time).

        :param nodes: List of nodes to read values from.
        :param from_time: Starting time to begin reading (included in output).
        :param to_time: Time to stop reading at (not included in output).
        :param interval: interval between time steps. It is interpreted as seconds if given as integer.
        :param kwargs: additional argument list, to be defined by subclasses.
        :return: pandas.DataFrame containing the data read from the connection.
        """
        pass

    def subscribe_series(
        self,
        handler: SubscriptionHandler,
        req_interval: TimeStep,
        offset: TimeStep | None = None,
        nodes: Nodes | None = None,
        interval: TimeStep = 1,
        data_interval: TimeStep = 1,
        **kwargs: Any,
    ) -> None:
        """Subscribe to nodes and call handler when new data is available. This will always return a series of values.
        If nodes with different intervals should be subscribed, multiple connection objects are needed.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs.
        :param req_interval: Duration covered by requested data (time interval). Interpreted as seconds if given as int
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
            Interpreted as seconds if given as int. Use negative values to go to past timestamps.
        :param data_interval: Time interval between values in returned data. Interpreted as seconds if given as int.
        :param interval: interval (between requests) for receiving new data. It is interpreted as seconds
            when given as an integer.
        :param nodes: Identifiers for the nodes to subscribe to.
        :param kwargs: Any additional arguments required by subclasses.
        """
        pass
