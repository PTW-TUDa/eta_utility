""" Base classes for the connectors

"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd


class BaseConnection(ABC):
    """Base class with a common interface for all connection objects

    :param str url: URL of the server to connect to.
    :param str usr: Username for login to server
    :param str pwd: Password for login to server
    :param nodes: List of nodes to select as a standard case
    :type nodes: List[Node], Set[Node]
    """

    def __init__(self, url, usr=None, pwd=None, *, nodes=None):
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
    def from_node(cls, node, **kwargs):
        """Initialize the object from a node with corresponding protocol

        :return: Initialized connection object
        :rtype: cls
        """
        pass

    @abstractmethod
    def read(self, nodes=None):
        """Read data from nodes

        :param nodes: List of nodes to read from
        :type nodes: List[Node]
        :return: Pandas DataFrame with resulting values
        :rtype: pd.DataFrame
        """
        pass

    @abstractmethod
    def write(self, values):
        """Write data to a list of nodes

        :param values: Dictionary of nodes and data to write. {node: value}
        :type values: Dict[Node, Any]
        """
        pass

    @abstractmethod
    def subscribe(self, handler, nodes=None, interval=1):
        """Subscribe to nodes and call handler when new data is available.

        :param nodes: identifiers for the nodes to subscribe to
        :type nodes: Set or List
        :param handler: function to be called upon receiving new values, must accept attributes: node, val
        :type handler: SubscriptionHandler
        :param interval: interval for receiving new data. Interpreted as seconds when given as integer.
        :type interval: int or timedelta
        """
        pass

    @abstractmethod
    def close_sub(self):
        """Close an open subscription. This should gracefully handle non-existant subscriptions."""
        pass

    @property
    def url(self):
        return self._url.geturl()

    def _validate_nodes(self, nodes):
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

    def __init__(self, url, usr=None, pwd=None, *, nodes=None):
        super().__init__(url, usr, pwd, nodes=nodes)

    @abstractmethod
    def read_series(self, from_time, to_time, nodes=None, **kwargs):
        """Read time series data from the connection, within a specified time interval (from_time until to_time).

        :param nodes: List of nodes to read values from
        :type nodes: List[Node]
        :param from_time: Starting time to begin reading (included in output)
        :type from_time: datetime.datetime
        :param to_time: To to stop reading at (not included in output)
        :type to_time: datetime.datetime
        :param kwargs: additional argument list, to be defined by subclasses
        :return: Pandas DataFrame containing the data read from the connection
        :rtype: pd.DataFrame
        """
        pass

    def subscribe_series(self, handler, time_interval, nodes=None, interval=1):
        """Continuously read time series data from the connection, starting at current time and going back read
        interval

        :return:
        :rtype:
        """
        pass


class SubscriptionHandler(ABC):
    """Subscription handlers do stuff to subscribed data points after they are received. Every handler must have a
    push method which can be called when data is received.

    :param write_interval: Interval for writing data to csv file
    :type write_interval: float or int
    """

    def __init__(self, write_interval=1):
        self._write_interval = write_interval

    def _round_timestamp(self, timestamp):
        """Helper method for rounding datime objects to the specified write interval

        :param datetime timestamp: Datetime object to be rounded
        :return: Rounded datetime object without timezone information
        :rtype: datetime
        """
        # if 60 <= self._write_interval < 3600:
        #     intervals = timestamp.minute // (self._write_interval / 60)
        #     timestamp = timestamp.replace(minute = int(intervals * (self._write_interval / 60)), second = 0,
        #                                   microsecond = 0, tzinfo = None)
        # elif 1 <= self._write_interval < 60:
        #     intervals = timestamp.second // self._write_interval
        #     timestamp = timestamp.replace(second = intervals * self._write_interval, microsecond = 0, tzinfo = None)
        # elif 0 < self._write_interval < 1:
        #     intervals = timestamp.microsecond // (self._write_interval * 1000000)
        #     timestamp = timestamp.replace(microsecond = intervals * self._write_interval * 1000000, tzinfo = None)

        intervals = timestamp.timestamp() // self._write_interval
        return datetime.fromtimestamp(intervals * self._write_interval)

    def _convert_series(self, value, timestamp):
        """Helper function to convert a value, timestamp pair in which value is a Series or list to a Series with
        datetime index according to the given timestamp(s).

        :param value: Series of values. There must be corresponding timestamps for each value.
        :type value: pd.Series[Any] or list-like[Any]
        :param timestamp: DatetimeIndex of the provided values. Alternatively an integer/timedelta can be provided to
                          determine the interval between data points. Use negative numbers to describe past data.
                          Integers are interpreted as seconds. If value is a pd.Series and has a pd.DatetimeIndex,
                          timestamp can be None.
        :type timestamp: pd.DatetimeIndex or int or timedelta or None
        :return: pd.Series with corresponding DatetimeIndex
        :rtype: pd.Series
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
    def push(self, node, value, timestamp=None):
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to
        :type node: Node
        :param value: Value of the data
        :type value: Any
        :param timestamp: Timestamp of receiving the data
        :type timestamp: datetime
        """
        pass
