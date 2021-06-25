""" Utility functions for connecting to the EnEffCo database and reading data.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Mapping, Optional

import pandas as pd
import requests
import tzlocal
from pytz import BaseTzInfo

from eta_utility import get_logger
from eta_utility.type_hints.custom_types import Node, Nodes, TimeStep

from .base_classes import BaseSeriesConnection, SubscriptionHandler

log = get_logger("connectors.eneffco")


class EnEffCoConnection(BaseSeriesConnection):
    """
    EnEffCoConnection is a class to download and upload multiple features from and to the EnEffCo database as
    timeseries.

    :param url: Url of the server with scheme (https://)
    :param usr: Username in EnEffco for login
    :param pwd: Password in EnEffco for login
    :param api_token: Token for API authentication
    :param nodes: Nodes to select in connection
    """

    API_PATH: str = "/API/v1.0"

    def __init__(self, url: str, usr: str, pwd: str, *, api_token: str, nodes: Optional[Nodes] = None) -> None:
        url = url + self.API_PATH
        self._api_token: str = api_token
        super().__init__(url, usr, pwd, nodes=nodes)

        self._node_ids: Optional[str] = None
        self._node_ids_raw: Optional[str] = None

        self._sub: Optional[asyncio.Task] = None
        self._subscription_nodes = set()
        self._subscription_open: bool = False
        self._local_tz: BaseTzInfo = tzlocal.get_localzone()

    @classmethod
    def from_node(cls, node: Node, *, usr: str, pwd: str, api_token: str) -> "EnEffCoConnection":
        """Initialize the connection object from an EnEffCo protocol node object

        :param node: Node to initialize from
        :param usr: Username for EnEffCo login
        :param pwd: Password for EnEffCo login
        :param api_token: Token for API authentication
        :return: EnEffCoConnection object
        """

        if node.protocol == "eneffco":
            return cls(node.url, usr, pwd, api_token=api_token, nodes=[node])

        else:
            raise ValueError(
                "Tried to initialize EnEffCoConnection from a node that does not specify eneffco as its"
                "protocol: {}.".format(node.name)
            )

    @staticmethod
    def round_time(time: datetime, interval: int) -> datetime:
        """Rounds time to full intervals in seconds (also supports decimals of seconds)"""
        intervals = time.timestamp() // interval
        return datetime.fromtimestamp(intervals * interval)

    def _round_timestamp(self, timestamp: datetime, interval: int = 1) -> datetime:
        """Helper method for rounding date time objects to specified interval in seconds.
        The method will also add local timezone information if not already present.

        :param datetime timestamp: Datetime object to be rounded
        :param interval: Interval in seconds to be rounded to
        :return: Rounded datetime object with timezone information
        """
        ts_store = self._assert_tz_awareness(timestamp)  # store previous information, include timezone information

        # Round timestamp
        intervals = timestamp.timestamp() // interval
        timestamp = datetime.fromtimestamp(intervals * interval)

        # Restore timezone information
        timestamp = timestamp.replace(tzinfo=ts_store.tzinfo)
        return timestamp

    def _assert_tz_awareness(self, timestamp: datetime) -> datetime:
        """Helper function to check if timestamp has timezone and if not assign local time zone.

        :param datetime timestamp: Datetime object to be rounded
        :return: Rounded datetime object with timezone information"""
        if timestamp.tzinfo is None:
            timestamp = self._local_tz.localize(timestamp)
        return timestamp

    def read(self, nodes: Optional[Nodes] = None) -> pd.DataFrame:
        """Download current value from the EnEffCo Database

        :param nodes: List of nodes to read values from
        :return: Pandas DataFrame containing the data read from the connection
        """
        # Make sure that latest value is read depending on base time of node in EnEffCo
        nodes = self._validate_nodes(nodes)
        values = pd.DataFrame()
        for node in nodes:
            request_url = "datapoint/{}/live".format(self.id_from_code(node.eneffco_code))
            response = self._raw_request("GET", request_url)
            response = response.json()

            data = pd.DataFrame(
                data=(r["Value"] for r in response),
                index=pd.to_datetime([r["From"] for r in response], utc=True, format="%Y-%m-%dT%H:%M:%SZ").tz_convert(
                    self._local_tz
                ),
                columns=[node.name],
                dtype="float64",
            )
            data.index.name = "Time (with timezone)"
            values = pd.concat([values, data], axis=1, sort=False)
        return values

    def write(
        self, values: Mapping[Node, Mapping[datetime, Any]], time_interval: timedelta = timedelta(seconds=1)
    ) -> None:
        """Writes some values to the EnEffCo Database

        :param values: Dictionary of nodes and data to write. {node: value}
        :param time_interval: Interval between datapoints (i.e. between "From" and "To" in EnEffCo Upload), default 1s
        """
        nodes = self._validate_nodes(values.keys())

        for node in nodes:
            request_url = "rawdatapoint/{}/value".format(self.id_from_code(node.eneffco_code, raw_datapoint=True))
            response = self._raw_request(
                "POST",
                request_url,
                data=self._prepare_raw_data(values[node], time_interval),
                headers={
                    "Content-Type": "application/json",
                    "cache-control": "no-cache",
                    "Postman-Token": self._api_token,
                },
                params={"comment": ""},
            )
            log.info(response.text)

    def _prepare_raw_data(self, data: Mapping[datetime, Any], time_interval: timedelta) -> str:
        """Change the input format into a compatible format with EnEffCo

        :param data: Data to write to node {time: value}. Could be a dictionary or a pandas Series.
        :param time_interval: Interval between datapoints (i.e. between "From" and "To" in EnEffCo Upload)

        :return upload_data: String from dictionary in the format for the upload to EnEffCo
        """

        if type(data) is Dict or isinstance(data, pd.Series):
            upload_data = {"Values": []}
            for time, val in data.items():
                time = self._assert_tz_awareness(time)
                time = pd.Timestamp(time).tz_convert("UTC")
                upload_data["Values"].append(
                    {
                        "Value": float(val),
                        "From": time.strftime("%Y-%m-%d %H:%M:%SZ"),
                        "To": (time + time_interval).strftime("%Y-%m-%d %H:%M:%SZ"),
                    }
                )

        else:
            raise ValueError("Unrecognized data format for EnEffCo upload. Provide dictionary or pandas series.")

        return str(upload_data)

    def read_info(self, nodes: Optional[Nodes] = None) -> pd.DataFrame:
        """Read additional datapoint information from Database.

        :param nodes: List of nodes to read values from
        :return: Pandas DataFrame containing the data read from the connection
        """

        nodes = self._validate_nodes(nodes)
        values = []

        for node in nodes:
            request_url = "datapoint/{}".format(self.id_from_code(node.eneffco_code))
            response = self._raw_request("GET", request_url)
            values.append(pd.Series(response.json(), name=node.name))

        return pd.concat(values, axis=1)

    def subscribe(self, handler: SubscriptionHandler, nodes: Optional[Nodes] = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This will return only the
        last available values.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param interval: interval for receiving new data. It it interpreted as seconds when given as an integer.
        :param nodes: identifiers for the nodes to subscribe to
        """
        self.subscribe_series(handler, 1, nodes, interval=interval, data_interval=interval)

    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Optional[Nodes] = None, interval: TimeStep = 1
    ) -> pd.DataFrame:
        """Download timeseries data from the EnEffCo Database

        :param nodes: List of nodes to read values from
        :param from_time: Starting time to begin reading (included in output)
        :param to_time: To to stop reading at (not included in output)
        :param interval: interval between time steps. It is interpreted as seconds if given as integer.
        :return: Pandas DataFrame containing the data read from the connection

        **Example - Download some EnEffCo-codes**::

            from eta_utility.connectors import Node, EnEffCoConnection
            from datetime import datetime

            nodes = Node.get_eneffco_nodes_from_codes(
                ["Namespace1.Code1", "Namespace2.Code2"]
            )
            connection = EnEffCoConnection.from_node(
                nodes[0], usr="username", pwd="pw", api_token="token"
            )
            from_time = datetime.fromisoformat("2019-01-01 00:00:00")
            to_time = datetime.fromisoformat("2019-01-02 00:00:00")
            data = connection.read_series(
                from_time, to_time, nodes=nodes, interval=900
            )
        """
        nodes = self._validate_nodes(nodes)
        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        values = pd.DataFrame()

        for node in nodes:
            request_url = "datapoint/{}/value?from={}&to={}&timeInterval={}&includeNanValues=True".format(
                self.id_from_code(node.eneffco_code),
                self.timestr_from_datetime(from_time),
                self.timestr_from_datetime(to_time),
                str(int(interval.total_seconds())),
            )

            response = self._raw_request("GET", request_url)
            response = response.json()

            data = pd.DataFrame(
                data=(r["Value"] for r in response),
                index=pd.to_datetime([r["From"] for r in response], utc=True, format="%Y-%m-%dT%H:%M:%SZ").tz_convert(
                    self._local_tz
                ),
                columns=[node.name],
                dtype="float64",
            )
            data.index.name = "Time (with timezone)"
            values = pd.concat([values, data], axis=1, sort=False)
        return values

    def subscribe_series(
        self,
        handler: SubscriptionHandler,
        req_interval: TimeStep,
        offset: TimeStep = None,
        nodes: Optional[Nodes] = None,
        interval: TimeStep = 1,
        data_interval: TimeStep = 1,
    ) -> None:
        """Subscribe to nodes and call handler when new data is available. This will always return a series of values.
        If nodes with different intervals should be subscribed, multiple connection objects are needed.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs
        :param req_interval: Duration covered by requested data (time interval). Interpreted as seconds if given as int
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
                       Interpreted as seconds if given as int. Use negative values to go to past timestamps.
        :param data_interval: Time interval between values in returned data. Interpreted as seconds if given as int.
        :param interval: interval (between requests) for receiving new data.
                         It it interpreted as seconds when given as an integer.
        :param nodes: identifiers for the nodes to subscribe to
        """

        # todo umbenennen: req_interval = data_interval, data_interval = resolution; take these attributes from node;
        #  same in subscribe
        nodes = self._validate_nodes(nodes)

        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)
        req_interval = req_interval if isinstance(req_interval, timedelta) else timedelta(seconds=req_interval)
        if offset is None:
            offset = -req_interval
        else:
            offset = offset if isinstance(offset, timedelta) else timedelta(seconds=offset)
        data_interval = data_interval if isinstance(data_interval, timedelta) else timedelta(seconds=data_interval)

        self._subscription_nodes.update(nodes)

        if self._subscription_open:
            # Adding nodes to subscription is enough to include them in the query. Do not start an additional loop
            # if one already exists
            return

        self._subscription_open = True
        loop = asyncio.get_event_loop()
        self._sub = loop.create_task(
            self._subscription_loop(
                handler,
                int(interval.total_seconds()),
                req_interval,
                offset,
                data_interval,
            )
        )

    def close_sub(self) -> None:
        try:
            self._sub.cancel()
            self._subscription_open = False
        except Exception:
            pass

    async def _subscription_loop(
        self,
        handler: SubscriptionHandler,
        interval: TimeStep,
        req_interval: TimeStep,
        offset,
        data_interval: TimeStep,
    ) -> None:
        """The subscription loop handles requesting data from the server in the specified interval

        :param handler: Handler object with a push function to receive data
        :param interval: Interval for requesting data in seconds
        :param req_interval: Duration covered by the requested data
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
                       Use negative values to go to past timestamps.
        :param data_interval: Interval between data points
        """

        try:
            while self._subscription_open:
                from_time = datetime.now() + offset
                to_time = from_time + req_interval

                for node in self._subscription_nodes:
                    value = self.read_series(from_time, to_time, node, interval=data_interval)
                    handler.push(node, value[node.name], value[node.name].index)

                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            pass

    def id_from_code(self, code: str, raw_datapoint: bool = False) -> str:
        """
        Function to get the raw EnEffCo ID corresponding to a specific (raw) datapoint

        :param code: exact EnEffCo code
        :param raw_datapoint: returns raw datapoint id
        """

        # Only build lists of ids if they are not available yet
        if self._node_ids is None:
            response = self._raw_request("GET", "/datapoint")
            self._node_ids = pd.DataFrame(data=response.json())

        if self._node_ids_raw is None:
            response = self._raw_request("GET", "/rawdatapoint")
            self._node_ids_raw = pd.DataFrame(data=response.json())

        if raw_datapoint:
            return self._node_ids_raw.loc[self._node_ids_raw["Code"] == code, "Id"].values.item()
        else:
            return self._node_ids.loc[self._node_ids["Code"] == code, "Id"].values.item()

    def timestr_from_datetime(self, dt: datetime) -> str:
        """Create an EnEffCo compatible time string

        :param dt: datetime object to convert to string
        :return: EnEffCo compatible time string
        """

        return dt.isoformat(sep="T", timespec="seconds").replace(":", "%3A")

    def _raw_request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Perform EnEffCo request and handle possibly resulting errors.

        :param method: HTTP request method
        :param endpoint: endpoint for the request (server uri is added automatically
        :param kwargs: Additional arguments for the request.
        """

        response = requests.request(method, self.url + "/" + str(endpoint), auth=(self.usr, self.pwd), **kwargs)

        # Check for request errors
        status_code = response.status_code
        if status_code == 400:
            raise ConnectionError(f"EnEffCo Error {status_code}: API is unavailable or insufficient user permissions.")
        elif status_code == 404:
            raise ConnectionError(
                "EnEffCo Error {}: Endpoint not found '{}'".format(status_code, self.url + str(endpoint))
            )
        elif status_code == 401:
            raise ConnectionError(f"EnEffCo Error {status_code}: Invalid login info")
        elif status_code == 500:
            raise ConnectionError(f"EnEffCo Error {status_code}: Server is unavailable")

        return response
