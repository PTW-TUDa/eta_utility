""" Utility functions for connecting to the EnEffCo database and reading data.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Mapping

import numpy as np
import pandas as pd
import requests

from eta_utility import get_logger
from eta_utility.connectors.node import Node, NodeEnEffCo

if TYPE_CHECKING:
    from typing import Any, Sequence
    from eta_utility.type_hints import AnyNode, Nodes, TimeStep

from .base_classes import BaseSeriesConnection, SubscriptionHandler

log = get_logger("connectors.eneffco")


class EnEffCoConnection(BaseSeriesConnection):
    """
    EnEffCoConnection is a class to download and upload multiple features from and to the EnEffCo database as
    timeseries.

    :param url: URL of the server with scheme (https://).
    :param usr: Username in EnEffco for login.
    :param pwd: Password in EnEffco for login.
    :param api_token: Token for API authentication.
    :param nodes: Nodes to select in connection.
    """

    _PROTOCOL = "eneffco"
    API_PATH: str = "/API/v1.0"

    def __init__(self, url: str, usr: str, pwd: str, *, api_token: str, nodes: Nodes | None = None) -> None:
        url = url + self.API_PATH
        self._api_token: str = api_token
        super().__init__(url, usr, pwd, nodes=nodes)

        self._node_ids: pd.DataFrame | None = None
        self._node_ids_raw: pd.DataFrame | None = None

        self._sub: asyncio.Task | None = None
        self._subscription_nodes: set[NodeEnEffCo] = set()
        self._subscription_open: bool = False

    @classmethod
    def from_node(cls, node: AnyNode, **kwargs: Any) -> EnEffCoConnection:
        """Initialize the connection object from an EnEffCo protocol node object

        :param node: Node to initialize from.
        :param usr: Username for EnEffCo login (default: None, uses usr from Node).
        :param pwd: Password for EnEffCo login.
        :param api_token: Token for API authentication.
        :return: EnEffCoConnection object.
        """
        if "api_token" not in kwargs:
            raise AttributeError("Keyword parameter 'api_token' is missing.")
        api_token = kwargs["api_token"]

        usr = node.usr if node.usr is not None else kwargs.get("usr", None)
        pwd = node.pwd if node.pwd is not None else kwargs.get("pwd", None)
        if usr is None or pwd is None:
            raise ValueError(
                "The node used for instantiation does not contain a username or password. "
                "Please provide 'usr' and 'pwd' keyword arguments."
            )

        if node.protocol == "eneffco" and isinstance(node, NodeEnEffCo):
            return cls(node.url, usr, pwd, api_token=api_token, nodes=[node])
        else:
            raise ValueError(
                "Tried to initialize EnEffCoConnection from a node that does not specify eneffco as its"
                "protocol: {}.".format(node.name)
            )

    @classmethod
    def from_ids(cls, ids: Sequence[str], url: str, usr: str, pwd: str, api_token: str) -> EnEffCoConnection:
        """Initialize the connection object from an EnEffCo protocol through the node IDs

        :param ids: Identification of the Node.
        :param url: URL for EnEffco connection.
        :param usr: Username for EnEffCo login.
        :param pwd: Password for EnEffCo login.
        :param api_token: Token for API authentication.
        :return: EnEffCoConnection object.
        """
        nodes = [Node(name=name, url=url, protocol="eneffco", eneffco_code=name) for name in ids]
        return cls(url=url, usr=usr, pwd=pwd, api_token=api_token, nodes=nodes)

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Download current value from the EnEffCo Database

        :param nodes: List of nodes to read values from.
        :return: pandas.DataFrame containing the data read from the connection.
        """
        nodes = self._validate_nodes(nodes)
        base_time = 1  # seconds
        the_time = self._round_timestamp(datetime.now(), base_time).replace(tzinfo=None)
        value = self.read_series(the_time - timedelta(seconds=base_time), the_time, nodes, base_time)
        return value

    def write(
        self, values: Mapping[AnyNode, Any] | pd.Series[datetime, Any], time_interval: timedelta | None = None
    ) -> None:
        """Writes some values to the EnEffCo Database

        :param values: Dictionary of nodes and data to write {node: value}.
        :param time_interval: Interval between datapoints (i.e. between "From" and "To" in EnEffCo Upload) (default 1s).
        """
        nodes = self._validate_nodes(values.keys())

        if time_interval is None:
            time_interval = timedelta(seconds=1)

        for node in nodes:
            request_url = f"rawdatapoint/{self.id_from_code(node.eneffco_code, raw_datapoint=True)}/value"
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

    def _prepare_raw_data(
        self, data: Mapping[datetime, Any] | pd.Series[datetime, Any], time_interval: timedelta
    ) -> str:
        """Change the input format into a compatible format with EnEffCo and filter NaN values.

        :param data: Data to write to node {time: value}. Could be a dictionary or a pandas Series.
        :param time_interval: Interval between datapoints (i.e. between "From" and "To" in EnEffCo Upload).

        :return upload_data: String from dictionary in the format for the upload to EnEffCo.
        """

        if isinstance(data, dict) or isinstance(data, pd.Series):
            upload_data: dict[str, list[Any]] = {"Values": []}
            for time, val in data.items():
                # Only write values if they are not nan
                if not np.isnan(val):
                    time = self._assert_tz_awareness(time).astimezone(timezone.utc)
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

    def read_info(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Read additional datapoint information from Database.

        :param nodes: List of nodes to read values from.
        :return: pandas.DataFrame containing the data read from the connection.
        """

        nodes = self._validate_nodes(nodes)
        values = []

        for node in nodes:
            request_url = f"datapoint/{self.id_from_code(node.eneffco_code)}"
            response = self._raw_request("GET", request_url)
            values.append(pd.Series(response.json(), name=node.name))

        return pd.concat(values, axis=1)

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This will return only the
        last available values.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs.
        :param interval: Interval for receiving new data. It is interpreted as seconds when given as an integer.
        :param nodes: Identifiers for the nodes to subscribe to.
        """
        self.subscribe_series(handler=handler, req_interval=1, nodes=nodes, interval=interval, data_interval=interval)

    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Nodes | None = None, interval: TimeStep = 1, **kwargs: Any
    ) -> pd.DataFrame:
        """Download timeseries data from the EnEffCo Database

        :param nodes: List of nodes to read values from.
        :param from_time: Starting time to begin reading (included in output).
        :param to_time: Time to stop reading at (not included in output).
        :param interval: Interval between time steps. It is interpreted as seconds if given as integer.
        :param kwargs: Other parameters (ignored by this connector).
        :return: Pandas DataFrame containing the data read from the connection.
        """
        _interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        nodes = self._validate_nodes(nodes)

        def read_node(node: NodeEnEffCo) -> pd.DataFrame:
            request_url = "datapoint/{}/value?from={}&to={}&timeInterval={}&includeNanValues=True".format(
                self.id_from_code(node.eneffco_code),
                self.timestr_from_datetime(from_time),
                self.timestr_from_datetime(to_time),
                str(int(_interval.total_seconds())),
            )

            response = self._raw_request("GET", request_url).json()

            data = pd.DataFrame(
                data=(r["Value"] for r in response),
                index=pd.to_datetime([r["From"] for r in response], utc=True, format="%Y-%m-%dT%H:%M:%SZ").tz_convert(
                    self._local_tz
                ),
                columns=[node.name],
                dtype="float64",
            )
            data.index.name = "Time (with timezone)"
            return data

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(read_node, nodes)

        values = pd.concat(results, axis=1, sort=False)
        return values

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
        :param req_interval: Duration covered by requested data (time interval). Interpreted as seconds if given as int.
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
                       Interpreted as seconds if given as int. Use negative values to go to past timestamps.
        :param data_interval: Time interval between values in returned data. Interpreted as seconds if given as int.
        :param interval: interval (between requests) for receiving new data.
                         It is interpreted as seconds when given as an integer.
        :param nodes: Identifiers for the nodes to subscribe to.
        :param kwargs: Other, ignored parameters.
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
        """Close an open subscription."""
        self._subscription_open = False

        if self.exc:
            raise self.exc

        try:
            self._sub.cancel()  # type: ignore
        except Exception:
            pass

    async def _subscription_loop(
        self,
        handler: SubscriptionHandler,
        interval: TimeStep,
        req_interval: TimeStep,
        offset: TimeStep,
        data_interval: TimeStep,
    ) -> None:
        """The subscription loop handles requesting data from the server in the specified interval

        :param handler: Handler object with a push function to receive data.
        :param interval: Interval for requesting data in seconds.
        :param req_interval: Duration covered by the requested data.
        :param offset: Offset from datetime.now from which to start requesting data (time interval).
                       Use negative values to go to past timestamps.
        :param data_interval: Interval between data points.
        """
        interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)
        req_interval = req_interval if isinstance(req_interval, timedelta) else timedelta(seconds=req_interval)
        data_interval = data_interval if isinstance(data_interval, timedelta) else timedelta(seconds=data_interval)
        offset = offset if isinstance(offset, timedelta) else timedelta(seconds=offset)

        try:
            while self._subscription_open:
                from_time = datetime.now() + offset
                to_time = from_time + req_interval

                values = self.read_series(from_time, to_time, self._subscription_nodes, interval=data_interval)

                for node in self._subscription_nodes:
                    handler.push(node, values[node.name])

                await asyncio.sleep(interval.total_seconds())
        except BaseException as e:
            self.exc = e

    def id_from_code(self, code: str, raw_datapoint: bool = False) -> str:
        """
        Function to get the raw EnEffCo ID corresponding to a specific (raw) datapoint

        :param code: Exact EnEffCo code.
        :param raw_datapoint: Returns raw datapoint ID.
        """

        # Only build lists of IDs if they are not available yet
        if self._node_ids is None:
            response = self._raw_request("GET", "/datapoint")
            self._node_ids = pd.DataFrame(data=response.json())

        if self._node_ids_raw is None:
            response = self._raw_request("GET", "/rawdatapoint")
            self._node_ids_raw = pd.DataFrame(data=response.json())

        if raw_datapoint:
            if len(self._node_ids_raw.loc[self._node_ids_raw["Code"] == code, "Id"]) > 0:
                return self._node_ids_raw.loc[self._node_ids_raw["Code"] == code, "Id"].values.item()
            else:
                raise ValueError(f"Code {code} does not exist on server {self.url}.")
        else:
            if len(self._node_ids.loc[self._node_ids["Code"] == code, "Id"]) > 0:
                return self._node_ids.loc[self._node_ids["Code"] == code, "Id"].values.item()
            else:
                raise ValueError(f"Code {code} does not exist on server {self.url}.")

    def timestr_from_datetime(self, dt: datetime) -> str:
        """Create an EnEffCo compatible time string.

        :param dt: Datetime object to convert to string.
        :return: EnEffCo compatible time string.
        """

        return dt.isoformat(sep="T", timespec="seconds").replace(":", "%3A").replace("+", "%2B")

    def _raw_request(self, method: str, endpoint: str, **kwargs: Any) -> requests.Response:
        """Perform EnEffCo request and handle possibly resulting errors.

        :param method: HTTP request method.
        :param endpoint: Endpoint for the request (server URI is added automatically).
        :param kwargs: Additional arguments for the request.
        """
        response = requests.request(
            method, self.url + "/" + str(endpoint), auth=requests.auth.HTTPBasicAuth(self.usr, self.pwd), **kwargs
        )

        # Check for request errors
        if response.status_code not in [200, 204]:  # Status 200 for GET requests, 204 for POST requests
            error = f"EnEffco Error {response.status_code}"
            if hasattr(response, "json") and "Message" in response.json():
                error = f"{error}: {response.json()['Message']}"
            elif response.status_code == 401:
                error = f"{error}: Access Forbidden, Invalid login info"
            elif response.status_code == 404:
                error = f"{error}: Endpoint not found '{self.url + str(endpoint)}'"
            elif response.status_code == 500:
                error = f"{error}: Server is unavailable"

            raise ConnectionError(error)

        return response

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeEnEffCo]:  # type: ignore
        vnodes = super()._validate_nodes(nodes)
        _nodes = set()
        for node in vnodes:
            if isinstance(node, NodeEnEffCo):
                _nodes.add(node)

        return _nodes
