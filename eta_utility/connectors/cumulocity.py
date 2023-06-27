from __future__ import annotations

import asyncio
import base64
import concurrent.futures
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Mapping

import pandas as pd
import requests

from eta_utility import get_logger
from eta_utility.connectors.node import NodeCumulocity

if TYPE_CHECKING:
    from typing import Any
    from eta_utility.type_hints import AnyNode, Nodes, TimeStep

from .base_classes import BaseSeriesConnection, SubscriptionHandler

log = get_logger("connectors.cumulocity")


class CumulocityConnection(BaseSeriesConnection, protocol="cumulocity"):
    """
    CumulocityConnection is a class to download and upload multiple features from and to the Cumulocity database as
    timeseries.

    :param url: URL of the server without scheme (https://).
    :param usr: Username in Cumulocity for login.
    :param pwd: Password in Cumulocity for login.
    :param tenant: Cumulocity tenant.
    :param nodes: Nodes to select in connection.

    """

    def __init__(self, url: str, usr: str | None, pwd: str | None, *, tenant: str, nodes: Nodes | None = None) -> None:
        self._tenant = tenant

        super().__init__(url, usr, pwd, nodes=nodes)

        if self.usr is None:
            raise ValueError("Username must be provided for the Cumulocity connector.")
        if self.pwd is None:
            raise ValueError("Password must be provided for the Cumulocity connector.")

        self._node_ids: pd.DataFrame | None = None
        self._node_ids_raw: pd.DataFrame | None = None

        self._sub: asyncio.Task | None = None
        self._subscription_nodes: set[NodeCumulocity] = set()
        self._subscription_open: bool = False

    @classmethod
    def _from_node(
        cls, node: AnyNode, usr: str | None = None, pwd: str | None = None, **kwargs: Any
    ) -> CumulocityConnection:
        """Initialize the connection object from an Cumulocity protocol node object

        :param node: Node to initialize from.
        :param usr: Username for Cumulocity login.
        :param pwd: Password for Cumulocity login.
        :param kwargs: Keyword arguments for API authentication, where "tenant" is required
        :return: CumulocityConnection object.
        """

        if "tenant" not in kwargs:
            raise AttributeError("Keyword parameter 'tenant' is missing.")
        tenant = kwargs["tenant"]

        if node.protocol == "cumulocity" and isinstance(node, NodeCumulocity):
            return cls(node.url, usr, pwd, tenant=tenant, nodes=[node])
        else:
            raise ValueError(
                "Tried to initialize CumulocityConnection from a node that does not specify cumulocity as its"
                "protocol: {}.".format(node.name)
            )

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Download current value from the Cumulocity Database

        :param nodes: List of nodes to read values from.
        :return: pandas.DataFrame containing the data read from the connection.
        """
        nodes = self._validate_nodes(nodes)
        base_time = 1  # minutes
        the_time = datetime.now()
        value = self.read_series(the_time - timedelta(minutes=base_time), the_time, nodes, base_time)
        return value[-1:]

    def write(
        self, values: Mapping[AnyNode, Any] | pd.Series[datetime, Any], time_interval: timedelta | None = None
    ) -> None:
        raise NotImplementedError("Not implemented yet.")

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """Subscribe to nodes and call handler when new data is available. This will return only the
        last available values.

        :param handler: SubscriptionHandler object with a push method that accepts node, value pairs.
        :param interval: Interval for receiving new data. It is interpreted as seconds when given as an integer.
        :param nodes: Identifiers for the nodes to subscribe to.
        """
        self.subscribe_series(handler=handler, req_interval=1, nodes=nodes, interval=interval, data_interval=interval)

    def read_series(
        self,
        from_time: datetime,
        to_time: datetime,
        nodes: Nodes | None = None,
        interval: TimeStep | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Download timeseries data from the Cumulocity Database

        :param nodes: List of nodes to read values from.
        :param from_time: Starting time to begin reading.
        :param to_time: Time to stop reading at.
        :param interval: Interval between time steps.
                        It is interpreted as seconds if given as integer (ignored by this connector).
        :param kwargs: Other parameters (ignored by this connector).
        :return: Pandas DataFrame containing the data read from the connection.
        """

        # get correct utc time for cumulocity
        from_time = from_time.astimezone(timezone.utc).replace(tzinfo=None)
        to_time = to_time.astimezone(timezone.utc).replace(tzinfo=None)
        nodes = self._validate_nodes(nodes)

        def read_node(node: NodeCumulocity) -> pd.DataFrame:
            request_url = "{}?dateFrom={}&dateTo={}&source={}&valueFragmentSeries={}&pageSize=2000".format(
                node.url,
                self.timestr_from_datetime(from_time),
                self.timestr_from_datetime(to_time),
                node.measurement_id,
                node.value_fragment_series,
            )

            headers = self.get_auth_header(node)

            data_list = []

            while True:
                response = self._raw_request("GET", request_url, headers).json()

                data_tmp = pd.DataFrame(
                    data=(r[r["type"]][node.value_fragment_series]["value"] for r in response["measurements"]),
                    index=pd.to_datetime(
                        [r["time"] for r in response["measurements"]], utc=True, format="%Y-%m-%dT%H:%M:%S.%fZ"
                    ).tz_convert(self._local_tz),
                    columns=[node.name],
                    dtype="float64",
                )
                data_list.append(data_tmp)
                if data_tmp.empty:
                    data = pd.concat(data_list)
                    break
                else:
                    request_url = response["next"]

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
            from_time = datetime.now() + offset
            while self._subscription_open:
                to_time = datetime.now() + offset + req_interval

                values = self.read_series(from_time, to_time, self._subscription_nodes, interval=data_interval)

                for node in self._subscription_nodes:
                    handler.push(node, values[node.name])

                from_time = to_time

                await asyncio.sleep(interval.total_seconds())
        except BaseException as e:
            self.exc = e

    def timestr_from_datetime(self, dt: datetime) -> str:
        """Create an Cumulocity compatible time string.

        :param dt: Datetime object to convert to string.
        :return: Cumulocity compatible time string.
        """

        return dt.isoformat() + "Z"

    def _raw_request(self, method: str, endpoint: str, headers: dict, **kwargs: Any) -> requests.Response:
        """Perform Cumulocity request and handle possibly resulting errors.

        :param method: HTTP request method.
        :param endpoint: Endpoint for the request (server URI is added automatically).
        :param kwargs: Additional arguments for the request.
        """

        response = requests.request(method, endpoint, headers=headers, verify=False, **kwargs)

        # Check for request errors
        if response.status_code not in [200, 204]:  # Status 200 for GET requests, 204 for POST requests
            error = f"Cumulocity Error {response.status_code}"
            if hasattr(response, "json") and "Message" in response.json():
                error = f"{error}: {response.json()['Message']}"
            elif response.status_code == 401:
                error = f"{error}: Access Forbidden, Invalid login info"
            elif response.status_code == 404:
                error = f"{error}: Endpoint not found '{str(endpoint)}'"
            elif response.status_code == 500:
                error = f"{error}: Server is unavailable"

            raise ConnectionError(error)

        return response

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeCumulocity]:  # type: ignore
        vnodes = super()._validate_nodes(nodes)
        _nodes = set()
        for node in vnodes:
            if isinstance(node, NodeCumulocity):
                _nodes.add(node)

        return _nodes

    def get_auth_header(self, node: NodeCumulocity) -> dict:
        auth_header = str(base64.b64encode(bytes(f"{self._tenant}/{self.usr}:{self.pwd}", encoding="utf-8")), "utf-8")
        headers = {"Authorization": f"Basic {auth_header}"}
        return headers
