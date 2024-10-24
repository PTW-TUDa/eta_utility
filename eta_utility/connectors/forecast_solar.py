"""
This module provides a read-only REST API connector to the forecast.solar API.

You can obtain an estimate of solar production for a specific location, defined by latitude and longitude,
and a specific plane orientation, defined by declination and azimuth, based on the installed module power.

Supported endpoints include: "Estimate", "Historic", and "Clearsky":

**Estimate Solar Production**
The `estimate` endpoint provides the forecast for today and the upcoming days, depending on the account model.

**Historic Solar Production**
The `historic` endpoint calculates the average solar production for a given day based on historical weather data,
excluding current weather conditions.

**Clear Sky Solar Production**
The `clearsky` endpoint calculates the theoretically possible solar production assuming no cloud cover.

For more information, visit the `forecast.solar API documentation <https://doc.forecast.solar/start>`_.

"""

# Consider Caching for at least 15 minutes and don't exceed rate limit
from __future__ import annotations

import concurrent.futures
import traceback
from collections.abc import Mapping
from datetime import datetime, timedelta
from logging import getLogger
from typing import TYPE_CHECKING

import pandas as pd
import requests
from requests_cache import DO_NOT_CACHE, CachedSession

from eta_utility.connectors.node import NodeForecastSolar
from eta_utility.timeseries import df_interpolate
from eta_utility.util import round_timestamp

from .base_classes import SeriesConnection, SubscriptionHandler

if TYPE_CHECKING:
    from types import TracebackType
    from typing import Any, ClassVar

    from typing_extensions import Self

    from eta_utility.type_hints import AnyNode, Nodes, TimeStep


log = getLogger(__name__)


class ForecastSolarConnection(SeriesConnection, protocol="forecast_solar"):
    """
    ForecastSolarConnection is a class to download and upload multiple features from and to the Forecast.Solar database
    as timeseries.

    :param url: URL of the server with scheme (https://).
    :param usr: Not needed for Forecast.Solar.
    :param pwd: Not needed for Forecast.Solar.
    :param api_key: Token for API authentication.
    :param nodes: Nodes to select in connection.
    """

    _baseurl: ClassVar[str] = "https://api.forecast.solar"
    _time_format: ClassVar[str] = "%Y-%m-%dT%H:%M:%SZ"
    _headers: ClassVar[dict[str, str]] = {"Content-Type": "application/json"}

    def __init__(
        self,
        url: str = _baseurl,
        *,
        api_key: str = "None",
        url_params: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
        nodes: Nodes | None = None,
    ) -> None:
        super().__init__(url, None, None, nodes=nodes)

        #: Url parameters for the forecast.Solar api
        self.url_params: dict[str, Any] | None = url_params
        #: Query parameters for the forecast.Solar api
        self.query_params: dict[str, Any] | None = query_params
        #: Key to use the Forecast.Solar api. If API key is none, only the public functions are usable.
        self._api_key: str = api_key
        #: Cached session to handle the requests
        self._session: CachedSession = CachedSession(
            cache_name="eta_utility/connectors/requests_cache/forecast_solar_cache",
            urls_expire_after={
                "https://api.forecast.solar*": 900,  # 15 minutes
                "*": DO_NOT_CACHE,  # Don't cache other URLs
            },
            allowable_codes=(200, 400, 401, 403),
            use_cache_dir=True,
        )

    @classmethod
    def _from_node(cls, node: AnyNode, **kwargs: Any) -> ForecastSolarConnection:
        """Initialize the connection object from an Forecast.Solar protocol node object

        :param node: Node to initialize from.
        :param kwargs: Keyword arguments for API authentication, where "api_key" is required
        :return: ForecastSolarConnection object.
        """
        api_key = kwargs.get("api_key", "None")

        return super()._from_node(node, api_key=api_key)

    def _read_node(self, node: NodeForecastSolar) -> pd.DataFrame:
        """Download data from the Forecast.Solar Database.

        :param node: Node to read values from.
        :return: pandas.DataFrame containing the data read from the connection.
        """
        url, query_params = node.url, node._query_params

        raw_response = self._raw_request("GET", url, params=query_params, headers=self._headers)
        response = raw_response.json()

        timestamps = pd.to_datetime(list(response["result"].keys()))
        watts = response["result"].values()

        data = pd.DataFrame(
            data=watts,
            index=timestamps.tz_convert(self._local_tz),
            dtype="float64",
        )
        data.index.name = "Time (with timezone)"

        return data

    def _select_data(
        self, results: pd.DataFrame, from_time: pd.Timestamp | None = None, to_time: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, pd.Timestamp]:
        """Forecast.solar api returns the data for the whole day. Select data only for the time interval.

        :param nodes: pandas.DataFrame containing the raw data read from the connection.
        :param from_time: Starting time to begin reading (included in output).
        :param to_time: Time to stop reading at (included in output).
        :return: pandas.DataFrame containing the selected data read from the connection and the current timestamp.
        """
        now = pd.Timestamp.now().tz_localize(self._local_tz)

        if isinstance(from_time, pd.Timestamp):
            previous_time = from_time.floor("15T") if self._api_key != "None" else from_time.floor("h")
        else:
            # When from_time is None, no series is selected
            previous_time = now.floor("15T") if self._api_key != "None" else now.floor("h")

        if isinstance(to_time, pd.Timestamp):
            next_time = (
                to_time.floor("15T") + timedelta(minutes=15)
                if self._api_key != "None"
                else to_time.floor("h") + timedelta(minutes=60)
            )
        else:
            # When to_time is None, no series is selected
            next_time = (
                previous_time + timedelta(minutes=15)
                if self._api_key != "None"
                else previous_time + timedelta(minutes=60)
            )

        if previous_time not in results:
            results.loc[previous_time] = 0

        if next_time not in results:
            results.loc[next_time] = 0

        results = results.sort_index()

        return results.loc[previous_time:next_time], now

    def read(self, nodes: Nodes | None = None) -> pd.DataFrame:
        """Return current value from the Forecast.Solar Database.

        :param nodes: List of nodes to read values from.
        :return: pandas.DataFrame containing the data read from the connection.
        """
        nodes = self._validate_nodes(nodes)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._read_node, nodes)

        values = pd.concat(results, axis=1, keys=[n.name for n in nodes], sort=False)

        values, now = self._select_data(values)

        # Insert the current timestamp _now and sort the index column to finish with the linear interpolation method
        values.loc[now] = pd.NA
        values = values.sort_index()

        return pd.DataFrame(values.interpolate(method="linear").loc[now])

    def write(self, values: Mapping[AnyNode, Any]) -> None:
        """
        .. warning::
            Cannot read single values from the Forecast.Solar API. Use read_series instead

        :raises NotImplementedError:
        """
        raise NotImplementedError("Write is not implemented for Forecast.Solar.")

    def subscribe(self, handler: SubscriptionHandler, nodes: Nodes | None = None, interval: TimeStep = 1) -> None:
        """
        .. warning::
            Cannot read single values from the Forecast.Solar API. Use read_series instead

        :raises NotImplementedError:
        """
        raise NotImplementedError("Subscribe is not implemented for Forecast.Solar.")

    def read_series(
        self, from_time: datetime, to_time: datetime, nodes: Nodes | None = None, interval: TimeStep = 1, **kwargs: Any
    ) -> pd.DataFrame:
        """Download timeseries data from the Forecast.Solar Database

        :param nodes: List of nodes to read values from.
        :param from_time: Starting time to begin reading (included in output).
        :param to_time: Time to stop reading at (not included in output).
        :param interval: Interval between time steps. It is interpreted as seconds if given as integer.
        :param kwargs: Other parameters (ignored by this connector).
        :return: Pandas DataFrame containing the data read from the connection.
        """
        _interval = interval if isinstance(interval, timedelta) else timedelta(seconds=interval)

        nodes = self._validate_nodes(nodes)

        from_time = pd.Timestamp(round_timestamp(from_time, _interval.total_seconds())).tz_convert(self._local_tz)
        to_time = pd.Timestamp(round_timestamp(to_time, _interval.total_seconds())).tz_convert(self._local_tz)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._read_node, nodes)

        values = pd.concat(results, axis=1, keys=[n.name for n in nodes], sort=False)

        values, _ = self._select_data(values, from_time, to_time)

        values = df_interpolate(values, _interval)

        return values.loc[from_time:to_time]  # type: ignore

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
        """
        .. warning::
            Cannot read single values from the Forecast.Solar  API. Use read_series instead

        :raises NotImplementedError:
        """
        raise NotImplementedError("Subscribe series is not implemented for Forecast.Solar.")

    def close_sub(self) -> None:
        """
        .. warning::
            Cannot read single values from the Forecast.Solar  API. Use read_series instead

        :raises NotImplementedError:
        """
        raise NotImplementedError("Close subscription is not implemented for Forecast.Solar.")

    async def _subscription_loop(
        self,
        handler: SubscriptionHandler,
        interval: TimeStep,
        req_interval: TimeStep,
        offset: TimeStep,
        data_interval: TimeStep,
    ) -> None:
        """
        .. warning::
            Cannot read single values from the Forecast.Solar  API. Use read_series instead

        :raises NotImplementedError:
        """
        raise NotImplementedError("Subscription loop is not implemented for Forecast.Solar.")

    def timestr_from_datetime(self, dt: datetime) -> str:
        """Create an Forecast.Solar compatible time string.

        :param dt: Datetime object to convert to string.
        :return: Forecast.Solar compatible time string.
        """

        return dt.isoformat(sep="T", timespec="seconds").replace(":", "%3A").replace("+", "%2B")

    def _raw_request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Perform Forecast.Solar request and handle possibly resulting errors.

        :param method: HTTP request method.
        :param endpoint: Endpoint for the request (server URI is added automatically).
        :param kwargs: Additional arguments for the request.
        """
        if self._api_key != "None":
            log.info("The api_key is None and only the public functions are available of the forecastsolar.api.")

        response = self._session.request(method, url, **kwargs)
        # Check for request errors
        response.raise_for_status()

        return response

    def _validate_nodes(self, nodes: Nodes | None) -> set[NodeForecastSolar]:  # type: ignore
        vnodes = super()._validate_nodes(nodes)
        _nodes = set()
        for node in vnodes:
            if isinstance(node, NodeForecastSolar):
                _nodes.add(node)

        return _nodes

    @classmethod
    def route_valid(cls, nodes: Nodes) -> bool:
        """Check if node routes make up a valid route, by using the Forecast.Solar API's check endpoint.

        :param nodes: List of nodes to check.
        :return: Boolean if the nodes are on the same route.
        """
        conn = ForecastSolarConnection()
        nodes = conn._validate_nodes(nodes)

        for node in nodes:
            _check_node = node.evolve(api_key=None, endpoint="check", data=None)
            try:
                conn._raw_request("GET", _check_node.url, headers=conn._headers)
            except requests.exceptions.HTTPError as e:
                log.error(f"\nRoute of node: {node.name} could not be verified: \n{e}")
                return False

        # If no request error occurred, the routes are valid
        return True

    @classmethod
    def calculate_watt_hours_period(cls, df: pd.DataFrame, watts_column: str) -> pd.DataFrame:
        """
        Calculates watt hours for each period based on the average watts provided.
        Assumes the DataFrame is indexed by timestamps.
        """
        # Calculate the duration of each period in hours
        df["period_hours"] = df.index.to_series().diff().dt.total_seconds() / 3600
        df["period_hours"].iloc[0] = df["period_hours"].mean()  # Handle the first NaN

        # Calculate watt_hours for each period
        df["watt_hours_period"] = df[watts_column] * df["period_hours"]
        return df.drop(columns=["period_hours"])

    @classmethod
    def summarize_watt_hours_over_day(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sums up watt hours over each day.
        """
        df["date"] = df.index.date
        daily_energy = df.groupby("date")["watt_hours_period"].sum().reset_index()
        daily_energy.columns = ["date", "watt_hours"]
        return daily_energy

    @classmethod
    def get_dataframe_of_values(
        cls, df: pd.DataFrame, watts_column: str = "watts"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the original DataFrame to return a DataFrame with
        watt_hours_period, watt_hours (summarized over the day), and watt_hours_day.
        """
        watt_hours = cls.calculate_watt_hours_period(df, watts_column)
        daily_sum = cls.summarize_watt_hours_over_day(watt_hours)

        return watt_hours, daily_sum

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
            return False
        try:
            self._session.close()
        except Exception as e:
            log.error(f"Error closing the connection: {e}")
        return True

    def __del__(self) -> None:
        try:
            self._session.close()
        finally:
            pass
