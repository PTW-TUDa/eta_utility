from datetime import datetime, timedelta

import pandas as pd
import pytest
import requests
import requests_cache
from attrs import validators

from eta_utility.connectors import ForecastSolarConnection, Node
from eta_utility.connectors.node import NodeForecastSolar

from ..utilities.requests.forecast_solar_request import request


# Sample node
@pytest.fixture
def forecast_solar_nodes(config_forecast_solar: dict[str, str]):
    return {
        "node": NodeForecastSolar(
            name="node_forecast_solar1",
            url=config_forecast_solar["url"],
            protocol="forecast_solar",
            endpoint="estimate",
            latitude=51.15,
            longitude=10.45,
            declination=20,
            azimuth=0,
            kwp=12.34,
        ),
        "node2": NodeForecastSolar(
            name="node_forecast_solar2",
            url=config_forecast_solar["url"],
            protocol="forecast_solar",
            latitude=51.15,
            longitude=10.45,
            declination=20,
            azimuth=0,
            kwp=12.34,
        ),
        "node3": NodeForecastSolar(
            name="node_forecast_solar3",
            url=config_forecast_solar["url"],
            protocol="forecast_solar",
            latitude=51.15,
            longitude=10.45,
            declination=20,
            azimuth=0,
            kwp=12.34,
        ),
        "node4": NodeForecastSolar(
            name="node_forecast_solar3",
            url=config_forecast_solar["url"],
            protocol="forecast_solar",
            api_key="A1B2C3D4E5F6G7H8",
            latitude=49.86381,
            longitude=8.68105,
            declination=[14, 10, 10],
            azimuth=[90, -90, 90],
            kwp=[23.31, 23.31, 23.31],
        ),
    }


@pytest.fixture
def _local_requests(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(requests_cache.CachedSession, "request", request)


@pytest.fixture
def connector():
    with ForecastSolarConnection() as connector:
        yield connector


def test_node_from_dict():
    nodes = NodeForecastSolar.from_dict(
        {
            "name": "node_forecast_solar1",
            "ip": "",
            "protocol": "forecast_solar",
            "endpoint": "estimate",
            "latitude": 51.15,
            "longitude": 10.45,
            "declination": 20,
            "azimuth": 0,
            "kwp": 12.34,
        },
        {
            "name": "node_forecast_solar2",
            "ip": "",
            "protocol": "forecast_solar",
            "latitude": 51.15,
            "longitude": 10.45,
            "declination": 20,
            "azimuth": 0,
            "kwp": 12.34,
        },
    )

    for node in nodes:
        assert (
            node.endpoint == "estimate"
        ), "Invalid endpoint for the forecastsolar.api, default endpoint is 'estimate'."


@pytest.mark.usefixtures("_local_requests")
def test_raw_connection(connector):
    get_url = connector._baseurl + "/help"

    result = connector._raw_request("GET", get_url)

    assert result.status_code == 200, "Connection failed"


@pytest.mark.xfail(reason="This test is expected to fail due to rate limiting")
def test_check_route():
    # Check if URL params location and plane are valid
    node = NodeForecastSolar.from_dict(
        {
            "name": "node_forecast_solar1",
            "ip": "",
            "protocol": "forecast_solar",
            "endpoint": "estimate",
            "latitude": 51.15,
            "longitude": 10.45,
            "declination": 20,
            "azimuth": 0,
            "kwp": 12.34,
        }
    )[0]

    assert ForecastSolarConnection.route_valid(node)

    with validators.disabled():
        invalid_node = node.evolve(latitude=91)  # latitude invalid
        assert not ForecastSolarConnection.route_valid(invalid_node)


@pytest.mark.usefixtures("_local_requests")
def test_read(forecast_solar_nodes, connector):
    nodes = [forecast_solar_nodes["node"], forecast_solar_nodes["node2"]]
    result = connector.read(nodes)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 1), "The result has the wrong size of data"


@pytest.mark.usefixtures("_local_requests")
def test_read_series(forecast_solar_nodes, connector):
    nodes = [forecast_solar_nodes["node"], forecast_solar_nodes["node2"]]

    start = datetime(2024, 9, 18, 12, 0)
    end = start + timedelta(days=4)
    interval = timedelta(minutes=15)

    res = connector.read_series(
        start,
        end,
        nodes,
        interval,
    )

    assert isinstance(res, pd.DataFrame)
    assert res.shape == (385, 2), "The result has the wrong size of data"
    assert connector._api_key == "None", "The api_key is not set correctly"


@pytest.mark.usefixtures("_local_requests")
def test_read_multiple_nodes(forecast_solar_nodes, connector):
    api_key = "A1B2C3D4E5F6G7H8"
    n = forecast_solar_nodes["node"].evolve(api_key=api_key)
    nodes = [n]
    nodes.append(n.evolve(declination=30, azimuth=90, kwp=10))
    nodes.append(n.evolve(declination=10, azimuth=60, kwp=40))
    nodes.append(n.evolve(latitude=37.6, longitude=-116.8))

    # Range of possible values for query parameters (except horizon)
    query_params = {
        "no_sun": (0, 1),
        "damping_morning": (0.0, 0.33, 0.99),
        "damping_evening": (0.0, 0.33, 0.99),
        "inverter": (10, 1000),
        "actual": (0, 100, 1000),
    }
    # Create nodes with different query parameters
    for k, v in query_params.items():
        for val in v:
            kwargs = {k: val}
            nodes.append(n.evolve(**kwargs))

    start = datetime(2024, 9, 18, 12, 0)
    end = start + timedelta(hours=2)
    interval = timedelta(minutes=1)

    result = connector.read_series(start, end, nodes, interval)
    assert isinstance(result, pd.DataFrame)
    assert len(result.columns) == len(nodes), "The result has the wrong number of columns"


@pytest.mark.usefixtures("_local_requests")
def test_connection_from_node(forecast_solar_nodes: dict[str, Node]):
    # Test connection from node
    nodes = [forecast_solar_nodes["node3"], forecast_solar_nodes["node4"]]
    start = datetime(2024, 9, 18, 12, 0)
    end = start + timedelta(hours=2)
    interval = timedelta(minutes=1)

    connector = ForecastSolarConnection.from_node(nodes)
    result = connector.read_series(start, end, nodes, interval)

    assert connector._baseurl is not None, "Base URL is empty"
    assert result.shape == (121, 2)


def test_cached_responses(forecast_solar_nodes: dict[str, Node]):
    # Test connection from node
    node = forecast_solar_nodes["node"]
    connector: ForecastSolarConnection = ForecastSolarConnection.from_node(node)

    url, query_params = node.url, node._query_params
    for i in range(10):
        try:
            response = connector._raw_request("GET", url, params=query_params, headers=connector._headers)
            if i != 0:
                assert response.from_cache is True
        except requests.exceptions.HTTPError as e:
            if i == 0 and e.response.status_code == 429:
                pytest.skip("Rate limit reached")
