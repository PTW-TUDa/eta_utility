from datetime import datetime, timedelta

import pandas as pd
import pytest
import requests
from attrs import validators

from eta_utility.connectors import ForecastSolarConnection, Node
from eta_utility.connectors.node import NodeForecastSolar

from ..utilities.requests.forecast_solar_request import request
from ..utilities.util_test import suppress_logging


# Sample node
@pytest.fixture()
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
    }


@pytest.fixture()
def _local_requests(monkeypatch):
    monkeypatch.setattr(requests, "request", request)


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
def test_raw_connection():
    node = NodeForecastSolar.from_dict({"name": "node_forecast_solar1", "ip": "", "protocol": "forecast_solar"})[0]

    connector = ForecastSolarConnection(node.url)

    get_url = connector.baseurl + "/help"

    result = connector._raw_request("GET", get_url)

    assert result.status_code == 200, "Invalid location or plane parameters"


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

    with validators.disabled(), suppress_logging():
        invalid_node = node.evolve(latitude=91)  # latitude invalid
        assert not ForecastSolarConnection.route_valid(invalid_node)


@pytest.mark.usefixtures("_local_requests")
def test_estimate(forecast_solar_nodes):
    nodes = [forecast_solar_nodes["node"], forecast_solar_nodes["node2"]]
    connector = ForecastSolarConnection()

    result = connector.read(nodes)

    assert isinstance(result, pd.DataFrame)


@pytest.mark.usefixtures("_local_requests")
def test_read_series(forecast_solar_nodes):
    nodes = [forecast_solar_nodes["node"], forecast_solar_nodes["node2"]]
    connector = ForecastSolarConnection()

    res = connector.read_series(
        datetime.now() - timedelta(seconds=10),
        datetime.now(),
        nodes[0],
        timedelta(seconds=1),
    )

    assert isinstance(res, pd.DataFrame)
    assert connector._api_key == "None", "The api_key is not set correctly"
    assert len(res) >= 10, "The result has the wrong size of data"


@pytest.mark.usefixtures("_local_requests")
def test_connection_from_node(forecast_solar_nodes: dict[str, Node]):
    # Test connection from node
    connector = ForecastSolarConnection.from_node(forecast_solar_nodes["node3"])
    assert connector.baseurl is not None, "Base URL is empty"


def test_cached_responses(forecast_solar_nodes: dict[str, Node]):
    # Test connection from node
    node = forecast_solar_nodes["node"]
    connector: ForecastSolarConnection = ForecastSolarConnection.from_node(node)

    url, query_params = node.url, node._query_params
    for i in range(10):
        try:
            response = connector._raw_request("GET", url, params=query_params, headers=connector.headers)
            if i != 0:
                assert response.from_cache is True
        except requests.exceptions.HTTPError as e:
            if i == 0 and e.response.status_code == 429:
                pytest.skip("Rate limit reached")
