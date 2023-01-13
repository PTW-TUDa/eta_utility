import pathlib
from datetime import datetime

import pandas as pd
import pytest
import requests
from lxml import etree

from eta_utility.connectors import ENTSOEConnection, Node
from eta_utility.connectors.entso_e import _ConnectionConfiguration
from eta_utility.util import dict_search

ENTSOE_TOKEN = ""


def create_node(endpoint: str, name: str = "CH1.Elek_U.L1-N") -> Node:
    return Node(
        name,
        "https://transparency.entsoe.eu/",
        "entsoe",
        endpoint=endpoint,
        bidding_zone="DEU-LUX",
    )


@pytest.fixture()
def _local_requests(monkeypatch, config_entsoe):
    monkeypatch.setattr(requests, "post", Postable(config_entsoe["path"]))


@pytest.fixture()
def two_price_day_ahead_nodes():
    return (create_node("Price", "name1"), create_node("Price", "name2"))


def test_entsoe_price_ahead(_local_requests):
    node = create_node("Price")

    server = ENTSOEConnection.from_node(node, api_token=ENTSOE_TOKEN)

    from_datetime = datetime.strptime("2022-02-15T13:18:12", "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime("2022-02-15T14:15:31", "%Y-%m-%dT%H:%M:%S")

    res = server.read_series(from_time=from_datetime, to_time=to_datetime)
    assert isinstance(res, pd.DataFrame)
    assert isinstance(res.index, pd.MultiIndex)
    assert node.name in res.columns[0]


def test_entsoe_actual_generation_per_type(_local_requests):
    node = create_node("ActualGenerationPerType")

    server = ENTSOEConnection.from_node(node, api_token=ENTSOE_TOKEN)

    from_datetime = datetime.strptime("2022-02-15T13:18:12", "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime("2022-02-15T14:00:00", "%Y-%m-%dT%H:%M:%S")

    res = server.read_series(from_time=from_datetime, to_time=to_datetime)
    assert isinstance(res, pd.DataFrame)
    assert isinstance(res.index, pd.DatetimeIndex)
    assert node.name in res.columns[0]


def test_multiple_nodes(_local_requests, two_price_day_ahead_nodes):
    server = ENTSOEConnection.from_node(two_price_day_ahead_nodes, api_token=ENTSOE_TOKEN)

    from_datetime = datetime.strptime("2022-02-15T13:18:12", "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime("2022-02-15T14:15:31", "%Y-%m-%dT%H:%M:%S")

    res = server.read_series(from_time=from_datetime, to_time=to_datetime)

    assert len(res.columns) == 2
    assert "name1" in res.columns[0] or res.columns[1]
    assert "name2" in res.columns[0] or res.columns[1]


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_interval(_local_requests, interval):
    """Considering interval of one second, should return
    the number of seconds between from_time and to_time
    """
    node = create_node("Price")

    server = ENTSOEConnection.from_node(node, api_token=ENTSOE_TOKEN)

    from_datetime = datetime.strptime("2022-02-15T13:18:12", "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime("2022-02-15T14:15:31", "%Y-%m-%dT%H:%M:%S")

    res = server.read_series(from_time=from_datetime, to_time=to_datetime, interval=interval)
    number_of_resolutions = len(res.index.levels[1])

    total_seconds = (to_datetime - from_datetime).seconds // interval + 1  # including the last datetime point
    assert total_seconds * number_of_resolutions == res.shape[0]


class MockResponse(requests.Response):
    def __init__(self, endpoint: str, path: pathlib.Path):
        super().__init__()
        self.status_code = 200
        self.endpoint = endpoint
        self.path = path

    @property
    def content(self):
        with open(self.path / f"{self.endpoint}_sample.xml") as f:
            return f.read().encode()


class Postable:
    def __init__(self, path):
        self.path = path

    def __call__(self, url, data, headers, **kwargs):
        parser = etree.XMLParser(load_dtd=False, ns_clean=True, remove_pis=True)
        e_msg = etree.XML(data, parser)
        ns = e_msg.nsmap

        endpoint_code = (
            e_msg.find(".//AttributeInstanceComponent", namespaces=ns).find("attributeValue", namespaces=ns).text
        )
        doc_types = _ConnectionConfiguration().doc_types

        endpoint = dict_search(doc_types, endpoint_code)

        return MockResponse(endpoint, self.path)
