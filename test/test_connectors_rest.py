import pytest

from eta_utility.connectors import Node, RESTConnection

SERVER_URL = "http://192.168.178.87:5000"

node = Node("TestNode1", SERVER_URL, "rest", rest_endpoint="Test")
node2 = Node("TestNode2", SERVER_URL, "rest", rest_endpoint="Test2")
node_fail = Node("TestNode2", "someurl", "bad_protocol", rest_endpoint="Test")
node_fail2 = Node("TestNode2", "someurl", "rest", rest_endpoint="Test")
node_fail3 = Node("TestNode2", SERVER_URL, "bad_protocol", rest_endpoint="Test")

test_dict = {"Test": "Ok"}


class MockResponse:
    """custom class to mock the return value of requests.Response"""

    def __init__(self, output):
        self.output = output

    def json(self):
        return self.output


def test_rest_from_node_failure():
    """ "testing from_node()-method (exception behaviour)"""
    with pytest.raises(ValueError) as excinfo:
        RESTConnection.from_node(node_fail)
    assert "TestNode2" in str(excinfo.value)


def test_rest_from_node_sucessful():
    """ "testing from_node()-method (normal behaviour)"""
    output = RESTConnection.from_node(node)
    assert isinstance(output, RESTConnection)
    assert output.url == SERVER_URL


def test_rest_read(monkeypatch):
    """ "testing read()-method"""

    def mock_get(url):
        """custom method to mock requests.get"""
        assert url == SERVER_URL + "/Test/GetJson"
        return MockResponse(test_dict)

    conn = RESTConnection(SERVER_URL)
    monkeypatch.setattr("eta_utility.connectors.rest.requests.get", mock_get)
    response = conn.read(node)
    assert response == test_dict


def test_rest_write(monkeypatch):
    """ "testing write()-method"""

    def mock_put(url, json):
        """cutom method to mock requests.put"""
        assert url == SERVER_URL + "/Test/PutJson"
        assert json == test_dict
        return MockResponse("OK")

    conn = RESTConnection(SERVER_URL)
    monkeypatch.setattr("eta_utility.connectors.rest.requests.put", mock_put)
    response = conn.write(node, test_dict)
    assert response.json() == "OK"


def test_rest_init_conn_single():
    """ "testing init() RESTConnection with single node as input"""
    conn = RESTConnection(SERVER_URL, node)
    assert isinstance(conn, RESTConnection)
    assert conn.url == SERVER_URL
    assert conn.selected_nodes == {node}


def test_rest_init_conn_multiple():
    """ "testing init() RESTConnection with multiple nodes as input"""
    conn = RESTConnection(SERVER_URL, [node, node2])
    assert isinstance(conn, RESTConnection)
    assert conn.url == SERVER_URL
    assert conn.selected_nodes == {node, node2}


def test_rest_init_conn_validate_nodes():
    """ "testing init() RESTConnection with invalid nodes as input"""
    conn = RESTConnection(SERVER_URL, [node, node_fail3, node_fail2, node2])
    assert isinstance(conn, RESTConnection)
    assert conn.url == SERVER_URL
    assert conn.selected_nodes == {node, node2}


def test_rest_init_conn_badnode():
    """ "testing init() RESTConnection failure due to invalid nodes"""
    with pytest.raises(ValueError):
        conn = RESTConnection(SERVER_URL, node_fail)
        assert conn is None
