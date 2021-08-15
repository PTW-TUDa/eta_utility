import unittest
from unittest.mock import Mock, patch

from eta_utility.connectors import Node
from eta_utility.connectors.rest import RESTConnection

SERVER_URL = "http://192.168.178.87:5000/"

node = Node("TestNode1", SERVER_URL, "rest", rest_endpoint="/Test")
node2 = Node("TestNode2", SERVER_URL, "rest", rest_endpoint="/Test2")
node_fail = Node("TestNode2", "someurl", "bad_protocol", rest_endpoint="/Test")
node_fail2 = Node("TestNode2", "someurl", "rest", rest_endpoint="/Test")
node_fail3 = Node("TestNode2", SERVER_URL, "bad_protocol", rest_endpoint="/Test")

test_dict = {"Test": "Ok"}


class rest_connector_tests(unittest.TestCase):

    # testing exception behaivor
    def test_rest_from_node_failure(self):
        with self.assertRaises(ValueError):
            RESTConnection.from_node(node_fail)

    # testing behavior on correct input
    def test_rest_from_node_sucessful(self):
        output = RESTConnection.from_node(node)
        self.assertIsInstance(output, RESTConnection)
        self.assertEqual(output.url, SERVER_URL)

    # testing read method
    @patch("eta_utility.connectors.rest.requests.get")
    def test_rest_read(self, mock_get):
        conn = RESTConnection(SERVER_URL)
        # configure mock return to let requests.get return a Response filled with test_dict as content
        mock_get.return_value = Mock(ok=True)
        mock_get.return_value.json.return_value = test_dict
        response = conn.read(node)
        self.assertDictEqual(response, test_dict)

    # testing write method
    @patch("eta_utility.connectors.rest.requests.put")
    def test_rest_write(self, mock_put):
        conn = RESTConnection(SERVER_URL)
        mock_put.return_value = Mock(ok=True)
        mock_put.return_value.json.return_value = "OK"
        response = conn.write(node, test_dict)
        self.assertEqual(response.json(), "OK")

    # testing init RESTConnection with single node as input
    def test_rest_init_conn_single(self):
        conn = RESTConnection(SERVER_URL, node)
        self.assertIsInstance(conn, RESTConnection)
        self.assertEqual(conn.url, SERVER_URL)
        self.assertEqual(conn.selected_nodes, {node})

    # testing init RESTConnection with multiple nodes as input
    def test_rest_init_conn_multiple(self):
        conn = RESTConnection(SERVER_URL, [node, node2])
        self.assertIsInstance(conn, RESTConnection)
        self.assertEqual(conn.url, SERVER_URL)
        self.assertEqual(conn.selected_nodes, {node, node2})

    # testing init RESTConnection with invalid nodes as input
    def test_rest_init_conn_validate_nodes(self):
        conn = RESTConnection(SERVER_URL, [node, node_fail3, node_fail2, node2])
        self.assertIsInstance(conn, RESTConnection)
        self.assertEqual(conn.url, SERVER_URL)
        self.assertEqual(conn.selected_nodes, {node, node2})

    # testing init RESTConnection failure due to invalid nodes
    def test_rest_init_conn_badnode(self):
        with self.assertRaises(ValueError):
            conn = RESTConnection(SERVER_URL, node_fail)
            self.assertIsNone(conn)


if __name__ == "__main__":
    unittest.main()
