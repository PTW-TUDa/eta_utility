from opcua import Client

from eta_utility.connectors.common import Node


class OPCUAUtilities:

    node = Node(
        "Serv.NodeName",
        "opc.tcp://10.0.0.1:48050",
        "opcua",
        opc_id="ns=6;s=.Some_Namespace.Node1",
    )

    node2 = Node(
        "Serv.NodeName2",
        "opc.tcp://10.0.0.1:48050",
        "opcua",
        opc_id="ns=6;s=.Test_Namespace.Node2",
    )

    node_fail = Node(
        "Serv.NodeName",
        "opc.tcp://someurl:48050",
        "opcua",
        opc_id="ns=6;s=.Test_Namespace.Node" ".Drehzahl",
    )

    class MockClient(Client):
        class MockNode:
            def get_value(self) -> float:
                return 2858.0

        @staticmethod
        def connect() -> None:
            pass

        @classmethod
        def get_node(cls, node) -> MockNode:
            return cls.MockNode()
