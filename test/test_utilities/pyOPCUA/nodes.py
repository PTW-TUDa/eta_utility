from eta_utility.connectors.common import Node


class OPCUANodes:

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
