from eta_utility.connectors.common import Node


class ModbusNodes:
    node = Node(
        "Serv.NodeName",
        "modbus.tcp://10.0.0.1:502",
        "modbus",
        mb_channel=3861,
        mb_register="Holding",
        mb_slave=32,
    )

    node2 = Node(
        "Serv.NodeName2",
        "modbus.tcp://10.0.0.2:502",
        "modbus",
        mb_channel=3866,
        mb_register="Holding",
        mb_slave=32,
    )
    node_fail = Node(
        "Test.Name",
        "modbus.tcp://someurl:502",
        "modbus",
        mb_channel=19026,
        mb_register="Holding",
        mb_slave=32,
    )
