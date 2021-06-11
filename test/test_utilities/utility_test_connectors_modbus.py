from typing import List

from pyModbusTCP.client import ModbusClient

from eta_utility.connectors.common import Node


class ModBusUtilities:

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

    class MockMBClient(ModbusClient):
        @staticmethod
        def open() -> bool:
            return True

        @staticmethod
        def is_open() -> bool:
            return True

        @staticmethod
        def close() -> None:
            pass

        @staticmethod
        def mode(val) -> None:
            pass

        @staticmethod
        def unit_id(val) -> None:
            pass

        @staticmethod
        def read_holding_registers(val, num) -> List[int]:
            return [17171, 34363]
