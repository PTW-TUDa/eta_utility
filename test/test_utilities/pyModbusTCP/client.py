from typing import List

from pyModbusTCP.client import ModbusClient as MBC


class ModbusClient(MBC):
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
