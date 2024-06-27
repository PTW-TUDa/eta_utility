import pandas as pd

from eta_utility.connectors.node import NodeEmonio


def live_from_dict(url: str) -> pd.DataFrame:
    # --live--
    from eta_utility.connectors.live_connect import LiveConnect

    live = {
        "system": [
            {
                "name": "emonio",
                "servers": {"ac_supply": {"url": url, "protocol": "emonio"}},
                "nodes": [
                    {"name": "V_RMS", "server": "ac_supply"},
                    {"name": "I_RMS", "server": "ac_supply", "phase": "a"},
                ],
            }
        ]
    }
    # Create the connection object with classmethod from_dict
    connection = LiveConnect.from_dict(None, None, 1, 10, **live)

    # Read the values of the nodes we defined in the dictionary
    result = connection.read("V_RMS", "I_RMS")
    # --live--
    return result


def emonio_manuell(url: str) -> pd.DataFrame:
    # --emonio--
    from eta_utility.connectors import EmonioConnection

    voltage_node = NodeEmonio("V_RMS", url, "emonio")
    current_node = NodeEmonio("I_RMS", url, "emonio", phase="a")

    # Initialize the connection object with both nodes
    connection = EmonioConnection.from_node([voltage_node, current_node])

    # Read values of selected nodes
    if isinstance(connection, EmonioConnection):
        result = connection.read()
    else:
        raise TypeError("The connection must be an ModbusConnection.")
    # --emonio--
    return result


def modbus_manuell(url: str) -> pd.DataFrame:
    # --modbus--
    from eta_utility.connectors import ModbusConnection
    from eta_utility.connectors.emonio import NodeModbusFactory

    factory = NodeModbusFactory(url)

    # V_RMS for all phases
    voltage_node = factory.get_default_node("Spannung", 300)
    # I_RMS for phase a
    current_node = factory.get_default_node("Strom", 2)

    connection = ModbusConnection.from_node([voltage_node, current_node])

    if isinstance(connection, ModbusConnection):
        result = connection.read()
    else:
        raise TypeError("The connection must be an ModbusConnection.")
    # --modbus--
    return result