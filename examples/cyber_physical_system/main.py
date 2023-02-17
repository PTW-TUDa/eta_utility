from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from eta_utility import get_logger
from eta_utility.connectors import Node
from eta_utility.eta_x import ETAx
from eta_utility.servers import OpcUaServer

if TYPE_CHECKING:
    from typing import Any, Mapping, Sequence


def main() -> None:
    nodes = {
        "temp_tank": {"id": "ns=2;s=Application.general.Tanktemperatur.senT", "dtype": "float"},
        "tankheater": {"id": "ns=2;s=Application.outputs.Tankheizung", "dtype": "bool"},
        "mode_tankheater": {"id": "ns=2;s=Application.IL_4EEApplication.bTankheizungONOFF", "dtype": "bool"},
        "heating_register": {"id": "ns=2;s=Application.general.Scada_Heissblasen", "dtype": "bool"},
        "motor_nozzles": {"id": "ns=2;s=Application.outputs.Antrieb_Duesen_Freigabe", "dtype": "bool"},
        "motor_basket": {"id": "ns=2;s=Application.outputs.Antrieb_Korb_Freigabe", "dtype": "bool"},
        "pump": {"id": "ns=2;s=Application.outputs.Spritzpumpe", "dtype": "bool"},
        "fan": {"id": "ns=2;s=Application.outputs.Abluftgeblaese", "dtype": "bool"},
        "valve": {"id": "ns=2;s=Application.general.Scada_Impulsblasen", "dtype": "bool"},
    }

    get_logger(format="logname")
    local_server(local_nodes(nodes))

    root_path = get_path()
    experiment(root_path)


def experiment(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> None:
    """Perform a conventionally controlled experiment with the cleaning machine environment.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    experiment = ETAx(
        root_path=root_path, config_overwrite=overwrite, relpath_config=".", config_name="experiment_cleaning_machine"
    )
    experiment.play("cleaning_machine", "test_run")


def local_nodes(definitions: Mapping[str, Mapping[str, str]]) -> list[Node]:
    """Create the specified list of local nodes.

    :param definitions: Node definition mapping
    :return: The created nodes
    """
    nodes = []

    for name, node in definitions.items():
        nodes.append(
            Node(name=name, url="opc.tcp://localhost:4840", protocol="opcua", opc_id=node["id"], dtype=node["dtype"])
        )

    return nodes


def local_server(nodes: Sequence[Node]) -> OpcUaServer:
    """Create a local server with the specified nodes.

    :param Nodes: Sequence of nodes
    """
    server = OpcUaServer(namespace=9, ip="localhost")
    server.create_nodes(nodes)
    return server


def get_path() -> pathlib.Path:
    """Get the path of this file.

    :return: Path to this file.
    """
    return pathlib.Path(__file__).parent


if __name__ == "__main__":
    import sys

    sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

    main()
