from datetime import datetime

import pandas as pd

from eta_utility.connectors import Node, WetterdienstConnection
from eta_utility.connectors.base_classes import Connection


def main() -> None:
    read_series()


def read_series() -> pd.DataFrame:
    # --begin_wetterdienst_doc_example--

    # Construct a node with the necessary information to request data from the Wetterdienst API
    node = (
        Node(
            "Temperature_Darmstadt",
            "https://opendata.dwd.de",
            "wetterdienst_observation",
            parameter="TEMPERATURE_AIR_MEAN_200",
            station_id="00917",  # Darmstadt observation station ID
            interval=600,  # 10 minutes interval
        ),
    )

    # start connection from one or multiple nodes
    # The from_node() method can be used for initializing the connection
    connection = Connection.from_node(node)

    # Define time interval as datetime values
    from_datetime = datetime(2024, 1, 16, 12, 00)
    to_datetime = datetime(2024, 1, 16, 18, 00)

    # read_series will request data from specified connection and time interval
    # The DataFrame will have index with time delta of the specified interval in seconds
    # If a node  has a different interval than the requested interval, the data will be resampled.
    if isinstance(connection, WetterdienstConnection):
        df = connection.read_series(from_time=from_datetime, to_time=to_datetime, interval=1200)
    else:
        raise TypeError("The connection must be an WetterdienstConnection, to be able to call read_series.")
    # Check out the WetterdienstConnection documentation for more information
    # https://wetterdienst.readthedocs.io/en/latest/data/introduction.html
    # --end_wetterdienst_doc_example--

    return df


if __name__ == "__main__":
    main()
