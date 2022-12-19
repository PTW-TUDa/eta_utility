from datetime import datetime

import pandas as pd

from eta_utility.connectors import ENTSOEConnection, Node


def main() -> None:
    read_series()


def read_series() -> pd.DataFrame:
    # --begin_entsoe_doc_example--
    # Define your ENTSO-E Token
    entsoe_token = ""

    # Check out NodeEntsoE documentation for endpoint and bidding zone information
    node = Node(
        "CH1.Elek_U.L1-N",
        "https://transparency.entsoe.eu/",
        "entsoe",
        endpoint="Price",
        bidding_zone="DEU-LUX",
    )

    # start connection from one or multiple nodes
    server = ENTSOEConnection.from_node(node, api_token=entsoe_token)

    # Define time interval as datetime values
    from_datetime = datetime.strptime("2022-02-15T13:18:12", "%Y-%m-%dT%H:%M:%S")
    to_datetime = datetime.strptime("2022-02-15T14:00:00", "%Y-%m-%dT%H:%M:%S")

    # read_series will request data from specified connection and time interval
    # The DataFrame will have index with time delta of the specified interval in seconds
    df = server.read_series(from_time=from_datetime, to_time=to_datetime, interval=1)
    # --end_entsoe_doc_example--

    return df


if __name__ == "__main__":
    main()
