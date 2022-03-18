""" The script can be used to read data from different servers in the ETA-Factory.
It can write output to CSV files and/or publish to a different OPC UA server.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
from datetime import timedelta
from typing import TYPE_CHECKING

try:
    import keyboard
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "For the data_recorder example, the pygame module is required. Install eta_utility with the "
        "[examples] option to get all packages required for running examples.",
        name="keyboard",
    )
from eta_utility import get_logger
from eta_utility.connectors import Node, connections_from_nodes, sub_handlers

if TYPE_CHECKING:
    from eta_utility.type_hints import Path, TimeStep


log = get_logger(level=2, format="logname")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments (see help for description)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("nodes_file", action="store", type=str, help="Excel file to read nodes from.")
    parser.add_argument("nodes_sheet", action="store", type=str, help="Name of the Excel sheet specifying the nodes.")
    parser.add_argument("--output_file", action="store", type=str, default=None, help="Path to the CSV output file.")
    parser.add_argument("--stop_after", action="store", type=int, default=None, help="Stop recording after X seconds.")
    parser.add_argument(
        "--sub_interval",
        action="store",
        type=float,
        default=1,
        help="Subscript read interval (or polling interval) in seconds.",
    )
    parser.add_argument(
        "--write_interval",
        action="store",
        type=float,
        default=1,
        help="Writing interval in seconds for writing to CSV file.",
    )
    parser.add_argument("--eneffco_usr", action="store", type=str, default=None, help="EnEffCo user name.")
    parser.add_argument("--eneffco_pw", action="store", type=str, default=None, help="EnEffCo password.")
    parser.add_argument(
        "--verbosity",
        action="store",
        type=int,
        default=2,
        help="Verbosity level (between 0 - no output and 4 - debug).",
    )

    return parser.parse_args()


async def logger(interval: int) -> None:
    """Print info message every interval seconds to show that the program continues to work.

    :param interval: Interval for printing the message in seconds.
    """
    step = 0
    while True:
        await asyncio.sleep(interval)
        step += interval
        log.info(f"Logging data for {step} s")


async def stop_execution(sleep_time: TimeStep) -> None:
    """Stop execution after the specified time interval.

    :param sleep_time: Time interval in seconds.
    """
    _time = sleep_time.total_seconds() if isinstance(sleep_time, timedelta) else sleep_time
    await asyncio.sleep(_time)


async def stop_keyboard(key: str = "q") -> None:
    """Stop execution if a key is pressed.

    :param key: Key to be pressed (default: "q").
    """
    while True:
        if keyboard.is_pressed(key):
            break
        await asyncio.sleep(0)


def execution_loop(
    nodes_file: Path,
    nodes_sheet: str,
    output_file: Path | None = None,
    stop_after: TimeStep | None = None,
    sub_interval: TimeStep = 1,
    write_interval: TimeStep = 1,
    eneffco_usr: str | None = None,
    eneffco_pw: str | None = None,
    eneffco_api_token: str | None = None,
    verbosity: int = 2,
) -> None:
    """Execute the subscription and publishing loop.

    :param nodes_file: Path to Excel sheet with node specification.
    :param nodes_sheet: Excel sheet name.
    :param output_file: Path to the CSV output file (optional) - One of output_file or publish_opcua is required.
    :param stop_after: Stop recording data automatically after X seconds
    :param sub_interval: Interval for subscription data.
    :param write_interval: Interval for writing to CSV file
    :param eneffco_usr: EnEffCo username.
    :param eneffco_pw: EnEffCo password.
    :param eneffco_api_token: API token for the EnEffCo connector.
    :param verbosity: Verbosity level (between 0 - no output and 4 - debug).
    """
    log.setLevel(verbosity * 10)

    nodes = Node.from_excel(nodes_file, nodes_sheet)
    connections = connections_from_nodes(nodes, usr=eneffco_usr, pwd=eneffco_pw, eneffco_api_token=eneffco_api_token)

    # Start handler
    subscription_handler = sub_handlers.MultiSubHandler()

    if output_file is None:
        raise ValueError("Specify at least an output_file")

    if output_file is not None:
        output_file = pathlib.Path(output_file)
        if output_file.is_file() or output_file.is_dir():
            try:
                os.remove(output_file)
            except FileNotFoundError:
                pass

        subscription_handler.register(sub_handlers.CsvSubHandler(output_file, write_interval=write_interval))

    loop = asyncio.get_event_loop()
    loop.create_task(logger(10))

    try:
        for host, connection in connections.items():
            # Start connections without passing on interrupt signals
            try:
                connection.subscribe(subscription_handler, interval=sub_interval)
            except ConnectionError as e:
                log.warning(str(e))

        log.warning("Starting processing loop")
        if stop_after is not None:
            log.info(f"Process will stop after {stop_after} s.")
            loop.run_until_complete(stop_execution(stop_after))
        else:
            log.warning("Use q to stop recording data (It might take some time to react).")
            loop.run_until_complete(stop_keyboard("q"))
            log.warning("Detected key press, stopping execution.")

    finally:
        log.info("Closing connections and handlers")
        for host, connection in connections.items():
            connection.close_sub()

        subscription_handler.close()


if __name__ == "__main__":
    args = parse_args()

    execution_loop(
        args.nodes_file,
        args.nodes_sheet,
        args.output_file,
        args.stop_after,
        args.sub_interval,
        args.write_interval,
        args.eneffco_usr,
        args.eneffco_pw,
        args.verbosity,
    )
