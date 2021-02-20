""" This module implements some commonly used subscription handlers, for example for writing to csv files.

"""

import csv
import functools as ft
import pathlib
import signal
from datetime import datetime, timedelta
from multiprocessing import Pipe, Process, connection

import numpy as np
import pandas as pd

from eta_utility import get_logger

from .base_classes import SubscriptionHandler

log = get_logger("eta_utilty.connectors")


class MultiSubHandler(SubscriptionHandler):
    """The MultiSubHandler can be used to distribute subcribed values to multiple different subscription handlers.
    The handlers can be registered using the register method.

    """

    def __init__(self):
        super().__init__()

        self._handlers = list()

    def register(self, sub_handler):
        """Register a subscription handler.

        :param SubscriptionHandler sub_handler: SubscriptionHandler object to use for handling subscriptions.
        """
        if not isinstance(sub_handler, SubscriptionHandler):
            raise TypeError("Subscription Handler should be an instance of the SubscriptionHandler slass.")

        self._handlers.append(sub_handler)

    def push(self, node, value, timestamp=None):
        """Receive data from a subcription. This should contain the node that was requested, a value and a timestemp
        when data was received. Push data to all registered sub-handlers

        :param node: Node object the data belongs to
        :type node: Node
        :param value: Value of the data
        :type value: Any
        :param timestamp: Timestamp of receiving the data
        :type timestamp: datetime
        """
        for handler in self._handlers:
            handler.push(node, value, timestamp)

    def close(self):
        """Finalize and close all subscription handlers."""
        for handler in self._handlers:
            try:
                handler.close()
            except:
                pass


class CsvSubHandler(SubscriptionHandler):
    """Handle data for a subscription and save it as a CSV file.

    :param output_file: CSV file to write data to
    :type output_file: pathlib.Path or str
    :param write_interval: Interval for writing data to csv file
    :type write_interval: float or int
    :param write_buffer: Number of lines to keep in buffer
    :type write_buffer: int
    :param ignore_sigint: Use this to ignore the SIGINT signal (Useful if this should be handled in a calling process)
    :type ignore_sigint: bool
    """

    def __init__(self, output_file, write_interval=1, write_buffer=500, ignore_sigint=False):
        super().__init__(write_interval=write_interval)

        self._recv, self._send = Pipe(duplex=False)

        run_func = ft.partial(self._run, self._recv, output_file, write_buffer, ignore_sigint)
        self._proc = Process(target=run_func, daemon=True)
        self._proc.start()
        log.debug(f"CSV Subscription Handler process started: {self._proc.name}")

    def push(self, node, value, timestamp=None):
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to
        :type node: Node
        :param value: Value of the data
        :type value: Any
        :param timestamp: Timestamp of receiving the data
        :type timestamp: datetime
        """
        timestamp = timestamp if timestamp is not None else datetime.now()
        self._send.send((node, value, timestamp))

    def _run(self, receiver, output_file, write_buffer=30, ignore_sigint=False):
        """Create the output file and periodically write data to it.

        :param receiver: Receiving end of a pipe
        :type receiver: connection.Connection
        :param output_file: File to write data to
        :type output_file: pathlib.Path or str
        :param int write_buffer: Minimum number of lines in buffer, until writing should begin
        :param ignore_sigint: Use this to ignore the SIGINT signal (Useful if this should be handled in a calling
        process)
        :type ignore_sigint: bool
        """
        if ignore_sigint:
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        def write_data(buffer, length, write_started, header):
            write_times = sorted(buffer.keys())

            if len(write_times) < 1:
                return None

            if write_started is False:
                header = set()
                for dikt in buffer.values():
                    header.update(dikt.keys())
                header = ["Time"] + list(header)
                writer.writerow(header)

            for time in write_times[: length + 1]:
                row = buffer.pop(time)

                write = [time]
                for name in header:
                    try:
                        write.append(row[name])
                    except KeyError:
                        if name != "Time":
                            write.append("")
                writer.writerow(write)

                if len(row) > len(header):
                    raise RuntimeWarning("Some values were missed while writing to csv.")

            log.debug(f"CSV Subscription wrote {length} lines")

            return header

        output_file = output_file if isinstance(output_file, pathlib.Path) else pathlib.Path(output_file)
        buffer = {}
        write_started = False
        header = []

        with open(output_file, "x", newline="") as file:
            log.info(f"CSV output file created: {output_file}")
            writer = csv.writer(file)
            while True:
                data = receiver.recv()

                # Check for sentinel and write final data
                if data is None:
                    write_data(buffer, len(buffer), write_started, header)
                    break

                node, value, timestamp = data

                if hasattr(value, "__len__"):
                    value = value[0]
                if hasattr(timestamp, "__len__"):
                    timestamp = timestamp[0]
                timestamp = self._round_timestamp(timestamp)

                if timestamp not in buffer:
                    buffer[timestamp] = {}
                buffer[timestamp][node.name] = value

                # write to file when write buffer is full, making
                if len(buffer) > write_buffer:
                    header = write_data(buffer, write_buffer // 2, write_started, header)
                    write_started = True if header is not None else False

    def close(self):
        """Finalize and close the subscription handler."""
        self._send.send(None)  # Send sentinel to subprocess to force writing of buffered data
        self._proc.join(timeout=50)
        self._send.close()
        log.debug(f"CSV Subscription Handler process stopped: {self._proc.name}")


class DFSubHandler(SubscriptionHandler):
    """Subscription handler for returning pandas data frames when requested

    :param int write_interval: Interval for writing data
    :param int keep_data_rows: Number of rows to keep in internal _data memory. Default 100.
    """

    def __init__(self, write_interval=1, keep_data_rows=100):
        super().__init__(write_interval=write_interval)
        self._data = pd.DataFrame()
        self.keep_data_rows = keep_data_rows

    def push(self, node, value, timestamp=None):
        """Append values to the dataframe

        :param node: Node object the data belongs to
        :type node: Node
        :param value: Value of the data or Series of values. There must be corresponding timestamps for each value.
        :type value: Any or pd.Series[Any] or list-like[Any]
        :param timestamp: Timestamp of receiving the data or DatetimeIndex if pushing multiple values. Alternatively
                          an integer/timedelta can be provided to determine the interval between data points. Use
                          negative numbers to describe past data. Integers are interpreted as seconds. If value is a
                          pd.Series and has a pd.DatetimeIndex, timestamp is ignored.
        :type timestamp: datetime or pd.DatetimeIndex or int or timedelta or None
        """
        # Check if node.name is in _data.columns
        if node.name not in self._data.columns:
            self._data[node.name] = np.nan

        # Multiple values
        if hasattr(value, "__len__"):
            value = self._convert_series(value, timestamp)
            # Push Series
            for _timestamp, _value in value.items():
                # todo assert tz-awareness
                self._data.loc[_timestamp, node.name] = _value

        # Single value
        else:
            if not isinstance(timestamp, datetime) and timestamp is not None:
                raise ValueError("Timestamp must be a datetime object or None.")
            timestamp = self._round_timestamp(timestamp if timestamp is not None else datetime.now())
            self._data.loc[timestamp, node.name] = value

        # Housekeeping (Keep internal data short)
        self.housekeeping()

    def get_latest(self):
        """Return a copy of the dataframe, this ensures they can be worked on freely. Returns None if data is empty."""
        if len(self._data.index) == 0:
            return None  # If no data in self._data, return None
        else:
            return self._data.iloc[[-1]].copy()

    @property
    def data(self):
        """This contains the interval dataframe and will return a copy of that."""
        return self._data.copy()

    def reset(self):
        """Reset the internal data and restart collection"""
        self._data = pd.DataFrame()
        log.info("Subscribed DataFrame {} was reset successfully.".format(hash(self._data)))

    def housekeeping(self):
        """Keep internal data short by only keeping last rows as specified in self.keep_data_rows"""
        self._data.drop(index=self._data.index[: -self.keep_data_rows], inplace=True)

    def close(self):
        """This is just here to satisfy the interface, not needed in this case."""
        pass
