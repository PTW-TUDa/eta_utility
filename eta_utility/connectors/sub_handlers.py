""" This module implements some commonly used subscription handlers, for example for writing to csv files.

"""

import csv
import io
import pathlib
import queue
import re
import threading
from collections import deque
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any, Deque, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from dateutil import tz

from eta_utility import get_logger
from eta_utility.type_hints import Node, Number, Path, TimeStep

from .base_classes import SubscriptionHandler

log = get_logger("eta_utilty.connectors")


class MultiSubHandler(SubscriptionHandler):
    """The MultiSubHandler can be used to distribute subcribed values to multiple different subscription handlers.
    The handlers can be registered using the register method.

    """

    def __init__(self) -> None:
        super().__init__()

        self._handlers: List = []

    def register(self, sub_handler: SubscriptionHandler) -> None:
        """Register a subscription handler.

        :param SubscriptionHandler sub_handler: SubscriptionHandler object to use for handling subscriptions.
        """
        if not isinstance(sub_handler, SubscriptionHandler):
            raise TypeError("Subscription Handler should be an instance of the SubscriptionHandler class.")

        self._handlers.append(sub_handler)

    def push(self, node: Node, value: Any, timestamp: Optional[datetime] = None) -> None:
        """Receive data from a subcription. This should contain the node that was requested, a value and a timestemp
        when data was received. Push data to all registered sub-handlers

        :param node: Node object the data belongs to
        :param value: Value of the data
        :param timestamp: Timestamp of receiving the data
        """
        for handler in self._handlers:
            handler.push(node, value, timestamp)

    def close(self) -> None:
        """Finalize and close all subscription handlers."""
        for handler in self._handlers:
            try:
                handler.close()
            except Exception:
                pass


class CsvSubHandler(SubscriptionHandler):
    """Handle data for a subscription and save it as a CSV file.

    :param output_file: CSV file to write data to
    :param write_interval: Interval for writing data to csv file
    :param size_limit: Size limit for the csv file. A new file with a unique name will be created when the size
        is exceeded.
    :param dialect: Dialect of the csv file. This takes objects, which correspond to the csv.Dialect interface from the
        python csv module.
    """

    def __init__(
        self,
        output_file: Path,
        write_interval: Union[float, int] = 1,
        size_limit: int = 1024,
        dialect: csv.Dialect = csv.excel,
    ) -> None:
        super().__init__(write_interval=write_interval)

        # Create the csv file handler object which writes data to disc
        self._csv_file = _CSVFileDB(output_file, write_interval, size_limit, dialect)

        # Enable propagation of exceptions
        self.exc = None

        # Create the queue and thread
        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def push(self, node: Node, value: Any, timestamp: Optional[datetime] = None) -> None:
        """Receive data from a subcription. THis should contain the node that was requested, a value and a timestemp
        when data was received. If the timestamp is not provided, current time will be used.

        :param node: Node object the data belongs to
        :param value: Value of the data
        :param timestamp: Timestamp of receiving the data
        """
        timestamp = timestamp if timestamp is not None else datetime.now()
        self._queue.put_nowait((node, value, timestamp))

        # Reraise exceptions if any
        if self.exc:
            raise self.exc

    def _run(self) -> None:
        """Take data from the queue, preprocess it and write to output file."""

        with self._csv_file as f:
            try:
                while True:
                    data = self._queue.get()
                    if data is None:
                        self._queue.task_done()
                        break

                    node, value, timestamp = data

                    # Make sure not to send lists to the file handler
                    if hasattr(value, "__len__"):
                        value = value[0]
                    if hasattr(timestamp, "__len__"):
                        timestamp = timestamp[0]
                    timestamp = self._round_timestamp(timestamp).astimezone(self._local_tz)

                    f.write(timestamp, node.name, value)
                    self._queue.task_done()
            except BaseException as e:
                self.exc = e

    def close(self) -> None:
        """Finalize and close the subscription handler."""
        # Reraise exceptions if any
        if self.exc:
            raise self.exc

        self._queue.put_nowait(None)
        self._queue.join()
        self._thread.join()


class _CSVFileDB(AbstractContextManager):
    """Handle CSV file content

    :param file: Path to the csv file
    :param write_interval: interval in seconds between values (rows in csv file).
    :param file_size_limit: Size limit for the file in MB. A new file will be created, once the limit is exceeded.
    :param dialect: Dialect of the csv file. This takes objects, which correspond to the csv.Dialect interface from the
        python csv module.
    """

    def __init__(
        self, file: Path, write_interval: Number, file_size_limit: int = 1024, dialect: csv.Dialect = csv.excel
    ):
        #: Path to the file that is being written to
        self.filepath: pathlib.Path = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        #: File descriptor
        self._file: Optional[io.TextIOWrapper] = None

        #: Interval between values in seconds
        self.write_interval: Number = write_interval
        #: Size limit for written files in bytes
        self.file_size_limit: int = file_size_limit * 1024 * 1024
        #: CSV dialect to be used for reading and writing data
        self.dialect: csv.Dialect = dialect

        #: List of header fields
        self._header: List[str] = []
        #: Ending position of the header in the file stream (used for extending the header)
        self._endof_header: int = 0
        #: Write buffer
        self._buffer: Deque[Dict[str, Union[datetime, Number]]] = deque()
        #: Latest timestamp in the write buffer
        self._latest_ts: datetime = datetime.fromtimestamp(10000, tz=tz.tzlocal())
        #: Latest known value for each of the names in the header
        self._latest_values: Dict[str, Number] = {}
        #: Length of the line terminator in bytes (for finding file positions)
        self._len_lineterminator: int = len(bytes(self.dialect.lineterminator, "UTF-8"))

    def __enter__(self):
        """Enter the context managed file database."""
        self._open_file()
        return self

    def _open_file(self, exclusive_creation: bool = False):
        """Open a new file and check whether it is writable. If the file exists, try to figure out the dialect and
        header of the existing file.

        :param exclusive_creation: Set to True, to request exclusive creation of a new file. If set to False, an
            existing file may be updated.
        :return:
        """
        # Try opening or creating the specified file.
        try:
            if exclusive_creation:
                raise FileNotFoundError

            self._file = self.filepath.open("r+t", newline="", encoding="UTF-8")
            log.debug(f"Opened existing '.csv' file for updating: {self.filepath}.")
        except FileNotFoundError:
            try:
                self._file = self.filepath.open("x+t", newline="", encoding="UTF-8")
                log.debug(f"Created a new '.csv' file: {self.filepath}.")
            except OSError:
                raise OSError(f"Unable to read or write the requested '.csv' file: {self.filepath}.")

        # Check whether the file is accessible in the required ways.
        if not self._file.readable() or not self._file.seekable() or not self._file.writable():
            raise ValueError("Output file for writing to '.csv' is not readable or writable.")
        else:
            log.debug("Successfully verified full '.csv' file access")

        # If the file is not empty, go to the beginning and try to figure out, whether existing data could be extended.
        if self._file.readline() in "":
            valid = True
            self._header = ["Timestamp"]
            self._write_file(self._header)
            self._endof_header = self._file.tell() - self._len_lineterminator
            log.debug(f"The '.csv' file was empty, dialect set to {self.dialect}, started writing header.")
        else:
            self._file.seek(0)

            # Read a maximum of 30 lines from the file to use as a sample for figuring out, whether the file
            # is valid csv
            sample_lines = []
            for _ in range(30):
                sample_lines.append(self._file.readline())

                if sample_lines[-1] == "":
                    break

            sample = "".join(sample_lines)
            try:
                valid = csv.Sniffer().has_header(sample)
                self.dialect = csv.Sniffer().sniff(sample)
                self.dialect.delimiter = "," if self.dialect.delimiter not in {",", ";"} else self.dialect.delimiter
                self._len_lineterminator = len(bytes(self.dialect.lineterminator, "UTF-8"))
            except csv.Error:
                valid = False
                self.dialect = csv.excel

            # Determine the header of the existing csv file
            self._file.seek(0)
            self._header = list(re.sub(r"\s+", "", self._file.readline()).split(self.dialect.delimiter))
            self._endof_header = self._file.tell() - self._len_lineterminator

        if not valid:
            raise ValueError(f"Output file for writing to '.csv' is not a valid '.csv' file: {self.filepath}.")

        # Go to the end to get ready for updating/extending the file.
        self._file.seek(0, 2)

    def _write_file(self, field_list: List[str], insert_pos: Optional[int] = None) -> int:
        """Write data to the file.

        :param field_list: List of strings to be inserted into the csv file
        :param insert_pos: Position to insert the fields (stream position). If None, insertion will be at end of file.
        :return: ending position of the last insertion (stream position)
        """

        if insert_pos is None:
            string = self.dialect.delimiter.join(field_list) + self.dialect.lineterminator

            try:
                self._file.write(string)
            except ValueError:
                self._open_file()
                self._file.write(string)

            pos = self._file.tell()
        else:
            # When inserting, everything else in the file must be moved along as well
            # (otherwise it would be overwritten). Therefore, a chunk of the file is read before inserting
            string = self.dialect.delimiter + self.dialect.delimiter.join(field_list)
            chunksize = max(100000, len(string))

            self._file.seek(insert_pos)
            chunk = self._file.read(chunksize)
            nextpos_read = self._file.tell()
            self._file.seek(insert_pos)
            self._file.write(string)
            pos = nextpos_insert = self._file.tell()

            while nextpos_insert < nextpos_read:
                self._file.seek(nextpos_read)
                newchunk = self._file.read(chunksize)
                nextpos_read = self._file.tell()
                # Insert last chunk into file and store next chunk
                self._file.seek(nextpos_insert)
                self._file.write(chunk)

                # Store current position for next insertion, then read next chunk
                nextpos_insert = self._file.tell()
                chunk = newchunk
            else:
                self._file.seek(nextpos_insert)
                self._file.write(chunk)

        return pos

    def write(
        self,
        timestamp: Optional[datetime] = None,
        name: Optional[str] = None,
        value: Optional[Number] = None,
        flush: bool = False,
        _len_buffer: int = 20,
    ) -> None:
        """Write value to the file and manage the data buffer.

        :param timestamp: Timestamp of the value to be written (can be empty if only flushing the buffer is intended)
        :param name: Name/Header for the value to be written (can be empty if only flushing the buffer is intended)
        :param value: Value to be written to the file (can be empty if only flushing the buffer is intended)
        :param flush: Flush the entire buffer to file if set to True (default: False)
        :param _len_buffer: Length of the buffer in lines (default: 20). Does not usually need to be changed.
        """
        if self._file is None:
            raise RuntimeError("Enter context manager before trying to write to CSVFileDB.")

        # Check whether the file size limit is exceeded to initiate switching to a new file.
        size_limit_exceeded = True if self.filepath.stat().st_size > self.file_size_limit else False

        # Determine how large the buffer should be, depending on whether data is being flushed to file or preparing to
        # switch to a new file.
        if flush:
            buffer_target = 0
        elif size_limit_exceeded and not len(self._buffer) >= 2 * _len_buffer:
            buffer_target = 2 * _len_buffer
            log.debug("Preparing to switch files due to exceeded CSV file size limit.")
        else:
            buffer_target = _len_buffer

        # New values are inserted into the buffer, depending on their timestamp in relation to self._latest_ts.
        if timestamp is not None and name is not None and value is not None:
            if name not in self._header:
                self._header.append(name)
                self._endof_header = self._write_file([name], self._endof_header)

            if timestamp > self._latest_ts:
                self._buffer.append({name: value})
                self._latest_ts = timestamp
            else:
                idx = int(len(self._buffer) - ((self._latest_ts - timestamp).total_seconds() // self.write_interval))
                try:
                    self._buffer[idx].update({name: value})
                except IndexError:
                    if timestamp == self._latest_ts:
                        self._buffer.append({name: value})
                    else:
                        self._buffer.appendleft({name: value})

        # Write any rows in the buffer which exceed the size of buffer_target to the file.
        while len(self._buffer) >= buffer_target and len(self._buffer) > 0:
            row = self._buffer.popleft()
            row[self._header[0]] = (
                self._latest_ts - timedelta(seconds=len(self._buffer) * self.write_interval)
            ).strftime("%Y-%m-%d %H:%M:%S.%f")

            processed_row = [""] * len(self._header)
            for idx, col in enumerate(self._header):
                if col in row:
                    processed_row[idx] = self._latest_values[col] = str(row[col])
                else:
                    processed_row[idx] = self._latest_values.get(col, "")

            log.debug(f"Writing line with index {processed_row[0]} to CSV file .")
            self._write_file(processed_row)

        # Close current file and create a new file with a different name if the size limit was exceeded.
        if size_limit_exceeded and buffer_target <= _len_buffer:
            log.info(f"CSV File size limit exceeded. Closing current file {self._filepath}.")
            self._file.close()
            self._file = None
            self.filepath = self.filepath.with_name(f"{self.filepath.stem}_{datetime.now().strftime('%y%m%d%H%M')}.csv")

            self._open_file(exclusive_creation=True)

            if flush:
                self.write(flush=True)

    def __exit__(self, *exc_details):
        """Exit the context manager

        :param exc_details: Execution details
        """
        self.write(flush=True)
        self._file.close()


class DFSubHandler(SubscriptionHandler):
    """Subscription handler for returning pandas data frames when requested

    :param write_interval: Interval for writing data
    :param size_limit: Number of rows to keep in internal _data memory. Default 100.
    """

    def __init__(self, write_interval: int = 1, size_limit: int = 100) -> None:
        super().__init__(write_interval=write_interval)
        self._data: pd.DataFrame = pd.DataFrame()
        self.keep_data_rows: int = size_limit

    def push(
        self,
        node: Node,
        value: Union[Any, pd.Series, Sequence[Any]],
        timestamp: Union[datetime, pd.DatetimeIndex, TimeStep, None] = None,
    ) -> None:
        """Append values to the dataframe

        :param node: Node object the data belongs to
        :param value: Value of the data or Series of values. There must be corresponding timestamps for each value.
        :param timestamp: Timestamp of receiving the data or DatetimeIndex if pushing multiple values. Alternatively
                          an integer/timedelta can be provided to determine the interval between data points. Use
                          negative numbers to describe past data. Integers are interpreted as seconds. If value is a
                          pd.Series and has a pd.DatetimeIndex, timestamp is ignored.
        """
        # Check if node.name is in _data.columns
        if node.name not in self._data.columns:
            self._data[node.name] = np.nan

        # Multiple values
        if hasattr(value, "__len__"):
            value = self._convert_series(value, timestamp)
            # Push Series
            # Values are rounded to self.write_interval in _convert_series
            for _timestamp, _value in value.items():
                _timestamp = self._assert_tz_awareness(_timestamp)
                self._data.loc[_timestamp, node.name] = _value

        # Single value
        else:
            if not isinstance(timestamp, datetime) and timestamp is not None:
                raise ValueError("Timestamp must be a datetime object or None.")
            timestamp = self._round_timestamp(timestamp if timestamp is not None else datetime.now())
            self._data.loc[timestamp, node.name] = value

        # Housekeeping (Keep internal data short)
        self._housekeeping()

    def get_latest(self) -> Union[pd.DataFrame, None]:
        """Return a copy of the dataframe, this ensures they can be worked on freely. Returns None if data is empty."""
        if len(self._data.index) == 0:
            return None  # If no data in self._data, return None
        else:
            self._data.fillna(method="ffill", inplace=True)
            return self._data.iloc[[-1]].copy()

    @property
    def data(self) -> Union[pd.DataFrame, None]:
        """This contains the interval dataframe and will return a copy of that."""
        self._data.fillna(method="ffill", inplace=True)
        return self._data.copy()

    def reset(self) -> None:
        """Reset the internal data and restart collection"""
        self._data = pd.DataFrame()
        log.info(f"Subscribed DataFrame {hash(self._data)} was reset successfully.")

    def _housekeeping(self) -> None:
        """Keep internal data short by only keeping last rows as specified in self.keep_data_rows"""
        self._data.drop(index=self._data.index[: -self.keep_data_rows], inplace=True)

    def close(self) -> None:
        """This is just here to satisfy the interface, not needed in this case."""
        pass
