""" Simple helpers for reading timeseries data from a csv file and getting slices or resampled data. This module
handles data using pandas dataframe objects.
"""
from __future__ import annotations

import csv
import operator as op
import pathlib
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Sequence

    from eta_utility.type_hints import Path, TimeStep

log = get_logger("timeseries")


def df_from_csv(
    path: Path,
    *,
    delimiter: str = ";",
    infer_datetime_from: str | Sequence[int] | tuple[int, int] = "dates",
    time_conversion_str: str = "%Y-%m-%d %H:%M",
) -> pd.DataFrame:
    """Take data from a csv file, process it and return a Timeseries (pandas Data Frame) object.

    Open and read the .csv file, perform error checks and ensure that valid float values are obtained. This
    assumes that the first column is always the date and time column and provides multiple methods to convert
    this column. It also assumes that the first row is the header row.
    The header row is converted to lower case and spaces are converted to _. If header values contain special
    characters, everything starting from the first special character is discarded.

    :param path: Path to the .csv file.
    :param delimiter: Delimiter used between csv fields.
    :param infer_datetime_from: Specify how date and time values should be inferred. This can be 'dates' or 'string'
                                or a tuple/list with two values.

                                * If 'dates' is specified, pandas will be used to automatically infer the datetime
                                  format from the file.
                                * If 'string' is specified, the parameter 'time_conversion_str' must specify the
                                  string (in python strptime format) to convert datetime values.
                                * If a tuple/list of two values is given, the time format specification (according to
                                  python strptime format) will be read from the specified field in the .csv
                                  file ('row', 'column').

    :param time_conversion_str: Time conversion string according to the python (strptime) format.
    """
    path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

    conversion_string = None
    infer_datetime_format = False
    parse_dates = False
    if type(infer_datetime_from) is str and infer_datetime_from == "dates":
        infer_datetime_format = True
    elif type(infer_datetime_from) is str and infer_datetime_from == "string":
        conversion_string = time_conversion_str
    elif type(infer_datetime_from) in {list, tuple}:
        if not len(infer_datetime_from) == 2:
            raise ValueError(
                "Field for date format must be specified in the format ['row', 'col']. "
                "Got {}".format(infer_datetime_from)
            )

    else:
        raise ValueError(
            "infer_datetime_from must be one of 'dates', 'string', or a tuple of ('row', 'col'), "
            "Got: {}".format(infer_datetime_from)
        )

    # Read names from header and format them such that they can be used easily as dataframe indices.
    # If required by infer_datetime_from also read time format from the file.
    with path.open("r") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first_line = next(reader)
            if isinstance(infer_datetime_from, list) or isinstance(infer_datetime_from, tuple):
                if infer_datetime_from[0] > 0:
                    for _ in range(1, infer_datetime_from[0]):
                        conversion_line = next(reader)
                else:
                    conversion_line = first_line
                conversion_string = "%" + conversion_line[infer_datetime_from[1]].split("%", 1)[1].strip()
        except StopIteration:
            raise EOFError(
                "The CSV file does not contain the specified date format field {}. \n"
                "File path: {}".format(infer_datetime_from, path)
            )

        # Find number of fields, names of fields and a conversion string for time
        length = len(first_line)
        splitter = re.compile("[^A-Za-z0-9 _-]")
        names = [splitter.split(s.strip(), 1)[0].lower().strip().replace(" ", "_") for s in first_line]

    # Load the CSV file
    if infer_datetime_format:
        parse_dates = True

    def converter(val: str) -> float:
        val = str(val).strip().replace(" ", "").replace(",", ".")
        if len(val) > 0:
            fval = float(val)
        else:
            fval = float("nan")
        return fval

    data = pd.read_csv(
        path,
        header=0,
        names=names,
        delimiter=delimiter,
        index_col=0,
        parse_dates=parse_dates,
        converters={y: converter for y in range(1, length)},
    )

    if conversion_string:
        data.index = pd.to_datetime(data.index, format=conversion_string)

    log.info(f"Loaded data from csv file: {path}")
    return data


def find_time_slice(
    time_begin: datetime,
    time_end: datetime | None = None,
    total_time: TimeStep | None = None,
    round_to_interval: TimeStep | None = None,
    random: bool | np.random.Generator = False,
) -> tuple[datetime, datetime]:
    """Return a (potentially random) slicing interval that can be used to slice a data frame.

    :param time_begin: Date and time of the beginning of the interval to slice from.
    :param time_end: Date and time of the ending of the interval to slice from.
    :param total_time: Specify the total time of the sliced interval. An integer will be interpreted as seconds.
                       If this argument is None, the complete interval between beginning and end will be returned.
    :param round_to_interval: Round times to a specified interval, this value is interpreted as seconds if given as
                              an int. Default is no rounding.
    :param random: If this value is true, or a random generator is supplied, it will be used to generate a
                   random slice of length total_time in the interval between time_begin and time_end.
    :return: Tuple of slice_begin time and slice_end time. Both times are datetime objects.
    """
    # Determine ending time and total time depending on what was supplied.
    if time_end is not None and total_time is None:
        total_time = time_end - time_begin
    elif total_time is not None:
        total_time = timedelta(seconds=total_time) if not isinstance(total_time, timedelta) else total_time
        time_end = time_begin + total_time if time_end is None else time_end
    else:
        raise ValueError("At least one of time_end and total_time must be specified to fully constrain the interval.")

    round_to_interval = (
        round_to_interval.total_seconds() if isinstance(round_to_interval, timedelta) else round_to_interval
    )

    # Determine the (possibly random) beginning time of the time slice and round it if necessary
    if random:
        if isinstance(random, bool):
            random = np.random.default_rng()
            log.info(
                "Using an unseeded random generator for time slicing. This will not produce deterministic " "results."
            )
        time_gap = max(0, (time_end - time_begin - total_time).total_seconds())
        if time_gap <= 0:
            log.warn(
                "Could not use random time sampling because the gap between the required starting and ending "
                "times is too small."
            )
        slice_begin = time_begin + timedelta(seconds=random.uniform() * time_gap)
    else:
        slice_begin = time_begin
    if round_to_interval is not None and round_to_interval > 0:
        slice_begin = datetime.fromtimestamp((slice_begin.timestamp() // round_to_interval) * round_to_interval)

    # Determine the ending time of the time slice and round it if necessary
    slice_end = slice_begin + total_time
    if round_to_interval is not None and round_to_interval > 0:
        slice_end = datetime.fromtimestamp((slice_end.timestamp() // round_to_interval) * round_to_interval)

    return slice_begin, slice_end


def df_time_slice(
    df: pd.DataFrame,
    time_begin: datetime,
    time_end: datetime | None = None,
    total_time: TimeStep | None = None,
    round_to_interval: TimeStep | None = None,
    random: bool | np.random.Generator = False,
) -> pd.DataFrame:
    """Return a data frame which has been sliced starting at time_begin and ending at time_end, from df.

    :param df: Original data frame to be sliced.
    :param time_begin: Date and time of the beginning of the interval to slice from.
    :param time_end: Date and time of the ending of the interval to slice from.
    :param total_time: Specify the total time of the sliced interval. An integer will be interpreted as seconds.
                       If this argument is None, the complete interval between beginning and end will be returned.
    :param round_to_interval: Round times to a specified interval, this value is interpreted as seconds if given as
                              an int. Default is no rounding.
    :param random: If this value is true, or a random generator is supplied, it will be used to generate a
                   random slice of length total_time in the interval between time_begin and time_end.
    :return: Sliced data frame.
    """

    slice_begin, slice_end = find_time_slice(time_begin, time_end, total_time, round_to_interval, random)
    return df[slice_begin:slice_end].copy()  # type: ignore


def df_resample(
    df: pd.DataFrame, *periods_deltas: TimeStep | Sequence[TimeStep], missing_data: str | None = None
) -> pd.DataFrame:
    """Resample the time index of a data frame. This method can be used for resampling in multiple different
    periods with multiple different deltas between single time entries.

    :param df: DataFrame for processing.
    :param periods_deltas: If one argument is specified, this will resample the data to the specified interval
                           in seconds. If more than one argument is specified, they will be interpreted as
                           (periods, interval) pairs. The first argument specifies a number of periods that should
                           be resampled, the second value specifies the interval that these periods should be
                           resampled to. A third argument would determine the next number of periods that should
                           be resampled to the interval specified by the fourth argument and so on.
    :param missing_data: Specify a method for handling missing data values. If this is not specified, missing data
                         will not be handled. All missing data handling functions for pandas dataframes are valid.
                         See also: https://pandas.pydata.org/docs/reference/frame.html#missing-data-handling. Some
                         examples: 'interpolate', 'fillna' (default: asfreq).
    :return: Copy of the DataFrame.
    """
    if missing_data == "fillna":
        interpolation_method = op.methodcaller(missing_data, method="pad")
    elif missing_data == "interpolate":
        interpolation_method = op.methodcaller(missing_data, method="time")
    elif missing_data is None:
        interpolation_method = op.methodcaller("asfreq")
    else:
        interpolation_method = op.methodcaller(missing_data)

    if len(periods_deltas) == 1:
        delta = str(
            periods_deltas[0].total_seconds() if isinstance(periods_deltas[0], timedelta) else periods_deltas[0]
        )
        new_df = interpolation_method(interpolation_method(df.resample(str(delta) + "S")))
    else:
        new_df = pd.DataFrame()
        total_periods = 0
        for i in range(len(periods_deltas) // 2):
            key = i * 2
            delta = str(
                periods_deltas[key + 1].total_seconds()  # type: ignore
                if isinstance(periods_deltas[key + 1], timedelta)
                else periods_deltas[key + 1]
            )
            new_df = pd.concat(
                df,
                interpolation_method(
                    interpolation_method(
                        df.iloc[total_periods : periods_deltas[key]].resample(str(delta) + "S")  # type: ignore
                    )
                ),
            )
            total_periods += periods_deltas[key]  # type: ignore

    return new_df
