from __future__ import annotations

import pathlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Sequence, Sized

import numpy as np
import pandas as pd

from eta_utility import timeseries

if TYPE_CHECKING:
    from typing import Mapping, SupportsFloat

    from eta_utility.type_hints import Path, TimeStep


def scenario_from_csv(
    paths: Path | Sequence[Path],
    data_prefixes: Sequence[str] | None = None,
    *,
    start_time: datetime,
    end_time: datetime | None = None,
    total_time: TimeStep | None = None,
    random: np.random.Generator | bool | None = False,
    resample_time: TimeStep | None = None,
    interpolation_method: Sequence[str | None] | str | None = None,
    rename_cols: Mapping[str, str] | None = None,
    prefix_renamed: bool = True,
    infer_datetime_from: str | Sequence[Sequence[int]] | Sequence[str] = "string",
    time_conversion_str: str | Sequence[str] = "%Y-%m-%d %H:%M",
    scaling_factors: Sequence[Mapping[str, SupportsFloat]] | Mapping[str, SupportsFloat] | None = None,
) -> pd.DataFrame:
    """Import (possibly multiple) scenario data files from csv files and return them as a single pandas
    data frame. The import function supports column renaming and will slice and resample data as specified.

    :raises ValueError: If start and/or end times are outside the scope of the imported scenario files.

    .. note::
        The ValueError will only be raised when this is true for all files. If only one file is outside
        the range, an empty series will be returned for that file.

    :param paths: Path(s) to one or more CSV data files. The paths should be fully qualified.
    :param data_prefixes: If more than one file is imported, a list of data_prefixes must be supplied such that
                          ambiguity of column names between the files can be avoided. There must be one prefix
                          for every imported file, such that a distinct prefix can be prepended to all columns
                          of a file.
    :param start_time: Starting time for the scenario import.
    :param end_time: Latest ending time for the scenario import (default: inferred from start_time and total_time).
    :param total_time: Total duration of the imported scenario. If given as int this will be
                       interpreted as seconds (default: inferred from start_time and end_time).
    :param random: Set to true if a random starting point (within the interval determined by
                   start_time and end_time) should be chosen. This will use the environments' random generator.
    :param resample_time: Resample the scenario data to the specified interval. If this is specified
                          one of 'upsample_fill' or downsample_method' must be supplied as well to determine how
                          the new data points should be determined. If given as an int, this will be interpreted as
                          seconds (default: no resampling).
    :param interpolation_method: Method for interpolating missing data values. Pandas missing data
                                 handling methods are supported. If a list with one value per file is given, the
                                 specified method will be selected according to the order of paths.
    :param rename_cols: Rename columns of the imported data. Maps the columns as they appear in the
                        data files to new names. Format: {old_name: new_name}.

                        .. note::
                            The column names are normalized to lowercase and underscores are added in place of spaces.
                            Additionally, everything after the first symbol is removed. For example
                            "Water Temperature #2" becomes "water_temperature". So if you want to rename the column,
                            you need to specify for example: {"water_temperature": "T_W"}.


    :param prefix_renamed: Should prefixes be applied to renamed columns as well?
                           When setting this to false make sure that all columns in all loaded scenario files
                           have different names. Otherwise, there is a risk of overwriting data.
    :param infer_datetime_from: Specify how datetime values should be converted. 'dates' will use
                                pandas to automatically determine the format. 'string' uses the conversion string
                                specified in the 'time_conversion_str' parameter. If a two-tuple of the format
                                (row, col) is given, data from the specified field in the data files will be used
                                to determine the date format.
    :param time_conversion_str: Time conversion string. This must be specified if the
                                infer_datetime_from parameter is set to 'string'. The string should specify the
                                datetime format in the python strptime format.
    :param scaling_factors: Scaling factors for each imported column.
    :return: Imported and processed data as pandas.DataFrame.
    """

    if not isinstance(paths, Sized):
        paths = [paths]
    _paths = []
    for path in paths:
        _paths.append(path if isinstance(path, pathlib.Path) else pathlib.Path(path))

    # interpolation methods needs to be a list, so in case of None create a list of Nones
    if type(interpolation_method) is str or not isinstance(interpolation_method, Sized):
        _interpolation_method = [interpolation_method] * len(_paths)
    elif len(interpolation_method) != len(_paths):
        raise ValueError("The number of interpolation methods does not match the number of paths.")
    else:
        _interpolation_method = list(interpolation_method)

    # scaling needs to be a list, so on case of None create a list of Nones
    if not isinstance(scaling_factors, Sequence):
        if len(_paths) > 1:
            raise ValueError("The scaling factors need to be defined for each path")
        else:
            _scaling_factors = [scaling_factors]
    elif len(scaling_factors) != len(_paths):
        raise ValueError("The number of scaling factors does not match the number of paths.")
    else:
        _scaling_factors = list(scaling_factors)

    # time conversion string needs to be a list, so on case of None create a list of Nones
    if isinstance(time_conversion_str, str):
        _time_conversion_str = [time_conversion_str] * len(_paths)
    elif len(time_conversion_str) != len(_paths):
        raise ValueError("The number of time conversion strings does not match the number of paths.")
    else:
        _time_conversion_str = list(time_conversion_str)

    # columns to consider as datetime values (infer_datetime_from)
    # needs to be a list, so on case of None create a list of Nones
    if isinstance(infer_datetime_from, str):
        _infer_datetime_from: list[str | Sequence[int]] = [infer_datetime_from] * len(_paths)
    else:
        _infer_datetime_from = list(infer_datetime_from)

    # Set defaults and convert values where necessary
    if total_time is not None:
        total_time = total_time if isinstance(total_time, timedelta) else timedelta(seconds=total_time)

    resample = True if resample_time is not None else False
    if resample and resample_time is not None:
        _resample_time = resample_time if isinstance(resample_time, timedelta) else timedelta(seconds=resample_time)
    else:
        _resample_time = timedelta(seconds=0)

    _random = False if random is None else random

    slice_begin, slice_end = timeseries.find_time_slice(
        start_time,
        end_time,
        total_time=total_time,
        random=_random,
        round_to_interval=_resample_time,
    )

    df = pd.DataFrame()
    for i, path in enumerate(_paths):
        data = timeseries.df_from_csv(
            path,
            infer_datetime_from=_infer_datetime_from[i],
            time_conversion_str=_time_conversion_str[i],
        )
        if resample:
            data = timeseries.df_resample(
                data,
                _resample_time,
                missing_data=_interpolation_method[i],
            )
        else:
            data = data.fillna(method=_interpolation_method[i])
        data = data[slice_begin:slice_end].copy()  # type: ignore

        col_names = {}
        for col in data.columns:
            prefix = data_prefixes[i] if data_prefixes else None
            col_names[col] = _fix_col_name(col, prefix, prefix_renamed, rename_cols)

            # Scale data values in the column
            if _scaling_factors[i] is not None and col in _scaling_factors[i]:  # type: ignore
                data[col] = data[col].multiply(_scaling_factors[i][col])  # type: ignore

        # rename all columns with the name mapping determined above
        data.rename(columns=col_names, inplace=True)
        df = pd.concat((data, df), axis=1)

    # Make sure that the resulting file corresponds to the requested time slice
    if (
        len(df) <= 0
        or df.first_valid_index() > slice_begin + _resample_time
        or df.last_valid_index() < slice_end - _resample_time
    ):
        raise ValueError(
            "The loaded scenario file does not contain enough data for the entire selected time slice. Or the set "
            "scenario times do not correspond to the provided data."
        )

    return df


def _fix_col_name(
    name: str,
    prefix: str | None = None,
    prefix_renamed: bool = False,
    rename_cols: Mapping[str, str] | None = None,
) -> str:
    """Figure out correct name for the column.

    :param name: Name to rename.
    :param prefix: Prefix to preprend to the name.
    :param prefix_renamed: Prepend prefix if name is renamed?
    :param rename_cols: Mapping of old names to new names.
    """
    if not prefix_renamed and rename_cols is not None and name in rename_cols:
        pre = ""
        name = str(rename_cols[name])
    elif prefix_renamed and rename_cols is not None and name in rename_cols:
        pre = f"{prefix}_" if prefix else ""
        name = str(rename_cols[name])
    else:
        pre = f"{prefix}_" if prefix is not None else ""
        name = str(name)

    name = f"{pre}{name}"
    return name
