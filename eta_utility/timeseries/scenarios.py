import pathlib
from datetime import datetime, timedelta
from typing import Mapping, Optional, Sequence, SupportsFloat, Union

import numpy as np
import pandas as pd

from eta_utility import timeseries
from eta_utility.type_hints.custom_types import TimeStep


def scenario_from_csv(
    paths: Union[pathlib.Path, Sequence[pathlib.Path]],
    data_prefixes: Optional[Sequence[str]] = None,
    *,
    start_time: datetime,
    end_time: Optional[datetime] = None,
    total_time: Optional[TimeStep] = None,
    random: Union[np.random.BitGenerator, bool, None] = False,
    resample_time: Optional[TimeStep] = None,
    resample_method: Optional[Union[Sequence[str], str]] = "asfreq",
    interpolation_method: Union[Sequence[str], str, None] = None,
    rename_cols: Optional[Mapping[str, str]] = None,
    prefix_renamed: Optional[bool] = True,
    infer_datetime_from: Optional[Union[str, Sequence[int]]] = "string",
    time_conversion_str: str = "%Y-%m-%d %H:%M",
    scaling_factors: Union[Sequence[Mapping[str, SupportsFloat]], SupportsFloat, None] = None,
) -> pd.DataFrame:
    """Import (possibly multiple) scenario data files from csv files and return them as a single pandas
    data frame. The import function supports column renaming and will slice and resample data as specified.

    :raises ValueError: If start and/or end times are outside of the scope of the imported scenario files.

    .. note::
        The ValueError will only be raised when this is true for all files. If only one file is outside of
        the range, and empty series will be returned for that file. TODO - Implement additional checks.

    :param paths: Path(s) to one or more CSV data files. The paths should be fully qualified.
    :param data_prefixes: If more than file is imported, a list of data_prefixes must be supplied such that
                          ambiquity of column names between the files can be avoided. There must be one prefix
                          for every imported file, such that a distinct prefix can be prepended to all columns
                          of a file.
    :param start_time: (Keyword only) Starting time for the scenario import. Default value is scenario_time_begin.
    :param end_time: (Keyword only) Latest ending time for the scenario import. Default value is scenario_time_end.
    :param total_time: (Keyword only) Total duration of the imported scenario. If given as int this will be
                       interpreted as seconds. The default is episode_duration.
    :param random: (Keyword only) Set to true if a random starting point (within the interval determined by
                   start_time and end_time should be chosen. This will use the environments random generator.
                   The default is false.
    :param resample_time: (Keyword only) Resample the scenario data to the specified interval. If this is specified
                          one of 'upsample_fill' or downsample_method' must be supplied as well to determin how
                          the new data points should be determined. If given as an in, this will be interpreted as
                          seconds. The default is no resampling.
    :param resample_method: (Keyword only) Method for filling in / aggregating data when resampling. Pandas
                            resampling methods are supported. Default is None (no resampling)
    :param interpolation_method: (Keyword only) Method for interpolating missing data values. Pandas missing data
                                 handling methods are supported. If a list with one value per file is given, the
                                 specified method will be selected according the order of paths.
    :param rename_cols: (Keyword only) Rename columns of the imported data. Maps the colunms as they appear in the
                        data files to new names. Format: {old_name: new_name}
    :param prefix_renamed: (Keyword only) Should prefixes be applied to renamed columns as well? Default: True.
                           When setting this to false make sure that all columns in all loaded scenario files
                           have different names. Otherwise there is a risk of overwriting data.
    :param infer_datetime_from: (Keyword only) Specify how datetime values should be converted. 'dates' will use
                                pandas to automatically determine the format. 'string' uses the conversion string
                                specified in the 'time_conversion_str' parameter. If a two-tuple of the format
                                (row, col) is given, data from the specified field in the data files will be used
                                to determine the date format. The default is 'string'
    :param time_conversion_str: (Keyword only) Time conversion string. This must be specified if the
                                infer_datetime_from parameter is set to 'string'. The string should specify the
                                datetime format in the python strptime format. The default is: '%Y-%m-%d %H:%M'.
    :param scaling_factors:
    :return:
    """

    if not hasattr(paths, "__len__"):
        paths = (paths,)

    # resample methods needs to be a list, so on case of None create a list of Nones
    if not hasattr(resample_method, "__len__"):
        if paths.__len__() > 1:
            resample_method = [resample_method] * len(paths)
        else:
            resample_method = (resample_method,)
    elif resample_method.__len__() != paths.__len__():
        raise ValueError("The number of resample methods does not match the number of paths.")

    # interpolation methods needs to be a list, so in case of None create a list of Nones
    if not hasattr(interpolation_method, "__len__"):
        if paths.__len__() > 1:
            interpolation_method = [interpolation_method] * len(paths)
        else:
            interpolation_method = (interpolation_method,)
    elif interpolation_method.__len__() != paths.__len__():
        raise ValueError("The number of interpolation methods does not match the number of paths.")

    # scaling needs to be a list, so on case of None create a list of Nones
    if not hasattr(scaling_factors, "__len__"):
        if paths.__len__() > 1:
            raise ValueError("The scaling factors need to be defined for each path")
        else:
            scaling_factors = (scaling_factors,)
    elif scaling_factors.__len__() != paths.__len__():
        raise ValueError("The number of scaling factors does not match the number of paths.")

    # Set defaults and convert values where necessary
    if total_time:
        total_time = total_time if isinstance(total_time, timedelta) else timedelta(seconds=total_time)
    resample = True if resample_time is not None else False

    if resample:
        resample_time = resample_time if isinstance(resample_time, timedelta) else timedelta(seconds=resample_time)

    slice_begin, slice_end = timeseries.find_time_slice(
        start_time,
        end_time,
        total_time=total_time,
        random=random,
        round_to_interval=resample_time,
    )

    df = pd.DataFrame()
    for i, path in enumerate(paths):
        data = timeseries.df_from_csv(
            path,
            infer_datetime_from=infer_datetime_from,
            time_conversion_str=time_conversion_str,
        )
        if resample:
            data = timeseries.df_resample(
                data,
                resample_time,
                resample_method=resample_method[i],
                missing_data=interpolation_method[i],
            )
        else:
            data = data.fillna(method=interpolation_method[i])
        data = data[slice_begin:slice_end].copy()

        col_names = {}
        for col in data.columns:
            # Figure out correct name for the column
            if not prefix_renamed and rename_cols is not None and col in rename_cols:
                pre = ""
                name = str(rename_cols[col])
            elif prefix_renamed and rename_cols is not None and col in rename_cols:
                pre = "{}_".format(data_prefixes[i]) if data_prefixes is not None else ""
                name = str(rename_cols[col])
            else:
                pre = "{}_".format(data_prefixes[i]) if data_prefixes is not None else ""
                name = str(col)
            col_names[col] = pre + name

            # Scale data values in the column
            if scaling_factors[i] and col in scaling_factors[i]:
                data[col] = data[col].multiply(scaling_factors[i][col])

        # rename all columns with the name mapping determined above
        data.rename(columns=col_names, inplace=True)
        df = pd.concat((data, df), 1)

    # Make sure that the resulting file corresponds to the requested time slice
    if (
        len(df) <= 0 or df.first_valid_index() > slice_begin + resample_time
        if resample_time is not None
        else timedelta(seconds=0) or df.last_valid_index() < slice_end - resample_time
        if resample_time is not None
        else timedelta(seconds=0)
    ):
        raise ValueError(
            "The loaded scenario file does not contain enough data for the entire selected time slice. Or the set "
            "scenario times do not correspond to the provided data."
        )

    return df
