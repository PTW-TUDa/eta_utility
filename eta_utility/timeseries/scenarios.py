import pathlib
from datetime import datetime, timedelta
from typing import Mapping, Optional, Sequence, Union

import pandas as pd

from eta_utility import timeseries


def scenario_from_csv(
    paths: Union[pathlib.Path, Sequence[pathlib.Path]],
    data_prefixes: Sequence[str] = None,
    *,
    start_time: datetime,
    end_time: datetime,
    total_time: Union[timedelta, int],
    random: Optional[bool] = False,
    resample_time: Optional[Union[timedelta, int]] = None,
    resample_method: Optional[str] = None,
    interpolation_method: Optional[Union[Sequence[str], str]] = None,
    rename_cols: Optional[Mapping[str, str]] = None,
    prefix_renamed: Optional[bool] = True,
    infer_datetime_from: Optional[Union[str, Sequence[int]]] = "string",
    time_conversion_str: str = "%Y-%m-%d %H:%M",
):
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
    :return:
    """
    if hasattr(paths, "__len__") and len(paths) > 1 and (data_prefixes is None or len(paths) != len(data_prefixes)):
        raise ValueError(
            "The number of paths and data_prefixes does not correspond to "
            "each other: {}\n{}".format(paths, data_prefixes)
        )
    elif not hasattr(paths, "__len__"):
        paths = (paths,)

    if (
        hasattr(paths, "__len__")
        and len(paths) > 1
        and hasattr(interpolation_method, "__len__")
        and len(paths) != len(interpolation_method)
    ):
        raise ValueError(
            "The number of interpolation methods does not match the number of paths. Specify 0, 1 or"
            "'number of paths' interpolation methods."
        )
    elif not hasattr(interpolation_method, "__len__"):
        interpolation_method = [interpolation_method] * len(paths)

    # Set defaults and convert values where necessary
    total_time = total_time if isinstance(total_time, timedelta) else timedelta(seconds=total_time)
    resample = True if resample_time is not None else False
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
                resample_method=resample_method,
                missing_data=interpolation_method[i],
            )
        data = data[slice_begin:slice_end].copy()

        col_names = {}
        for col in data.columns:
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
        data.rename(columns=col_names, inplace=True)
        df = pd.concat((data, df), 1)

    # Make sure that the resulting file corresponds to the requested time sclie
    if (
        len(df) <= 0
        or df.first_valid_index() > slice_begin + resample_time
        or df.last_valid_index() < slice_end - resample_time
    ):
        raise ValueError(
            "The loaded scenario file does not contain enough data for the entire selected time slice. Or the set "
            "scenario times do not correspond to the provided data."
        )

    return df
