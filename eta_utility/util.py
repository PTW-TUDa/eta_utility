from __future__ import annotations

import copy
import csv
import json
import logging
import pathlib
import re
import sys
from typing import TYPE_CHECKING, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd

if TYPE_CHECKING:
    from typing import Any
    from urllib.parse import ParseResult

    from .type_hints import Path


def get_logger(
    name: str | None = None, level: int | None = None, format: str | None = None  # noqa: A002
) -> logging.Logger:
    """Get eta_utility specific logger.

    Call this without specifying a name to initiate logging. Set the "level" and "format" parameters to determine
    log output.

    .. note::
        When using this function internally (inside eta_utility) a name should always be specified to avoid leaking
        logging info to external loggers. Also note that this can only be called once without specifying a name!
        Subsequent calls will have no effect.

    :param name: Name of the logger.
    :param level: Logging level (higher is more verbose between 0 - no output and 4 - debug).
    :param format: Format of the log output. One of: simple, logname. (default: simple).
    :return: The *eta_utility* logger.
    """
    prefix = "eta_utility"

    if name is not None and name != prefix:
        # Child loggers
        name = ".".join((prefix, name))
        log = logging.getLogger(name)
    else:
        # Main logger (only add handler if it does not have one already)
        log = logging.getLogger(prefix)
        log.propagate = False

        formats = {"simple": "[%(levelname)s] %(message)s", "logname": "[%(name)s: %(levelname)s] %(message)s"}
        fmt = formats[format] if format in formats else formats["simple"]

        if not log.hasHandlers():
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(fmt=fmt))
            log.addHandler(handler)

    if level is not None:
        log.setLevel(int(level * 10))
    return log


log = get_logger("util")


def json_import(path: Path) -> dict[str, Any]:
    """Extend standard JSON import to allow '//' comments in JSON files.

    :param path: Path to JSON file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

    try:
        # Remove comments from the JSON file (using regular expression), then parse it into a dictionary
        cleanup = re.compile(r"^\s*(.*?)(?=/{2}|$)", re.MULTILINE)
        with path.open("r") as f:
            file = "".join(cleanup.findall(f.read()))
        result = json.loads(file)
        log.info(f"JSON file {path} loaded successfully.")
    except OSError as e:
        log.error(f"JSON file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise

    return result


def url_parse(url: str) -> tuple[ParseResult, str | None, str | None]:
    """Extend parsing of URL strings to find passwords and remove them from the original URL.

    :param url: URL string to be parsed.
    :return: Tuple of ParseResult object and two strings for username and password.
    """
    _url = urlparse(url)

    # Get username and password either from the arguments or from the parsed URL string
    usr = _url.username if _url.username is not None else None
    pwd = _url.password if _url.password is not None else None

    # Find the "password-free" part of the netloc to prevent leaking secret info
    if usr is not None:
        match = re.search("(?<=@).+$", _url.netloc)
        if match:
            _url = urlparse(
                urlunparse((_url.scheme, match.group(), _url.path, _url.query, _url.fragment, _url.fragment))
            )

    return _url, usr, pwd


def dict_get_any(dikt: dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
    """Get any of the specified items from dictionary, if any are available. The function will return
    the first value it finds, even if there are multiple matches.

    :param dikt: Dictionary to get values from.
    :param names: Item names to look for.
    :param fail: Flag to determine, if the function should fail with a KeyError, if none of the items are found.
                 If this is False, the function will return the value specified by 'default'.
    :param default: Value to return, if none of the items are found and 'fail' is False.
    :return: Value from dictionary.
    :raise: KeyError, if none of the requested items are available and fail is True.
    """
    for name in names:
        if name in dikt:
            # Return first value found in dictionary
            return dikt[name]

    if fail is True:
        raise KeyError(f"Did not find one of the required keys in the configuration: {names}")
    else:
        return default


def dict_pop_any(dikt: dict[str, Any], *names: str, fail: bool = True, default: Any = None) -> Any:
    """Pop any of the specified items from dictionary, if any are available. The function will return
    the first value it finds, even if there are multiple matches. This function removes the found values from the
    dictionary!

    :param dikt: Dictionary to pop values from.
    :param names: Item names to look for.
    :param fail: Flag to determine, if the function should fail with a KeyError, if none of the items are found.
                 If this is False, the function will return the value specified by 'default'.
    :param default: Value to return, if none of the items are found and 'fail' is False.
    :return: Value from dictionary.
    :raise: KeyError, if none of the requested items are available and fail is True.
    """
    for name in names:
        if name in dikt:
            # Return first value found in dictionary
            return dikt.pop(name)

    if fail is True:
        raise KeyError(f"Did not find one of the required keys in the configuration: {names}")
    else:
        return default


def deep_mapping_update(
    source: Mapping[str, str | Mapping[str, Any]], overrides: Mapping[str, str | Mapping[str, Any]]
) -> dict[str, str | Mapping[str, Any]]:
    """Perform a deep update of a nested dictionary or similar mapping.

    :param source: Original mapping to be updated.
    :param overrides: Mapping with new values to integrate into the new mapping.
    :return: New Mapping with values from the source and overrides combined.
    """
    output = dict(copy.deepcopy(source))
    for key, value in overrides.items():
        if isinstance(value, Mapping):
            output[key] = deep_mapping_update(dict(source).get(key, {}), value)  # type: ignore
        else:
            output[key] = value
    return output


def csv_export(
    path: Path,
    data: Mapping[str, Any] | Sequence[Mapping[str, Any] | Any] | pd.DataFrame,
    names: Sequence[str] | None = None,
    *,
    sep: str = ";",
    decimal: str = ".",
) -> None:
    """Export data to CSV file.

    :param path: Directory path to export data.
    :param data: Data to be saved.
    :param names: Field names used when data is a Matrix without column names.
    :param sep: Separator to use between the fields.
    :param decimal: Sign to use for decimal points.
    """
    _path = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    if _path.suffix != ".csv":
        _path.with_suffix(".csv")

    if isinstance(data, Mapping):
        exists = True if _path.exists() else False

        with _path.open("a") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys(), delimiter=sep)
            if not exists:
                writer.writeheader()

            writer.writerow({key: replace_decimal_str(val, decimal) for key, val in data.items()})

    elif isinstance(data, pd.DataFrame):
        data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    elif isinstance(data, Sequence):
        if names is not None:
            cols = names
        elif isinstance(data[0], Mapping):
            cols = list(data[0].keys())
        else:
            raise ValueError("Column names for csv export not specified.")

        _data = pd.DataFrame(data=data, columns=cols)
        _data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    log.info(f"Exported CSV data to {_path}.")


def replace_decimal_str(value: str | float, decimal: str = ".") -> str:
    """Replace the decimal sign in a string.

    :param value: The value to replace in.
    :param decimal: New decimal sign.
    """
    return str(value).replace(".", decimal)
