import json
import logging
import re
import sys
from typing import Any, Dict, Tuple
from urllib.parse import ParseResult, urlparse, urlunparse

from .type_hints import Path


def get_logger(name: str = None, level: int = None, format: str = None) -> logging.Logger:
    """Get eta_utility specific logger

    Call this without specifying a name to initiate logging. Set the "level" and "format" parameters to determine
    log output.

    .. note::
        When using this function internally (inside eta_utility) a name should always be specified to avoid leaking
        logging info to external loggers. Also note that this can only be called once without specifying a name!
        Subsequent calls will have no effect.

    :param name: Name of the logger
    :param level: Logging level (higher is more verbose)
    :param format: Format of the log output. One of: simple, logname. (default: simple)
    :return: logger
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


log = get_logger("util", 2)


def json_import(path: Path) -> Dict[str, Any]:
    """Extend standard json import to allow comments in json files

    :param path: path to json file
    :return: Parsed dictionary
    """
    try:
        # Remove comments from the json file (using regular expression), then parse it into a dictionary
        cleanup = re.compile(r"^\s*(.*?)(?=/{2}|$)", re.MULTILINE)
        with path.open("r") as f:
            file = "".join(cleanup.findall(f.read()))
        result = json.loads(file)
        log.info(f"JSON file {path} loaded successfully.")
    except OSError as e:
        log.error(f"JSON file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise

    return result


def url_parse(url: str) -> Tuple[ParseResult, str, str]:
    """Extend parsing of url strings to find passwords and remove them from the original URL

    :param url: URL string to be parsed
    :return: Tuple of ParseResult object and two strings for username and password
    """
    _url = urlparse(url)

    # Get username and password either from the arguments or from the parsed url string
    usr = _url.username if _url.username is not None else None
    pwd = _url.password if _url.password is not None else None

    # Find the "password-free" part of the netloc to prevent leaking secret info
    if usr is not None:
        match = re.search("(?<=@).+$", _url.netloc)
        _url = urlparse(urlunparse((_url.scheme, match.group(), _url.path, _url.query, _url.fragment, _url.fragment)))

    return _url, usr, pwd
