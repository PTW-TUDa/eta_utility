from __future__ import annotations

import copy
import csv
import io
import json
import locale
import logging
import math
import os
import pathlib
import re
import socket
import sys
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Mapping, Sequence
from urllib.parse import urlparse, urlunparse

import pandas as pd
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from dateutil import tz

if TYPE_CHECKING:
    import types
    from tempfile import _TemporaryFileWrapper
    from typing import Any, Generator
    from urllib.parse import ParseResult

    from .type_hints import Path, PrivateKey


LOG_DEBUG = 1
LOG_INFO = 2
LOG_WARNING = 3
LOG_ERROR = 4
LOG_PREFIX = "eta_utility"
LOG_FORMATS = {
    "simple": "[%(levelname)s] %(message)s",
    "logname": "[%(name)s: %(levelname)s] %(message)s",
    "time": "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s",
}


def get_logger(
    name: str | None = None,
    level: int | None = None,
    format: str | None = None,  # noqa: A002
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
    :param format: Format of the log output. One of: simple, logname, time. (default: simple).
    :return: The *eta_utility* logger.
    """

    if name is not None and name != LOG_PREFIX:
        # Child loggers
        name = ".".join((LOG_PREFIX, name))
        log = logging.getLogger(name)
    else:
        # Main logger (only add handler if it does not have one already)
        log = logging.getLogger(LOG_PREFIX)
        log.propagate = False

        fmt = LOG_FORMATS[format] if format in LOG_FORMATS else LOG_FORMATS["simple"]

        if not log.hasHandlers():
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(logging.Formatter(fmt=fmt))
            log.addHandler(handler)

    if level is not None:
        log.setLevel(int(level * 10))
    return log


def log_add_filehandler(
    filename: Path,
    level: int | None = None,
    format: str | None = None,  # noqa: A002
) -> logging.Logger:
    """Add a file handler to the logger to save the log output.

    :param logger: Logger where file handler is added.
    :param filename: File path where logger is stored.
    :param level: Logging level (higher is more verbose between 0 - no output and 4 - debug).
    :param format: Format of the log output. One of: simple, logname, time. (default: time).
    :return: The *FileHandler* logger.
    """
    log = logging.getLogger(LOG_PREFIX)
    _format = format if format in LOG_FORMATS else LOG_FORMATS["time"]
    _filename = filename if isinstance(filename, pathlib.Path) else pathlib.Path(filename)

    filehandler = logging.FileHandler(filename=_filename)
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(logging.Formatter(fmt=_format))
    log.addHandler(filehandler)

    if level is not None:
        filehandler.setLevel(int(level * 10))

    return log


def log_add_streamhandler(
    level: int | None = None,
    format: str | None = None,  # noqa: A002
) -> logging.Logger:
    """Add a stream handler to the logger to show the log output.

    :param level: Logging level (higher is more verbose between 0 - no output and 4 - debug).
    :param format: Format of the log output. One of: simple, logname, time. (default: time).
    :return: The eta_utility logger with an attached StreamHandler
    """
    log = logging.getLogger(LOG_PREFIX)
    _format = format if format in LOG_FORMATS else LOG_FORMATS["time"]

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(fmt=_format))
    log.addHandler(handler)

    if level is not None:
        handler.setLevel(int(level * 10))

    return log


log = get_logger("util")


def json_import(path: Path) -> list[Any] | dict[str, Any]:
    """Extend standard JSON import to allow '//' comments in JSON files.

    :param path: Path to JSON file.
    :return: Parsed dictionary.
    """
    path = pathlib.Path(path) if not isinstance(path, pathlib.Path) else path

    try:
        # Remove comments from the JSON file (using regular expression), then parse it into a dictionary
        cleanup = re.compile(r"^((?:(?:[^\/\"])*(?:\"[^\"]*\")*(?:\/[^\/])*)*)", re.MULTILINE)
        with path.open("r") as f:
            file = "\n".join(cleanup.findall(f.read()))
        try:
            result = json.loads(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Error while decoding file {path}: {e.msg}", e.doc, e.pos)
        log.info(f"JSON file {path} loaded successfully.")
    except OSError as e:
        log.error(f"JSON file couldn't be loaded: {e.strerror}. Filename: {e.filename}")
        raise

    return result


def url_parse(url: str | None, scheme: str = "") -> tuple[ParseResult, str | None, str | None]:
    """Extend parsing of URL strings to find passwords and remove them from the original URL.

    :param url: URL string to be parsed.
    :return: Tuple of ParseResult object and two strings for username and password.
    """
    if url is None or url == "":
        _url = urlparse("")
    else:
        _url = urlparse(f"//{url.strip()}" if "//" not in url else url.strip(), scheme=scheme)

    # Get username and password either from the arguments or from the parsed URL string
    usr = str(_url.username) if _url.username is not None else None
    pwd = str(_url.password) if _url.password is not None else None

    # Find the "password-free" part of the netloc to prevent leaking secret info
    if usr is not None:
        match = re.search("(?<=@).+$", str(_url.netloc))
        if match:
            _url = urlparse(
                str(urlunparse((_url.scheme, match.group(), _url.path, _url.query, _url.fragment, _url.fragment)))
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


def dict_search(dikt: dict[str, str], val: str) -> str:
    """Function to get key of _psr_types dictionary, given value.
    Raise ValueError in case of value not specified in data.

    :param val: value to search
    :param data: dictionary to search for value
    :return: key of the dictionary
    """
    for key, value in dikt.items():
        if val == value:
            return key
    raise ValueError(f"Value: {val} not specified in specified dictionary")


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
    index: Sequence[int] | pd.DatetimeIndex | None = None,
    *,
    sep: str = ";",
    decimal: str = ".",
) -> None:
    """Export data to CSV file.

    :param path: Directory path to export data.
    :param data: Data to be saved.
    :param names: Field names used when data is a Matrix without column names.
    :param index: Optional sequence to set an index
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
        data.index = index
        data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    elif isinstance(data, Sequence):
        if names is not None:
            cols = names
        elif isinstance(data[-1], Mapping):
            cols = list(data[-1].keys())
        else:
            raise ValueError("Column names for csv export not specified.")

        _data = pd.DataFrame(data=data, columns=cols)
        _data.index = index
        _data.to_csv(path_or_buf=str(_path), sep=sep, decimal=decimal)

    log.info(f"Exported CSV data to {_path}.")


def replace_decimal_str(value: str | float, decimal: str = ".") -> str:
    """Replace the decimal sign in a string.

    :param value: The value to replace in.
    :param decimal: New decimal sign.
    """
    return str(value).replace(".", decimal)


def ensure_timezone(dt_value: datetime) -> datetime:
    """Helper function to check if datetime has timezone and if not assign local time zone.

    :param dt_value: Datetime object
    :return: datetime object with timezone information"""
    if dt_value.tzinfo is None:
        return dt_value.replace(tzinfo=tz.tzlocal())
    return dt_value


def round_timestamp(dt_value: datetime, interval: float = 1, ensure_tz: bool = True) -> datetime:
    """Helper method for rounding date time objects to specified interval in seconds.
    The method will also add local timezone information is None in datetime and
    if ensure_timezone is True.

    :param dt_value: Datetime object to be rounded
    :param interval: Interval in seconds to be rounded to
    :param ensure_tz: Boolean value to ensure or not timezone info in datetime
    :return: Rounded datetime object
    """
    if ensure_tz:
        dt_value = ensure_timezone(dt_value)
    timezone_store = dt_value.tzinfo

    rounded_timestamp = math.ceil(dt_value.timestamp() / interval) * interval

    return datetime.fromtimestamp(rounded_timestamp).replace(tzinfo=timezone_store)


class KeyCertPair(ABC):
    """KeyCertPair is a wrapper for an RSA private key file and a corresponding x509 certificate. Implementations
    provide a contextmanager "tempfiles", which provides access to the certificate files and the
    properties key and cert, which contain the RSA key and certificate information."""

    def __init__(self, key: PrivateKey, cert: x509.Certificate):
        self._key = key
        self._cert = cert

    @property
    def key(self) -> PrivateKey:
        """RSA private key for the certificate."""
        return self._key

    @property
    def cert(self) -> x509.Certificate:
        """x509 certificate information."""
        return self._cert

    @contextmanager
    @abstractmethod
    def tempfiles(self) -> Generator:
        """Accessor for temporary certificate files."""
        raise NotImplementedError

    @property
    @abstractmethod
    def key_path(self) -> str:
        """Path to the key file."""
        raise NotImplementedError

    @property
    @abstractmethod
    def cert_path(self) -> str:
        """Path to the certificate file."""
        raise NotImplementedError


class SelfsignedKeyCertPair(KeyCertPair):
    """Self signed key and certificate pair for use with the connectors.

    :param common_name: Common name the certificate should be valid for.
    :param passphrase: Pass phrase for encryption of the private key.
    :param country: Country code for the certificate owner, for example "DE" oder "US".
    :param province: Province name of the certificate owner. Empty by default.
    :param city: City name of the certificate owner. Empty by default.
    :param organization: Name of the certificate owner's organization. "OPC UA Client" by default.
    """

    def __init__(
        self,
        common_name: str,
        passphrase: str | None = None,
        country: str | None = None,
        province: str | None = None,
        city: str | None = None,
        organization: str | None = None,
    ) -> None:
        super().__init__(*self.generate_cert(common_name, country, province, city, organization))

        self._key_tempfile: _TemporaryFileWrapper[bytes] | None = None
        self._cert_tempfile: _TemporaryFileWrapper[bytes] | None = None

        self.passphrase = passphrase

    def generate_cert(
        self,
        common_name: str,
        country: str | None = None,
        province: str | None = None,
        city: str | None = None,
        organization: str | None = None,
    ) -> tuple[rsa.RSAPrivateKey, x509.Certificate]:
        """Generate a self signed key and certificate pair for use with the connectors.

        :param common_name: Common name the certificate should be valid for.
        :param country: Country code for the certificate owner, for example "DE" oder "US".
        :param province: Province name of the certificate owner. Empty by default.
        :param city: City name of the certificate owner. Empty by default.
        :param organization: Name of the certificate owner's organization. "OPC UA Client" by default.
        :return: Tuple of RSA private key and x509 certificate.
        """
        # Set some default values for the parameters
        if country is None:
            country = locale.setlocale(locale.LC_ALL, locale.getdefaultlocale()[0]).split("_")[-1]

        if province is None:
            province = ""

        if city is None:
            city = ""

        if organization is None:
            organization = "OPC UA Client"

        # Generate the private key
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        # Determine certificate subject and issuer from input values.
        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COUNTRY_NAME, country),
                x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, province),
                x509.NameAttribute(NameOID.LOCALITY_NAME, city),
                x509.NameAttribute(NameOID.ORGANIZATION_NAME, organization),
                x509.NameAttribute(NameOID.COMMON_NAME, common_name),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=10))  # Certificate valid for 10 days
            .add_extension(
                x509.SubjectAlternativeName(
                    [x509.DNSName("localhost"), x509.DNSName(socket.gethostbyname(socket.gethostname()))]
                ),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )  # Sign certificate with our private key

        return key, cert

    def store_tempfile(self) -> tuple[str, str]:
        """Store the key and certificate as named temporary files. The function returns the names
        of the two files.

        :return: Tuple of name of the key file and name of the certificate file.
        """
        # store key
        if self.passphrase is not None:
            encryption: serialization.KeySerializationEncryption = serialization.BestAvailableEncryption(
                bytes(self.passphrase, "utf-8")
            )
        else:
            encryption = serialization.NoEncryption()

        self._key_tempfile = NamedTemporaryFile("w+b", delete=False, suffix=".pem")
        assert self._key_tempfile is not None
        self._key_tempfile.write(
            self.key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=encryption,
            )
        )

        # store cert
        self._cert_tempfile = NamedTemporaryFile("w+b", delete=False, suffix=".pem")
        assert self._cert_tempfile is not None
        self._cert_tempfile.write(self.cert.public_bytes(serialization.Encoding.PEM))

        return self.key_path, self.cert_path

    @contextmanager
    def tempfiles(self) -> Generator:
        """Accessor for temporary certificate files."""
        try:
            self.store_tempfile()
            assert self._key_tempfile is not None
            assert self._cert_tempfile is not None

            self._key_tempfile.close()
            self._cert_tempfile.close()
            yield self
        finally:
            try:
                assert self._key_tempfile is not None
                os.unlink(self._key_tempfile.name)
            except BaseException:
                pass

            try:
                assert self._cert_tempfile is not None
                os.unlink(self._cert_tempfile.name)
            except BaseException:
                pass

    @property
    def key_path(self) -> str:
        """Path to the key file."""
        if self._key_tempfile is not None:
            return self._key_tempfile.name
        else:
            raise RuntimeError("Create the key file before trying to reference the filename")

    @property
    def cert_path(self) -> str:
        """Path to the certificate file."""
        if self._cert_tempfile is not None:
            return self._cert_tempfile.name
        else:
            raise RuntimeError("Create the certificate file before trying to reference the filename")


class PEMKeyCertPair(KeyCertPair):
    """Load a PEM formatted key and certificate pair from files.

    :param key_path: Path to the PEM formatted RSA private key file.
    :param cert_path: Path to the PEM formatted certificate file.
    :param passphrase: Pass phrase for encryption of the private key.
    """

    def __init__(self, key_path: Path, cert_path: Path, passphrase: str | None) -> None:
        self._key_path = pathlib.Path(key_path) if not isinstance(key_path, pathlib.Path) else key_path
        self._cert_path = pathlib.Path(cert_path) if not isinstance(cert_path, pathlib.Path) else cert_path

        with self._cert_path.open("rb") as _c:
            cert = x509.load_pem_x509_certificate(_c.read())

        _passphrase = bytes(passphrase, "utf-8") if passphrase is not None else None

        with self._key_path.open("rb") as _k:
            key = serialization.load_pem_private_key(_k.read(), password=_passphrase)

        super().__init__(key, cert)

        self.passphrase = passphrase

    @contextmanager
    def tempfiles(self) -> Generator:
        """Accessor for temporary certificate files."""
        yield self

    @property
    def key_path(self) -> str:
        """Path to the key file."""
        return self._key_path.as_posix()

    @property
    def cert_path(self) -> str:
        """Path to the certificate file."""
        return self._cert_path.as_posix()


class Suppressor(io.TextIOBase):
    """Context manager to suppress standard output."""

    def __enter__(self) -> Suppressor:
        self.stderr = sys.stderr
        sys.stderr = self  # type: ignore
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None
    ) -> None:
        sys.stderr = self.stderr
        if exc_type is not None:
            raise exc_type(exc_val).with_traceback(exc_tb)

    def write(self, x: Any) -> int:
        return 0
