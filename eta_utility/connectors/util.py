from __future__ import annotations

import struct
from asyncio import sleep as async_sleep
from time import sleep
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable, Sequence


class RetryWaiter:
    """Helper class which keeps track of waiting time before retrying a connection."""

    values = [0, 1, 3, 5, 5, 10, 20, 30, 40, 60]

    def __init__(self) -> None:
        self.counter = 0

    def tried(self) -> None:
        """Register a retry with the RetryWaiter."""
        self.counter += 1

    def success(self) -> None:
        """Register a successful connection with the RetryWaiter."""
        self.counter = 0

    @property
    def wait_time(self) -> int:
        """Return the time to wait for."""
        if self.counter >= len(self.values) - 1:
            return self.values[-1]
        else:
            return self.values[self.counter]

    def wait(self) -> None:
        """Wait/sleep synchronously."""
        sleep(self.wait_time)

    async def wait_async(self) -> None:
        """Wait/sleep asynchronously - must be awaited."""
        await async_sleep(self.wait_time)


def decode_modbus_value(value: Sequence[int], byteorder: str, type_: Callable | None = None) -> Any:
    r"""Method to decode incoming modbus values. Strings are always decoded as utf-8 values. If you do not
    want this behaviour specify 'bytes' as the data type.

    :param value: Current value to be decoded into float.
    :param byteorder: Byteorder for decoding i.e. 'little' or 'big' endian.
    :param type\_: Type of the output value. See `Python struct format character documentation
                  <https://docs.python.org/3/library/struct.html#format-characters>` for all possible
                  format strings (default: f).
    :return: Decoded value as a python type.
    """
    if byteorder == "little":
        bo = "<"
    elif byteorder == "big":
        bo = ">"
    else:
        raise ValueError(f"Specified an invalid byteorder: '{byteorder}'")

    if type_ is str or type_ is bytes:
        dtype = "s"
        _len = len(value)
    elif type_ is bool:
        dtype = "?"
        _len = 1
        if _len != len(value):
            raise ValueError(f"The length of the received value ({len(value)})does not match the data type {type_}")
    elif type_ is int:
        _int_types = {1: "b", 2: "h", 4: "i", 8: "q"}
        _len = 1
        try:
            dtype = _int_types[len(value)]
        except KeyError:
            raise ValueError(f"The length of the received value ({len(value)})does not match the data type {type_}")
    elif type_ is float or type_ is None:
        _float_types = {2: "e", 4: "f", 8: "d"}
        _len = 1
        try:
            dtype = _float_types[len(value)]
        except KeyError:
            raise ValueError(f"The length of the received value ({len(value)}) does not match the data type: {type_}")
    else:
        raise ValueError(f"The given modbus data type was not recognized: {type_}")

    # Determine the format strings for packing and unpacking the received byte sequences. These format strings
    # depend on the endianness (determined by bo), the length of the value in bytes and the data type.
    pack = f"{len(value):1d}B"
    unpack = f"{bo}{_len}{dtype}"

    # Convert the value into the appropriate format
    val = struct.unpack(unpack, struct.pack(pack, *value))[0]
    if type_ is str:
        val = type_(val, "utf-8")
    elif type_ is not None:
        val = type_(val)
    else:
        val = float(val)

    return val


def encode_modbus_value(
    value: str | int | float | bytes, byteorder: str, byte_length: int, type_: Callable | None = None
) -> list[int]:
    r"""Method to encode python data type to modbus value.

    :param value: Current value to be decoded into float.
    :param byteorder: Byteorder for decoding i.e. 'little' or 'big' endian.
    :param byte_length: Length of the value in bytes.
    :param type\_: Type of the output value. See `Python struct format character documentation
                  <https://docs.python.org/3/library/struct.html#format-characters>` for all possible
                  format strings (default: f).
    :return: Decoded value as a python type.
    """
    # Make sure that value is of the type specified by the node.
    if type_ is not None:
        value = type_(value)

    if isinstance(value, int):
        if value < 0:
            _types = {1: "b", 2: "h", 4: "i", 8: "q"}
        else:
            _types = {1: "B", 2: "H", 4: "I", 8: "Q"}
        try:
            _type = _types[byte_length]
        except KeyError:
            raise ValueError(f"Byte length for integers must be either 1, 2, 4 or 8. Got {byte_length}.")
        _len: str | int = ""

    elif isinstance(value, float):
        _types = {2: "e", 4: "f", 8: "d"}
        try:
            _type = _types[byte_length]
        except KeyError:
            raise ValueError(f"Byte length for floats must be either 4 or 8. Got {byte_length}.")
        _len = ""

    else:
        _type = "s"
        _len = byte_length
        if not isinstance(value, bytes):
            value = bytes(value, "utf-8")

    _order = {"big": ">", "little": "<"}
    try:
        bo = _order[byteorder]
    except KeyError:
        raise ValueError(f"Unknown byte order specified: {byteorder}")

    try:
        byte = struct.pack(f"{bo}{_len}{_type}", value)
    except struct.error:
        raise ValueError(f"Could not convert value {value!r} to bits.")

    bitstrings = [f"{bin(x)[2:]:0>8}" for x in byte]

    return [int(z) for z in "".join(bitstrings)]


def bitarray_to_bytearray(bits: list[int | bool]) -> list[int]:
    """Convert a list of bits into a list of bytes."""
    # Make sure that _bits is a list of integers, not bools.
    _bits = [int(x) for x in bits] if isinstance(bits[0], bool) else bits

    b_size = (len(_bits) + 7) // 8
    bytes_list = [0] * b_size
    for i in range(0, b_size):
        start = i * 8
        bytes_list[i] = int("".join([str(v) for v in _bits[start : start + 8]]), 2)

    return bytes_list
