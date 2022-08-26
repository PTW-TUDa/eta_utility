import pytest

from eta_utility.connectors.util import encode_modbus_value

modbus_values = (
    (5, "big", 1, bytes([0x05])),
    (1001, "big", 4, bytes([0x00, 0x00, 0x03, 0xE9])),
    (1001, "little", 4, bytes([0xE9, 0x03, 0x00, 0x00])),
    (129387, "big", 4, bytes([0x00, 0x01, 0xF9, 0x6B])),
    (129387, "little", 4, bytes([0x6B, 0xF9, 0x01, 0x00])),
    (2.3782, "big", 4, bytes([0x40, 0x18, 0x34, 0x6E])),
    (2.3782, "little", 4, bytes([0x6E, 0x34, 0x18, 0x40])),
    ("string", "big", 6, b"string"),
    ("string", "little", 6, b"string"),
    (b"string", "little", 8, b"string\x00\x00"),
)


@pytest.mark.parametrize(("value", "byteorder", "bytelength", "expected"), modbus_values)
def test_encode_modbus_value(value, byteorder, bytelength, expected):
    result = encode_modbus_value(value, byteorder, bytelength)

    assert int("".join(str(v) for v in result), 2).to_bytes(bytelength, "big") == expected
