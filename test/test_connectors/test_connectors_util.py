import pytest

from eta_utility.connectors.util import encode_bits

modbus_values = (
    (5, "big", 8, bytes([0x05])),
    (1001, "big", 32, bytes([0x00, 0x00, 0x03, 0xE9])),
    (1001, "little", 32, bytes([0xE9, 0x03, 0x00, 0x00])),
    (129387, "big", 32, bytes([0x00, 0x01, 0xF9, 0x6B])),
    (129387, "little", 32, bytes([0x6B, 0xF9, 0x01, 0x00])),
    (2.3782, "big", 32, bytes([0x40, 0x18, 0x34, 0x6E])),
    (2.3782, "little", 32, bytes([0x6E, 0x34, 0x18, 0x40])),
    ("string", "big", 48, b"string"),
    ("string", "little", 48, b"string"),
    (b"string", "little", 64, b"string\x00\x00"),
)


@pytest.mark.parametrize(("value", "byteorder", "bitlength", "expected"), modbus_values)
def test_encode_modbus_value(value, byteorder, bitlength, expected):
    result = encode_bits(value, byteorder, bitlength)

    assert int("".join(str(v) for v in result), 2).to_bytes(bitlength // 8, "big") == expected
