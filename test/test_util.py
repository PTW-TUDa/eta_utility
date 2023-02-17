import json
import pathlib
from datetime import datetime

import pytest

from eta_utility.util import dict_search, json_import, round_timestamp


@pytest.mark.parametrize(
    ("datetime_str", "interval", "expected"),
    [
        ("2016-01-01T02:02:02", 1, "2016-01-01T02:02:02"),
        ("2016-01-01T02:02:02", 60, "2016-01-01T02:03:00"),
        ("2016-01-01T02:02:00", 60, "2016-01-01T02:02:00"),
        ("2016-01-01T02:02:02", 60 * 60, "2016-01-01T03:00:00"),
        ("2016-01-01T02:00:00", 60 * 60, "2016-01-01T02:00:00"),
    ],
)
def test_round_timestamp(datetime_str, interval, expected):
    dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S")

    result = round_timestamp(dt, interval, False).isoformat(sep="T", timespec="seconds")

    assert result == expected


def test_dict_search():
    assert dict_search({"key": "value"}, "value") == "key"


def test_dict_search_fail():
    with pytest.raises(ValueError, match=r".*not specified in specified dictionary"):
        dict_search({}, "value")


def test_remove_comments_json():
    with open(pathlib.Path(__file__).parent / "resources/remove_comments/removed_comments.json") as f:
        control = json.load(f)

    assert json_import(pathlib.Path(__file__).parent / "resources/remove_comments/with_comments.json") == control
