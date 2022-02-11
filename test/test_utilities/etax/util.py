import csv
import json
import pathlib
import shutil
from typing import Any, Dict


def read_file(path: pathlib.Path) -> Dict[str, Any]:
    file_type = path.suffix
    if file_type == ".csv":
        with open(path) as f:
            reader = csv.reader(f)
            file = []
            for line in reader:
                if line:
                    file.append(line)
    elif file_type == ".json":
        with open(path) as f:
            file = json.load(f)
    else:
        raise Exception("File path not available")
    return file


def remove_data(path: pathlib.Path) -> None:
    shutil.rmtree(path / "data/")
