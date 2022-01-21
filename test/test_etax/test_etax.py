import csv
import json
import pathlib
import shutil
from typing import Any, Dict

import pytest

from eta_utility.eta_x import ETAx


@pytest.fixture()
def pendulum_conventional_eta():
    root_path = pathlib.Path(__file__).parent
    etax = ETAx(root_path=root_path, config_name="pendulum_conventional", relpath_config="../test_resources/")
    yield etax
    remove_data(root_path)


def test_results_generated(pendulum_conventional_eta):
    series_name = "simple_controller"
    pendulum_conventional_eta.play(series_name, "1", "Test of gym pendulum.")

    assert (pendulum_conventional_eta.path_results / series_name).is_dir()


def test_execution(pendulum_conventional_eta):
    pendulum_conventional_eta.play("simple_controller", "1", "Test of gym pendulum.")

    report = read_file(pendulum_conventional_eta.path_series_results / "report.csv")
    config = read_file(pendulum_conventional_eta.path_config)

    n_steps = int(config["settings"]["episode_duration"] // config["settings"]["sampling_time"])
    assert int(report[-1][1]) == config["settings"]["n_episodes_play"]
    assert int(report[-1][0]) == n_steps


# helper function
def read_file(path: pathlib.Path) -> Dict[str, Any]:
    file_type = path.suffix
    if file_type == ".csv":
        with open(path) as f:
            reader = csv.reader(f)
            file = []
            for line in reader:
                file.append(line)
    elif file_type == ".json":
        with open(path) as f:
            file = json.load(f)
    else:
        raise Exception("File path not available")
    return file


def remove_data(path: pathlib.Path) -> None:
    shutil.rmtree(path / "data/")
