import pathlib

import pytest

from eta_utility import get_logger
from eta_utility.eta_x import ETAx

from .test_utilities.etax import read_file, remove_data


@pytest.fixture()
def pendulum_conventional_eta():
    get_logger()
    root_path = pathlib.Path(__file__).parent
    etax = ETAx(root_path=root_path, config_name="pendulum_conventional", relpath_config="test_resources/etax/config/")
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
