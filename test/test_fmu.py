import pathlib

import pytest

from eta_utility.eta_x import ETAx

from .test_utilities.etax.util import read_file, remove_data


@pytest.fixture()
def damped_oscillator_eta():
    root_path = pathlib.Path(__file__).parent
    etax = ETAx(root_path=root_path, config_name="damped_oscillator", relpath_config="test_resources/etax/config/")
    yield etax
    remove_data(root_path)


def test_sim_steps_per_sample(damped_oscillator_eta):
    damped_oscillator_eta.play("test_fmu", "1", "Test damped oscillator model from FMU.")

    config = read_file(damped_oscillator_eta.path_config)["settings"]
    report = read_file(damped_oscillator_eta.path_series_results / "report_fmu.csv")
    expected_env_iteractions = (
        config["n_episodes_play"]
        * int(config["episode_duration"] / config["sampling_time"])
        * config["sim_steps_per_sample"]
        + 1
    )
    actual_env_interactions = int(report[-1][0])

    assert expected_env_iteractions == actual_env_interactions
