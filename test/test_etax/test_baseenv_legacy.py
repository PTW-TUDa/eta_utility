import datetime
import pathlib
import shutil

import pandas as pd
import pytest

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import episode_results_path


@pytest.fixture()
def pendulum_conventional_eta():
    get_logger()
    root_path = pathlib.Path(__file__).parent
    etax = ETAx(root_path=root_path, config_name="pendulum", relpath_config="../resources/pendulum/")
    yield etax
    shutil.rmtree(root_path / "data/")


def test_results_generated(pendulum_conventional_eta):
    series_name = "simple_controller"
    pendulum_conventional_eta.play(series_name, "run1", "Test of gym pendulum.")

    assert (pendulum_conventional_eta.config.path_results / series_name).is_dir()


def test_execution(pendulum_conventional_eta):
    pendulum_conventional_eta.play("simple_controller", "run2", "Test of gym pendulum.")

    report = pd.read_csv(
        episode_results_path(pendulum_conventional_eta.config_run.path_series_results, "run2", 1, 1), sep=";"
    )
    config = pendulum_conventional_eta.config

    n_steps = int(config.settings.episode_duration // config.settings.sampling_time)
    assert report.iloc[-1]["n_episode"] == config.settings.n_episodes_play
    assert report.iloc[-1]["n_iter"] == n_steps


@pytest.fixture()
def damped_oscillator_eta():
    root_path = pathlib.Path(__file__).parent
    etax = ETAx(root_path=root_path, config_name="damped_oscillator", relpath_config="../resources/damped_oscillator/")
    yield etax
    shutil.rmtree(root_path / "data/")


def test_sim_steps_per_sample(damped_oscillator_eta):
    damped_oscillator_eta.play("test_fmu", "run1", "Test damped oscillator model from FMU.")

    config = damped_oscillator_eta.config
    report = pd.read_csv(episode_results_path(damped_oscillator_eta.config_run.path_series_results, "run1", 1, 1))
    expected_env_iteractions = (
        config.settings.n_episodes_play
        * int(config.settings.episode_duration / config.settings.sampling_time)
        * config.settings.sim_steps_per_sample
        + 1
    )

    assert expected_env_iteractions == len(report)


def test_export_state_log_with_time_index(damped_oscillator_eta):
    damped_oscillator_eta.play("test_fmu", "run1", "Test damped oscillator model from FMU.")

    config = damped_oscillator_eta.config
    report = pd.read_csv(
        episode_results_path(damped_oscillator_eta.config_run.path_series_results, "run1", 1, 1),
        sep=";",
        index_col=0,
    )
    report.index = pd.to_datetime(report.index)
    step = config.settings.sampling_time / config.settings.sim_steps_per_sample

    assert (report.index[1] - report.index[0]) == datetime.timedelta(seconds=step)
