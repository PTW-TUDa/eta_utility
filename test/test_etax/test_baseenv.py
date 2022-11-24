import shutil
from datetime import timedelta

import pandas as pd
import pytest

from eta_utility.eta_x.common import episode_results_path
from examples.damped_oscillator.main import (  # noqa: I900
    experiment_conventional as ex_oscillator,
)
from examples.damped_oscillator.main import (  # noqa: I900
    get_path as get_oscillator_path,
)


class TestStateLog:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_oscillator_path()
        yield path
        shutil.rmtree(path / "results")

    @pytest.fixture(scope="class")
    def results_path(self, experiment_path):
        return experiment_path / "results/conventional_series"

    @pytest.fixture(scope="class")
    def damped_oscillator_eta(self, experiment_path):
        return ex_oscillator(experiment_path)

    def test_export_state_log(self, damped_oscillator_eta, results_path):
        assert episode_results_path(results_path, "run1", 1, 1).exists()

    def test_export_with_datetime_index(self, damped_oscillator_eta, results_path):
        config = damped_oscillator_eta.config
        report = pd.read_csv(
            episode_results_path(results_path, "run1", 1, 1),
            sep=";",
            index_col=0,
        )
        report.index = pd.to_datetime(report.index)
        step = config.settings.sampling_time / config.settings.sim_steps_per_sample

        assert (report.index[1] - report.index[0]) == timedelta(seconds=step)
