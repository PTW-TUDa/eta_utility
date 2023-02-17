import logging
import shutil

import pytest

from examples.cyber_physical_system.main import get_path as get_cps_path  # noqa: I900
from examples.damped_oscillator.main import (  # noqa: I900
    experiment_conventional as ex_oscillator,
)
from examples.damped_oscillator.main import (  # noqa: I900
    get_path as get_oscillator_path,
)
from examples.pendulum.main import (  # noqa: I900
    conventional as ex_pendulum_conventional,
)
from examples.pendulum.main import get_path as get_pendulum_path  # noqa: I900
from examples.pendulum.main import (  # noqa: I900
    machine_learning as ex_pendulum_learning,
)


class TestPendulumExample:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_pendulum_path()
        yield path
        shutil.rmtree(path / "results")

    def test_conventional(self, experiment_path):
        ex_pendulum_conventional(experiment_path, {"environment_specific": {"do_render": False}})

    def test_learning(self, experiment_path):
        ex_pendulum_learning(
            experiment_path,
            {
                "settings": {"n_episodes_learn": 4, "save_model_every_x_episodes": 2, "n_environments": 2},
                "environment_specific": {"do_render": False},
            },
        )


class TestOscillatorExample:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_oscillator_path()
        yield path
        logging.shutdown()
        shutil.rmtree(path / "results")

    def test_oscillator(self, experiment_path):
        ex_oscillator(experiment_path)


class TestCPSExample:
    @pytest.fixture(scope="class")
    def experiment_path(self):
        path = get_cps_path()
        yield path
        try:
            shutil.rmtree(path / "results")
        except FileNotFoundError:
            pass

    def test_cps(self, experiment_path):
        """Text for example to be added. This is currently impossible because simulation model
        cannot be compiled on linux.

            from examples.cyber_physical_system.main import experiment as ex_cps
            ex_cps(expriment_path, {"settings": {"sampling_time": 2, "episode_duration": 20}})
        """
        pass
