import shutil

import pytest

from examples.damped_oscillator.main import experiment as ex_oscillator  # noqa: I900
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
            experiment_path, {"settings": {"n_episodes_learn": 2}, "environment_specific": {"do_render": False}}
        )


class TestOscillatorExample:
    @pytest.fixture(scope="class")
    def expriment_path(self):
        path = get_oscillator_path()
        yield path
        shutil.rmtree(path / "results")

    def test_oscillator(self, expriment_path):
        ex_oscillator(expriment_path)
