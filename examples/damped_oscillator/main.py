from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from eta_utility import get_logger
from eta_utility.eta_x import ETAx

if TYPE_CHECKING:
    from typing import Any


def main() -> None:
    get_logger()
    root_path = get_path()

    experiment(root_path)


def experiment(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> None:
    """Perform a conventionally controlled experiment with the pendulum environment.
    This uses the pendulum_conventional config file.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    # --main--
    experiment = ETAx(root_path, "damped_oscillator", overwrite, relpath_config=".")
    experiment.play("example_series", "run1")
    # --main--


def get_path() -> pathlib.Path:
    """Get the path of this file.

    :return: Path to this file.
    """
    return pathlib.Path(__file__).parent


if __name__ == "__main__":
    main()
