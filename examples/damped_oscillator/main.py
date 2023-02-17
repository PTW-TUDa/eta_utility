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

    experiment_conventional(root_path)
    experiment_learning(root_path)


def experiment_conventional(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> ETAx:
    """Perform a conventionally controlled experiment with the damped oscillator environment.
    This uses the damped_oscillator_conventional.json config file.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    :return: The ETAx object containing the experiment
    """
    # --main--
    experiment = ETAx(root_path, "damped_oscillator_conventional", overwrite, relpath_config=".")
    experiment.play("conventional_series", "run1")
    # --main--
    return experiment


def experiment_learning(root_path: pathlib.Path, overwrite: dict[str, Any] | None = None) -> None:
    """Perform machine learning experiment with the damped oscillator environment.
    This uses the damped_oscillator_learning.json config file.

    :param root_path: Root path of the experiment.
    :param overwrite: Additional config values to overwrite values from JSON.
    """
    # --main--
    experiment = ETAx(root_path, "damped_oscillator_learning", overwrite, relpath_config=".")
    experiment.learn("learning_series", "run1")
    experiment.play("learning_series", "run1")
    # --main--


def get_path() -> pathlib.Path:
    """Get the path of this file.

    :return: Path to this file.
    """
    return pathlib.Path(__file__).parent


def plot() -> None:
    """Load results from both runs and create a plot to compare them."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import cm

    data = (
        pd.concat(
            (
                pd.read_csv("results/conventional_series/run1_000_01.csv", sep=";").add_prefix("conv_"),
                pd.read_csv("results/learning_series/run1_000_01.csv", sep=";").add_prefix("rl_"),
            ),
            axis=1,
        )
        .rolling(30)
        .mean()
    )

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = "9"
    linestyles = ["--", "-"]

    def greys(x: int) -> tuple[int]:
        return cm.Greys(int(255 - ((255 - 100) / 3) * x))

    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    fig.set_tight_layout(True)

    x = data.index
    columns = {
        "mass deviation conventional": "conv_s",
        "input conventional": "conv_u",
        "mass deviation DRL": "rl_s",
        "input DRL": "rl_u",
    }

    lines: list[mpl.lines.Line2D] = []
    labels: list[str] = []
    for name, col in columns.items():
        hdl = ax.plot(x, data[col], color=greys(len(lines)), linestyle=linestyles[len(lines) % len(linestyles)])[0]
        lines.append(hdl)
        labels.append(name)

    ax.legend(lines, labels, loc="upper right")
    ax.yaxis.grid(color="gray", linestyle="dashed")

    ax.set_xlabel("time")
    ax.set_ylabel("distance")

    plt.savefig("training_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    import sys

    sys.path.append(pathlib.Path(__file__).parent.parent.parent.as_posix())

    main()
