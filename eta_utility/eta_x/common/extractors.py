from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device

from eta_utility import get_logger

from .common import deserialize_net_arch

if TYPE_CHECKING:
    from typing import Any

    import gym

log = get_logger("eta_x")


class CustomExtractor(BaseFeaturesExtractor):
    """
    Advanced feature extractor which allows the definition of arbitrary network structures. Layers can be any
    of the layers defined in `torch.nn <https://pytorch.org/docs/stable/nn.html>`_. The net_arch parameter will
    be interpreted by the function :py:func:`eta_utility.eta_x.common.common.deserialize_net_arch`.

    :param observation_space: gym space.
    :param net_arch: The architecture of the Advanced Feature Extractor. See
        :py:func:`eta_utility.eta_x.common.deserialize_net_arch` for syntax.
    :param device: Torch device for training.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        net_arch: Sequence[Mapping[str, Any]],
        device: th.device | str = "auto",
    ):
        device = get_device(device)
        network = deserialize_net_arch(net_arch, in_features=observation_space.shape[0], device=device)

        # Check output dimension of the network
        with th.no_grad():
            output = network(th.as_tensor(observation_space.sample()[None]).float())
        super().__init__(observation_space, output.shape[1])

        self.network = network

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Perform a forward pass through the network.

        :param observations: Observations to pass through network.
        :return: Output of network.
        """
        return self.network(observations)
