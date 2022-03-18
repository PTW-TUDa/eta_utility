from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import torch as th
from attrs import define, field  # noqa: I900
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Union

    import gym
    from attrs import Attribute  # noqa: I900

    _MLPCNNNetworkMapping = Sequence[Mapping[str, Sequence[Union[int, Sequence[Union[int, str]]]]]]

log = get_logger("eta_x")


def _cnn_layers_converter(layers: Sequence[int | Sequence[int | str]]) -> Sequence[Sequence[str | int]]:
    _new_layers = []
    for layer in layers:
        if isinstance(layer, int):
            raise ValueError("CNN layers must be defined by lists of values.")

        _layer: list[str | int]
        if layer[0] in {"max_pooling", "average_pooling"}:
            _layer = [str(layer[0]), int(layer[1]), int(layer[2])]
        else:
            _layer = [int(v) for v in layer]
        _new_layers.append(_layer)

    return _new_layers


def _mlp_layers_converter(layers: Sequence[int | Sequence[int | str]]) -> Sequence[int]:
    _new_layers = []
    for layer in layers:
        if isinstance(layer, Sequence):
            raise ValueError("MLP layers must be defined by values.")
        else:
            _new_layers.append(layer)

    return _new_layers


@define(frozen=True, kw_only=True)
class MLPCNNNetArch:
    #: CNN layers, where the inputs are a part of all observations - all observations can be used as well.
    cnn_layers: Sequence[Sequence[str | int]] = field(converter=_cnn_layers_converter)
    #: Number of inputs for the CNN layer.
    cnn_inputs: int = field(converter=int)
    #: Number of CNN input channels.
    cnn_input_channels: int = field(converter=int)

    #: MLP layers, where the inputs are the remaining observations.
    mlp_layers: Sequence[int] = field(converter=_mlp_layers_converter)

    @classmethod
    def from_serialized(cls, sequence: _MLPCNNNetworkMapping) -> MLPCNNNetArch:
        """To construct a network a nested dictionary must be specified, with the required entry 'cnn' and the
        optional entry 'mlp'.

        * Each cnn-layer is specified by: [kernel size, output channels, stride, padding].
        * The cnn dict must start with an entry for overall settings:
          [number of observations to use for the cnn part, number of time-series].
        * Each pooling layer is specified by: [pooling type, pooling size, stride].

        An exemplary network with 288 values for the cnn part, which consists of 6 independent time-series is defined
        as follows: [dict(cnn=[[288, 6],[8, 8, 1, 'valid'],['max_pooling', 4, 2]], mlp=[100])].

        See the documentation for a more detailed example.
        """
        _cnn_layers = []
        _mlp_layers = []
        for dikt in sequence:
            if not isinstance(dikt, Mapping):
                raise TypeError("Network architecture must contain only dicts.")

            if "cnn" in dikt:
                if not isinstance(dikt["cnn"], Sequence):
                    raise ValueError("CNN network architecture must consist of lists describing the layers.")
                _cnn_layers.extend(list(dikt["cnn"]))

            if "mlp" in dikt:
                if not isinstance(dikt["mlp"], Sequence):
                    raise ValueError("MLP network architecture must consist of a list describing the layers.")
                _mlp_layers.extend(list(dikt["mlp"]))

        if len(_cnn_layers) == 0:
            raise ValueError("No CNN layers where specified. Please use the specialized CNN policies for this case.")
        elif len(_cnn_layers) < 2:
            raise ValueError("The CNN layers must be specified by an input configuration and at least one layer.")

        cnn_layers = _cnn_layers_converter(_cnn_layers)
        mlp_layers = _mlp_layers_converter(_mlp_layers)

        cnn_input_config = _cnn_layers.pop(0)
        if not isinstance(cnn_input_config, Sequence):
            raise ValueError(
                "The CNN input configuration (first entry of CNN layer config) " "must be a list of 2 entries."
            )
        elif len(cnn_input_config) != 2:
            raise ValueError("The CNN input configuration (first entry of CNN layer config) must contain 2 entries.")
        cnn_inputs = cnn_input_config[0]
        if isinstance(cnn_inputs, Sequence):
            raise ValueError("Number of CNN inputs must be convertible to integers.")
        cnn_input_channels = cnn_input_config[1]
        if isinstance(cnn_input_channels, Sequence):
            raise ValueError("Number of CNN input channels must be convertible to integers.")

        return cls(
            cnn_layers=cnn_layers, cnn_inputs=cnn_inputs, cnn_input_channels=cnn_input_channels, mlp_layers=mlp_layers
        )

    @cnn_layers.validator
    def _cnn_layers_validator(self, attribute: Attribute, new_value: Sequence[int]) -> None:
        # Check CNN layer specifications
        for layer in new_value:
            if not isinstance(layer, Sequence) or len(layer) < 3:
                raise ValueError("A layer must be defined by a list of integers.")

            if layer[0] == "max_pooling":
                if len(layer) != 3:
                    raise ValueError("A max-pooling layer must be specified by 3 values.")
            elif layer[0] == "average_pooling":
                if len(layer) != 3:
                    raise ValueError("An average-pooling layer must be specified by 3 values.")
            else:
                if len(layer) != 4:
                    raise ValueError("A CNN-layer must be specified by 4 values.")

    @mlp_layers.validator
    def _mlp_layer_validator(self, attribute: Attribute, new_value: Sequence[int]) -> None:
        for layer in new_value:
            if not isinstance(layer, int):
                raise ValueError("An MLP layer must be defined by an integer for the output dimension.")


class MLPCNNExtractor(BaseFeaturesExtractor):
    """
    Advanced Feature Extractor that combines an MLP with an CNN Network. Pass this class to the agent as a
    ``features_extractor_class`` in ``policy_kwargs``.

    :param observation_space: gym space.
    :param extractor_arch: The architecture of the Advanced Feature Extractor. See explanation above for syntax.
    :param activation_fn_mlp: Activation function for the mlp part of the network.
    :param activation_fn_cnn: Activation function for the cnn part of the network.
    :param device: Device for training.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        extractor_arch: _MLPCNNNetworkMapping | MLPCNNNetArch,
        activation_fn_mlp: type[th.nn.Module] = th.nn.Tanh,
        activation_fn_cnn: type[th.nn.Module] = th.nn.Tanh,
        device: th.device | str = "auto",
    ):
        device = get_device(device)

        #: Definition of the combined MLP and CNN network.
        self.net_def: MLPCNNNetArch
        if not isinstance(extractor_arch, MLPCNNNetArch):
            self.net_def = MLPCNNNetArch.from_serialized(extractor_arch)
        else:
            self.net_def = extractor_arch

        last_mlp_layer_dim = self._init_cnn_config(observation_space)

        #: Torch sequential CNN network.
        self.cnn_net: th.nn.Sequential
        self._init_cnn(activation_fn_cnn)
        self.cnn_net.to(device)

        #: Torch sequential MLP network.
        self.mlp_net: th.nn.Sequential | None
        self._init_mlp(activation_fn_mlp, last_mlp_layer_dim)
        if self.mlp_net is not None:
            self.mlp_net.to(device)

        # Check dimension of extractor output
        #: Output dimension of the CNN network.
        self.cnn_output_dim: int
        #: Output dimension of the MLP network.
        self.mlp_output_dim: int
        #: Combined output dimension.
        self.combined_output: int

        with th.no_grad():
            my_tensor = th.as_tensor(observation_space.sample()[None]).float()
            my_mlp_tensor, my_cnn_tensor = th.split(
                my_tensor, [int(my_tensor.shape[1]) - self.net_def.cnn_inputs, self.net_def.cnn_inputs], dim=1
            )
            my_cnn_tensor = th.reshape(
                my_cnn_tensor,
                [-1, self.net_def.cnn_input_channels, int(my_cnn_tensor.shape[1] // self.net_def.cnn_input_channels)],
            )

            self.cnn_output_dim = self.cnn_net(my_cnn_tensor).shape[1]
            if self.mlp_net is not None:
                self.mlp_output_dim = self.mlp_net(my_mlp_tensor).shape[1]
            else:
                self.mlp_output_dim = 0
            self.combined_output = self.cnn_output_dim + self.mlp_output_dim

            log.info(f"MLP/CNN network, CNN output dimension: {self.cnn_output_dim}.")
            log.info(f"MLP/CNN network, MLP output dimension: {self.mlp_output_dim}.")

        super().__init__(observation_space, self.combined_output)

    def _init_cnn_config(self, observation_space: gym.Space) -> int:
        """Initialize the configuration for the CNN network.

        :param observation_space: Observation space of the agent.
        :return: Number of nodes in the first CNN layer.
        """
        # Check whether the input configuration for the CNN works in combination with the observation space.
        if self.net_def.cnn_inputs > int(observation_space.shape[0]):
            raise ValueError(
                f"The number of observations used for the CNN part ({self.net_def.cnn_inputs})"
                f"must be less or equal than the number of all observations "
                f"({int(observation_space.shape[0])})."
            )

        if self.net_def.cnn_inputs % self.net_def.cnn_input_channels != 0:
            raise ValueError(
                f"The number of observations used for the CNN part ({self.net_def.cnn_inputs})"
                f"must be evenly divisible by the input channels used for the first layer "
                f"({self.net_def.cnn_input_channels})."
            )

        # Split flat observations for the cnn-only and mlp-only part
        if self.net_def.cnn_inputs == int(observation_space.shape[0]) or self.net_def.cnn_inputs == -1:
            if len(self.net_def.mlp_layers) != 0:
                raise ValueError("All observations are used for the CNN part but MLP only layers were also specified.")

        # Define values for following layers
        self.in_channels = self.net_def.cnn_input_channels
        last_mlp_layer_dim = int(observation_space.shape[0]) - self.net_def.cnn_inputs
        return last_mlp_layer_dim

    def _init_cnn(self, activation_fn_cnn: type[th.nn.Module]) -> None:
        """Initialize the CNN part of the network.

        :param activation_fn_cnn: Activation function for the CNN layers.
        """
        cnn_net: list[th.nn.Module] = []
        for layer in self.net_def.cnn_layers:
            if layer[0] == "max_pooling":
                max_pooling_layer = th.nn.MaxPool1d(
                    kernel_size=int(layer[1]),
                    stride=int(layer[2]),
                    padding=0,
                    dilation=1,
                    return_indices=False,
                    ceil_mode=False,
                )
                cnn_net.append(max_pooling_layer)

            elif layer[0] == "average_pooling":
                avg_pooling_layer = th.nn.AvgPool1d(
                    kernel_size=int(layer[1]), stride=int(layer[2]), padding=0, ceil_mode=False, count_include_pad=True
                )
                cnn_net.append(avg_pooling_layer)

            else:
                conv_layer = th.nn.Conv1d(
                    self.in_channels,
                    kernel_size=int(layer[0]),  # width of filter
                    out_channels=int(layer[1]),  # number of filter/ output channels
                    stride=int(layer[2]),  # stride of filter
                    padding=int(layer[3]),  # padding: 'same' or 'valid'
                    dilation=(1,),
                    groups=1,
                    bias=True,
                    padding_mode="zeros",
                    device=None,
                    dtype=None,
                )

                cnn_net.append(conv_layer)
                cnn_net.append(activation_fn_cnn())
        # flatten the last cnn_layer
        cnn_net.append(th.nn.Flatten())

        self.cnn_net = th.nn.Sequential(*cnn_net)

    def _init_mlp(self, activation_fn_mlp: type[th.nn.Module], last_mlp_layer_dim: int) -> None:
        """Initialize the MLP part of the network.

        :param activation_fn_mlp: Activation function for the MLP layers.
        :param last_mlp_layer_dim: Dimension of the last MLP layer.
        """
        # iterate through mlp_layers - mlp_layer only specified if mlp_only_layers != []
        mlp_net: list[th.nn.Module] = []
        for layer in self.net_def.mlp_layers:
            mlp_net.append(th.nn.Linear(last_mlp_layer_dim, layer))
            mlp_net.append(activation_fn_mlp())
            last_mlp_layer_dim = layer

        if len(mlp_net):
            self.mlp_net = th.nn.Sequential(*mlp_net)
        else:
            self.mlp_net = None

    def forward(self, observations: th.Tensor) -> th.Tensor:
        """Perform a forward pass through the network.

        :param observations: Observations to pass through network.
        :return: Output of network.
        """
        if self.mlp_net is not None:
            obs_mlp, obs_cnn = th.split(
                observations, [int(observations.shape[1]) - self.net_def.cnn_inputs, self.net_def.cnn_inputs], dim=1
            )

            reshaped_obs_cnn = th.reshape(
                obs_cnn, [-1, self.net_def.cnn_input_channels, int(obs_cnn.shape[1] // self.net_def.cnn_input_channels)]
            )

            mlp_out = self.mlp_net(obs_mlp)
            cnn_out = self.cnn_net(reshaped_obs_cnn)
            output = th.cat((mlp_out, cnn_out), dim=1)

        else:
            obs_cnn = observations
            reshaped_obs_cnn = th.reshape(
                obs_cnn, [-1, self.net_def.cnn_input_channels, int(obs_cnn.shape[1] // self.net_def.cnn_input_channels)]
            )
            output = self.cnn_net(reshaped_obs_cnn)

        return output
