from __future__ import annotations

from typing import TYPE_CHECKING

import torch as th

if TYPE_CHECKING:
    from typing import Iterable, Sequence


class Split1d(th.nn.ModuleList):
    """Split1d defines a pytorch module which splits the 1D input tensor into multiple parts and passes each
    of the parts through a separate network. After the pass through the network, the output from all networks
    is joined together. Thus, Split1d will return a 1d observation vector.

    When configuring the network architecture, it is important to ensure that the output of all networks is 1D.
    Use torch.nn.Flatten to flatten the output of networks where the output is not one dimensional.

    Use the parameters 'sizes' and 'net_arch' to determine how many of the input features should be passed through
    which network. Each value in sizes must have a correstponding value in net_arch. For the following examples, let's
    assume that 'in_features' is 15. If 'sizes' is [3, 10, None], a valid configuration for
    net_arch could be [th.nn.Linear(out_features=10), th.nn.Conv1d(out_channels:2), th.nn.Linear(out_features=2)].
    The last value of 'sizes' will automatically be calculated to be 2 (15 - 3 - 10 = 2). With this, 3 values would
    be passed to the first *Linear* layer, 10 values would be passed to the "Conv1d" layer and the final 2 values would
    be passed to the third layer in net_arch (which is the *Linear* layer with 2 output features).

    If you would like to use dictionaries to configure the net_arch, you can use the function
    :py:func:`eta_utility.eta_x.common.common.deserialize_net_arch` to create the torch network architecture.

    :param in_features: Number of input features for the Module
    :param sizes: List of sizes for splitting the input features. This list can contain the value "None" once. If the
        list contains None, this will be evaluated to contain all remaining input features.
    :param net_arch: List of torch.nn Modules. Each value of this list corresponds to one value of the 'sizes' list.
    """

    def __init__(self, in_features: int, sizes: Sequence[None | int], net_arch: Sequence[th.nn.Module]):
        super().__init__()

        self.sizes = self.get_full_sizes(in_features, sizes)
        self.in_features = in_features

        # Check that the number of extractor architectures is equal to the  number of sizes specified.
        if len(net_arch) != len(self.sizes):
            raise ValueError(
                f"There must be one extractor architecture (there are {len(net_arch)}) "
                f"for each split in the data (there are {len(self.sizes)})."
            )

        for net in net_arch:
            self.append(net)

    def extra_repr(self) -> str:
        """Add info about the module to its torch representation.

        :return: String representation of the object.
        """
        return f"in_features={self.in_features}, sizes={self.sizes}"

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """Perform a forward pass through the layer.

        :param tensor: Input tensor
        :return: Output tensor
        """
        if tensor.shape[1] != self.in_features:
            raise ValueError(
                f"The tensor is shorter ({len(tensor)}) than the number of elements specified "
                f"for the split process ({self.in_features})"
            )

        tensors = th.split(tensor, self.sizes, dim=1)
        outputs = [self[item](tensor) for item, tensor in enumerate(tensors)]

        return th.cat(outputs, dim=1)

    @staticmethod
    def get_full_sizes(in_features: int, sizes: Iterable[None | int]) -> list[int]:
        """Use in_features and the sizes list to determine the missing value in 'sizes' in case 'sizes' contains a
        None value (see class description for more information on how a None value in 'sizes' is interpreted.

        :param in_features: Number of input features for the Module.
        :param sizes: List of sizes for splitting the input features. This list can contain the value "None" once. If
            the list contains None, this will be evaluated to contain all remaining input features.
        :return: List of sizes without the missing value.
        """
        # Check if the sizes list contains None and sum all elements that are not None.
        nones = 0
        none_idx: int = 0
        _sum = 0
        for idx, s in enumerate(sizes):
            if s is None:
                none_idx = idx
                nones += 1
            else:
                _sum += s

        if nones > 1:
            raise ValueError(
                "Please only specify None once in the configuration for the split process. None is where "
                "all remaining elements will be processed."
            )
        elif nones == 1:
            # mypy does not correctly understand how we are removing None values.
            _sizes: list[int] = list(sizes)  # type: ignore
            _sizes[none_idx] = in_features - _sum
        else:
            if _sum != in_features:
                raise ValueError(
                    f"If None is not specified in the split process configuration, the sum of elements "
                    f"specified in 'sizes' ({_sum}) must be equal to in_features ({in_features})."
                )
            # mypy does not correctly understand how we are removing None values.
            _sizes = list(sizes)  # type: ignore

        return _sizes


class Fold1d(th.nn.Module):
    """Fold a 1D tensor to create a multi-dimensional tensor. The parameter 'out_channels' determines, how many
    dimensions the output tensor will have.

    :param out_channels: Number of dimensions of the output tensor.
    """

    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels

    def extra_repr(self) -> str:
        """Add info about the module to its torch representation.

        :return: String representation of the object.
        """
        return f"out_channels: {self.out_channels}"

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """Perform a forward pass through the layer.

        :param tensor: Input tensor
        :return: Output tensor
        """
        return th.reshape(tensor, [-1, self.out_channels, tensor.shape[1] // self.out_channels])
