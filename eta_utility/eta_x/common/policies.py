from __future__ import annotations

from typing import TYPE_CHECKING

import torch as th
from stable_baselines3.common import policies

if TYPE_CHECKING:
    from typing import Any


class NoPolicy(policies.BasePolicy):
    """No Policy allows for the creation of agents which do not use neural networks. It does not implement any of
    the typical policy functions but is a simple interface that can be used and ignored. There is no need
    to worry about the implementation details of policies.
    """

    def forward(self, *args: Any, **kwargs: Any) -> None:
        """No Policy allows for the creation of agents which do not use neural networks. It does not implement any of
        the typical policy functions but is a simple interface that can be used and ignored. There is no need
        to worry about the implementation details of policies.
        """
        raise NotImplementedError("'NoPolicy' should be used only, when predictions are calculated otherwise.")

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Get the action according to the policy for a given observation.

        Not implemented in NoPolicy.

        :param observation: Observations of the agent.
        :param deterministic: Whether to use stochastic or deterministic actions.
        :return: Taken action according to the policy.
        """
        raise NotImplementedError("'NoPolicy' should be used only, when predictions are calculated otherwise.")

    # type ignored because mypy doesn't seem to think the following is equivalent to the super class...
    def state_dict(  # type: ignore
        self, *, destination: dict[str, Any] | None = None, prefix: str = "", keep_vars: bool = False
    ) -> dict[str, Any]:
        """Returns a dictionary containing a whole state of the module. The dictionary is empty in NoPolicy.

        :param destination: If provided, the state will be updated into the dictionary and the same object returned.
        :param prefix: Prefix added to parameter and buffer names when composing keys in state_dict.
        :param keep_vars: Determine, which variables will be detached from torch.autograd.
        :return: Dictionary with the module/policy state.
        """
        return {}
