from stable_baselines3.common import policies


class NoPolicy(policies.BasePolicy):
    """No Policy allows for the creation of agents which do not use neural networks. It does not implement any of
    the typical policy functions but is a simple interface that can be used and ignored. There is no need
    to worry about the implementation details of policies.

    :param args: Any arguments that should be passed to the BasePolicy
    :param kwargs: Any keywords that should be passed to the Base Policy
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        """NoPolicy should be used only, when steps results are calculated otherwise."""
        raise NotImplementedError("NoPolicy should be used only, when steps results are calculated otherwise.")

    def proba_step(self) -> None:
        """NoPolicy should be used only, when probabilities are calculated otherwise."""
        raise NotImplementedError("NoPolicy should be used only, when probabilities are calculated otherwise.")
