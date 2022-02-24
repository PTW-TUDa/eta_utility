import numpy as np
from gym import spaces


def continous_obs_space_from_state(state_config) -> spaces.Box:
    """Use the state_config to generate the observation space according to the format required by the OpenAI
    specification. This will set the observation_space attribute and return the corresponding space object.
    The generated observation space is continous.

    :return: Observation Space
    """
    state_low = state_config.loc[state_config.is_agent_observation == True].low_value.values  # noqa: E712
    state_high = state_config.loc[
        state_config.is_agent_observation == True  # noqa: E712
    ].high_value.values  # noqa: E712
    observation_space = spaces.Box(state_low, state_high, dtype=np.float)

    return observation_space
