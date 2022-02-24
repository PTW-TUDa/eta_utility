import numpy as np
from gym import spaces


def continous_action_space_from_state(state_config) -> spaces.Space:
    """Use the state_config to generate the action space according to the format required by the OpenAI
    specification. This will set the action_space attribute and return the corresponding space object.
    The generated action space is continous.

    :return: Action space
    """
    action_low = state_config.loc[state_config.is_agent_action == True].low_value.values  # noqa: E712
    action_high = state_config.loc[state_config.is_agent_action == True].high_value.values  # noqa: E712
    action_space = spaces.Box(action_low, action_high, dtype=np.float)

    return action_space
