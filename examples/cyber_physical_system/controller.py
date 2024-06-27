from __future__ import annotations

import numpy as np

from eta_utility.eta_x.agents import RuleBased


class DirectControl(RuleBased):
    """Simple controller for input signal of the cleaning machine experiment.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the cleaning machine. For this case, function uses tank temperature, status of the
        tank heater and the market price.

        :param observation: observation of the environment given one action
        :returns: On or Off as action (u) for the tank heating of the cleaning machine
        """

        temp_tank_sim = observation[0]
        tankheater_status = observation[1]
        market_price = observation[2]

        # Set target temperature depending on the market price
        temp_set = 273.15 + (65 if market_price <= 100.08 else 62)

        # Three-point control for setting controlled variable ON/OFF of the tank heater
        actions: np.ndarray = np.zeros((1), dtype=float)
        if temp_tank_sim > temp_set:
            actions[0] = 0
        elif temp_tank_sim > (temp_set - 3) and temp_tank_sim < temp_set:
            actions[0] = 1 if tankheater_status == 1 else 0
        elif temp_tank_sim <= (temp_set - 3):
            actions[0] = 1

        return actions
