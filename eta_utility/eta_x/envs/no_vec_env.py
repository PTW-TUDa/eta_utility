from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.util import obs_space_info

if TYPE_CHECKING:
    from typing import Callable

    import gym
    from stable_baselines3.common.vec_env.base_vec_env import (
        VecEnvIndices,
        VecEnvObs,
        VecEnvStepReturn,
    )


class NoVecEnv(DummyVecEnv):
    """
    NoVecEnv is an environment vectorizer which hands the implementation of multithreading off to the environment.
    The environment must specify the attribute "is_multithreaded" and set it to True. NoVecEnv will hand all actions
    it receives over to the environment, even if they were meant for multiple environments. Therefore, the environment
    has to specifically support parallel evaluation of multiple action sets.

    This vectorizer is useful for environments implemented for example in julia, where we do not want to create multiple
    environments and could potentially implement multithreading inside the environment.

    .. note::
        The reset function of the environment should only return a single set of observations. The order in which the
        resetfunction is called will always be the same. For example, if subenvironments 5, 10 and 15 return done, the
        reset function will be called three times, first for environment 5, then for 10 and finally for 15.

        If the environment returns done (even if it is just for a single subenvironment / action set) the reset function
        of the environment will be called to retrieve initial observations for each one of the done returns separately.

    All other functionality is directly derived from DummyVecEnv. Since there is only a single

    See also: :ref:`stable_baselines.common.vec_env.DummyVecEnv`

    :param env_fns: A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns: list[Callable]) -> None:
        super().__init__([env_fns[0]])

        # Correct the number of environments and re-initialize all values which depend on the number of environments.
        self.num_envs = len(env_fns)

        self.keys, shapes, dtypes = obs_space_info(self.envs[0].observation_space)

        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys]
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        # Fix typing error in base class (should be "ndarray" and None, not just None)
        self.actions: np.ndarray | None  # type: ignore

        # Check if the environment itself is multithreaded and raise an error if it is noz
        if not (hasattr(self.envs[0], "is_multithreaded") and self.envs[0].is_multithreaded is True):
            raise ValueError(
                "The given environment cannot be used with NoVecEnv because it does not specify the attribute "
                "is_multithreaded (or the attribute is not set to True)."
            )

    def step_wait(self) -> VecEnvStepReturn:
        """Store observations and reset environments

        :return: Tuple with stepping result sequences (observations, rewards, dones, infos)
        """
        assert self.actions is not None, "Stepping the environment is only possible when actions are set."

        # Re-initialize the observation buffers (necessary because the number of action sets is not known beforehand).
        obs, self.buf_rews, self.buf_dones, self.buf_infos = self.envs[0].step(self.actions)
        for idx in range(self.num_envs):
            if self.buf_dones[idx]:
                self.buf_infos[idx]["terminal_observation"] = obs[idx]
                obs[idx] = self.envs[0].reset()
            self._save_obs(idx, obs[idx])

        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self) -> VecEnvObs:
        """Reset all sub environments and return their observations.

        :return: Observations from all sub environments.
        """
        for env_idx in range(self.num_envs):
            obs = self.envs[0].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def _get_target_envs(self, indices: VecEnvIndices) -> list[gym.Env]:
        """Return the 0 target environment (because there can only ever be one...)

        :param indices: Indices of the environments. Values don't not really matter here, only length is important.
        :return: List of environments to target.
        """
        indices = self._get_indices(indices)
        return [self.envs[0] for i in indices]
