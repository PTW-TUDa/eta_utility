from copy import deepcopy
from typing import Any, Callable, Sequence, Tuple

import numpy as np
from stable_baselines.common.vec_env import (
    AlreadySteppingError,
    DummyVecEnv,
    NotSteppingError,
)

from .. import ProcessPool


class MultiVecEnv(DummyVecEnv):
    """
    Creates a vectorized wrapper for multiple environments, which allows calling of multiple environments in batches
    using multiple python processes.

    The environment borrows from DummyVecEnv and will behave very similar to a DummyVecEnv if set_proc_pool is not
    called before stepping the environment. This means that it will just execute the evaluation of all environments
    in sequence.

    .. seealso:: stable_baselines.common.vec_env.DummyVecEnv

    :param Sequence[Callable] env_fns: A list of functions that will create the environments
        (each callable returns a `Gym.Env` instance when called).
    """

    def __init__(self, env_fns: Sequence[Callable]):
        super().__init__(env_fns)
        for key, env in enumerate(self.envs):
            env.env_id = key

        self._running = False
        self._processes = None

        self._buf_results = [()] * self.num_envs

    def step_async(self, actions: Sequence[Sequence[Any]]) -> None:
        """Perform an asynchronous step using the ProcessPool concurrency mapping interface provided by agents module

        :param actions: Sequence of actions for environments to perform
        :return:
        """
        # Make sure only one step occurs at a time
        if self._running:
            raise AlreadySteppingError()
        self._running = True

        # Store actions and tell environments which actions to execute if necessary
        self.actions = actions
        if hasattr(self.envs[0], "actions"):
            chunksize, remain = divmod(len(self.actions), len(self.envs))
            chunksize += 1 if remain else 0
            for key, env in enumerate(self.envs):
                env.actions = range(key * chunksize, chunksize * (key + 1))

        if len(self.actions) == 1:
            self._buf_results = [item for sublist in self.envs[0].step(self.actions) for item in sublist]
        elif not isinstance(self._processes, ProcessPool):
            self._buf_results = [
                item for env_idx in range(self.num_envs) for item in self.envs[env_idx].step(self.actions)
            ]
        else:
            self._buf_results = [
                item
                for sublist in self._processes.map("step", self.envs, args=(self.actions,), method="return")
                for item in sublist
            ]

    def step_wait(self) -> Tuple[Sequence, Sequence, Sequence, Sequence]:
        """Store observations and reset environments

        :return: Tuple with stepping result sequences (observations, rewards, dones, infos)
        :rtype: Tuple[Sequence, Sequence, Sequence, Sequece]
        """
        if not self._running:
            raise NotSteppingError()

        # Change size of buffers depending on the number of calculated results
        self.buf_obs = [None] * len(self._buf_results)
        self.buf_dones = np.zeros((len(self._buf_results),), dtype=np.bool)
        self.buf_infos = [{}] * len(self._buf_results)
        if not hasattr(self._buf_results[0][1], "__len__") or len(self._buf_results[0][1]) <= 1:
            self.buf_rews = np.zeros(len(self._buf_results), dtype=np.float32)
        else:
            self.buf_rews = [None] * len(self._buf_results)

        for env_idx, res in enumerate(self._buf_results):
            (
                obs,
                self.buf_rews[env_idx],
                self.buf_dones[env_idx],
                self.buf_infos[env_idx],
            ) = res

            if self.num_envs == len(self._buf_results) and self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self.buf_obs[env_idx] = obs

        self._running = False
        return (
            np.copy(self.buf_obs),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(self):
        """Reset all environments"""
        self.buf_obs = [None] * self.num_envs
        for env_idx in range(self.num_envs):
            self.buf_obs[env_idx] = self.envs[env_idx].reset()
        return self.buf_obs

    def set_proc_pool(self, process_pool: ProcessPool) -> None:
        """Set the process pool to use an existing process pool.

        :param agents.common.concurrent.ProcessPool process_pool: Process pool to use for parallel calculations
        """
        self._processes = process_pool

    def __getstate__(self):
        self._processes.close()
        self._processes.join()
        self._processes = None
        return self
