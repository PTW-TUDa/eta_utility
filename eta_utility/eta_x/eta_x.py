from __future__ import annotations

import os
import pathlib
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from eta_utility import get_logger
from eta_utility.eta_x import ConfigOpt, ConfigOptRun
from eta_utility.eta_x.common import (
    CallbackEnvironment,
    initialize_model,
    is_env_closed,
    load_model,
    log_net_arch,
    log_run_info,
    log_to_file,
    merge_callbacks,
    vectorize_environment,
)

if TYPE_CHECKING:
    from typing import Any, Generator, Mapping

    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.type_aliases import MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

log = get_logger("eta_x")


class ETAx:
    """Initialize an optimization model and provide interfaces for optimization, learning and execution (play).

    :param root_path: Root path of the eta_x application (the configuration will be interpreted relative to this).
    :param config_name: Name of configuration .ini file in configuration directory (should be JSON format).
    :param config_overwrite: Dictionary to overwrite selected configurations.
    :param relpath_config: Relative path to configuration file, starting from root path.
    """

    def __init__(
        self,
        root_path: str | os.PathLike,
        config_name: str,
        config_overwrite: Mapping[str, Any] | None = None,
        relpath_config: str | os.PathLike = "config/",
    ) -> None:
        # Load configuration for the optimization
        _root_path = root_path if isinstance(root_path, pathlib.Path) else pathlib.Path(root_path)
        _relpath_config = relpath_config if isinstance(relpath_config, pathlib.Path) else pathlib.Path(relpath_config)
        #: Path to the configuration file.
        self.path_config = _root_path / _relpath_config / f"{config_name}.json"
        #: ConfigOpt object for the optimization run.
        self.config: ConfigOpt = ConfigOpt.from_json(self.path_config, root_path, config_overwrite)
        log.setLevel(int(self.config.settings.verbose * 10))

        #: Configuration for an optimization run.
        self.config_run: ConfigOptRun | None = None

        #: The vectorized environments.
        self.environments: VecEnv | VecNormalize | None = None
        #: Vectorized interaction environments.
        self.interaction_env: VecEnv | None = None
        #: The model or algorithm.
        self.model: BaseAlgorithm | None = None

    def prepare_run(self, series_name: str, run_name: str, run_description: str = "") -> None:
        """Prepare the learn and play methods by reading configuration, creating results folders and the model.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :return: Boolean value indicating successful preparation.
        """
        self.config_run = ConfigOptRun(
            series=series_name,
            name=run_name,
            description=run_description,
            path_root=self.config.path_root,
            path_results=self.config.path_results,
            path_scenarios=self.config.path_scenarios,
        )
        self.config_run.create_results_folders()

        # Add file handler to parent logger to log the terminal output
        log_to_file(config=self.config, config_run=self.config_run)

        log.info("Run prepared successfully.")

    def prepare_model(self, reset: bool = False) -> None:
        """Check for existing model and load it or back it up and create a new model.

        :param reset: Flag to determine whether an existing model should be reset.
        """
        self._prepare_model(reset)

    def _prepare_model(self, reset: bool = False) -> None:
        """Check for existing model and load it or back it up and create a new model.

        :param reset: Flag to determine whether an existing model should be reset.
        """
        assert self.config_run is not None, (
            "Set the config_run attribute before trying to initialize the model "
            "(for example by calling prepare_run)."
        )
        assert self.environments is not None, (
            "Initialize the environments before trying to initialize the model" "(for example by calling prepare_run)."
        )

        path_model = self.config_run.path_run_model
        if path_model.is_file() and reset:
            log.info(f"Existing model detected: {path_model}")

            bak_name = path_model / f"_{datetime.fromtimestamp(path_model.stat().st_mtime).strftime('%Y%m%d_%H%M')}.bak"
            path_model.rename(bak_name)
            log.info(f"Reset is active. Existing model will be backed up. Backup file name: {bak_name}")
        elif path_model.is_file():
            log.info(f"Existing model detected: {path_model}. Loading model.")

            self.model = load_model(
                self.config.setup.agent_class,
                self.environments,
                self.config.settings.agent,
                self.config_run.path_run_model,
                tensorboard_log=self.config.setup.tensorboard_log,
                log_path=self.config_run.path_series_results,
            )
            return

        # Initialize the model if it wasn't loaded from a file
        self.model = initialize_model(
            self.config.setup.agent_class,
            self.config.setup.policy_class,
            self.environments,
            self.config.settings.agent,
            self.config.settings.seed,
            tensorboard_log=self.config.setup.tensorboard_log,
            log_path=self.config_run.path_series_results,
        )

    @contextmanager
    def prepare_environments(self, training: bool = True) -> Generator:
        """Context manager which prepares the environments and closes them after it exits.

        :param training: Should preparation be done for training (alternative: playing)?
        """
        try:
            self._prepare_environments(training)
            yield

        finally:
            # close all environments when done (kill processes)
            log.debug("Closing environments.")
            assert self.environments is not None, "Initialized environments could not be found."
            self.environments.close()
            if self.config.settings.interact_with_env:
                assert self.interaction_env is not None, "Initialized interaction environments could not be found."
                self.interaction_env.close()

    def _prepare_environments(self, training: bool = True) -> None:
        """Vectorize and prepare the environments and potentially the interaction environments.

        :param training: Should preparation be done for training (alternative: playing)?
        """
        assert self.config_run is not None, (
            "Set the config_run attribute before trying to initialize the environments "
            "(for example by calling prepare_run)."
        )

        env_class = self.config.setup.environment_class
        self.config_run.set_env_info(env_class)

        callback = CallbackEnvironment(self.config.settings.plot_interval)
        # Vectorize the environments
        self.environments = vectorize_environment(
            env_class,
            self.config_run,
            self.config.settings.environment,
            callback,
            self.config.settings.seed,
            self.config.settings.verbose,
            self.config.setup.vectorizer_class,
            self.config.settings.n_environments,
            training=training,
            monitor_wrapper=self.config.setup.monitor_wrapper,
            norm_wrapper_obs=self.config.setup.norm_wrapper_obs,
            norm_wrapper_reward=self.config.setup.norm_wrapper_reward,
        )

        if self.config.settings.interact_with_env:
            # Perform some checks to ensure the interaction environment is configured correctly.
            if self.config.setup.interaction_env_class is None:
                raise ValueError(
                    "If 'interact_with_env' is specified, an interaction env class must be specified as well."
                )
            elif self.config.settings.interaction_env is None:
                raise ValueError(
                    "If 'interact_with_env' is specified, the interaction_env settings must be specified as well."
                )
            interaction_env_class = self.config.setup.interaction_env_class
            self.config_run.set_interaction_env_info(interaction_env_class)

            # Vectorize the environment
            self.interaction_env = vectorize_environment(
                interaction_env_class,
                self.config_run,
                self.config.settings.interaction_env,
                callback,
                self.config.settings.seed,
                self.config.settings.verbose,
                training=training,
            )

    def learn(
        self,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        reset: bool = False,
        callbacks: MaybeCallback = None,
    ) -> None:
        """Start the learning job for an agent with the specified environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        :param callbacks: Provide additional callbacks to send to the model.learn() call.
        """
        if is_env_closed(self.environments) or self.model is None:
            _series_name = series_name if series_name is not None else ""
            _run_name = run_name if run_name is not None else ""
            self.prepare_run(_series_name, _run_name, run_description)

        assert self.config_run is not None, "Run configuration could not be found. Call prepare_run first."

        with self.prepare_environments(training=True):
            assert (
                self.environments is not None
            ), "Initialized environments could not be found. Call prepare_environments first."

            self.prepare_model(reset)
            assert self.model is not None, "Initialized model could not be found. Call prepare_model first."

            # Log some information about the model and configuration
            log_net_arch(self.model, self.config_run)
            log_run_info(self.config, self.config_run)

            # Genetic algorithm has a slightly different concept for saving since it does not stop between time steps
            if "n_generations" in self.config.settings.agent:
                save_freq = self.config.settings.save_model_every_x_episodes
                total_timesteps = self.config.settings.agent["n_generations"]
            else:
                # Check if all required config values are present
                if self.config.settings.episode_duration is None:
                    raise ValueError("Missing configuration values for learning: 'episode_duration'.")
                elif self.config.settings.sampling_time is None:
                    raise ValueError("Missing configuration values for learning: 'sampling_time'.")
                elif self.config.settings.n_episodes_learn is None:
                    raise ValueError("Missing configuration values for learning: 'n_episodes_learn'.")

                # define callback for periodically saving models
                save_freq = int(
                    self.config.settings.episode_duration
                    / self.config.settings.sampling_time
                    * self.config.settings.save_model_every_x_episodes
                )
                total_timesteps = int(
                    self.config.settings.n_episodes_learn
                    * self.config.settings.episode_duration
                    / self.config.settings.sampling_time
                )

            callback_learn = merge_callbacks(
                CheckpointCallback(
                    save_freq=save_freq,
                    save_path=str(self.config_run.path_series_results / "models"),
                    name_prefix=self.config_run.name,
                ),
                callbacks,
            )

            # Start learning
            log.info("Start learning process of agent in environment.")
            try:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback_learn,
                    tb_log_name=self.config_run.name,
                )
            except OSError:
                filename = str(self.config_run.path_series_results / f"{self.config_run.name}_model_before_error.pkl")
                log.info(f"Saving model to file: {filename}.")
                self.model.save(filename)
                raise

            # reset environment one more time to call environment callback one last time
            self.environments.reset()

            # save model
            log.debug(f"Saving model to file: {self.config_run.path_run_model}.")
            self.model.save(self.config_run.path_run_model)
            if isinstance(self.environments, VecNormalize):
                log.debug(f"Saving environment normalization data to file: {self.config_run.path_vec_normalize}.")
                self.environments.save(str(self.config_run.path_vec_normalize))

        log.info(f"Learning finished: {series_name} / {run_name}")

    def play(self, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        """
        if is_env_closed(self.environments) or self.model is None:
            _series_name = series_name if series_name is not None else ""
            _run_name = run_name if run_name is not None else ""
            self.prepare_run(_series_name, _run_name, run_description)

        assert self.config_run is not None, "Run configuration could not be found. Call prepare_run first."

        with self.prepare_environments(training=True):
            assert (
                self.environments is not None
            ), "Initialized environments could not be found. Call prepare_environments first."

            self.prepare_model(reset=False)
            assert self.model is not None, "Initialized model could not be found. Call prepare_model first."

            if self.config.settings.n_episodes_play is None:
                raise ValueError("Missing configuration value for playing: 'n_episodes_play' in section 'settings'")

            # Log some information about the model and configuration
            log_net_arch(self.model, self.config_run)
            log_run_info(self.config, self.config_run)

            n_episodes_stop = self.config.settings.n_episodes_play

            # Reset the environments before starting to play
            try:
                log.debug("Resetting environments before starting to play.")
                observations = self._reset_envs()
            except ValueError as e:
                raise ValueError(
                    "It is likely that returned observations do not conform to the specified state config."
                ) from e
            n_episodes = 0

            log.debug("Start playing process of agent in environment.")
            if self.config.settings.interact_with_env:
                log.info("Starting agent with environment/optimization interaction.")
            else:
                log.info("Starting without an additional interaction environment.")

            _round_actions = self.config.settings.round_actions
            _scale_actions = self.config.settings.scale_actions if self.config.settings.scale_actions is not None else 1

            while n_episodes < n_episodes_stop:
                try:
                    observations, dones = self._play_step(_round_actions, _scale_actions, observations)
                except BaseException as e:
                    log.error(
                        "Exception occurred during an environment step. Aborting and trying to reset environments."
                    )
                    observations = self._reset_envs()
                    log.debug("Environment reset successful - re-raising exception")
                    raise e

                n_episodes += sum(dones)

    def _play_step(
        self, _round_actions: int | None, _scale_actions: float, observations: VecEnvObs
    ) -> tuple[VecEnvObs, np.ndarray]:
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."

        action, _states = self.model.predict(observation=observations, deterministic=False)  # type: ignore
        # Type ignored because typing in stable_baselines appears to be incorrect
        # Round and scale actions if required
        if _round_actions is not None:
            action = np.round(action * _scale_actions, _round_actions)
        else:
            action *= _scale_actions
        # Some agents (i.e. MPC) can interact with an additional environment
        if self.config.settings.interact_with_env:
            assert (
                self.interaction_env is not None
            ), "Initialized interaction environments could not be found. Call prepare_run first."

            # Perform a step  with the interaction environment and update the normal environment with
            # its observations
            observations, rewards, dones, info = self.interaction_env.step(action)
            observations = np.array(self.environments.env_method("update", observations, indices=0))
            # Make sure to also reset the environment, if the interaction_env says it's done. For the interaction
            # env this is done inside the vectorizer.
            for idx in range(self.environments.num_envs):
                if dones[idx]:
                    info[idx]["terminal_observation"] = observations
                    observations[idx] = self._reset_env_interaction(observations)
        else:
            observations, rewards, dones, info = self.environments.step(action)
        return observations, dones

    def _reset_envs(self) -> VecEnvObs:
        """Reset the environments when interaction with another environment is taking place.

        :param observations: Observations from the interaction env.
        :return: observations after reset.
        """
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."
        log.debug("Resetting environments.")

        if self.config.settings.interact_with_env:
            assert (
                self.interaction_env is not None
            ), "Initialized interaction environments could not be found. Call prepare_run first."
            observations = self.interaction_env.reset()
            return self._reset_env_interaction(observations)
        else:
            return self.environments.reset()

    def _reset_env_interaction(self, observations: VecEnvObs) -> VecEnvObs:
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."
        log.debug("Resetting main environment during environment interaction.")

        try:
            observations = np.array(self.environments.env_method("first_update", observations, indices=0))
        except AttributeError as e:
            if "first_update" in str(e):
                observations = self.environments.reset()
            else:
                raise e

        return observations
