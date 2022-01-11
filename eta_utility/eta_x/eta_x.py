import copy
import importlib
import inspect
import json
import os
import pathlib
import re
from datetime import datetime
from functools import partial
from typing import Any, Mapping, MutableMapping, Optional, Type, Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eta_utility import get_logger
from eta_utility.type_hints import BaseEnv, DefSettings, Path, ReqSettings

log = get_logger("eta_x", 2)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#                     CALLBACKS                         #
# - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def callback_environment(environment: BaseEnv) -> None:
    """
    This callback will be called at the end of each episode.
    When multiprocessing is used, no global variables are available (as an own python instance is created).

    :param environment: the instance of the environment where the callback was triggered
    :type environment: BaseEnv
    """
    log.info(
        "Environment callback triggered (env_id = {}, n_episodes = {}, run_name = {}.".format(
            environment.env_id, environment.n_episodes, environment.run_name
        )
    )

    # render first episode
    if environment.n_episodes == 1:
        environment.render()

    plot_interval = int(environment.env_settings["plot_interval"])

    # render progress over episodes (for each environment individually)
    if environment.n_episodes % plot_interval == 0:
        environment.render()
        if hasattr(environment, "render_episodes"):
            environment.render_episodes()


# - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                       CLASS                          #
# - - - - - - - - - - - - - - - - - - - - - - - - - - -#


class ETAx:
    """Initialize an optimization model and provide interfaces for optimization, learning and execution (play)

    :param root_path: Root path of the eta_x application (the configuration will be interpreted relative to this)
    :param config_name: Name of configuration .ini file in configuration directory (should be json format)
    :param config_overwrite: Dictionary to overwrite selected configurations
    :param relpath_config: Relative path to configuration file, starting from root path (default: config/)
    """

    _req_settings: ReqSettings = {
        "setup": {"agent_package", "agent_class", "environment_package", "environment_class"},
        "settings": {},
        "paths": {"relpath_results"},
        "environment_specific": {},
        "agent_specific": {},
    }

    _default_settings: DefSettings = {
        "setup": {
            "tensorboard_log": False,
            "monitor_wrapper": False,
            "norm_wrapper_obs": False,
            "norm_wrapper_reward": False,
            "policy_package": "eta_utility.eta_x.agents.common",
            "policy_class": "NoPolicy",
            "vectorizer_package": "stable_baselines3.common.vec_env.dummy_vec_env",
            "vectorizer_class": "DummyVecEnv",
        },
        "settings": {"verbose": 2, "seed": None},
        "environment_specific": {"plot_interval": 1},
        "agent_specific": {},
    }

    def __init__(
        self,
        root_path: Union[str, os.PathLike],
        config_name: str,
        config_overwrite: Optional[Mapping[str, Any]] = None,
        relpath_config: Union[str, os.PathLike] = "config/",
    ) -> None:
        # initial states
        self._modules_imported: bool = False
        self._environment_vectorized: bool = False
        self._model_initialized: bool = False
        self._prepared: bool = False

        self.info: MutableMapping[str, str] = {}  #: Information about the run

        # default series/run
        self.series_name: Optional[str] = None  #: Name of the series of runs
        self.run_name: Optional[str] = None  #: Name of the run
        self.run_description: Optional[str] = None  #: Description of the run

        # default paths
        #: Root path of the application
        self.path_root: pathlib.Path = root_path if isinstance(root_path, pathlib.Path) else pathlib.Path(root_path)
        self.path_config: pathlib.Path = (
            self.path_root / relpath_config if not isinstance(relpath_config, pathlib.Path) else relpath_config
        ) / (
            config_name + ".json"
        )  #: Config file path

        self.path_results: Optional[pathlib.Path] = None  #: General results folder
        self.path_series_results: Optional[pathlib.Path] = None  #: Results folder for the series
        self.path_run_model: Optional[pathlib.Path] = None  #: Model file path
        self.path_run_info: Optional[pathlib.Path] = None  #: Info file path
        self.path_run_monitor: Optional[pathlib.Path] = None  #: Monitor file path

        self.config: Optional[Mapping[str, Any]] = None  #: Configuration dictionary

        # Classes and instances
        self.agent: Optional[Type[BaseAlgorithm]] = None  #: Agent class
        self.policy: Optional[Type[BasePolicy]] = None  #: Policy class
        self.vectorizer: Optional[Type[DummyVecEnv]] = None  #: Environment vectorizer class
        self.environment: Optional[Type[BaseEnv]] = None  #: Environment class
        self.environments: Optional[DummyVecEnv] = None  #: Vectorized environment object
        self.model: Optional[BaseAlgorithm] = None  #: Instantiated model
        self.interaction_env_class: Optional[Type[BaseEnv]] = None  #: Environment class for interactions
        self.interaction_env: Optional[BaseEnv] = None  #: Environment object for interactions

        # load and overwrite config
        self.load_config(config_name, config_overwrite)

        self.path_results = self.path_root / self.config["paths"]["relpath_results"]
        log.setLevel(int(self.config["settings"]["verbose"] * 10))
        self.config["paths"].update(
            {
                "path_root": str(self.path_root),
                "path_root_results": str(self.path_results),
            }
        )
        self.config["environment_specific"].setdefault("seed", self.config["settings"]["seed"])
        self.config["environment_specific"].setdefault("verbose", self.config["settings"]["verbose"])
        self.config["agent_specific"].setdefault("seed", self.config["settings"]["seed"])
        self.config["agent_specific"].setdefault("verbose", self.config["settings"]["verbose"])

        self.import_modules()

    def load_config(self, config_name: str, config_overwrite: Optional[Mapping[str, Any]] = None) -> None:
        """Load configuration  from file

        :param config_name: name of the configuration file in data/configurations/
        :param config_overwrite: Config parameters to overwrite
        """

        def deep_update(source: DefSettings, overrides: Mapping[str, Any]) -> DefSettings:
            """Update a nested dictionary or similar mapping."""
            output = copy.deepcopy(source)
            for key, value in overrides.items():
                if isinstance(value, Mapping):
                    output[key] = deep_update(source.get(key, {}), value)
                else:
                    output[key] = value
            return output

        config_overwrite = {} if config_overwrite is None else config_overwrite
        try:
            # Remove comments from the json file (using regular expression), then parse it into a dictionary
            cleanup = re.compile(r"^\s*(.*?)(?=/{2}|$)", re.MULTILINE)
            with self.path_config.open("r") as f:
                file = "".join(cleanup.findall(f.read()))
            config = json.loads(file)
            del file
            log.info(f"Configuration {config_name} loaded successfully.")
            config = deep_update(self._default_settings, config)
        except OSError as e:
            log.error(
                "Configuration {} couldn't be loaded: {}. \n"
                "\t Filename: {}".format(config_name, e.strerror, e.filename)
            )
            raise

        self.config = deep_update(config, config_overwrite)

        errors = False
        for sect in self._req_settings:
            if sect not in self.config:
                log.error(f"Required section '{sect}' not found in configuration.")
                errors = True
            else:
                for name in self._req_settings[sect]:
                    if name not in self.config[sect]:
                        log.error(f"Required parameter '{name}' not found in config section '{sect}'")
                        errors = True

        if errors:
            log.error("Not all required config parameters were found. Exiting.")
            exit(1)

    def save_config(self, config_name: str) -> None:
        """Save configuration for to file

        :param config_name: name of the configuration file in data/configurations/
        """

        try:
            with self.path_config.open("w") as f:
                json.dump(self.config, f)

            log.info(f"Configuration {config_name} saved successfully.")
        except OSError as e:
            log.error(
                "Configuration {} couldn't be saved: {}. \n"
                "\t Filename: {}".format(config_name, e.strerror, e.filename)
            )

    def import_modules(self) -> None:
        """Import all modules of agent, policy, vectorizer, environment"""
        log.debug("Trying to import modules for agent, policy, vectorizer and environment.")

        try:
            self.agent = getattr(
                importlib.import_module(self.config["setup"]["agent_package"]),
                self.config["setup"]["agent_class"],
            )
            self.policy = getattr(
                importlib.import_module(self.config["setup"]["policy_package"]),
                self.config["setup"]["policy_class"],
            )
            self.vectorizer = getattr(
                importlib.import_module(self.config["setup"]["vectorizer_package"]),
                self.config["setup"]["vectorizer_class"],
            )
            self.environment = getattr(
                importlib.import_module(self.config["setup"]["environment_package"]),
                self.config["setup"]["environment_class"],
            )

            if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
                if not {"interaction_env_package", "interaction_env_class"} < self.config["setup"].keys():
                    raise ValueError(
                        "If 'interact_with_env' is specified, 'interaction_env_package and "
                        "'interaction_env_class' must also be specified in config section 'setup'."
                    )
                self.interaction_env_class = getattr(
                    importlib.import_module(self.config["setup"]["interaction_env_package"]),
                    self.config["setup"]["interaction_env_class"],
                )

            self._modules_imported = True
            log.info("Modules imported successfully.")
        except OSError:
            raise

        self._environment_vectorized = False

    def prepare_run(
        self, series_name: str, run_name: str, run_description: str = "", reset: bool = False, training: bool = True
    ) -> bool:
        """Prepare the learn and play methods

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        :param reset: Should an exisiting model be overwritten if it exists?
        :param training: Should preparation be done for training or playing?
        :return: Boolean value indicating successful preparation
        """

        self.series_name = series_name if series_name is not None else self.series_name
        self.run_name = run_name if run_name is not None else self.run_name
        self.run_description = run_description if run_description != "" else self.run_description

        if (series_name is None and self.series_name is None) or (run_name is None and self.run_name is None):
            raise ValueError("series_name and run_name must be specified and not None!")

        # set paths for model, config, logs, trajectories and plot storage
        self.path_series_results = self.path_results / self.series_name
        self.config["paths"]["path_results"] = str(self.path_series_results)
        self.path_run_model = self.path_series_results / (self.run_name + "_model.zip")
        self.path_run_info = self.path_series_results / (self.run_name + "_info.json")
        self.path_run_monitor = self.path_series_results / (self.run_name + "_monitor.csv")

        if not self.path_results.is_dir():
            for p in reversed(self.path_results.parents):
                if not p.is_dir():
                    p.mkdir()
            self.path_results.mkdir()

        if not self.path_series_results.is_dir():
            log.debug("Path for result series doesn't exist on your OS. Trying to create directories.")
            try:
                self.path_series_results.mkdir()
                log.info("Directory created successfully: \n" "\t {}".format(self.path_series_results))
            except OSError:
                raise

        # load modules
        if not self._modules_imported:
            self.import_modules()

        # vectorize environment
        if not self._environment_vectorized:
            self.vectorize_environment(training=training)

        if not self._model_initialized:
            self.initialize_model(reset=reset)

        # set run info
        self.info["series_name"] = self.series_name
        self.info["run_name"] = self.run_name
        self.info["run_description"] = self.run_description
        (
            self.info["env_version"],
            self.info["env_description"],
        ) = self.environment.get_info(self.environment)

        # set prepared variable when all conditions are met
        if all(
            {
                self._modules_imported,
                self._environment_vectorized,
                self._model_initialized,
                self.path_run_model.is_file(),
            }
        ):
            self._prepared = True
            log.info("Run prepared successfully.")

        return self._prepared

    def vectorize_environment(self, training: bool = True) -> bool:
        """Instantiate and vectorize the environment

        :param training: Should preparation be done for training or playing?
        :return: Boolean value indicating successful completion of vectorization
        """

        if not self._modules_imported:
            raise RuntimeError("Cannot vectorize the environment without importing corresponding modules first.")

        log.debug("Trying to vectorize the environment.")
        if self.vectorizer.__class__.__name__ == "DummyVecEnv":
            n_environments = int(1)
        else:
            n_environments = int(self.config["settings"]["n_environments"])

        # Create the vectorized environment
        self.environments = self.vectorizer(
            [
                lambda env_id=i + 1: self.environment(
                    env_id=env_id,
                    run_name=self.run_name,
                    general_settings=self.config["settings"],
                    path_settings=self.config["paths"],
                    env_settings=self.config["environment_specific"],
                    verbose=self.config["environment_specific"]["verbose"],
                    callback=callback_environment,
                )
                for i in range(n_environments)
            ]
        )

        if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
            self.interaction_env = self.vectorizer(
                [
                    partial(
                        self.interaction_env_class,
                        env_id=0,
                        run_name=self.run_name,
                        general_settings=self.config["settings"],
                        path_settings=self.config["paths"],
                        env_settings=self.config["interaction_env_specific"]
                        if "interaction_env_specific" in self.config
                        else self.config["environment_specific"],
                        verbose=(
                            self.config["interaction_env_specific"]
                            if "interaction_env_specific" in self.config
                            else self.config["environment_specific"]
                        )["verbose"],
                        callback=callback_environment,
                    )
                ]
            )

        if self.config["setup"]["monitor_wrapper"]:
            log.error("Monitoring is not supported for vectorized environments!")
            log.warning("The monitor_wrapper parameter will be ignored.")

        # Automatically normalize the input features
        if self.config["setup"]["norm_wrapper_obs"] or self.config["setup"]["norm_wrapper_reward"]:
            # check if normalization data are available; then load
            file_vec_normalize = os.path.join(self.path_series_results, "vec_normalize.pkl")
            if os.path.exists(file_vec_normalize):
                log.info(
                    "Normalization data detected. Loading running averages into "
                    "normalization wrapper: \n"
                    "\t {}".format(file_vec_normalize)
                )
                self.environments = VecNormalize.load(file_vec_normalize, self.environments)
                self.environments.training = (training,)
                self.environments.norm_obs = (self.config["setup"]["norm_wrapper_obs"],)
                self.environments.norm_reward = (self.config["setup"]["norm_wrapper_reward"],)
                self.environments.clip_obs = self.config["setup"]["norm_wrapper_clip_obs"]
            else:
                log.info("No Normalization data detected.")
                self.environments = VecNormalize(
                    self.environments,
                    training=training,
                    norm_obs=self.config["setup"]["norm_wrapper_obs"],
                    norm_reward=self.config["setup"]["norm_wrapper_reward"],
                    clip_obs=self.config["setup"]["norm_wrapper_clip_obs"],
                )

        self._environment_vectorized = True
        log.info("Environment vectorized successfully.")
        return self._environment_vectorized

    def initialize_model(self, reset: bool = False) -> bool:
        """Load or initialize a model

        :param reset: Should an exisiting model be overwritten if it exists?
        :return: Boolean value indicating successful completion of initialization
        """
        if not self._modules_imported:
            raise RuntimeError("Cannot initialize the model without importing corresponding modules first.")
        self._model_initialized = False

        # check for existing model
        if self.path_run_model.is_file():
            log.info(f"Existing model detected: \n \t {self.path_run_model}")
            if reset:
                new_name = str(
                    self.path_run_model
                    / (
                        "_"
                        + (datetime.fromtimestamp(self.path_run_model.stat().st_mtime).strftime("%Y%m%d_%H%M"))
                        + ".bak"
                    )
                )
                self.path_run_model.rename(new_name)
                log.info(
                    "Reset is active. Existing model will be backed up and ignored. \n"
                    "\t Backup file name: {}".format(new_name)
                )
            else:
                self.load_model()

        if not self._model_initialized:
            agent_kwargs = {}
            if self.config["setup"]["tensorboard_log"]:
                tensorboard_log = self.path_series_results
                log.info("Tensorboard logging is enabled. \n" "\t Log file: {}".format(tensorboard_log))
                log.info(
                    "Please run the following command in the console to start tensorboard: \n"
                    '\t tensorboard --logdir "{}" --port 6006'.format(tensorboard_log)
                )
                agent_kwargs = {"tensorboard_log": self.path_series_results}

            # check if the agent takes all of the default parameters.
            agent_params = inspect.signature(self.agent).parameters
            if "seed" not in agent_params and inspect.Parameter.VAR_KEYWORD not in {
                p.kind for p in agent_params.values()
            }:
                del self.config["agent_specific"]["seed"]
                log.debug(
                    "'seed' is not a valid parameter for agent {}. This default parameter will be ignored.".format(
                        self.agent.__name__
                    )
                )

            # create model instance
            self.model = self.agent(
                self.policy,
                self.environments,
                **self.config["agent_specific"],
                **agent_kwargs,
            )

            self._model_initialized = True
            log.info("Model initialized successfully.")
            return self._model_initialized

    def load_model(self, path_run_model: Path = None) -> bool:
        """Load an existing model

        :param path_run_model: Load model from specified path instead of the path defined in configuration
        :return: Boolean value indicating successful completion of initialization
        """
        if path_run_model is not None:
            self.path_run_model = (
                pathlib.Path(path_run_model) if not isinstance(path_run_model, pathlib.Path) else path_run_model
            )
        elif self.path_run_model is None:
            log.error("Model path for loading is not specified. Loading failed.")
            return False

        log.debug(f"Trying to load existing model: {self.path_run_model}")
        self._model_initialized = False

        if not self.path_run_model.exists():
            log.error("Model couldn't be loaded. Path not found: \n" "\t {}".format(self.path_run_model))
            return

        # tensorboard logging
        agent_kwargs = {}
        if self.config["setup"]["tensorboard_log"]:
            tensorboard_log = self.path_series_results
            log.info("Tensorboard logging is enabled. Log file: \n" "\t {}".format(tensorboard_log))
            log.info(
                "Please run the following command in the console to start tensorboard: \n"
                '\t tensorboard --logdir "{}" --port 6006'.format(tensorboard_log)
            )
            agent_kwargs = {"tensorboard_log": self.path_series_results}

        try:
            self.model = self.agent.load(
                self.path_run_model, self.environments, **self.config["agent_specific"], **agent_kwargs
            )
            self._model_initialized = True
            log.info("Model loaded successfully.")
        except OSError as e:
            raise OSError(f"Model couldn't be loaded: {e.strerror}. Filename: {e.filename}") from e

        return self._model_initialized

    def save_model(self, path_run_model: Path = None) -> None:
        """Save model to file

        :param path_run_model: Save model to specified path instead of the path defined in configuration
        """
        if path_run_model:
            self.path_run_model = (
                pathlib.Path(path_run_model) if not isinstance(path_run_model, pathlib.Path) else path_run_model
            )

        self.model.save(self.path_run_model)

    def log_run_info(self, path_run_info: Path = None) -> None:
        """Save run config to result series directory

        :param path_run_info: Save run information to specified path instead of the path defined in configuration
        """
        if path_run_info:
            self.path_run_info = (
                pathlib.Path(path_run_info) if not isinstance(path_run_info, pathlib.Path) else path_run_info
            )

        with self.path_run_info.open("w") as f:
            try:
                json.dump({**self.info, **self.config}, f)
                log.info("Log file successfully created.")
            except TypeError:
                log.warning("Log file could not be created because of non serializable input in config_overwrite.")

    def pretrain(
        self,
        series_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_description: str = "",
        path_expert_dataset: Optional[Path] = None,
        reset: bool = False,
    ) -> bool:
        """Pretrain the agent with an expert defined trajectory data set or from data generated by
        an environment method.

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        :param path_expert_dataset: Path to the expert dataset. This overwrites the value in 'agent_specifc' config.
        :type path_expert_dataset: str or pathlib.Path
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        :return: Boolean value indicating successful pretraining
        """
        if not self._prepared:
            self.prepare_run(series_name, run_name, run_description, reset=reset, training=True)
        pretrained = False

        if "pretrain_method" not in self.config["agent_specific"] or (
            "pretrain_method" in self.config["agent_specific"]
            and self.config["agent_specific"]["pretrain_method"] == "expert_dataset"
        ):

            # Evaluate path to the expert data set.
            if path_expert_dataset is not None:
                path_expert_dataset = (
                    pathlib.Path(path_expert_dataset)
                    if not isinstance(path_expert_dataset, pathlib.Path)
                    else path_expert_dataset
                )
            elif "path_expert_dataset" in self.config["agent_specific"]:
                path_expert_dataset = pathlib.Path(self.config["agent_specific"]["path_expert_dataset"])
            else:
                path_expert_dataset = self.path_series_results / (self.run_name + "_expert-dataset.npz")
        elif "pretrain_env_method" in self.config["agent_specific"]:
            # Pretrain the agent with data generated by an environment method.
            if "pretrain_kwargs" not in self.config["agent_specific"]:
                log.error(
                    "Missing configuration value for pretraining: 'pretrain_kwargs' in section 'agent_specific'."
                    "Pretraining will be skipped"
                )
            else:
                self.model.pretrain(
                    self.config["agent_specific"]["pretrain_env_method"],
                    self.config["agent_specific"]["pretrain_kwargs"],
                )
                pretrained = True

        return pretrained

    def learn(
        self,
        series_name: Optional[str] = None,
        run_name: Optional[str] = None,
        run_description: str = "",
        pretrain: bool = False,
        reset: bool = False,
    ) -> None:
        """Start the learning job for an agent with the specified environment.

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        :param pretrain: Indication whether pretraining should be performed
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        """
        if not self._prepared:
            self.prepare_run(series_name, run_name, run_description, reset=reset, training=True)

        if pretrain:
            self.pretrain()

        self.log_run_info()

        if "save_model_every_x_episodes" not in self.config["settings"]:
            raise ValueError(
                "Missing configuration value for learning: 'save_model_every_x_episodes' " "in section 'settings'"
            )

        # Genetic algorithm has a slightly different concept for saving since it does not stop between time steps
        if "n_generations" in self.config["settings"]:
            save_freq = int(self.config["settings"]["save_model_every_x_episodes"])

            total_timesteps = self.config["settings"]["n_generations"]
        else:
            # Check if all required config values are present
            errors = False
            req_settings = {"episode_duration", "sampling_time", "n_episodes_learn"}
            for name in req_settings:
                if name not in self.config["settings"]:
                    log.error(f"Missing configuration value for learning: {name} in section 'settings'")
                    errors = True
            if errors:
                raise ValueError("Missing configuration values for learning.")

            # define callback for periodically saving models
            save_freq = int(
                self.config["settings"]["episode_duration"]
                / self.config["settings"]["sampling_time"]
                * self.config["settings"]["save_model_every_x_episodes"]
            )

            total_timesteps = int(
                self.config["settings"]["n_episodes_learn"]
                * self.config["settings"]["episode_duration"]
                / self.config["settings"]["sampling_time"]
            )

        # start learning
        log.info("Start learning process of agent in environment.")

        callback_learn = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.path_series_results / "models"),
            name_prefix=self.run_name,
        )

        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback_learn,
                tb_log_name=self.run_name,
            )
        except OSError:
            self.save_model(self.path_series_results / (self.run_name + "_model_before-error.pkl"))
            raise

        # reset environment one more time to call environment callback one last time
        self.environments.reset()

        # save model
        self.save_model()
        if isinstance(self.environments, VecNormalize):
            stats_path = os.path.join(self.path_series_results, "vec_normalize.pkl")
            self.environments.save(stats_path)

        # close all environments when done (kill processes)
        self.environments.close()

        log.info(f"Learning finished: {series_name} / {run_name}")

    def play(
        self, series_name: Optional[str] = None, run_name: Optional[str] = None, run_description: str = ""
    ) -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        """
        log.info("Start playing in environment with given agent.")

        if not self._prepared:
            self.prepare_run(series_name, run_name, run_description, reset=False, training=False)

        if "n_episodes_play" not in self.config["settings"]:
            raise ValueError("Missing configuration value for playing: 'n_episodes_play' in section 'settings'")

        self.log_run_info()

        n_episodes_stop = self.config["settings"]["n_episodes_play"]

        try:
            if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
                observations = self.interaction_env.reset()
                observations = np.array(self.environments.env_method("reset", observations, indices=0))
            else:
                observations = self.environments.reset()
        except ValueError as e:
            raise ValueError(
                "It is likely that returned observations do not conform to the specified state config."
            ) from e
        n_episodes = 0

        log.info("Start playing process of agent in environment.")
        if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
            log.info("Starting Agent with environment/optimization interaction.")
        else:
            log.info("Starting without an additional interaction environment.")

        while n_episodes < n_episodes_stop:

            # Some agents (i.e. MPC) can interact with an additional environment if required.
            if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
                action, _states = self.model.predict(observation=observations, deterministic=False)
                if "scale_interaction_actions" in self.config["settings"]:
                    action = (
                        np.round(
                            action * self.config["settings"]["scale_interaction_actions"],
                            self.config["settings"]["round_actions"],
                        )
                        if "round_actions" in self.config["settings"]
                        else (action * self.config["settings"]["scale_interaction_actions"])
                    )
                else:
                    action = (
                        np.round(action, self.config["settings"]["round_actions"])
                        if "round_actions" in self.config["settings"]
                        else action
                    )
                observations, rewards, dones, info = self.interaction_env.step(action)  # environment gets called here
                observations = np.array(self.environments.env_method("update", observations, indices=0))

            else:
                action, _states = self.model.predict(observation=observations, deterministic=False)
                observations, rewards, dones, info = self.environments.step(action)

            n_episodes += sum(dones)
