from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import pathlib
import re
from datetime import datetime
from functools import partial
from typing import TYPE_CHECKING, Mapping

import numpy as np
from attrs import define, field, validators, converters, Attribute
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from eta_utility import get_logger

if TYPE_CHECKING:
    from typing import Any, MutableMapping

    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

    from eta_utility.eta_x.envs import BaseEnv
    from eta_utility.type_hints import Path, EnvSettings

log = get_logger("eta_x", 2)


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


def _path_converter(path: Path) -> pathlib.Path:
    return pathlib.Path(path) if not isinstance(path, pathlib.Path) else path


def _get_class(instance: ConfigOptSetup, attrib: Attribute, new_value: str):
    # Find module and class name
    module, cls_name = new_value.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)

    cls_attr_name = f"{attrib.name.rsplit('_', 1)[0]}_class"
    setattr(instance, cls_attr_name, cls)


@define(frozen=False)
class ConfigOpt:
    #: Name of the configuration used for the series of run
    config_name: str = field(kw_only = True, validator = validators.instance_of(str))

    #: Root path for the optimization run (scenarios and results are relative to this)
    path_root: pathlib.Path = field(kw_only = True, converter = _path_converter)
    #: Relative path to the results folder
    relpath_results: str = field(kw_only = True, validator = validators.instance_of(str))

    #: Optimization run setup
    setup: ConfigOptSetup = field(kw_only = True)
    #: Optimization run settings
    settings: ConfigOptSettings = field(kw_only = True)

    @classmethod
    def from_json(cls, file: Path, overwrite: Mapping[str, Any] | None = None) -> ConfigOpt:
        """Load configuration  from file

        :param file: name of the configuration file in data/configurations/
        :param overwrite: Config parameters to overwrite
        """
        _file = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        _overwrite = {} if overwrite is None else overwrite

        try:
            # Remove comments from the json file (using regular expression), then parse it into a dictionary
            cleanup = re.compile(r"^\s*(.*?)(?=/{2}|$)", re.MULTILINE)
            with _file.open("r") as f:
                _data = "".join(cleanup.findall(f.read()))
            config = json.loads(_data)
            del _data

            log.info(f"Configuration file {_file} loaded successfully.")
        except OSError as e:
            log.error(f"Configuration file {_file} couldn't be loaded: {e.strerror}. \n")
            raise

        # Ensure all required sections are present in configuration
        if {"setup"} > config.keys():
            raise ValueError(
                    f"Not all required sections (setup, settings, agent_specific, environment_specific) "
                    f"are present in configuration file {_file}."
            )

        for section, conf in config.items():
            if section == "setup":
                setup = ConfigOptSetup.from_dict(conf)
            elif section == "settings":
                settings = ConfigOptSettings.from_dict(conf)

        self.config = deep_update(config, config_overwrite)

        return cls()

    @staticmethod
    def deep_update(source: DefaultSettings, overrides: Mapping[str, Any]) -> DefaultSettings:
        """Update a nested dictionary or similar mapping."""
        output = copy.deepcopy(source)
        for key, value in overrides.items():
            if isinstance(value, Mapping):
                output[key] = deep_update(source.get(key, {}), value)
            else:
                output[key] = value
        return output

    def check_complete(self, req_settings: RequiredSettings) -> bool:
        """Check whether all required settings are set in the class"""
        errors = False
        for sect in req_settings:
            if sect not in self.config:
                log.error(f"Required section '{sect}' not found in configuration.")
                errors = True
            else:
                for name in req_settings[sect]:
                    if name not in self.config[sect]:
                        log.error(f"Required parameter '{name}' not found in config section '{sect}'")
                        errors = True
        if errors:
            log.error("Not all required config parameters were found.")

        return errors

    def save_file(self, path: Path) -> None:
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


@define(frozen = False)
class ConfigOptSetup:
    """Configuration options as specified in the "setup" section of the configuration file.
    """

    #: Import description string for the agent class
    agent_import: str = field(on_setattr = _get_class)
    #: Agent class
    agent_class: type[BaseAlgorithm] = field(init = False)
    #: Import description string for the environment class
    environment_import: str = field(on_setattr = _get_class)
    #: Environment class
    environment_class: type[BaseEnv] = field(init = False)
    #: Import description string for the interaction environment (default: None)
    interaction_env_import: str | None = field(default = None, on_setattr = _get_class)
    #: Interaction environment class (default. None)
    interaction_env_class: type[BaseEnv] | None = field(init = False, default = None)

    #: Import description string for the environment vectorizer
    #: (default: stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv)
    vectorizer_import: str = field(
            default = "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv", on_setattr = _get_class
    )
    #: Environment vectorizer class
    vectorizer_class: type[VecEnv] = field(init = False)
    #: Import description string for the policy class (default: eta_utility.eta_x.agents.common.NoPolicy)
    policy_import: str = field(default = "eta_utility.eta_x.agents.common.NoPolicy", on_setattr = _get_class)
    #: Policy class
    policy_class: type[BasePolicy] = field(init = False)


    #: Flag which is true if the environment should be wrapped for monitoring (default: False)
    monitor_wrapper: bool = field(default = False)
    #: Flag which is true if the observations should be normalized (default: False)
    norm_wrapper_obs: bool = field(default = False)
    #: Flag which is true if the observations should be normalized and clipped (default: False)
    norm_wrapper_clip_obs: bool = field(default = False)
    #: Flag which is true if the rewards should be normalized (default: False)
    norm_wrapper_reward: bool = field(default = False)
    #: Flag to enable tensorboard logging (default: False)
    tensorboard_log: bool = field(default = False)

    @classmethod
    def from_dict(cls, dikt) -> ConfigOptSetup:
        if "agent_package" not in dikt or "agent_class" not in dikt:
            raise ValueError("'agent_package' and 'agent_class' parameters must always be specified.")

        if "environment_package" not in dikt or "environment_class" not in dikt:
            raise ValueError("'environment_package' and 'environment_class' parameters must always be specified.")

        if "interaction_env_package" in dikt:
            if "interaction_env_class" not in dikt:
                raise ValueError("If 'interaction_env_package is specified, "
                                 "interaction_env_class must also be specified.")
            interaction_env_import = f"{dikt['interaction_env_package']}.{dikt['interaction_env_class']}"
        else:
            interaction_env_import = None

        if "vectorizer_package" in dikt:
            vectorizer_import = f"{dikt['vectorizer_package']}.{dikt['vectorizer_class']}"
        else:
            vectorizer_import = None

        if "policy_package" in dikt:
            policy_import = f"{dikt['policy_package']}.{dikt['policy_class']}"
        else:
            policy_import = None

        monitor_wrapper = dikt["monitor_wrapper"] if "monitor_wrapper" in dikt else None
        norm_wrapper_obs = dikt["norm_wrapper_obs"] if "norm_wrapper_obs" in dikt else None
        norm_wrapper_reward = dikt["norm_wrapper_reward"] if "norm_wrapper_reward" in dikt else None
        tensorboard_log = dikt["tensorboard_log"] if "tensorboard_log" in dikt else None

        return ConfigOptSetup(
                agent_import = f"{dikt['agent_package']}.{dikt['agent_class']}",
                environment_import = f"{dikt['environment_package']}.{dikt['environment_class']}",
                interaction_env_import = interaction_env_import,
                vectorizer_import = vectorizer_import,
                policy_import = policy_import,
                monitor_wrapper = monitor_wrapper,
                norm_wrapper_obs = norm_wrapper_obs,
                norm_wrapper_reward = norm_wrapper_reward,
                tensorboard_log = tensorboard_log,
        )


@define(frozen = False)
class ConfigOptSettings:
    #: Seed for random sampling
    seed: int | None = field(kw_only = True, default = None)
    #: Logging verbosity of the framework
    verbose: int = field(kw_only = True, default = 2)
    #: Number of vectorized environments to instantiate (if not using DummyVecEnv
    n_environments: int = field(kw_only = True, default = 1)
    #: Flag to determine whether the interaction env is used or not
    interact_with_env: bool = field(kw_only = True, default = False)

    environment: dict[str, Any]

    "environment_specific": {"plot_interval": 1},
    self.config["environment_specific"].setdefault("seed", self.config["settings"]["seed"])
    self.config["environment_specific"].setdefault("verbose", self.config["settings"]["verbose"])
    self.config["agent_specific"].setdefault("seed", self.config["settings"]["seed"])
    self.config["agent_specific"].setdefault("verbose", self.config["settings"]["verbose"])

    def dikt(self) -> dict[str, Any]:


@define(frozen = False)
class ConfigOptRun:
    #: Name of the series of optimization runs
    series: str = field(kw_only = True, validator = validators.instance_of(str))
    #: Name of an optimization run
    name: str = field(kw_only = True, validator = validators.instance_of(str))
    #: Description of an optimization run
    description: str = field(
            kw_only = True,
            converter = lambda s: "" if s is None else s,
            validator = validators.optional(validators.instance_of(str))
    )
    #: Root path for the optimization run (scenarios and results are relative to this)
    path_root: pathlib.Path = field(kw_only = True, converter = _path_converter)
    #: Relative path to the results folder
    relpath_results: str = field(kw_only = True, validator = validators.instance_of(str))
    #: Path to results of the optimization run
    path_results: pathlib.Path = field(init = False, converter = _path_converter)
    #: Path for the results of the series of optimization runs
    path_series_results: pathlib.Path = field(init = False, converter = _path_converter)
    #: Path to the model of the optimization run
    path_run_model: pathlib.Path = field(init = False, converter = _path_converter)
    #: Path to information about the optimization run
    path_run_info: pathlib.Path = field(init = False, converter = _path_converter)
    #: Path to the monitoring information about the optimization run
    path_run_monitor: pathlib.Path = field(init = False, converter = _path_converter)
    #: Path to the normalization wrapper information
    path_vec_normalize: pathlib.Path = field(init = False, converter = _path_converter)

    # Information about the environments
    #: Version of the main environment
    env_version: str | None = field(init = False, default = None, validator = validators.instance_of(str))
    #: Description of the main environment
    env_description: str | None  = field(init = False, default = None, validator = validators.instance_of(str))

    #: Version of the secondary environment (interaction_env)
    interaction_env_version: str | None = field(
            init = False, default = None, validator = validators.optional(validators.instance_of(str))
    )
    #: Description of the secondary environment (interaction_env)
    interaction_env_description: str | None = field(
            init = False, default = None, validator = validators.optional(validators.instance_of(str))
    )

    def __attrs_post_init__(self):
        """Add default values to the derived paths"""
        object.__setattr__(self, "path_results", self.path_root / self.relpath_results)
        object.__setattr__(self, "path_series_results", self.path_results / self.series)
        object.__setattr__(self, "path_run_model", self.path_series_results / f"{self.name}_model.zip")
        object.__setattr__(self, "path_run_info", self.path_series_results / f"{self.name}_info.json")
        object.__setattr__(self, "path_run_monitor", self.path_series_results / f"{self.name}_monitor.csv")
        object.__setattr__(self, "path_vec_normalize", self.path_series_results / "vec_normalize.pkl")

    def create_results_folders(self):
        if not self.path_results.is_dir():
            for p in reversed(self.path_results.parents):
                if not p.is_dir():
                    p.mkdir()
                    log.info(f"Directory created successfully: \n\t {p}")
            self.path_results.mkdir()
            log.info(f"Directory created successfully: \n\t {self.path_results}")


        if not self.path_series_results.is_dir():
            log.debug("Path for result series doesn't exist on your OS. Trying to create directories.")
            self.path_series_results.mkdir()
            log.info(f"Directory created successfully: \n\t {self.path_series_results}")

    def set_env_info(self, env: BaseEnv):
       version, description = env.get_info()
       object.__setattr__(self, "env_version", version)
       object.__setattr__(self, "env_description", description)

    def set_interaction_env_info(self, env: BaseEnv):
       version, description = env.get_info()
       object.__setattr__(self, "interaction_env_version", version)
       object.__setattr__(self, "interaction_env_description", description)

    @property
    def paths(self) -> dict[str, pathlib.Path]:
        return {
            "path_root": self.path_root,
            "path_results": self.path_results,
            "path_series_results": self.path_series_results,
            "path_run_model": self.path_run_model,
            "path_run_info": self.path_run_info,
            "path_run_monitor": self.path_run_monitor,
        }


def vectorize_environment(
        config: ConfigOpt, config_run: ConfigOptRun, interaction_env: bool = False, training: bool = False
) -> VecNormalize | VecEnv:
    """Vectorize the environment and automatically apply normalization wrappers if configured. If the environment
    is initialized as an interaction_env it will not have normalization wrappers and use the appropriate configuration
    automatically

    :param config: Configuration options for setting up the framework
    :param config_run: Configuration for a specific optimization run
    :param interaction_env: Flag to identify whether the environment should be initialized as an interaction_env. If
                            this is true, the environment will not receive normalization wrappers and automatically
                            use the correct configuration information. (default: False)
    :param training: Flag to identify whether the environment should be initialized for training or playing. It true,
                     it will be initialized for training. (default: False)
    :return: vectorized environments possibly also wrapped in a normalizer.
    """
    # Create the vectorized environment
    if not interaction_env:
        envs = create_environments(
                config.setup.environment_class,
                config_run,
                config.settings,
                config.settings.env,
                config.setup.vectorizer_class,
                config.settings.n_environments
        )
    else:
        # Create the vectorized interaction environment if it is required
        envs = create_environments(
                config.setup.interaction_env_class,
                config_run,
                config.settings,
                config.settings.interaction_env,
                config.setup.vectorizer_class
        )

    if config.setup.monitor_wrapper:
        log.error("Monitoring is not supported for vectorized environments! "
                  "The monitor_wrapper parameter will be ignored.")

    # Automatically normalize the input features
    if not interaction_env and (config.setup.norm_wrapper_obs or config.setup.norm_wrapper_reward):
        # check if normalization data are available; then load
        if config_run.path_vec_normalize.is_file():
            log.info(
                    f"Normalization data detected. Loading running averages into normalization wrapper: \n"
                    f"\t {config_run.path_vec_normalize}"
            )
            envs = VecNormalize.load(str(config_run.path_vec_normalize), envs)
            envs.training = (training,)
            envs.norm_obs = (config.setup.norm_wrapper_obs,)
            envs.norm_reward = (config.setup.norm_wrapper_reward,)
            envs.clip_obs = config.setup.norm_wrapper_clip_obs
        else:
            log.info("No Normalization data detected.")
            envs = VecNormalize(
                    envs,
                    training = training,
                    norm_obs = config.setup.norm_wrapper_obs,
                    norm_reward = config.setup.norm_wrapper_reward,
                    clip_obs = config.setup.norm_wrapper_clip_obs,
            )

    return envs


def create_environments(
        env: type[BaseEnv],
        config_run: ConfigOptRun,
        settings: ConfigOptSettings,
        env_settings: EnvSettings,
        vectorizer: type[VecEnv] = DummyVecEnv,
        n: int = 1
    ) -> VecEnv:
    """ Create vectorized environments.

    :param env: Environment class to instantiate and vectorize
    :param config_run: Configuration for the optimization run
    :param settings: General framework settings
    :param env_settings: Settings for the environment
    :param vectorizer: Vectorizer to be used (default: DummyVecEnv)
    :param n: Number of vectorized environments to generate. (default: 1)
    :return: n vectorized environments
    """
    log.debug("Trying to vectorize the environment.")
    # Ensure n is one if the DummyVecEnv is used (it doesn't support more than one
    if vectorizer.__class__.__name__ == "DummyVecEnv" and n != 1:
        n = 1
        log.warning("Setting number of environments to 1 because DummyVecEnv (default) is used.")

    # Create the vectorized environment
    return vectorizer(
            [
                    lambda env_id = i + 1: env(
                            env_id = env_id,
                            run_name = config_run.name,
                            general_settings = settings.dict,
                            path_settings = config_run.paths,
                            env_settings = env_settings,
                            verbose = env_settings["verbose"],
                            callback = callback_environment,
                    )
                    for i in range(n)
            ]
        )


class ETAx:
    """Initialize an optimization model and provide interfaces for optimization, learning and execution (play)

    :param root_path: Root path of the eta_x application (the configuration will be interpreted relative to this)
    :param config_name: Name of configuration .ini file in configuration directory (should be json format)
    :param config_overwrite: Dictionary to overwrite selected configurations
    :param relpath_config: Relative path to configuration file, starting from root path (default: config/)
    """
    def __init__(
        self,
        root_path: str | os.PathLike,
        config_name: str,
        config_overwrite: Mapping[str, Any] | None = None,
        relpath_config: str | os.PathLike = "config/",
    ) -> None:
        self._environment_vectorized: bool = False
        self._model_initialized: bool = False
        self._prepared: bool = False

        #: Load configuration for the optimization
        _root_path = root_path if isinstance(root_path, pathlib.Path) else pathlib.Path(root_path)
        _relpath_config = relpath_config if isinstance(relpath_config, pathlib.Path) else pathlib.Path(relpath_config)
        self.config: ConfigOpt = ConfigOpt.from_json(
                (_root_path / _relpath_config / f"{config_name}.json"), config_overwrite
        )
        log.setLevel(int(self.config.settings.verbose * 10))

        #: Configuration for an optimization run
        self.config_run: ConfigOptRun

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
        self.config_run = ConfigOptRun(
                    series = series_name,
                    name = run_name,
                    description = run_description,
                    path_root = self.config.path_root,
                    relpath_results = self.config.relpath_results,
            )
        self.config_run.create_results_folders()

        # Vectorize the environments
        self.environments = vectorize_environment(self.config, self.config_run, training = training)
        if self.config.settings.interact_with_env:
            if self.config.setup.interaction_env_class is None:
                raise ValueError(
                        "If 'interact_with_env' is specified, an interaction env class must be specified as well."
                )
            self.interaction_env = vectorize_environment(
                    self.config, self.config_run, interaction_env = True, training = training
            )




        if not self._model_initialized:
            self.initialize_model(reset=reset)


        # set prepared variable when all conditions are met
        if all(
            {
                self._environment_vectorized,
                self._model_initialized,
                self.path_run_model.is_file(),
            }
        ):
            self._prepared = True
            log.info("Run prepared successfully.")

        return self._prepared

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

    def log_run_info(self) -> None:
        """Save run config to result series directory

        :param path_run_info: Save run information to specified path instead of the path defined in configuration
        """
        with self.path_run_info.open("w") as f:
            try:
                json.dump({**self.info, **self.config}, f)
                log.info("Log file successfully created.")
            except TypeError:
                log.warning("Log file could not be created because of non serializable input in config_overwrite.")

    def pretrain(
        self,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        path_expert_dataset: Path | None = None,
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
        series_name: str | None = None,
        run_name: str | None = None,
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

    def play(self, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
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

            action, _states = self.model.predict(observation=observations, deterministic=False)

            # Some agents (i.e. MPC) can interact with an additional environment if required.
            if "interact_with_env" in self.config["settings"] and self.config["settings"]["interact_with_env"]:
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
                observations, rewards, dones, info = self.environments.step(action)

            n_episodes += sum(dones)
