from __future__ import annotations

import importlib
import inspect
import json
import os
import pathlib
import re
from datetime import datetime
from typing import TYPE_CHECKING, Mapping

import numpy as np
from attrs import (  # noqa: I900
    Attribute,
    Factory,
    converters,
    define,
    field,
    fields,
    validators,
)
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from eta_utility import deep_mapping_update, dict_pop_any, get_logger

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.policies import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

    from eta_utility.eta_x.envs import BaseEnv
    from eta_utility.type_hints import AlgoSettings, EnvSettings, Path

log = get_logger("eta_x", 2)


def _path_converter(path: Path) -> pathlib.Path:
    return pathlib.Path(path) if not isinstance(path, pathlib.Path) else path


def _get_class(instance: ConfigOptSetup, attrib: Attribute, new_value: str | None) -> None:
    # Find module and class name
    if new_value is not None:
        module, cls_name = new_value.rsplit(".", 1)
        cls = getattr(importlib.import_module(module), cls_name)

        cls_attr_name = f"{attrib.name.rsplit('_', 1)[0]}_class"
        setattr(instance, cls_attr_name, cls)


@define(frozen=False)
class ConfigOpt:
    """Configuration for the optimization, which can be loaded from a json file."""

    #: Name of the configuration used for the series of run
    config_name: str = field(kw_only=True, validator=validators.instance_of(str))

    #: Root path for the optimization run (scenarios and results are relative to this)
    path_root: pathlib.Path = field(kw_only=True, converter=_path_converter)
    #: Relative path to the results folder
    relpath_results: str = field(kw_only=True, validator=validators.instance_of(str))
    #: relative path to the scenarios folder
    relpath_scenarios: str | None = field(
        kw_only=True, validator=validators.optional(validators.instance_of(str)), default=None
    )
    #: Path to the results folder
    path_results: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the scenarios folder
    path_scenarios: pathlib.Path | None = field(
        init=False, converter=converters.optional(_path_converter), default=None
    )

    #: Optimization run setup
    setup: ConfigOptSetup = field(kw_only=True)
    #: Optimization run settings
    settings: ConfigOptSettings = field(kw_only=True)

    def __attrs_post_init__(self):
        object.__setattr__(self, "path_results", self.path_root / self.relpath_results)

        if self.relpath_scenarios is not None:
            object.__setattr__(self, "path_scenarios", self.path_root / self.relpath_scenarios)

    @classmethod
    def from_json(cls, file: Path, path_root: Path, overwrite: Mapping[str, Any] | None = None) -> ConfigOpt:
        """Load configuration  from file

        :param file: name of the configuration file in data/configurations/
        :param overwrite: Config parameters to overwrite
        """
        _file = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        _path_root = path_root if isinstance(path_root, pathlib.Path) else pathlib.Path(path_root)
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

        config = dict(deep_mapping_update(config, _overwrite))

        # Ensure all required sections are present in configuration
        if {"setup", "settings", "paths"} > config.keys():
            raise ValueError(
                f"Not all required sections (setup, settings, paths) are present in configuration file {_file}."
            )

        if "environment_specific" not in config:
            config["environment_specific"] = {}
            log.info("Section 'environment_specific' not present in configuration, assuming it is empty.")

        if "agent_specific" not in config:
            config["agent_specific"] = {}
            log.info("Section 'agent_specific' not present in configuration, assuming it is empty.")

        # Load values from paths section
        errors = False
        paths = config.pop("paths")

        if "relpath_results" not in paths:
            log.error("'relpath_results' is required and could not be found in section 'paths' of the configuration.")
            errors = True
        relpath_results = paths.pop("relpath_results", None)

        relpath_scenarios = paths.pop("relpath_scenarios", None)

        # Load values from all other sections
        try:
            setup = ConfigOptSetup.from_dict(config.pop("setup"))
        except ValueError as e:
            log.error(e)
            errors = True

        settings_raw = {}
        settings_raw["settings"] = config.pop("settings")
        settings_raw["environment_specific"] = config.pop("environment_specific")
        if "interaction_env_specific" in config:
            settings_raw["interaction_env_specific"] = config.pop("interaction_env_specific")
        elif "interaction_environment_specific" in config:
            settings_raw["interaction_env_specific"] = config.pop("interaction_environment_specific")
        settings_raw["agent_specific"] = config.pop("agent_specific")
        try:
            settings = ConfigOptSettings.from_dict(settings_raw)
        except ValueError as e:
            log.error(e)
            errors = True

        # Log configuration values which were not recognized.
        for name in config:
            log.warning(
                f"Specified configuration value '{name}' in the setup section of the configuration was not "
                f"recognized and is ignored."
            )

        if errors:
            raise ValueError(
                "Not all required values were found in setup section (see log). " "Could not load config file."
            )

        return cls(
            config_name=str(file),
            path_root=_path_root,
            relpath_results=relpath_results,
            relpath_scenarios=relpath_scenarios,
            setup=setup,
            settings=settings,
        )


@define(frozen=False)
class ConfigOptSetup:
    """Configuration options as specified in the "setup" section of the configuration file."""

    #: Import description string for the agent class
    agent_import: str = field(on_setattr=_get_class)
    #: Agent class
    agent_class: type[BaseAlgorithm] = field(init=False)
    #: Import description string for the environment class
    environment_import: str = field(on_setattr=_get_class)
    #: Environment class
    environment_class: type[BaseEnv] = field(init=False)
    #: Import description string for the interaction environment (default: None)
    interaction_env_import: str | None = field(default=None, on_setattr=_get_class)
    #: Interaction environment class (default. None)
    interaction_env_class: type[BaseEnv] | None = field(init=False, default=None)

    #: Import description string for the environment vectorizer
    #: (default: stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv)
    vectorizer_import: str = field(
        default="stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv",
        on_setattr=_get_class,
        converter=converters.default_if_none("stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"),
    )
    #: Environment vectorizer class
    vectorizer_class: type[VecEnv] = field(init=False)
    #: Import description string for the policy class (default: eta_utility.eta_x.agents.common.NoPolicy)
    policy_import: str = field(
        default="eta_utility.eta_x.agents.common.NoPolicy",
        on_setattr=_get_class,
        converter=converters.default_if_none("eta_utility.eta_x.agents.common.NoPolicy"),
    )
    #: Policy class
    policy_class: type[BasePolicy] = field(init=False)

    #: Flag which is true if the environment should be wrapped for monitoring (default: False)
    monitor_wrapper: bool = field(default=False, converter=bool)
    #: Flag which is true if the observations should be normalized (default: False)
    norm_wrapper_obs: bool = field(default=False, converter=bool)
    #: Flag which is true if the observations should be normalized and clipped (default: False)
    norm_wrapper_clip_obs: bool = field(default=False, converter=bool)
    #: Flag which is true if the rewards should be normalized (default: False)
    norm_wrapper_reward: bool = field(default=False, converter=bool)
    #: Flag to enable tensorboard logging (default: False)
    tensorboard_log: bool = field(default=False, converter=bool)

    def __attrs_post_init__(self):
        _fields = fields(ConfigOptSetup)
        _get_class(self, _fields.agent_import, self.agent_import)
        _get_class(self, _fields.environment_import, self.environment_import)
        _get_class(self, _fields.interaction_env_import, self.interaction_env_import)
        _get_class(self, _fields.vectorizer_import, self.vectorizer_import)
        _get_class(self, _fields.policy_import, self.policy_import)

    @classmethod
    def from_dict(cls, dikt: dict[str, str]) -> ConfigOptSetup:
        errors = False

        if "agent_package" not in dikt or "agent_class" not in dikt:
            log.error("'agent_package' and 'agent_class' parameters must always be specified.")
            errors = True
        agent_import = f"{dikt.pop('agent_package', None)}.{dikt.pop('agent_class', None)}"

        if "environment_package" not in dikt or "environment_class" not in dikt:
            log.error("'environment_package' and 'environment_class' parameters must always be specified.")
            errors = True
        environment_import = f"{dikt.pop('environment_package', None)}.{dikt.pop('environment_class', None)}"

        if len({"interaction_env_package", "interaction_env_class"} & dikt.keys()) > 0:
            if "interaction_env_package" not in dikt or "interaction_env_class" not in dikt:
                log.error(
                    "If one of 'interaction_env_package' and 'interaction_env_class' is specified, "
                    "the other must also be specified."
                )
                errors = True
            interaction_env_import = (
                f"{dikt.pop('interaction_env_package', None)}.{dikt.pop('interaction_env_class', None)}"
            )
        else:
            interaction_env_import = None

        if len({"vectorizer_package", "vectorizer_class"} & dikt.keys()) > 0:
            if "vectorizer_package" not in dikt or "vectorizer_class" not in dikt:
                log.error(
                    "If one of 'vectorizer_package' and 'vectorizer_class' is specified, "
                    "the other must also be specified."
                )
                errors = True
            vectorizer_import = f"{dikt.pop('vectorizer_package', None)}.{dikt.pop('vectorizer_class', None)}"
        else:
            vectorizer_import = None

        if len({"policy_package", "policy_class"} & dikt.keys()) > 0:
            if "policy_package" not in dikt or "policy_class" not in dikt:
                log.error(
                    "If one of 'policy_package' and 'policy_class' is specified, " "the other must also be specified."
                )
                errors = True
            policy_import = f"{dikt.pop('policy_package', None)}.{dikt.pop('policy_class', None)}"
        else:
            policy_import = None

        monitor_wrapper = dikt.pop("monitor_wrapper", None)
        norm_wrapper_obs = dikt.pop("norm_wrapper_obs", None)
        norm_wrapper_reward = dikt.pop("norm_wrapper_reward", None)
        tensorboard_log = dikt.pop("tensorboard_log", None)

        # Log configuration values which were not recognized.
        for name in dikt:
            log.warning(
                f"Specified configuration value '{name}' in the setup section of the configuration was not "
                f"recognized and is ignored."
            )

        if errors:
            raise ValueError(
                "Not all required values were found in setup section (see log). " "Could not load config file."
            )

        return ConfigOptSetup(
            agent_import=agent_import,
            environment_import=environment_import,
            interaction_env_import=interaction_env_import,
            vectorizer_import=vectorizer_import,
            policy_import=policy_import,
            monitor_wrapper=monitor_wrapper,
            norm_wrapper_obs=norm_wrapper_obs,
            norm_wrapper_reward=norm_wrapper_reward,
            tensorboard_log=tensorboard_log,
        )


@define(frozen=False)
class ConfigOptSettings:
    #: Seed for random sampling (default: None)
    seed: int | None = field(kw_only=True, default=None, converter=converters.optional(int))
    #: Logging verbosity of the framework (default: 2)
    verbose: int = field(kw_only=True, default=2, converter=converters.pipe(converters.default_if_none(2), int))
    #: Number of vectorized environments to instantiate (if not using DummyVecEnv) (default: 1)
    n_environments: int = field(kw_only=True, default=1, converter=int)

    #: Number of episodes to execute when the agent is playing (default: None)
    n_episodes_play: int | None = field(kw_only=True, default=None, converter=converters.optional(int))
    #: Number of episodes to execute when the agent is learning (default: None)
    n_episodes_learn: int | None = field(kw_only=True, default=None, converter=converters.optional(int))
    #: Flag to determine whether the interaction env is used or not (default: False)
    interact_with_env: bool = field(
        kw_only=True, default=False, converter=converters.pipe(converters.default_if_none(False), bool)
    )
    #: How often to save the model during training (default: 1 - after every episode)
    save_model_every_x_episodes: int = field(
        kw_only=True, default=1, converter=converters.pipe(converters.default_if_none(1), int)
    )

    #: Duration of an episode in seconds (can be a float value)
    episode_duration: float = field(kw_only=True, converter=float)
    #: Duration between time samples in seconds (can be a float value)
    sampling_time: float = field(kw_only=True, converter=float)
    #: Simulation steps for every sample
    sim_steps_per_sample: int = field(
        kw_only=True, default=1, converter=converters.pipe(converters.default_if_none(1), int)
    )

    #: Multiplier for scaling the agent actions before passing them to the environment
    #: (especially useful with interaciton environments) (default: None)
    scale_actions: float | None = field(kw_only=True, default=None, converter=converters.optional(float))
    #: Number of digits to round actions to before passing them to the environment
    #: (especially useful with interaction environments) (default: None)
    round_actions: int | None = field(kw_only=True, default=None, converter=converters.optional(int))

    #: Settings dictionary for the environment
    environment: dict[str, Any] = field(
        kw_only=True, default=Factory(dict), converter=converters.default_if_none(Factory(dict))
    )
    #: Settings dictionary for the interaction environment (default: None
    interaction_env: dict[str, Any] | None = field(kw_only=True, default=None)
    #: Settings dictionary for the agent
    agent: dict[str, Any] = field(
        kw_only=True, default=Factory(dict), converter=converters.default_if_none(Factory(dict))
    )

    def __attrs_post_init__(self):
        self.environment.setdefault("seed", self.seed)
        self.environment.setdefault("verbose", self.verbose)
        self.agent.setdefault("seed", self.seed)
        self.agent.setdefault("verbose", self.verbose)

        # Set standards for interaction env settings or copy
        if self.interaction_env is not None:
            self.interaction_env.setdefault("seed", self.seed)
            self.interaction_env.setdefault("verbose", self.verbose)
        elif self.interact_with_env is True and self.interaction_env is None:
            log.warning(
                "Interaction with an environment has been requested, but no section 'interaction_env_specific' "
                "found in settings. Re-using 'environment_specific' section."
            )
            self.interaction_env = self.environment

        if self.n_episodes_play is None and self.n_episodes_learn is None:
            raise ValueError("At least one of 'n_episodes_play' or 'n_episodes_learn' must be specified in settings.")

    @classmethod
    def from_dict(cls, dikt: dict[str, dict[str | str]]) -> ConfigOptSettings:
        errors = False

        # Read general settings dictionary
        if "settings" not in dikt:
            raise ValueError("Settings section not found in configuration. Cannot import config file.")
        settings = dikt.pop("settings")

        if "seed" not in settings:
            log.info("'seed' not specified in settings, using default value 'None'")
        seed = settings.pop("seed", None)

        if "verbose" not in settings and "verbosity" not in settings:
            log.info("'verbose' or 'verbosity' not specified in settings, using default value '2'")
        verbose = dict_pop_any(settings, "verbose", "verbosity", fail=False, default=None)

        if "n_environments" not in settings:
            log.info("'n_environments' not specified in settings, using default value '1'")
        n_environments = settings.pop("n_environments", None)

        if "n_epsisodes_play" not in settings and "n_episodes_learn" not in settings:
            log.error("Neither 'n_episodes_play' nor 'n_episodes_learn' is specified in settings.")
            errors = True
        n_epsiodes_play = settings.pop("n_episodes_play", None)
        n_episodes_learn = settings.pop("n_episodes_learn", None)

        interact_with_env = settings.pop("interact_with_env", False)
        save_model_every_x_episodes = settings.pop("save_model_every_x_episodes", None)

        if "episode_duration" not in settings:
            log.error("'episode_duration' is not specified in settings.")
            errors = True
        episode_duration = settings.pop("episode_duration", None)

        if "sampling_time" not in settings:
            log.error("'sampling_time' is not specified in settings.")
            errors = True
        sampling_time = settings.pop("sampling_time", None)

        sim_steps_per_sample = settings.pop("sim_steps_per_sample", 1)
        scale_actions = dict_pop_any(settings, "scale_interaction_actions", "scale_actions", fail=False, default=None)
        round_actions = dict_pop_any(settings, "round_interaction_actions", "round_actions", fail=False, default=None)

        if "environment_specific" not in dikt:
            log.error("'environment_specific' section not defined in settings.")
            errors = True
        environment = dikt.pop("environment_specific", None)

        if "agent_specific" not in dikt:
            log.error("'agent_specific' section not defined in settings.")
            errors = True
        agent = dikt.pop("agent_specific", None)

        interaction_env = dict_pop_any(
            settings, "interaction_env_specific", "interaction_environment_specific", fail=False, default=None
        )

        # Log configuration values which were not recognized.
        for name in settings:
            log.warning(
                f"Specified configuration value '{name}' in the settings section of the configuration "
                f"was not recognized and is ignored."
            )

        if errors:
            raise ValueError("Not all required values were found in settings (see log). Could not load config file.")

        return cls(
            seed=seed,
            verbose=verbose,
            n_environments=n_environments,
            n_episodes_play=n_epsiodes_play,
            n_episodes_learn=n_episodes_learn,
            interact_with_env=interact_with_env,
            save_model_every_x_episodes=save_model_every_x_episodes,
            episode_duration=episode_duration,
            sampling_time=sampling_time,
            sim_steps_per_sample=sim_steps_per_sample,
            scale_actions=scale_actions,
            round_actions=round_actions,
            environment=environment,
            agent=agent,
            interaction_env=interaction_env,
        )


@define(frozen=False)
class ConfigOptRun:
    #: Name of the series of optimization runs
    series: str = field(kw_only=True, validator=validators.instance_of(str))
    #: Name of an optimization run
    name: str = field(kw_only=True, validator=validators.instance_of(str))
    #: Description of an optimization run
    description: str = field(
        kw_only=True,
        converter=lambda s: "" if s is None else s,
        validator=validators.optional(validators.instance_of(str)),
    )
    #: Root path of the framework run
    path_root: pathlib.Path = field(kw_only=True, converter=_path_converter)
    #: Path to results of the optimization run
    path_results: pathlib.Path = field(kw_only=True, converter=_path_converter)
    #: Path to scenarios used for the optimization run
    path_scenarios: pathlib.Path = field(kw_only=True, converter=_path_converter)
    #: Path for the results of the series of optimization runs
    path_series_results: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the model of the optimization run
    path_run_model: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to information about the optimization run
    path_run_info: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the monitoring information about the optimization run
    path_run_monitor: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the normalization wrapper information
    path_vec_normalize: pathlib.Path = field(init=False, converter=_path_converter)

    # Information about the environments
    #: Version of the main environment
    env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the main environment
    env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    #: Version of the secondary environment (interaction_env)
    interaction_env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the secondary environment (interaction_env)
    interaction_env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    def __attrs_post_init__(self):
        """Add default values to the derived paths"""
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

    def set_env_info(self, env: type[BaseEnv]):
        version, description = env.get_info()
        object.__setattr__(self, "env_version", version)
        object.__setattr__(self, "env_description", description)

    def set_interaction_env_info(self, env: type[BaseEnv]):
        version, description = env.get_info()
        object.__setattr__(self, "interaction_env_version", version)
        object.__setattr__(self, "interaction_env_description", description)

    @property
    def paths(self) -> dict[str, pathlib.Path]:
        return {
            "path_results": self.path_results,
            "path_series_results": self.path_series_results,
            "path_run_model": self.path_run_model,
            "path_run_info": self.path_run_info,
            "path_run_monitor": self.path_run_monitor,
        }


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


def vectorize_environment(
    env: type[BaseEnv],
    config_run: ConfigOptRun,
    env_settings: EnvSettings,
    seed: int | None = None,
    verbose: int = 2,
    vectorizer: type[VecEnv] = DummyVecEnv,
    n: int = 1,
    *,
    training=False,
    norm_wrapper_obs: bool = False,
    norm_wrapper_clip_obs: bool = False,
    norm_wrapper_reward: bool = False,
) -> VecNormalize | VecEnv:
    """Vectorize the environment and automatically apply normalization wrappers if configured. If the environment
    is initialized as an interaction_env it will not have normalization wrappers and use the appropriate configuration
    automatically.

    :param env: Environment class which will be instantiated and vectorized
    :param config_run: Configuration for a specific optimization run
    :param env_settings: Configuration settings dictionary for the environment which is being initialized
    :param seed: Random seed for the environment (default: None)
    :param verbose: Logging verbosity to use in the environment (default: 2)
    :param vectorizer: Vectorizer class to use for vectorizing the environments (default: DummyVecEnv)
    :param n: Number of vectorized environments to create (default: 1)
    :param training: Flag to identify whether the environment should be initialized for training or playing. It true,
                     it will be initialized for training. (default: False)
    :param norm_wrapper_obs: Flag to determine whether observations from the environments should be normalized
                             (default: False)
    :param norm_wrapper_clip_obs: Flag to determine whether a normalized observations should be clipped
                             (default: False)
    :param norm_wrapper_reward: Flag to determine whether rewards from the environments should be normalized
                             (default: False)
    :return: Vectorized environments, possibly also wrapped in a normalizer.
    """
    # Create the vectorized environment
    log.debug("Trying to vectorize the environment.")
    # Ensure n is one if the DummyVecEnv is used (it doesn't support more than one
    if vectorizer.__class__.__name__ == "DummyVecEnv" and n != 1:
        n = 1
        log.warning("Setting number of environments to 1 because DummyVecEnv (default) is used.")

    # Create the vectorized environment
    envs = vectorizer(
        [
            lambda env_id=i + 1: env(env_id, config_run, seed, verbose, callback_environment, **env_settings)
            for i in range(n)
        ]
    )

    # Automatically normalize the input features
    if norm_wrapper_obs or norm_wrapper_reward:
        # check if normalization data is available and load it if possible, otherwise
        # create a new normalization wrapper.
        if config_run.path_vec_normalize.is_file():
            log.info(
                f"Normalization data detected. Loading running averages into normalization wrapper: \n"
                f"\t {config_run.path_vec_normalize}"
            )
            envs = VecNormalize.load(str(config_run.path_vec_normalize), envs)
            envs.training = (training,)
            envs.norm_obs = (norm_wrapper_obs,)
            envs.norm_reward = (norm_wrapper_reward,)
            envs.clip_obs = norm_wrapper_clip_obs
        else:
            log.info("No Normalization data detected.")
            envs = VecNormalize(
                envs,
                training=training,
                norm_obs=norm_wrapper_obs,
                norm_reward=norm_wrapper_reward,
                clip_obs=norm_wrapper_clip_obs,
            )

    return envs


def initialize_model(
    algo: type[BaseAlgorithm],
    policy: type[BasePolicy],
    envs: VecEnv | VecNormalize,
    algo_settings: AlgoSettings,
    seed: int | None = None,
    *,
    tensorboard_log: bool = False,
    path_results: Path | None = None,
) -> BaseAlgorithm:
    """Initialize a new model or algorithm

    :param algo: Algorithm to initialize
    :param policy: The policy that should be used by the algorithm
    :param envs: The environment which the algorithm operates on
    :param algo_settings: Additional settings for the algorithm
    :param seed: Random seed to be used by the algorithm (default: None)
    :param tensorboard_log: Flag to enable logging to tensorboard (default: False)
    :param path_results: Path to store results in. Only required if logging is true. (default: None)
    :return: Initialized model
    """
    log.debug(f"Trying to initialize model: {algo.__name__}")
    _path_results = (
        path_results if path_results is None or isinstance(path_results, pathlib.Path) else pathlib.Path(path_results)
    )

    # tensorboard logging
    algo_kwargs = {}
    if tensorboard_log:
        if _path_results is None:
            raise ValueError("If tensorboard logging is enabled, a path for results must be specified as well.")
        log_path = _path_results
        log.info(f"Tensorboard logging is enabled. Log file: {log_path}")
        log.info(
            f"Please run the following command in the console to start tensorboard: \n"
            f"tensorboard --logdir '{log_path}' --port 6006"
        )
        algo_kwargs = {"tensorboard_log": str(_path_results)}

    # check if the agent takes all the default parameters.
    algo_settings.setdefault("seed", seed)

    algo_params = inspect.signature(algo).parameters
    if "seed" not in algo_params and inspect.Parameter.VAR_KEYWORD not in {p.kind for p in algo_params.values()}:
        del algo_settings["seed"]
        log.warning(
            f"'seed' is not a valid parameter for agent {algo.__name__}. This default parameter will be ignored."
        )

    # create model instance
    return algo(policy, envs, **algo_settings, **algo_kwargs)


def load_model(
    algo: type[BaseAlgorithm],
    envs: VecEnv | VecNormalize,
    algo_settings: AlgoSettings,
    path_model: Path,
    *,
    tensorboard_log: bool = False,
    path_results: Path | None = None,
) -> BaseAlgorithm:
    """Load an existing model

    :param algo: Algorithm type of the model to be loaded
    :param envs: The environment which the algorithm operates on
    :param algo_settings: Additional settings for the algorithm
    :param path_model: Path to load the model from
    :param tensorboard_log: Flag to enable logging to tensorboard (default: False)
    :param path_results: Path to store results in. Only required if logging is true. (default: None)
    :return: Initialized model
    """
    log.debug(f"Trying to load existing model: {path_model}")
    _path_model = path_model if isinstance(path_model, pathlib.Path) else pathlib.Path(path_model)
    _path_results = (
        path_results if path_results is None or isinstance(path_results, pathlib.Path) else pathlib.Path(path_results)
    )

    if not _path_model.exists():
        raise OSError(f"Model couldn't be loaded. Path not found: {_path_model}")

    # tensorboard logging
    algo_kwargs = {}
    if tensorboard_log:
        if _path_results is None:
            raise ValueError("If tensorboard logging is enabled, a path for results must be specified as well.")
        log_path = _path_results
        log.info(f"Tensorboard logging is enabled. Log file: {log_path}")
        log.info(
            f"Please run the following command in the console to start tensorboard: \n"
            f"tensorboard --logdir '{log_path}' --port 6006"
        )
        algo_kwargs = {"tensorboard_log": str(_path_results)}

    try:
        model = algo.load(_path_model, envs, **algo_settings, **algo_kwargs)
        log.debug("Model loaded successfully.")
    except OSError as e:
        raise OSError(f"Model couldn't be loaded: {e.strerror}. Filename: {e.filename}") from e

    return model


def log_run_info(config: ConfigOpt, config_run: ConfigOptRun) -> None:
    """Save run configuration to the run_info file

    :param config: Configuration for the framework
    :param config_run: Configuration for this optimization run
    """
    with config_run.path_run_info.open("w") as f:
        try:
            json.dump({**config_run.asdict(), **config.asdict()}, f)
            log.info("Log file successfully created.")
        except TypeError:
            log.warning("Log file could not be created because of non serializable input in config.")


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
        #: Load configuration for the optimization
        _root_path = root_path if isinstance(root_path, pathlib.Path) else pathlib.Path(root_path)
        _relpath_config = relpath_config if isinstance(relpath_config, pathlib.Path) else pathlib.Path(relpath_config)
        self.config: ConfigOpt = ConfigOpt.from_json(
            (_root_path / _relpath_config / f"{config_name}.json"), root_path, config_overwrite
        )
        log.setLevel(int(self.config.settings.verbose * 10))

        #: Configuration for an optimization run
        self.config_run: ConfigOptRun | None = None

        self.environments: VecEnv | VecNormalize | None = None
        self.interaction_env: VecEnv | None = None
        self.model: BaseAlgorithm | None = None

    def prepare_run(
        self, series_name: str, run_name: str, run_description: str = "", reset: bool = False, training: bool = True
    ) -> None:
        """Prepare the learn and play methods

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        :param reset: Should an exisiting model be backed up and overwritten (otherwise load it)? (default: False)
        :param training: Should preparation be done for training (alternative: playing)? (default: True)
        :return: Boolean value indicating successful preparation (
        """
        self.config_run = ConfigOptRun(
            series=series_name,
            name=run_name,
            description=run_description,
            path_results=self.config.path_results,
            path_scenarios=self.config.path_scenarios,
        )
        self.config_run.create_results_folders()

        if self.config.setup.monitor_wrapper:
            log.error(
                "Monitoring is not supported for vectorized environments! "
                "The monitor_wrapper parameter will be ignored."
            )

        # Vectorize the environments
        self.config_run.set_env_info(self.config.setup.environment_class)
        self.environments = vectorize_environment(
            self.config.setup.environment_class,
            self.config_run,
            self.config.settings.environment,
            self.config.settings.seed,
            self.config.settings.verbose,
            self.config.setup.vectorizer_class,
            self.config.settings.n_environments,
            training=training,
            norm_wrapper_obs=self.config.setup.norm_wrapper_obs,
            norm_wrapper_clip_obs=self.config.setup.norm_wrapper_clip_obs,
            norm_wrapper_reward=self.config.setup.norm_wrapper_reward,
        )

        if self.config.settings.interact_with_env:
            if self.config.setup.interaction_env_class is None:
                raise ValueError(
                    "If 'interact_with_env' is specified, an interaction env class must be specified as well."
                )

            self.config_run.set_interaction_env_info(self.config.setup.interaction_env_class)
            self.interaction_env = vectorize_environment(
                self.config.setup.interaction_env_class,
                self.config_run,
                self.config.settings.interaction_env,
                self.config.settings.seed,
                self.config.settings.verbose,
                training=training,
            )

        # Check for existing model and load it or back it up and create a new model
        path_model = self.config_run.path_run_model
        skip_init = False
        if path_model.is_file() and reset:
            log.info(f"Existing model detected: {path_model}")

            bak_name = path_model / f"_{datetime.fromtimestamp(path_model.stat().st_mtime).strftime('%Y%m%d_%H%M')}.bak"
            path_model.rename(bak_name)
            log.info(f"Reset is active. Existing model will be backed up. Backup file name: {bak_name}")
        elif path_model.is_file():
            skip_init = True
            log.info(f"Existing model detected: {path_model}. Loading model.")

            self.model = load_model(
                self.config.setup.agent_class,
                self.environments,
                self.config.settings.agent,
                self.config_run.path_run_model,
                tensorboard_log=self.config.setup.tensorboard_log,
                path_results=self.config_run.path_results,
            )

        # Initialize the model if it wasn't loaded from a file
        if not skip_init:
            self.model = initialize_model(
                self.config.setup.agent_class,
                self.config.setup.policy_class,
                self.environments,
                self.config.settings.agent,
                self.config.settings.seed,
                tensorboard_log=self.config.setup.tensorboard_log,
                path_results=self.config_run.path_results,
            )
        log.info("Run prepared successfully.")

    def learn(
        self,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        reset: bool = False,
    ) -> None:
        """Start the learning job for an agent with the specified environment.

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        """
        if self.environments is None or self.model is None:
            self.prepare_run(series_name, run_name, run_description, reset=reset, training=True)

        log_run_info(self.config, self.config_run)

        # Genetic algorithm has a slightly different concept for saving since it does not stop between time steps
        if "n_generations" in self.config.settings.agent:
            save_freq = self.config.settings.save_model_every_x_episodes
            total_timesteps = self.config.settings.agent["n_generations"]
        else:
            # Check if all required config values are present
            errors = False
            req_settings = {"episode_duration", "sampling_time", "n_episodes_learn"}
            for name in req_settings:
                if getattr(self.config, name) is None:
                    log.error(f"Missing configuration value for learning: {name} in section 'settings'")
                    errors = True
            if errors:
                raise ValueError("Missing configuration values for learning.")

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

        callback_learn = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(self.config_run.path_series_results / "models"),
            name_prefix=self.config_run.name,
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

        # close all environments when done (kill processes)
        log.debug("Closing environments.")
        self.environments.close()

        log.info(f"Learning finished: {series_name} / {run_name}")

    def play(self, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs
        :param run_name: Name for a specific run
        :param run_description: Description for a specific run
        """
        if self.environments is None or self.model is None:
            self.prepare_run(series_name, run_name, run_description, reset=False, training=False)

        if self.config.settings.n_episodes_play is None:
            raise ValueError("Missing configuration value for playing: 'n_episodes_play' in section 'settings'")

        log_run_info(self.config, self.config_run)

        n_episodes_stop = self.config.settings.n_episodes_play

        # Reset the environments before starting to play
        try:
            log.debug("Resetting environments before starting to play.")
            if self.config.settings.interact_with_env:
                observations = self.interaction_env.reset()
                self.environments.reset()
                observations = np.array(self.environments.env_method("update", observations, indices=0))
            else:
                observations = self.environments.reset()
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
            action, _states = self.model.predict(observation=observations, deterministic=False)

            # Round and scale actions if required
            if _round_actions is not None:
                action = np.round(action * _scale_actions, _round_actions)
            else:
                action = action * _scale_actions

            # Some agents (i.e. MPC) can interact with an additional environment
            if self.config.settings.interact_with_env:
                # Perform a step  with the interaction environment and update the normal environment with
                # its observations
                observations, rewards, dones, info = self.interaction_env.step(action)
                observations = np.array(self.environments.env_method("update", observations, indices=0))
            else:
                observations, rewards, dones, info = self.environments.step(action)

            n_episodes += sum(dones)
