from __future__ import annotations

import importlib
import itertools
import json
import pathlib
import re
from typing import TYPE_CHECKING

from attrs import Factory, converters, define, field, fields, validators  # noqa: I900

from eta_utility import deep_mapping_update, dict_pop_any, get_logger

if TYPE_CHECKING:
    from typing import Any, Mapping

    from attrs import Attribute  # noqa: I900
    from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
    from stable_baselines3.common.vec_env import DummyVecEnv

    from eta_utility.eta_x.envs import BaseEnv
    from eta_utility.type_hints import Path


log = get_logger("eta_x")


def _path_converter(path: Path) -> pathlib.Path:
    """Convert value to a class."""
    return pathlib.Path(path) if not isinstance(path, pathlib.Path) else path


def _get_class(instance: ConfigOptSetup, attrib: Attribute, new_value: str | None) -> str | None:
    """Find module and class name and import the specified class."""
    if new_value is not None:
        module, cls_name = new_value.rsplit(".", 1)
        try:
            cls = getattr(importlib.import_module(module), cls_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not find module '{e.name}'. " f"While importing class '{cls_name}' from '{attrib.name}' value."
            )

        cls_attr_name = f"{attrib.name.rsplit('_', 1)[0]}_class"
        setattr(instance, cls_attr_name, cls)

    return new_value


@define(frozen=False, kw_only=True)
class ConfigOpt:
    """Configuration for the optimization, which can be loaded from a JSON file."""

    #: Name of the configuration used for the series of run.
    config_name: str = field(validator=validators.instance_of(str))

    #: Root path for the optimization run (scenarios and results are relative to this).
    path_root: pathlib.Path = field(converter=_path_converter)
    #: Relative path to the results folder.
    relpath_results: str = field(validator=validators.instance_of(str))
    #: relative path to the scenarios folder (default: None).
    relpath_scenarios: str | None = field(validator=validators.optional(validators.instance_of(str)), default=None)
    #: Path to the results folder.
    path_results: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the scenarios folder (default: None).
    path_scenarios: pathlib.Path | None = field(
        init=False, converter=converters.optional(_path_converter), default=None
    )

    #: Optimization run setup.
    setup: ConfigOptSetup = field()
    #: Optimization run settings.
    settings: ConfigOptSettings = field()

    def __attrs_post_init__(self) -> None:
        object.__setattr__(self, "path_results", self.path_root / self.relpath_results)

        if self.relpath_scenarios is not None:
            object.__setattr__(self, "path_scenarios", self.path_root / self.relpath_scenarios)

    @classmethod
    def from_json(cls, file: Path, path_root: Path, overwrite: Mapping[str, Any] | None = None) -> ConfigOpt:
        """Load configuration from JSON file, which consists of the following sections:

        - **paths**: In this section, the (relative) file paths for results and scenarios are specified. The paths
          are deserialized directly into the :class:`ConfigOpt` object.
        - **setup**: This section specifies which classes and utilities should be used for optimization. The setup
          configuration is deserialized into the :class:`ConfigOptSetup` object.
        - **settings**: The settings section contains basic parameters for the optimization, it is deserialized
          into a :class:`ConfigOptSettings` object.
        - **environment_specific**: The environment section contains keyword arguments for the environment.
          This section must contain values for the arguments of the environment, the expected values are therefore
          different depending on the environment and not fully documented here.
        - **agent_specific**: The agent section contains keyword arguments for the control algorithm (agent).
          This section must contain values for the arguments of the agent, the expected values are therefore
          different depending on the agent and not fully documented here.

        :param file: Path to the configuration file.
        :param overwrite: Config parameters to overwrite.
        :return: ConfigOpt object.
        """
        _file = file if isinstance(file, pathlib.Path) else pathlib.Path(file)
        _path_root = path_root if isinstance(path_root, pathlib.Path) else pathlib.Path(path_root)
        _overwrite = {} if overwrite is None else overwrite

        try:
            # Remove comments from the JSON file (using regular expression), then parse it into a dictionary
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

        # Load values from all other sections.
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

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            raise KeyError(f"The key {name} does not exist - it cannot be set.")
        setattr(self, name, value)


@define(frozen=False, kw_only=True)
class ConfigOptSetup:
    """Configuration options as specified in the "setup" section of the configuration file."""

    #: Import description string for the agent class.
    agent_import: str = field(on_setattr=_get_class)
    #: Agent class (automatically determined from agent_import).
    agent_class: type[BaseAlgorithm] = field(init=False)
    #: Import description string for the environment class.
    environment_import: str = field(on_setattr=_get_class)
    #: Imported Environment class (automatically determined from environment_import).
    environment_class: type[BaseEnv] = field(init=False)
    #: Import description string for the interaction environment (default: None).
    interaction_env_import: str | None = field(default=None, on_setattr=_get_class)
    #: Interaction environment class (default: None) (automatically determined from interaction_env_import).
    interaction_env_class: type[BaseEnv] | None = field(init=False, default=None)

    #: Import description string for the environment vectorizer
    #: (default: stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv).
    vectorizer_import: str = field(
        default="stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv",
        on_setattr=_get_class,
        converter=converters.default_if_none(  # type: ignore
            "stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"
        ),
    )  # mypy currently does not recognize converters.default_if_none
    #: Environment vectorizer class  (automatically determined from vectorizer_import).
    vectorizer_class: type[DummyVecEnv] = field(init=False)
    #: Import description string for the policy class (default: eta_utility.eta_x.agents.common.NoPolicy).
    policy_import: str = field(
        default="eta_utility.eta_x.common.NoPolicy",
        on_setattr=_get_class,
        converter=converters.default_if_none("eta_utility.eta_x.common.NoPolicy"),  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none
    #: Policy class (automatically determined from policy_import).
    policy_class: type[BasePolicy] = field(init=False)

    #: Flag which is true if the environment should be wrapped for monitoring (default: False).
    monitor_wrapper: bool = field(default=False, converter=bool)
    #: Flag which is true if the observations should be normalized (default: False).
    norm_wrapper_obs: bool = field(default=False, converter=bool)
    #: Flag which is true if the observations should be normalized and clipped (default: False).
    norm_wrapper_clip_obs: bool = field(default=False, converter=bool)
    #: Flag which is true if the rewards should be normalized (default: False).
    norm_wrapper_reward: bool = field(default=False, converter=bool)
    #: Flag to enable tensorboard logging (default: False).
    tensorboard_log: bool = field(default=False, converter=bool)

    def __attrs_post_init__(self) -> None:
        _fields = fields(ConfigOptSetup)
        _get_class(self, _fields.agent_import, self.agent_import)
        _get_class(self, _fields.environment_import, self.environment_import)
        _get_class(self, _fields.interaction_env_import, self.interaction_env_import)
        _get_class(self, _fields.vectorizer_import, self.vectorizer_import)
        _get_class(self, _fields.policy_import, self.policy_import)

    @classmethod
    def from_dict(cls, dikt: dict[str, str]) -> ConfigOptSetup:
        errors = False

        if "agent_import" not in dikt and ("agent_package" not in dikt or "agent_class" not in dikt):
            log.error("'agent_import' or both of 'agent_package' and 'agent_class' parameters must " "be specified.")
            errors = True
        if "agent_import" not in dikt:
            agent_import = f"{dikt.pop('agent_package', None)}.{dikt.pop('agent_class', None)}"
        else:
            agent_import = dikt.pop("agent_import")

        if "environment_import" not in dikt and ("environment_package" not in dikt or "environment_class" not in dikt):
            log.error(
                "'environment_import' or both of 'environment_package' and 'environment_class' parameters must "
                "be specified."
            )
            errors = True
        if "environment_import" not in dikt:
            environment_import = f"{dikt.pop('environment_package', None)}.{dikt.pop('environment_class', None)}"
        else:
            environment_import = dikt.pop("environment_import")

        if (
            len({"interaction_env_package", "interaction_env_class"} & dikt.keys()) > 0
            or "interaction_env_import" in dikt
        ):
            if "interaction_env_import" not in dikt and (
                "interaction_env_package" not in dikt or "interaction_env_class" not in dikt
            ):
                log.error(
                    "If one of 'interaction_env_package' and 'interaction_env_class' is specified, "
                    "the other must also be specified."
                )
                errors = True
            if "interaction_env_import" not in dikt:
                interaction_env_import = (
                    f"{dikt.pop('interaction_env_package', None)}.{dikt.pop('interaction_env_class', None)}"
                )
            else:
                interaction_env_import = dikt.pop("interaction_env_import")
        else:
            interaction_env_import = None

        if len({"vectorizer_package", "vectorizer_class"} & dikt.keys()) > 0 or "vectorizer_import" in dikt:
            if "vectorizer_import" not in dikt and ("vectorizer_package" not in dikt or "vectorizer_class" not in dikt):
                log.error(
                    "If one of 'vectorizer_package' and 'vectorizer_class' is specified, "
                    "the other must also be specified."
                )
                errors = True
            if "vectorizer_import" not in dikt:
                vectorizer_import = f"{dikt.pop('vectorizer_package', None)}.{dikt.pop('vectorizer_class', None)}"
            else:
                vectorizer_import = dikt.pop("vectorizer_import")
        else:
            vectorizer_import = None

        if len({"policy_package", "policy_class"} & dikt.keys()) > 0 or "policy_import" in dikt:
            if "policy_import" not in dikt and ("policy_package" not in dikt or "policy_class" not in dikt):
                log.error(
                    "If one of 'policy_package' and 'policy_class' is specified, " "the other must also be specified."
                )
                errors = True
            if "policy_import" not in dikt:
                policy_import = f"{dikt.pop('policy_package', None)}.{dikt.pop('policy_class', None)}"
            else:
                policy_import = dikt.pop("policy_import")
        else:
            policy_import = None

        monitor_wrapper = dikt.pop("monitor_wrapper", None)
        norm_wrapper_obs = dikt.pop("norm_wrapper_obs", None)
        norm_wrapper_clip_obs = dikt.pop("norm_wrapper_clip_obs", None)
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
            norm_wrapper_clip_obs=norm_wrapper_clip_obs,
            norm_wrapper_reward=norm_wrapper_reward,
            tensorboard_log=tensorboard_log,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            raise KeyError(f"The key {name} does not exist - it cannot be set.")
        setattr(self, name, value)


def _env_defaults(instance: ConfigOptSettings, attrib: Attribute, new_value: dict[str, Any] | None) -> dict[str, Any]:
    """Set default values for the environment settings."""
    _new_value = {} if new_value is None else new_value

    _new_value.setdefault("seed", instance.seed)
    _new_value.setdefault("verbose", instance.verbose)
    _new_value.setdefault("sampling_time", instance.sampling_time)
    _new_value.setdefault("episode_duration", instance.episode_duration)

    if instance.sim_steps_per_sample is not None:
        _new_value.setdefault("sim_steps_per_sample", instance.sim_steps_per_sample)

    return _new_value


def _agent_defaults(instance: ConfigOptSettings, attrib: Attribute, new_value: dict[str, Any] | None) -> dict[str, Any]:
    """Set default values for the environment settings."""
    _new_value = {} if new_value is None else new_value

    _new_value.setdefault("seed", instance.seed)
    _new_value.setdefault("verbose", instance.verbose)

    return _new_value


@define(frozen=False, kw_only=True)
class ConfigOptSettings:
    #: Seed for random sampling (default: None).
    seed: int | None = field(default=None, converter=converters.optional(int))
    #: Logging verbosity of the framework (default: 2).
    verbose: int = field(
        default=2, converter=converters.pipe(converters.default_if_none(2), int)  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none
    #: Number of vectorized environments to instantiate (if not using DummyVecEnv) (default: 1).
    n_environments: int = field(
        default=1, converter=converters.pipe(converters.default_if_none(1), int)  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none

    #: Number of episodes to execute when the agent is playing (default: None).
    n_episodes_play: int | None = field(default=None, converter=converters.optional(int))
    #: Number of episodes to execute when the agent is learning (default: None).
    n_episodes_learn: int | None = field(default=None, converter=converters.optional(int))
    #: Flag to determine whether the interaction env is used or not (default: False).
    interact_with_env: bool = field(
        default=False, converter=converters.pipe(converters.default_if_none(False), bool)  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none
    #: How often to save the model during training (default: 1 - after every episode).
    save_model_every_x_episodes: int = field(
        default=1, converter=converters.pipe(converters.default_if_none(1), int)  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none
    #: How many episodes to pass between each render call (default: 1 - after every episode).
    plot_interval: int = field(
        default=1, converter=converters.pipe(converters.default_if_none(1), int)  # type: ignore
    )  # mypy currently does not recognize converters.default_if_none

    #: Duration of an episode in seconds (can be a float value).
    episode_duration: float = field(converter=float)
    #: Duration between time samples in seconds (can be a float value).
    sampling_time: float = field(converter=float)
    #: Simulation steps for every sample.
    sim_steps_per_sample: int | None = field(default=None, converter=converters.optional(int))

    #: Multiplier for scaling the agent actions before passing them to the environment
    #: (especially useful with interaction environments) (default: None).
    scale_actions: float | None = field(default=None, converter=converters.optional(float))
    #: Number of digits to round actions to before passing them to the environment
    #: (especially useful with interaction environments) (default: None).
    round_actions: int | None = field(default=None, converter=converters.optional(int))

    #: Settings dictionary for the environment.
    environment: dict[str, Any] = field(
        default=Factory(dict),
        converter=converters.default_if_none(Factory(dict)),  # type: ignore
        on_setattr=_env_defaults,
    )  # mypy currently does not recognize converters.default_if_none
    #: Settings dictionary for the interaction environment (default: None).
    interaction_env: dict[str, Any] | None = field(default=None, on_setattr=_env_defaults)
    #: Settings dictionary for the agent.
    agent: dict[str, Any] = field(
        default=Factory(dict),
        converter=converters.default_if_none(Factory(dict)),  # type: ignore
        # mypy currently does not recognize converters.default_if_none
        on_setattr=_agent_defaults,
    )

    def __attrs_post_init__(self) -> None:
        _fields = fields(ConfigOptSettings)
        _env_defaults(self, _fields.environment, self.environment)
        _agent_defaults(self, _fields.agent, self.agent)

        # Set standards for interaction env settings or copy settings from environment
        if self.interaction_env is not None:
            _env_defaults(self, _fields.interaction_env, self.interaction_env)
        elif self.interact_with_env is True and self.interaction_env is None:
            log.warning(
                "Interaction with an environment has been requested, but no section 'interaction_env_specific' "
                "found in settings. Re-using 'environment_specific' section."
            )
            self.interaction_env = self.environment

        if self.n_episodes_play is None and self.n_episodes_learn is None:
            raise ValueError("At least one of 'n_episodes_play' or 'n_episodes_learn' must be specified in settings.")

    @classmethod
    def from_dict(cls, dikt: dict[str, dict[str, str]]) -> ConfigOptSettings:
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

        if "n_episodes_play" not in settings and "n_episodes_learn" not in settings:
            log.error("Neither 'n_episodes_play' nor 'n_episodes_learn' is specified in settings.")
            errors = True
        n_epsiodes_play = settings.pop("n_episodes_play", None)
        n_episodes_learn = settings.pop("n_episodes_learn", None)

        interact_with_env = settings.pop("interact_with_env", False)
        save_model_every_x_episodes = settings.pop("save_model_every_x_episodes", None)
        plot_interval = settings.pop("plot_interval", None)

        if "episode_duration" not in settings:
            log.error("'episode_duration' is not specified in settings.")
            errors = True
        episode_duration = settings.pop("episode_duration", None)

        if "sampling_time" not in settings:
            log.error("'sampling_time' is not specified in settings.")
            errors = True
        sampling_time = settings.pop("sampling_time", None)

        sim_steps_per_sample = settings.pop("sim_steps_per_sample", None)
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
            dikt, "interaction_env_specific", "interaction_environment_specific", fail=False, default=None
        )

        # Log configuration values which were not recognized.
        for name in itertools.chain(settings, dikt):
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
            plot_interval=plot_interval,
            episode_duration=episode_duration,  # type: ignore # type is ensured through errors above
            sampling_time=sampling_time,  # type: ignore # type is ensured through errors above
            sim_steps_per_sample=sim_steps_per_sample,
            scale_actions=scale_actions,
            round_actions=round_actions,
            environment=environment,
            agent=agent,
            interaction_env=interaction_env,
        )

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        if not hasattr(self, name):
            raise KeyError(f"The key {name} does not exist - it cannot be set.")
        setattr(self, name, value)


@define(frozen=True, kw_only=True)
class ConfigOptRun:
    """Configuration for an optimization run, including the series and run names descriptions and paths
    for the run.
    """

    #: Name of the series of optimization runs.
    series: str = field(validator=validators.instance_of(str))
    #: Name of an optimization run.
    name: str = field(validator=validators.instance_of(str))
    #: Description of an optimization run.
    description: str = field(
        converter=lambda s: "" if s is None else s,  # type: ignore # mypy does not support unnamed functions
        validator=validators.instance_of(str),
    )
    #: Root path of the framework run.
    path_root: pathlib.Path = field(converter=_path_converter)
    #: Path to results of the optimization run.
    path_results: pathlib.Path = field(converter=_path_converter)
    #: Path to scenarios used for the optimization run.
    path_scenarios: pathlib.Path | None = field(default=None, converter=converters.optional(_path_converter))
    #: Path for the results of the series of optimization runs.
    path_series_results: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the model of the optimization run.
    path_run_model: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to information about the optimization run.
    path_run_info: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the monitoring information about the optimization run.
    path_run_monitor: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the normalization wrapper information.
    path_vec_normalize: pathlib.Path = field(init=False, converter=_path_converter)
    #: Path to the neural network architecture file.
    path_net_arch: pathlib.Path = field(init=False, converter=_path_converter)

    # Information about the environments
    #: Version of the main environment.
    env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the main environment.
    env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    #: Version of the secondary environment (interaction_env).
    interaction_env_version: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )
    #: Description of the secondary environment (interaction_env).
    interaction_env_description: str | None = field(
        init=False, default=None, validator=validators.optional(validators.instance_of(str))
    )

    def __attrs_post_init__(self) -> None:
        """Add default values to the derived paths."""
        object.__setattr__(self, "path_series_results", self.path_results / self.series)
        object.__setattr__(self, "path_run_model", self.path_series_results / f"{self.name}_model.zip")
        object.__setattr__(self, "path_run_info", self.path_series_results / f"{self.name}_info.json")
        object.__setattr__(self, "path_run_monitor", self.path_series_results / f"{self.name}_monitor.csv")
        object.__setattr__(self, "path_vec_normalize", self.path_series_results / "vec_normalize.pkl")
        object.__setattr__(self, "path_net_arch", self.path_series_results / "net_arch.txt")

    def create_results_folders(self) -> None:
        """Create the results folders for an optimization run (or check if they already exist)."""
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

    def set_env_info(self, env: type[BaseEnv]) -> None:
        """Set the environment information of the optimization run to represent the given environment.
        The information will default to None if this is never called.

        :param env: The environment whose description should be used.
        """
        version, description = env.get_info()
        object.__setattr__(self, "env_version", version)
        object.__setattr__(self, "env_description", description)

    def set_interaction_env_info(self, env: type[BaseEnv]) -> None:
        """Set the interaction environment information of the optimization run to represent the given environment.
        The information will default to None if this is never called.

        :param env: The environment whose description should be used.
        """
        version, description = env.get_info()
        object.__setattr__(self, "interaction_env_version", version)
        object.__setattr__(self, "interaction_env_description", description)

    @property
    def paths(self) -> dict[str, pathlib.Path]:
        """Dictionary of all paths for the optimization run. This is for easier access and contains all
        paths as mentioned above."""
        paths = {
            "path_root": self.path_root,
            "path_results": self.path_results,
            "path_series_results": self.path_series_results,
            "path_run_model": self.path_run_model,
            "path_run_info": self.path_run_info,
            "path_run_monitor": self.path_run_monitor,
            "path_vec_normalize": self.path_vec_normalize,
        }
        if self.path_scenarios is not None:
            paths["path_scenarios"] = self.path_scenarios

        return paths
