from __future__ import annotations

import inspect
import json
import pathlib
from functools import partial
from typing import TYPE_CHECKING

from attrs import asdict  # noqa: I900
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from eta_utility import get_logger

from .policies import NoPolicy

if TYPE_CHECKING:
    from typing import Callable

    from gym import Env
    from stable_baselines3.common.base_class import BaseAlgorithm, BasePolicy
    from stable_baselines3.common.vec_env import VecEnv

    from eta_utility.eta_x import ConfigOpt, ConfigOptRun
    from eta_utility.eta_x.envs import BaseEnv
    from eta_utility.type_hints import AlgoSettings, EnvSettings, Path

log = get_logger("eta_x")


class CallbackEnvironment:
    """This callback should be called at the end of each episode.
    When multiprocessing is used, no global variables are available (as an own python instance is created).

    :param plot_interval: How many episodes to pass between each render call.
    """

    def __init__(self, plot_interval: int) -> None:
        self.plot_interval = plot_interval

    def __call__(self, env: BaseEnv) -> None:
        """
        This callback should be called at the end of each episode.
        When multiprocessing is used, no global variables are available (as an own python instance is created).

        :param env: Instance of the environment where the callback was triggered.
        """
        log.info(
            "Environment callback triggered (env_id = {}, n_episodes = {}, run_name = {}.".format(
                env.env_id, env.n_episodes, env.run_name
            )
        )

        # render first episode
        if env.n_episodes == 1:
            env.render()
        # render progress over episodes (for each environment individually)
        elif env.n_episodes % self.plot_interval == 0:
            env.render()
            if hasattr(env, "render_episodes"):
                env.render_episodes()


def vectorize_environment(
    env: type[BaseEnv],
    config_run: ConfigOptRun,
    env_settings: EnvSettings,
    callback: Callable[[BaseEnv], None],
    seed: int | None = None,
    verbose: int = 2,
    vectorizer: type[DummyVecEnv] = DummyVecEnv,
    n: int = 1,
    *,
    training: bool = False,
    monitor_wrapper: bool = False,
    norm_wrapper_obs: bool = False,
    norm_wrapper_clip_obs: bool = False,
    norm_wrapper_reward: bool = False,
) -> VecNormalize | VecEnv:
    """Vectorize the environment and automatically apply normalization wrappers if configured. If the environment
    is initialized as an interaction_env it will not have normalization wrappers and use the appropriate configuration
    automatically.

    :param env: Environment class which will be instantiated and vectorized.
    :param config_run: Configuration for a specific optimization run.
    :param env_settings: Configuration settings dictionary for the environment which is being initialized.
    :param callback: Callback to call with an environment instance.
    :param seed: Random seed for the environment.
    :param verbose: Logging verbosity to use in the environment.
    :param vectorizer: Vectorizer class to use for vectorizing the environments.
    :param n: Number of vectorized environments to create.
    :param training: Flag to identify whether the environment should be initialized for training or playing. If true,
                     it will be initialized for training.
    :param norm_wrapper_obs: Flag to determine whether observations from the environments should be normalized.
    :param norm_wrapper_clip_obs: Flag to determine whether a normalized observations should be clipped.
    :param norm_wrapper_reward: Flag to determine whether rewards from the environments should be normalized.
    :return: Vectorized environments, possibly also wrapped in a normalizer.
    """
    # Create the vectorized environment
    log.debug("Trying to vectorize the environment.")
    # Ensure n is one, if the DummyVecEnv is used (it doesn't support more than one)
    if vectorizer.__class__.__name__ == "DummyVecEnv" and n != 1:
        n = 1
        log.warning("Setting number of environments to 1 because DummyVecEnv (default) is used.")

    if "seed" in env_settings and env_settings["seed"] is not None:
        seed = env_settings.pop("seed")
    else:
        seed = seed

    if "verbose" in env_settings and env_settings["verbose"] is not None:
        verbose = env_settings.pop("verbose")
    else:
        verbose = verbose

    # Create the vectorized environment
    def create_env(env_id: int) -> Env:
        env_id += 1
        return env(env_id, config_run, seed, verbose, callback, **env_settings)

    envs: VecEnv | VecNormalize
    envs = vectorizer([partial(create_env, i) for i in range(n)])

    # The VecMonitor knows the ep_reward and so this can be logged to tensorboard
    if monitor_wrapper:
        envs = VecMonitor(envs)

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
            envs.training = training
            envs.norm_obs = norm_wrapper_obs
            envs.norm_reward = norm_wrapper_reward
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
    """Initialize a new model or algorithm.

    :param algo: Algorithm to initialize.
    :param policy: The policy that should be used by the algorithm.
    :param envs: The environment which the algorithm operates on.
    :param algo_settings: Additional settings for the algorithm.
    :param seed: Random seed to be used by the algorithm.
    :param tensorboard_log: Flag to enable logging to tensorboard.
    :param path_results: Path to store results in. Only required if logging is true.
    :return: Initialized model.
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
    return algo(policy, envs, **algo_settings, **algo_kwargs)  # type: ignore


def load_model(
    algo: type[BaseAlgorithm],
    envs: VecEnv | VecNormalize,
    algo_settings: AlgoSettings,
    path_model: Path,
    *,
    tensorboard_log: bool = False,
    path_results: Path | None = None,
) -> BaseAlgorithm:
    """Load an existing model.

    :param algo: Algorithm type of the model to be loaded.
    :param envs: The environment which the algorithm operates on.
    :param algo_settings: Additional settings for the algorithm.
    :param path_model: Path to load the model from.
    :param tensorboard_log: Flag to enable logging to tensorboard.
    :param path_results: Path to store results in. Only required if logging is true.
    :return: Initialized model.
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
        model = algo.load(_path_model, envs, **algo_settings, **algo_kwargs)  # type: ignore
        log.debug("Model loaded successfully.")
    except OSError as e:
        raise OSError(f"Model couldn't be loaded: {e.strerror}. Filename: {e.filename}") from e

    return model


def log_run_info(config: ConfigOpt, config_run: ConfigOptRun) -> None:
    """Save run configuration to the run_info file.

    :param config: Configuration for the framework.
    :param config_run: Configuration for this optimization run.
    """
    with config_run.path_run_info.open("w") as f:
        try:
            json.dump({**asdict(config_run), **asdict(config)}, f)
            log.info("Log file successfully created.")
        except TypeError:
            log.warning("Log file could not be created because of non-serializable input in config.")


def log_net_arch(model: BaseAlgorithm, config_run: ConfigOptRun) -> None:
    """Store network architecture or policy information in a file. This requires for the model to be initialized,
    otherwise it will raise a ValueError.

    :param model: The algorithm whose network architecture is stored.
    :param config_run: Optimization run configuration (which contains info about the file to store info in).
    :raises: ValueError.
    """
    if not config_run.path_net_arch.exists() and model.policy.__class__ is not NoPolicy:
        with open(config_run.path_net_arch, "w") as f:
            f.write(str(model.policy))

        log.info(f"Net arch / Policy information store successfully in: {config_run.path_net_arch}.")
    elif config_run.path_net_arch.exists():
        log.info(f"Net arch / Policy information already exists in {config_run.path_net_arch}")


def is_vectorized_env(env: BaseEnv | VecEnv | VecNormalize | None) -> bool:
    """Check if an environment is vectorized.

    :param env: The environment to check.
    """
    if env is None:
        return False

    return True if hasattr(env, "num_envs") else False


def is_env_closed(env: BaseEnv | VecEnv | VecNormalize | None) -> bool:
    """Check whether an environment has been closed.

    :param env: The environment to check.
    """
    if env is None:
        return True

    if hasattr(env, "closed"):
        return env.closed  # type: ignore # hasattr not recognized correctly

    if hasattr(env, "venv"):
        return is_env_closed(env.venv)  # type: ignore # hasattr not recognized correctly

    return False


def episode_results_path(series_results_path: Path, run_name: str, episode: int, env_id: int = 1) -> pathlib.Path:
    """Generate a path which can be used for storing episode results of a specific environment.

    :param series_results_path: Path for results of the series of optimization runs.
    :param run_name: Name of the optimization run.
    :param episode: Number of the episode the environment is working on.
    :param env_id: Identification of the environment.
    """
    path = series_results_path if isinstance(series_results_path, pathlib.Path) else pathlib.Path(series_results_path)

    return path / f"{run_name}_{episode:0>#3}_{env_id:0>#2}.csv"
