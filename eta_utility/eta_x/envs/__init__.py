from eta_utility.util_julia import julia_extensions_available

from .base_env import BaseEnv as BaseEnv
from .base_env_live import BaseEnvLive as BaseEnvLive
from .base_env_mpc import BaseEnvMPC as BaseEnvMPC
from .base_env_sim import BaseEnvSim as BaseEnvSim
from .no_vec_env import NoVecEnv as NoVecEnv
from .state import (
    StateConfig as StateConfig,
    StateVar as StateVar,
)

# Import JuliaEnv if julia is available and ignore errors otherwise.
if julia_extensions_available():
    from .julia_env import JuliaEnv as JuliaEnv
