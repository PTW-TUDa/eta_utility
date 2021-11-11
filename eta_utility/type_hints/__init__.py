from .custom_types import Number, Path, TimeStep
from .types_connectors import Connection, Node, Nodes

# Only import eta_x types if it is installed
try:
    import gym
except ModuleNotFoundError:
    pass
else:
    from .types_eta_x import BaseEnv, DefSettings, ReqSettings, StepResult
