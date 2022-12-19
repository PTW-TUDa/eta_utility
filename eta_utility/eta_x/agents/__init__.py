from .mpc_basic import MPCBasic
from .rule_based import RuleBased

# Import Nsga2 algorithm if julia is available and ignore errors otherwise.
try:
    from eta_utility.util_julia import check_julia_package

    check_julia_package()
    from .nsga2 import Nsga2
except ImportError:
    pass
