from .math_solver import MathSolver, MPCBasic
from .rule_based import RuleBased

# Import Nsga2 algorithm if julia is available and ignore errors otherwise.
from eta_utility.util_julia import check_ju_extensions_installed
if check_ju_extensions_installed():
    from .nsga2 import Nsga2
