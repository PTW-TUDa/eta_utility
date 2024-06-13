from eta_utility.util_julia import julia_extensions_available

from .math_solver import (
    MathSolver as MathSolver,
    MPCBasic as MPCBasic,
)
from .rule_based import RuleBased as RuleBased

# Import Nsga2 algorithm if julia is available and ignore errors otherwise.
if julia_extensions_available():
    from .nsga2 import Nsga2 as Nsga2
