from .inefficient_policy import InefficientSolverPolicy
from .plan_policy import PlanPolicy
from .random_policy import RandomPolicy
from .solver_policy import SolverPolicy
from .stochastic_policy import StochasticSolverPolicy

REGISTRY = {"solver": SolverPolicy, "stochastic_solver": StochasticSolverPolicy,
            "inefficient_solver": InefficientSolverPolicy, "random": RandomPolicy, "plan": PlanPolicy}


def build_policy(name: str, params: dict):
    if name not in REGISTRY:
        raise KeyError(f"unknown policy {name!r}; known: {sorted(REGISTRY)} (astar/DRC sources are added by their loaders)")
    return REGISTRY[name](**params)
