from __future__ import annotations

from ..solvers.search import SolverConfig, solve
from .base import Policy


class SolverPolicy(Policy):
    """Policy A: follow a push-based A* plan computed from the initial state.

    params: objective ("steps"|"pushes"), max_nodes, time_limit_s. If no plan is found the policy emits random
    actions and records solve_status, so failures are visible instead of silently dropped.
    """

    policy_id = "solver"

    def start(self, level, env, rng):
        super().start(level, env, rng)
        cfg = SolverConfig(objective=self.params.get("objective", "steps"), max_nodes=self.params.get("max_nodes", 200_000),
                           time_limit_s=self.params.get("time_limit_s", 20.0))
        self.result = solve(level, None, cfg)
        self.plan = list(self.result.actions)
        self.i = 0

    def act(self, state):
        if self.i < len(self.plan):
            a = self.plan[self.i]
            self.i += 1
            return a
        from ..environments.actions import Action

        return Action(self.rng.randrange(4))

    def info(self):
        r = self.result
        return dict(solve_status=r.status, optimality=r.optimality, objective=r.objective, plan_steps=r.steps,
                    plan_pushes=r.pushes, nodes_expanded=r.nodes_expanded, search_seconds=round(r.seconds, 4))
