from __future__ import annotations

from ..environments.actions import Action
from ..solvers.search import SolverConfig, solve
from .base import Policy


class StochasticSolverPolicy(Policy):
    """Policy B: follow a solver plan, deviate with probability `deviation_prob` per step, then replan.

    After a deviation the policy replans from the actual state. Possible outcomes are all recorded in info():
    replanned, deadlocked (solver proves no solution from the new state), replan_failed (node/time limit), and
    plan_exceeds_horizon (a plan exists but is longer than the remaining step budget). When no plan exists it
    acts randomly. A deviation that changed nothing (blocked move) does not invalidate the plan.
    """

    policy_id = "stochastic_solver"

    def start(self, level, env, rng):
        super().start(level, env, rng)
        self.eps = self.params.get("deviation_prob", 0.1)
        self.cfg = SolverConfig(max_nodes=self.params.get("max_nodes", 50_000), time_limit_s=self.params.get("time_limit_s", 5.0))
        self.max_replans = self.params.get("max_replans", 10)
        self.plan, self.i = [], 0
        self.n_dev = self.n_replans = 0
        self.events = []
        self.dead = False
        self.need_replan = True
        self._plan_for(level.initial_state(), "initial")

    def _plan_for(self, state, why):
        self.n_replans += 1
        r = solve(self.level, state, self.cfg)
        self.need_replan = False
        if r.found:
            self.plan, self.i = list(r.actions), 0
            remaining = self.env.max_steps - state.t
            if r.steps > remaining:
                self.events.append((state.t, "plan_exceeds_horizon", r.steps, remaining))
        else:
            self.plan, self.i = [], 0
            self.dead = True
            self.events.append((state.t, "deadlocked" if r.status == "unsolvable_proven" else "replan_failed", r.status))

    def act(self, state):
        self._before = state
        if self.need_replan and not self.dead:
            if self.n_replans <= self.max_replans:
                self._plan_for(state, "deviation")
            else:
                self.need_replan, self.dead = False, True
                self.events.append((state.t, "replan_budget_exhausted"))
        self._deviated = False
        if self.rng.random() < self.eps:
            self._deviated = True
            self.n_dev += 1
            return Action(self.rng.randrange(4))
        if self.i < len(self.plan):
            a = self.plan[self.i]
            return a
        return Action(self.rng.randrange(4))

    def observe(self, action, result):
        if self._deviated:
            if result.state.key() != self._before.key():
                self.need_replan = True
                self.plan, self.i = [], 0
        elif self.i < len(self.plan) and action == self.plan[self.i]:
            self.i += 1

    def info(self):
        return dict(deviations=self.n_dev, replans=self.n_replans, events=self.events, ended_without_plan=self.dead)
