from __future__ import annotations

from ..environments.actions import Action
from .base import Policy


class PlanPolicy(Policy):
    """Replay an externally supplied action list (e.g. precomputed A* solutions). Source is recorded via policy_id."""

    def __init__(self, policy_id: str, plans: dict, **params):
        super().__init__(**params)
        self.policy_id, self.plans = policy_id, plans

    def start(self, level, env, rng):
        super().start(level, env, rng)
        self.plan, self.i = list(self.plans[level.level_id]), 0

    def act(self, state):
        if self.i >= len(self.plan):
            raise IndexError("external plan exhausted before the episode ended")
        a = Action(self.plan[self.i])
        self.i += 1
        return a
