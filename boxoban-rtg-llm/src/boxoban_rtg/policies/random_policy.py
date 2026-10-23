from __future__ import annotations

from ..environments.actions import Action
from .base import Policy


class RandomPolicy(Policy):
    """Policy D: uniform random actions, used for low-quality / unsuccessful behaviour."""

    policy_id = "random"

    def act(self, state):
        return Action(self.rng.randrange(4))
