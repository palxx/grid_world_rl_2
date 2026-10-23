"""Policy interface used by the collector and, later, by LLM rollouts.

A policy sees only the level and the current state (never future rewards or hidden teacher data at inference),
and returns one Action. `observe` receives the executed transition so policies can react to what really happened.
"""
from __future__ import annotations

from ..environments.actions import Action
from ..environments.boxoban import BoxobanEnv, StepResult
from ..environments.state import Level, State


class Policy:
    policy_id = "base"

    def __init__(self, **params):
        self.params = params

    def start(self, level: Level, env: BoxobanEnv, rng) -> None:
        self.level, self.env, self.rng = level, env, rng

    def act(self, state: State) -> Action:
        raise NotImplementedError

    def observe(self, action: Action, result: StepResult) -> None:
        pass

    def info(self) -> dict:
        return {}
