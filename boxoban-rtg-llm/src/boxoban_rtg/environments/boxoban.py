"""Boxoban environment adapter reproducing gym-sokoban 0.0.6 dynamics, rewards and horizon.

Differences from the reference, all behavior-preserving on official levels
(checked by the differential tests):
  * rewards are integer units of 0.1 (float reward = units / 10);
  * the initial boxes-on-goal count is computed from the board, not assumed 0
    (identical for all official levels, none starts with a box on a goal);
  * out-of-bounds counts as a wall (the reference relies on the outer wall ring);
  * `truncated` is reported separately: step budget exhausted and not solved.
"""
from __future__ import annotations

from dataclasses import dataclass

from .actions import parse_action
from .rewards import reward_units, units_to_float
from .state import Level, State, render_state

DEFAULT_MAX_STEPS = 120  # gym-sokoban default; always set explicitly in configs


@dataclass(frozen=True)
class StepResult:
    state: State  # next state
    reward_units: int
    terminated: bool  # puzzle solved
    truncated: bool  # horizon reached without solving
    moved_player: bool
    moved_box: bool
    boxes_on_goal: int

    @property
    def reward(self) -> float:
        return units_to_float(self.reward_units)

    @property
    def blocked(self) -> bool:
        return not self.moved_player

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class BoxobanEnv:
    def __init__(self, max_steps: int = DEFAULT_MAX_STEPS):
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps
        self.level: Level | None = None
        self.state: State | None = None
        self._done = True

    # -- queries -------------------------------------------------------
    def boxes_on_goal(self, state: State | None = None) -> int:
        s = state or self.state
        return len(s.boxes & self.level.goals)

    def is_solved(self, state: State | None = None) -> bool:
        s = state or self.state
        return self.level.goals <= s.boxes

    def render(self, state: State | None = None) -> str:
        return render_state(self.level, state or self.state)

    # -- dynamics ------------------------------------------------------
    def reset(self, level: Level) -> State:
        self.level = level
        self.state = level.initial_state()
        self._done = False
        return self.state

    def _free(self, p, boxes) -> bool:
        r, c = p
        lv = self.level
        return 0 <= r < lv.height and 0 <= c < lv.width and p not in lv.walls and p not in boxes

    def step(self, action) -> StepResult:
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset()")
        a = parse_action(action)
        s, d = self.state, a.delta
        before = self.boxes_on_goal(s)
        nxt = (s.player[0] + d[0], s.player[1] + d[1])
        player, boxes, moved_player, moved_box = s.player, s.boxes, False, False
        if nxt in s.boxes:
            dest = (nxt[0] + d[0], nxt[1] + d[1])
            if self._free(dest, s.boxes):
                boxes = (s.boxes - {nxt}) | {dest}
                player, moved_player, moved_box = nxt, True, True
        elif self._free(nxt, s.boxes):
            player, moved_player = nxt, True
        new = State(player, boxes, s.t + 1)
        after = self.boxes_on_goal(new)
        solved = self.is_solved(new)
        truncated = (not solved) and new.t >= self.max_steps
        self.state, self._done = new, solved or truncated
        return StepResult(new, reward_units(before, after, solved), solved, truncated,
                          moved_player, moved_box, after)


@dataclass(frozen=True)
class ReplayResult:
    steps: tuple
    return_units: int
    solved: bool
    truncated: bool

    @property
    def episode_return(self) -> float:
        return units_to_float(self.return_units)


def replay(level: Level, actions, max_steps: int = DEFAULT_MAX_STEPS) -> ReplayResult:
    """Execute actions from the exact initial board. Raises if actions continue after the episode ended."""
    env = BoxobanEnv(max_steps)
    env.reset(level)
    steps = []
    for i, a in enumerate(actions):
        if steps and steps[-1].done:
            raise ValueError(f"action {i} supplied after the episode ended")
        steps.append(env.step(a))
    last = steps[-1] if steps else None
    return ReplayResult(tuple(steps), sum(s.reward_units for s in steps),
                        bool(last and last.terminated), bool(last and last.truncated))
