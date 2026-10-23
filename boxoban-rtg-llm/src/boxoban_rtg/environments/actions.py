"""Four-direction action vocabulary.

Verified against gym-sokoban 0.0.6: ids 1-4 are "push up/down/left/right"; each
pushes an adjacent box if the box can move, otherwise it falls back to a plain
move. Ids 5-8 are move-only and add nothing, so the model only chooses among
UP/DOWN/LEFT/RIGHT, which map to gym ids 1-4.
"""
from enum import IntEnum


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    @property
    def delta(self) -> tuple[int, int]:
        return _DELTAS[self]

    @property
    def gym_id(self) -> int:
        return int(self) + 1


_DELTAS = {Action.UP: (-1, 0), Action.DOWN: (1, 0), Action.LEFT: (0, -1), Action.RIGHT: (0, 1)}
ACTION_NAMES = tuple(a.name for a in Action)


def parse_action(x) -> Action:
    """Strict parse of an Action, int, or exact upper-case name."""
    if isinstance(x, Action):
        return x
    if isinstance(x, str):
        if x in Action.__members__:
            return Action[x]
        raise ValueError(f"not an action name: {x!r}")
    if isinstance(x, (int,)) and not isinstance(x, bool) and 0 <= x < len(Action):
        return Action(x)
    raise ValueError(f"not an action: {x!r}")
