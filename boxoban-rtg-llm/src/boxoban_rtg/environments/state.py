"""Immutable level and state types. Coordinates are (row, col), 0-indexed, row 0 at the top."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

Pos = tuple[int, int]

_CHARS = set("# .$@*+")


@dataclass(frozen=True)
class Level:
    level_id: str
    rows: tuple[str, ...]  # canonical text, characters: # space . $ @ * +
    walls: frozenset
    goals: frozenset
    init_player: Pos
    init_boxes: frozenset

    @property
    def height(self) -> int:
        return len(self.rows)

    @property
    def width(self) -> int:
        return len(self.rows[0])

    @property
    def level_hash(self) -> str:
        return hashlib.sha256("\n".join(self.rows).encode()).hexdigest()

    def initial_state(self) -> "State":
        return State(self.init_player, self.init_boxes, 0)


@dataclass(frozen=True)
class State:
    player: Pos
    boxes: frozenset
    t: int = 0

    def key(self) -> tuple:
        """Board identity ignoring the step counter."""
        return (self.player, tuple(sorted(self.boxes)))


def parse_level(rows, level_id: str = "adhoc") -> Level:
    rows = tuple(r.rstrip("\n") for r in rows)
    if not rows or any(len(r) != len(rows[0]) for r in rows):
        raise ValueError("level rows must be non-empty and rectangular")
    walls, goals, boxes, players = set(), set(), set(), []
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch not in _CHARS:
                raise ValueError(f"unknown level character {ch!r} at ({r},{c})")
            if ch == "#":
                walls.add((r, c))
            if ch in ".*+":
                goals.add((r, c))
            if ch in "$*":
                boxes.add((r, c))
            if ch in "@+":
                players.append((r, c))
    if len(players) != 1:
        raise ValueError(f"expected exactly one player, found {len(players)}")
    if len(boxes) != len(goals):
        raise ValueError(f"{len(boxes)} boxes but {len(goals)} goals")
    return Level(level_id, rows, frozenset(walls), frozenset(goals), players[0], frozenset(boxes))


def render_state(level: Level, state: State) -> str:
    """Canonical text board; '*' box on goal, '+' player on goal."""
    out = []
    for r in range(level.height):
        line = []
        for c in range(level.width):
            p = (r, c)
            if p in level.walls:
                line.append("#")
            elif p == state.player:
                line.append("+" if p in level.goals else "@")
            elif p in state.boxes:
                line.append("*" if p in level.goals else "$")
            else:
                line.append("." if p in level.goals else " ")
        out.append("".join(line))
    return "\n".join(out)
