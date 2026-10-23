"""Sound deadlock tests: they only prune states from which no solution exists."""
from __future__ import annotations

from ..environments.actions import Action
from ..environments.state import Level

DIRS = [a.delta for a in Action]


def floor_cells(level: Level) -> frozenset:
    return frozenset((r, c) for r in range(level.height) for c in range(level.width) if (r, c) not in level.walls)


def live_cells_and_push_distance(level: Level) -> dict:
    """Cells from which a lone box can reach some goal, mapped to the minimum number of pushes.

    Reverse BFS from the goals over "pulls": a box at p could have come from q = p - d if both q and the
    player cell q - d are floor. Ignores other boxes, so it is a lower bound (admissible) and the key set is a
    superset of the truly live cells (never prunes a solvable state).
    """
    floor = floor_cells(level)
    dist = {g: 0 for g in level.goals}
    frontier = list(level.goals)
    while frontier:
        nxt = []
        for p in frontier:
            for dr, dc in DIRS:
                q = (p[0] - dr, p[1] - dc)
                player = (q[0] - dr, q[1] - dc)
                if q in floor and player in floor and q not in dist:
                    dist[q] = dist[p] + 1
                    nxt.append(q)
        frontier = nxt
    return dist


def frozen_square_deadlock(level: Level, boxes, moved) -> bool:
    """True if the box just moved to `moved` lies in a 2x2 block of walls/boxes with a box off a goal.

    Such a block can never move again, so if any of its boxes is off a goal the level is unsolvable.
    """
    r0, c0 = moved
    blocked = lambda p: p in level.walls or p in boxes
    for dr in (-1, 0):
        for dc in (-1, 0):
            cells = [(r0 + dr, c0 + dc), (r0 + dr, c0 + dc + 1), (r0 + dr + 1, c0 + dc), (r0 + dr + 1, c0 + dc + 1)]
            if all(blocked(p) for p in cells) and any(p in boxes and p not in level.goals for p in cells):
                return True
    return False
