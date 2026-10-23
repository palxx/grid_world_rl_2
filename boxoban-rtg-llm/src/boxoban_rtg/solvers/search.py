"""Push-based A* for Boxoban/Sokoban, from any state, with exact player-walk costs.

A macro-move is "walk to the cell behind a box, then push it one cell". Its cost is walk length + 1 steps
(objective="steps") or 1 (objective="pushes"). The search key is (boxes, player), and the player ends a push on
the box's old cell, so step costs are exact and the returned plan is the minimum number of environment steps
whenever weight == 1 and the search finishes. Pruning uses only sound tests (dead squares, frozen 2x2 blocks).

Optimality labels (field `optimality`):
  "proven_min_steps"   objective=steps, weight=1, no noise/bans, search completed
  "proven_min_pushes"  objective=pushes, weight=1, no noise/bans, search completed (steps NOT minimal)
  "heuristic"          anything else (weighted, noisy or restricted search)
Every plan must still be replayed in the environment; `solve_and_validate` does so.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field

from ..environments.actions import Action
from ..environments.boxoban import DEFAULT_MAX_STEPS, replay
from ..environments.state import Level, State
from .deadlocks import DIRS, floor_cells, frozen_square_deadlock, live_cells_and_push_distance
from .heuristics import box_push_lower_bound

_DIR_TO_ACTION = {a.delta: a for a in Action}


@dataclass(frozen=True)
class SolverConfig:
    objective: str = "steps"  # "steps" | "pushes"
    weight: float = 1.0  # f = g + weight * h ; > 1 gives suboptimal but faster/different plans
    noise: float = 0.0  # multiplicative deterministic heuristic noise in [0, noise]; diversifies plans
    noise_seed: int = 0
    max_nodes: int = 200_000
    time_limit_s: float = 20.0
    banned_root_pushes: frozenset = frozenset()  # {(box_cell, (dr, dc))} not allowed as the first push


@dataclass
class SolveResult:
    status: str  # solved | unsolvable_proven | node_limit | timeout
    actions: list = field(default_factory=list)
    steps: int = 0
    pushes: int = 0
    nodes_expanded: int = 0
    seconds: float = 0.0
    optimality: str = "none"
    objective: str = "steps"
    replay_ok: bool | None = None
    replay_return_units: int | None = None
    config: dict = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return self.status == "solved"


def _reach(level, floor, boxes, start):
    """BFS over floor cells not occupied by boxes: cell -> (distance, parent, action-delta)."""
    seen = {start: (0, None, None)}
    q = deque([start])
    while q:
        p = q.popleft()
        d0 = seen[p][0]
        for dlt in DIRS:
            n = (p[0] + dlt[0], p[1] + dlt[1])
            if n in floor and n not in boxes and n not in seen:
                seen[n] = (d0 + 1, p, dlt)
                q.append(n)
    return seen


def _walk(seen, target):
    deltas = []
    p = target
    while seen[p][1] is not None:
        deltas.append(seen[p][2])
        p = seen[p][1]
    return deltas[::-1]


def solve(level: Level, start: State | None = None, cfg: SolverConfig = SolverConfig()) -> SolveResult:
    t0 = time.perf_counter()
    start = start or level.initial_state()
    floor = floor_cells(level)
    push_dist = live_cells_and_push_distance(level)
    goals = level.goals
    res = SolveResult("node_limit", objective=cfg.objective, config=cfg.__dict__.copy() | {"banned_root_pushes": sorted(cfg.banned_root_pushes)})
    exact = cfg.weight == 1.0 and cfg.noise == 0.0 and not cfg.banned_root_pushes
    res.optimality = ("proven_min_steps" if cfg.objective == "steps" else "proven_min_pushes") if exact else "heuristic"

    def hval(boxes, player):
        h = box_push_lower_bound(boxes, push_dist)
        if h is None:
            return None
        h = h * cfg.weight
        if cfg.noise:
            h *= 1.0 + cfg.noise * ((hash((cfg.noise_seed, player, tuple(sorted(boxes)))) % 1000) / 1000.0)
        return h

    s0 = (frozenset(start.boxes), start.player)
    if s0[0] == goals:
        res.status, res.optimality = "solved", res.optimality
        res.seconds = time.perf_counter() - t0
        return res
    h0 = hval(*s0)
    if h0 is None:
        res.status, res.seconds = "unsolvable_proven", time.perf_counter() - t0
        return res
    best_g = {s0: 0}
    parent = {s0: None}  # state -> (prev_state, stand_cell, delta)
    counter = 0
    heap = [(h0, 0, counter, s0)]
    expanded = 0
    while heap:
        f, ng, _, st = heapq.heappop(heap)
        g = -ng
        if g > best_g.get(st, 1 << 60):
            continue
        boxes, player = st
        if boxes == goals:
            res.status = "solved"
            break
        expanded += 1
        if expanded > cfg.max_nodes:
            res.status = "node_limit"
            break
        if (expanded & 255) == 0 and time.perf_counter() - t0 > cfg.time_limit_s:
            res.status = "timeout"
            break
        seen = _reach(level, floor, boxes, player)
        for b in boxes:
            for dlt in DIRS:
                stand = (b[0] - dlt[0], b[1] - dlt[1])
                if stand not in seen:
                    continue
                if st is s0 and (b, dlt) in cfg.banned_root_pushes:
                    continue
                dest = (b[0] + dlt[0], b[1] + dlt[1])
                if dest not in floor or dest in boxes or dest not in push_dist:
                    continue
                nb = (boxes - {b}) | {dest}
                if frozen_square_deadlock(level, nb, dest):
                    continue
                cost = (seen[stand][0] + 1) if cfg.objective == "steps" else 1
                ns = (frozenset(nb), b)
                ng2 = g + cost
                if ng2 < best_g.get(ns, 1 << 60):
                    h = hval(nb, b)
                    if h is None:
                        continue
                    best_g[ns] = ng2
                    parent[ns] = (st, stand, dlt)
                    counter += 1
                    heapq.heappush(heap, (ng2 + h, -ng2, counter, ns))
    else:
        res.status = "unsolvable_proven" if exact or not cfg.banned_root_pushes else "unsolvable_proven"
    res.nodes_expanded = expanded
    res.seconds = time.perf_counter() - t0
    if res.status != "solved":
        if res.status == "unsolvable_proven" and not exact:
            res.optimality = "none"
        return res
    # reconstruct the step-level action list
    macros = []
    cur = (boxes, player)
    while parent[cur] is not None:
        prev, stand, dlt = parent[cur]
        macros.append((prev, stand, dlt))
        cur = prev
    macros.reverse()
    actions = []
    for prev, stand, dlt in macros:
        pb, pp = prev
        seen = _reach(level, floor, pb, pp)
        actions += [_DIR_TO_ACTION[d] for d in _walk(seen, stand)]
        actions.append(_DIR_TO_ACTION[dlt])
    res.actions, res.steps, res.pushes = actions, len(actions), len(macros)
    return res


def solve_and_validate(level: Level, cfg: SolverConfig = SolverConfig(), max_steps: int = DEFAULT_MAX_STEPS) -> SolveResult:
    """Solve from the initial board and replay the plan in the environment; replay_ok means solved within max_steps."""
    res = solve(level, None, cfg)
    if res.found:
        rep = replay(level, res.actions, max_steps)
        res.replay_ok = bool(rep.solved and len(rep.steps) == len(res.actions))
        res.replay_return_units = rep.return_units
    return res
