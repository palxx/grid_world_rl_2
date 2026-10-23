import itertools
from collections import deque
from dataclasses import replace

import pytest

from boxoban_rtg.environments.actions import Action
from boxoban_rtg.environments.boxoban import BoxobanEnv, replay
from boxoban_rtg.environments.level_loader import iter_levels
from boxoban_rtg.solvers.deadlocks import live_cells_and_push_distance
from boxoban_rtg.solvers.search import SolverConfig, solve, solve_and_validate
from conftest import lv

TOYS = [
    lv("#######", "#     #", "#@ $  #", "#   . #", "#######"),
    lv("#######", "#.  $ #", "#   @ #", "#     #", "#######"),
    lv("########", "#      #", "#@ $ $ #", "# .  . #", "########"),
]


def bfs_min_steps(level, limit=40):
    env = BoxobanEnv(10_000)
    seen = {level.initial_state().key()}
    q = deque([((), level.initial_state())])
    while q:
        acts, st = q.popleft()
        if len(acts) >= limit:
            continue
        for a in Action:
            e = BoxobanEnv(10_000)
            e.reset(replace(level, init_player=st.player, init_boxes=st.boxes))
            r = e.step(a)
            if r.terminated:
                return len(acts) + 1
            k = r.state.key()
            if k not in seen:
                seen.add(k)
                q.append((acts + (a,), r.state))
    return None


@pytest.mark.parametrize("level", TOYS)
def test_solver_matches_exhaustive_search_on_toys(level):
    res = solve_and_validate(level)
    assert res.found and res.replay_ok and res.optimality == "proven_min_steps"
    assert res.steps == bfs_min_steps(level)


def test_push_objective_is_labelled_and_not_claimed_step_optimal():
    res = solve(TOYS[2], None, SolverConfig(objective="pushes"))
    assert res.found and res.optimality == "proven_min_pushes" and res.steps >= solve(TOYS[2]).steps


def test_weighted_and_noisy_are_labelled_heuristic():
    assert solve(TOYS[0], None, SolverConfig(weight=3.0)).optimality == "heuristic"
    assert solve(TOYS[0], None, SolverConfig(noise=1.0)).optimality == "heuristic"


def test_unsolvable_is_proven_not_a_timeout():
    dead = lv("######", "#$@ .#", "######")  # box stuck in a corner, off its goal
    res = solve(dead)
    assert res.status == "unsolvable_proven" and not res.found and res.actions == []


def test_node_limit_reported():
    res = solve(TOYS[2], None, SolverConfig(max_nodes=1))
    assert res.status == "node_limit" and not res.found


def test_dead_square_distance_lower_bound():
    d = live_cells_and_push_distance(TOYS[0])
    assert d[(3, 4)] == 0 and (1, 1) not in d  # goal; a corner cell cannot reach a goal


def test_solve_from_arbitrary_state_replays():
    level = TOYS[0]
    env = BoxobanEnv(50)
    env.reset(level)
    env.step("DOWN")
    mid = env.state
    res = solve(level, mid)
    assert res.found
    rep = replay(replace(level, init_player=mid.player, init_boxes=mid.boxes), res.actions)
    assert rep.solved and len(rep.steps) == res.steps


def test_banned_root_push_changes_or_blocks_the_plan():
    level = TOYS[2]
    base = solve(level)
    env = BoxobanEnv(50)
    env.reset(level)
    first = next((a, s) for a, s in zip(base.actions, replay(level, base.actions).steps) if s.moved_box)
    banned = frozenset({(first[1].state.player, first[0].delta)})
    alt = solve(level, None, SolverConfig(banned_root_pushes=banned))
    assert alt.status in ("solved", "unsolvable_proven") and (not alt.found or alt.actions != base.actions)


@pytest.mark.levels
def test_official_levels_solved_and_replayed_with_return_identity(levels_root):
    for level in itertools.islice(iter_levels("unfiltered", "test", levels_root), 8):
        res = solve_and_validate(level, SolverConfig(time_limit_s=30))
        assert res.found and res.replay_ok and res.optimality == "proven_min_steps"
        # With 4 boxes, none initially on a goal: every solved episode returns exactly 14 - 0.1 * steps.
        assert res.replay_return_units == 140 - res.steps
