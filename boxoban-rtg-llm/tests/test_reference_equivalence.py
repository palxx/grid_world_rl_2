"""Lockstep comparison with gym-sokoban 0.0.6 (BoxobanEnv/SokobanEnv). Skipped if it is not installed."""
import random

import pytest

gym_sokoban = pytest.importorskip("gym_sokoban")
np = pytest.importorskip("numpy")

from boxoban_rtg.environments.actions import Action
from boxoban_rtg.environments.boxoban import BoxobanEnv
from boxoban_rtg.environments.level_loader import read_level_file
from boxoban_rtg.environments.state import parse_level

from gym_sokoban.envs.boxoban_env import BoxobanEnv as _RefEnv  # noqa: E402


class RefEnv(_RefEnv):
    def reset(self):  # the reference reset downloads levels; we install the board ourselves
        return None

pytestmark = pytest.mark.reference


def make_ref(level, max_steps):
    """Reference env with room arrays built directly from the level (its parser cannot read '*' or '+')."""
    ref = RefEnv(max_steps=max_steps)
    ref.num_boxes = len(level.goals)
    h, w = level.height, level.width
    fixed, state = np.ones((h, w), int), np.ones((h, w), int)
    for (r, c) in level.walls:
        fixed[r, c] = state[r, c] = 0
    for (r, c) in level.goals:
        fixed[r, c] = state[r, c] = 2
    for (r, c) in level.init_boxes:
        state[r, c] = 3 if (r, c) in level.goals else 4
    state[level.init_player] = 5
    ref.room_fixed, ref.room_state, ref.box_mapping = fixed, state, {}
    ref.player_position = np.array(level.init_player)
    ref.num_env_steps, ref.reward_last = 0, 0
    # The reference assumes 0 boxes start on goals (true for every official level); seed the true count.
    ref.boxes_on_target = len(level.init_boxes & level.goals)
    ref.render = lambda *a, **k: None  # images are irrelevant here
    return ref


def ref_board(ref):
    player = tuple(int(x) for x in ref.player_position)
    boxes = frozenset((int(r), int(c)) for r, c in np.argwhere(np.isin(ref.room_state, [3, 4])))
    return player, boxes


def run_lockstep(level, actions_fn, max_steps, n_steps):
    env, ref = BoxobanEnv(max_steps), make_ref(level, max_steps)
    env.reset(level)
    events = {"on": 0, "off": 0, "solved": 0, "blocked": 0, "push": 0, "trunc": 0}
    for _ in range(n_steps):
        a = actions_fn(env)
        mine = env.step(a)
        _, r, done, info = ref.step(Action(a).gym_id)
        assert abs(mine.reward - r) < 1e-9, (mine.reward, r)
        assert (mine.state.player, mine.state.boxes) == ref_board(ref)
        assert mine.done == done
        assert mine.moved_player == info["action.moved_player"] and mine.moved_box == info["action.moved_box"]
        if done:
            assert mine.terminated == info["all_boxes_on_target"]
            assert mine.truncated == (info["maxsteps_used"] and not info["all_boxes_on_target"])
        events["on"] += mine.reward_units in (9, 109)
        events["off"] += mine.reward_units == -11
        events["solved"] += mine.terminated
        events["blocked"] += mine.blocked
        events["push"] += mine.moved_box
        events["trunc"] += mine.truncated
        if done:
            break
    return events


def totals(*evs):
    out = {}
    for e in evs:
        for k, v in e.items():
            out[k] = out.get(k, 0) + v
    return out


@pytest.mark.levels
def test_random_play_on_official_levels(levels_root):
    levels = read_level_file(levels_root / "unfiltered/train/000.txt", "unfiltered", "train")[:60]
    rng = random.Random(0)
    evs = [run_lockstep(lv, lambda e: rng.randrange(4), 120, 120) for lv in levels for _ in range(3)]
    t = totals(*evs)
    assert t["push"] > 100 and t["blocked"] > 100 and t["trunc"] > 0  # the comparison exercised real events
    assert t["on"] > 0  # at least one box was pushed onto a goal (reward +1)


def test_tiny_levels_cover_finish_and_push_off():
    rng = random.Random(1)
    small = [parse_level(r) for r in (
        ["######", "#@ $.#", "#    #", "######"],
        ["#######", "#.  $ #", "#   @ #", "#######"],
        ["#######", "#@*   #", "#  .$ #", "#######"],
    )]
    evs = [run_lockstep(lv, lambda e: rng.randrange(4), 40, 40) for lv in small for _ in range(400)]
    t = totals(*evs)
    assert t["solved"] > 0 and t["on"] > 0 and t["off"] > 0 and t["trunc"] > 0, t
