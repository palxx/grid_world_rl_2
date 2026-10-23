import pytest

from boxoban_rtg.environments.boxoban import BoxobanEnv, replay
from boxoban_rtg.environments.level_loader import iter_levels, load_level, parse_level_id
from boxoban_rtg.environments.state import parse_level, render_state
from conftest import lv


def test_parse_goal_occupancy_and_roundtrip():
    level = lv("#######", "#@*. $#", "#######", level_id="x")
    assert level.goals == {(1, 2), (1, 3)} and level.init_boxes == {(1, 2), (1, 5)}
    s = level.initial_state()
    text = render_state(level, s).split("\n")
    assert text[1] == "#@*. $#"
    assert parse_level(text).initial_state() == s


def test_player_on_goal_preserved():
    level = lv("#######", "#+$ $.#", "#######")
    assert (1, 1) in level.goals and level.init_player == (1, 1)
    env = BoxobanEnv(5)
    env.reset(level)
    assert env.render().split("\n")[1] == "#+$ $.#"
    env.step("RIGHT")  # push; the goal under the departed player is shown again
    assert env.render().split("\n")[1] == "#.@$$.#"


def test_invalid_levels():
    for rows in [["##", "#@#", "#"], ["#@@#"], ["#@$#"], ["#@x#"]]:
        with pytest.raises(ValueError):
            parse_level(rows)


def test_moves_pushes_and_blocks():
    level = lv("#######", "#@$$..#", "#     #", "#######")
    env = BoxobanEnv(50)
    env.reset(level)
    r = env.step("RIGHT")  # box behind box: blocked
    assert r.blocked and not r.moved_box and r.state.player == (1, 1) and r.state.t == 1
    r = env.step("LEFT")  # wall
    assert r.blocked
    r = env.step("DOWN")
    assert r.moved_player and not r.moved_box and r.state.player == (2, 1)
    r = env.step("UP")
    assert r.state.player == (1, 1)


def test_push_against_wall_blocked():
    level = lv("#####", "#@$.#", "#####")
    env = BoxobanEnv(5)
    env.reset(level)
    env.step("DOWN")
    assert env.state.player == (1, 1)


def test_truncation_vs_termination(one_box):
    env = BoxobanEnv(3)
    env.reset(one_box)
    rs = [env.step("UP") for _ in range(3)]
    assert rs[-1].truncated and not rs[-1].terminated and not rs[0].done
    with pytest.raises(RuntimeError):
        env.step("UP")
    env = BoxobanEnv(1)  # solving on the last allowed step counts as solved, not truncated
    env.reset(one_box)
    r = env.step("RIGHT")
    assert r.terminated and not r.truncated


def test_replay_rejects_actions_after_end(one_box):
    with pytest.raises(ValueError):
        replay(one_box, ["RIGHT", "LEFT"])


def test_deterministic_replay(one_box):
    a = replay(one_box, ["UP", "RIGHT"])
    b = replay(one_box, ["UP", "RIGHT"])
    assert a == b


@pytest.mark.levels
def test_official_level_parsing(levels_root):
    level = load_level("unfiltered/test/000/0", levels_root)
    assert (level.height, level.width) == (10, 10)
    assert len(level.init_boxes) == len(level.goals) == 4
    assert not (level.init_boxes & level.goals) and level.init_player not in level.goals
    assert level.level_hash == load_level("unfiltered/test/000/0", levels_root).level_hash
    assert parse_level_id(level.level_id) == ("unfiltered", "test", "000", 0)
    assert render_state(level, level.initial_state()).split("\n") == list(level.rows)


@pytest.mark.levels
def test_test_partition_size_and_order(levels_root):
    ids = [l.level_id for l in iter_levels("unfiltered", "test", levels_root)]
    assert len(ids) == 1000 and len(set(ids)) == 1000
    assert ids == [l.level_id for l in iter_levels("unfiltered", "test", levels_root)]
