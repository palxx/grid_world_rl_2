import pytest

from boxoban_rtg.environments.boxoban import BoxobanEnv, replay
from boxoban_rtg.trajectories.rtg import float_to_units, rtg_units, update_remaining


def test_rtg_is_suffix_sum_with_zero_terminal():
    r = [-1, -1, 9, -1, 109]
    g = rtg_units(r)
    assert g == [115, 116, 117, 108, 109]
    assert all(g[t] == sum(r[t:]) for t in range(len(r))) and g[0] == sum(r)
    assert g[-1] - r[-1] == 0  # RTG after the last step is zero


def test_closed_loop_update_reaches_zero_exactly(one_box):
    res = replay(one_box, ["UP", "DOWN", "RIGHT"])
    rem = rtg_units([s.reward_units for s in res.steps])[0]
    for s in res.steps:
        rem = update_remaining(rem, s.reward_units)  # remaining(t+1) = remaining(t) - reward(t)
    assert rem == 0 and res.return_units == 107


def test_remaining_matches_rtg_at_every_step(one_box):
    res = replay(one_box, ["UP", "DOWN", "RIGHT"])
    rewards = [s.reward_units for s in res.steps]
    rem, g = sum(rewards), rtg_units(rewards)
    for t, r in enumerate(rewards):
        assert rem == g[t]
        rem = update_remaining(rem, r)


def test_float_units_conversion():
    assert float_to_units(10.7) == 107 and float_to_units(-0.1) == -1
    with pytest.raises(ValueError):
        float_to_units(0.05)
