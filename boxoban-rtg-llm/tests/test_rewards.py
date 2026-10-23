from boxoban_rtg.environments.boxoban import BoxobanEnv, replay
from boxoban_rtg.environments.rewards import reward_units
from conftest import lv


def test_event_rewards_units():
    assert reward_units(0, 0, False) == -1
    assert reward_units(0, 1, False) == 9
    assert reward_units(1, 0, False) == -11
    assert reward_units(3, 4, True) == 109


def test_step_blocked_and_finish(one_box):
    env = BoxobanEnv(10)
    env.reset(one_box)
    r = env.step("UP")  # wall
    assert r.blocked and r.reward_units == -1 and r.reward == -0.1 and not r.done
    r = env.step("RIGHT")  # push onto goal, completes the puzzle
    assert r.moved_box and r.terminated and not r.truncated and r.reward_units == 109 and r.boxes_on_goal == 1


def test_push_off_goal_penalty():
    level = lv("######", "#@*  #", "######")  # box starts on its goal
    env = BoxobanEnv(10)
    env.reset(level)
    assert env.boxes_on_goal() == 1
    r = env.step("RIGHT")
    assert r.moved_box and r.reward_units == -11 and r.boxes_on_goal == 0 and not r.done


def test_episode_return_is_sum_of_step_rewards(one_box):
    res = replay(one_box, ["UP", "DOWN", "RIGHT"])
    assert [s.reward_units for s in res.steps] == [-1, -1, 109]
    assert res.return_units == 107 and abs(res.episode_return - 10.7) < 1e-12
