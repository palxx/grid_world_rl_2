import pytest

from boxoban_rtg.policies import build_policy
from boxoban_rtg.trajectories.collector import derive_seed, run_episode
from test_trajectories import LEVEL


def run(name, seed, params=None):
    return run_episode(LEVEL, build_policy(name, params or {"max_nodes": 50_000, "time_limit_s": 1000.0}) if name != "random" else build_policy(name, {}), seed, 60)


@pytest.mark.parametrize("name", ["random", "stochastic_solver", "inefficient_solver", "solver"])
def test_same_seed_same_episode(name):
    a, b = run(name, 11), run(name, 11)
    assert a.actions == b.actions and a.reward_units == b.reward_units and a.episode_id == b.episode_id


@pytest.mark.parametrize("name", ["random", "stochastic_solver"])
def test_different_seeds_differ(name):
    params = {"deviation_prob": 0.4} if name == "stochastic_solver" else None
    assert len({tuple(run(name, s, params).actions) for s in range(8)}) > 1


def test_derived_seeds_are_stable_and_distinct():
    assert derive_seed(0, "random", "lvl", 1) == derive_seed(0, "random", "lvl", 1)
    assert derive_seed(0, "random", "lvl", 1) != derive_seed(0, "random", "lvl", 2)
    assert derive_seed(0, "random", "lvl", 1) != derive_seed(1, "random", "lvl", 1)
