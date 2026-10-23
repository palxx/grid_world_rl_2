import pytest

from boxoban_rtg.environments.actions import Action, parse_action


def test_mapping_to_gym_push_ids():
    assert [a.gym_id for a in Action] == [1, 2, 3, 4]
    assert [a.delta for a in Action] == [(-1, 0), (1, 0), (0, -1), (0, 1)]


def test_parse_strict():
    assert parse_action("UP") is Action.UP and parse_action(3) is Action.RIGHT
    for bad in ["up", "UP.", " UP", "NOOP", 4, -1, True, None, 1.0]:
        with pytest.raises(ValueError):
            parse_action(bad)
