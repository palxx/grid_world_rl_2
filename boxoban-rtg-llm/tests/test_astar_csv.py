import gzip

import pytest

from boxoban_rtg.environments.actions import Action
from boxoban_rtg.solvers.search import solve
from boxoban_rtg.trajectories.astar_csv import detect_digit_mapping, digits_to_actions, read_astar_csv
from test_solver import TOYS

TRUE = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)  # arbitrary stand-in encoding for the test


def digits_for(level, mapping):
    inv = {a: i for i, a in enumerate(mapping)}
    return "".join(str(inv[a]) for a in solve(level).actions)


def test_mapping_recovered_from_replays():
    pairs = [(t, digits_for(t, TRUE)) for t in TOYS[:3]]
    m, frac, _ = detect_digit_mapping(pairs)
    assert m == TRUE and frac == 1.0 and digits_to_actions(pairs[0][1], m) == solve(pairs[0][0]).actions


def test_mapping_refused_when_nothing_solves():
    with pytest.raises(ValueError):
        detect_digit_mapping([(TOYS[0], "0000"), (TOYS[1], "1111")])


def test_csv_reader_skips_rows_without_plans(tmp_path):
    p = tmp_path / "unfiltered_train.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("File, Level, Steps, SearchSteps, Actions\n0,3,4,10,0123\n0,4,,99,\n1,2,2,5,33\n")
    plans = read_astar_csv(p, "unfiltered", "train")
    assert plans == {"unfiltered/train/000/3": "0123", "unfiltered/train/001/2": "33"}
