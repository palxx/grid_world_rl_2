"""Adapter for AlignmentResearch/boxoban-astar-solutions (a Hugging Face *dataset*, not available offline here).

Format as read from learned-planner's plot/behavior_analysis.py and play_astar_actions.py (NOT yet checked against
the real file): `<difficulty>_<split>.csv.gz` with columns File, Level, Steps, SearchSteps, Actions; Actions is a
string of digits 0-3 in the project's own encoding. Because the digit encoding is unverified, `detect_digit_mapping`
picks the permutation under which the plans actually solve the levels when replayed in our environment, and
refuses to proceed unless one mapping solves (almost) everything. What the solver was minimising (steps vs
pushes) is unknown, so nothing from this source is labelled optimal.
"""
from __future__ import annotations

import csv
import gzip
import itertools

from ..environments.actions import Action
from ..environments.boxoban import replay
from ..environments.level_loader import make_level_id


def read_astar_csv(path, difficulty, split) -> dict:
    """Return {level_id: digit_string} for rows that contain a plan."""
    opener = gzip.open if str(path).endswith(".gz") else open
    plans = {}
    with opener(path, "rt", newline="") as f:
        rd = csv.DictReader(f)
        rd.fieldnames = [c.strip() for c in rd.fieldnames]
        for row in rd:
            acts = (row.get("Actions") or "").strip()
            if acts and acts.isdigit() and set(acts) <= set("0123"):
                lid = make_level_id(difficulty, split, f"{int(row['File']):03d}", int(row["Level"]))
                plans[lid] = acts
    return plans


def digits_to_actions(digits: str, mapping) -> list:
    return [mapping[int(c)] for c in digits]


def detect_digit_mapping(levels_and_digits, max_steps=120, min_fraction=0.99):
    """levels_and_digits: list of (Level, digit_string). Returns (mapping, solved_fraction, all_scores)."""
    scores = {}
    for perm in itertools.permutations(Action, 4):
        ok = 0
        for lv, digits in levels_and_digits:
            try:
                rep = replay(lv, digits_to_actions(digits, perm), max_steps=10_000)
                ok += rep.solved
            except ValueError:
                pass
        scores[tuple(a.name for a in perm)] = ok / len(levels_and_digits)
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    if scores[best] < min_fraction or (len(ranked) > 1 and ranked[1] >= scores[best] * 0.5):
        raise ValueError(f"no unambiguous digit mapping; best={best} fraction={scores[best]:.3f}, runner-up={ranked[1]:.3f}")
    return tuple(Action[n] for n in best), scores[best], scores
