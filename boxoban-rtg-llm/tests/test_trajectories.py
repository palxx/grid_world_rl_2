import json
import random
from dataclasses import replace

import pytest

from boxoban_rtg.environments.boxoban import replay
from boxoban_rtg.environments.state import parse_level
from boxoban_rtg.policies import build_policy
from boxoban_rtg.trajectories.collector import collect, run_episode
from boxoban_rtg.trajectories.schema import Episode
from boxoban_rtg.trajectories.statistics import episode_quality, summarize
from boxoban_rtg.trajectories.storage import find_episode, read_jsonl, step_rows, write_jsonl
from boxoban_rtg.trajectories.validators import ValidationError, validate_episode
from conftest import lv

LEVEL = parse_level(["########", "#      #", "#@ $ $ #", "# .  . #", "########"], "toy-a")


def episode(name="solver", seed=0, params=None, level=LEVEL, max_steps=60):
    return run_episode(level, build_policy(name, params or {}), seed, max_steps)


def test_solver_episode_is_valid_and_solved():
    ep = episode()
    validate_episode(ep, LEVEL)
    assert ep.solved and ep.rtg_units[0] == ep.return_units and ep.metadata["policy_info"]["optimality"] == "proven_min_steps"


def test_random_episode_runs_to_truncation_and_validates():
    ep = episode("random", seed=3, max_steps=25)
    validate_episode(ep, LEVEL)
    assert ep.length == 25 and ep.truncated and not ep.solved


def test_jsonl_roundtrip(tmp_path):
    eps = [episode(seed=1), episode("random", seed=2, max_steps=20)]
    assert write_jsonl(tmp_path / "e.jsonl", eps) == 2
    back = list(read_jsonl(tmp_path / "e.jsonl"))
    assert [b.to_dict() for b in back] == [e.to_dict() for e in eps]
    assert find_episode([tmp_path / "e.jsonl"], eps[1].episode_id).seed == 2
    assert Episode.from_dict(json.loads(json.dumps(eps[0].to_dict()))) == eps[0]


@pytest.mark.parametrize("tamper", ["reward", "action", "flag", "state", "truncate", "hash"])
def test_validator_rejects_corruption(tamper):
    ep = episode()
    if tamper == "reward":
        ep.reward_units[0] += 1
    elif tamper == "action":
        ep.actions[0] = (ep.actions[0] + 1) % 4
    elif tamper == "flag":
        ep.terminated, ep.truncated = False, True
    elif tamper == "state":
        ep.players[1] = [0, 0]
    elif tamper == "truncate":
        for k in ("actions", "reward_units", "moved_player", "moved_box"):
            setattr(ep, k, getattr(ep, k)[:-1])
        ep.players, ep.boxes = ep.players[:-1], ep.boxes[:-1]
    elif tamper == "hash":
        ep.level_hash = "0" * 64
    with pytest.raises(ValidationError):
        validate_episode(ep, LEVEL)


def test_step_rows_have_spec_fields_and_correct_rtg():
    ep = episode()
    rows = step_rows(ep, LEVEL)
    assert len(rows) == ep.length
    need = {"episode_id", "level_id", "level_hash", "dataset_partition", "policy_id", "policy_params", "environment_version",
            "seed", "time_step", "state", "observation", "action", "reward", "next_state", "terminated", "truncated",
            "episode_return", "rtg", "episode_length", "solved", "metadata"}
    assert need <= set(rows[0])
    assert rows[-1]["terminated"] and not rows[0]["terminated"]
    assert abs(rows[0]["rtg"] - ep.episode_return) < 1e-9 and abs(rows[-1]["rtg"] - rows[-1]["reward"]) < 1e-9


def test_parquet_roundtrip(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet", exc_type=ImportError)
    from boxoban_rtg.trajectories.storage import write_parquet

    eps = [episode(), episode("random", seed=5, max_steps=10)]
    n = write_parquet(tmp_path / "s.parquet", eps, {LEVEL.level_id: LEVEL})
    t = pq.read_table(tmp_path / "s.parquet")
    assert t.num_rows == n == sum(e.length for e in eps) and "rtg" in t.column_names


def test_quality_flags_back_and_forth():
    level = lv("#######", "#     #", "#@  $.#", "#######")
    from boxoban_rtg.policies.plan_policy import PlanPolicy

    plan = {level.level_id: [3, 2, 3, 2, 3, 2, 3, 3, 3]}  # RIGHT/LEFT shuffling, then push
    ep = run_episode(level, PlanPolicy("plan", plan), 0, 30)
    q = episode_quality(ep)
    assert q["reversal_frac"] > 0.4 and q["repeated_state_frac"] > 0.4
    assert episode_quality(episode())["repeated_state_frac"] == 0


def test_collector_accepts_dedupes_and_reports(tmp_path):
    by = {LEVEL.level_id: LEVEL}
    jobs = [(LEVEL.level_id, "solver", {}, 0, "solver"), (LEVEL.level_id, "solver", {}, 1, "solver"),
            (LEVEL.level_id, "random", {}, 2, "random")]
    acc, rej, rep = collect(jobs, by, tmp_path / "a.jsonl", 40, rejected_path=tmp_path / "r.jsonl")
    assert (rep.attempted, rep.accepted) == (3, 2) and rep.rejected["duplicate_trajectory"] == 1
    assert len(list(read_jsonl(tmp_path / "r.jsonl"))) == 1


def test_collector_quality_limit_rejects_with_reason(tmp_path):
    by = {LEVEL.level_id: LEVEL}
    jobs = [(LEVEL.level_id, "random", {}, s, "random") for s in range(4)]
    acc, rej, rep = collect(jobs, by, tmp_path / "a.jsonl", 40, quality_limits={"random": {"repeated_state_frac": 0.01}})
    assert rep.accepted == 0 and sum(rep.rejected.values()) == 4 and all("quality" in k for k in rep.rejected)


def test_stochastic_policy_records_deviations_and_outcomes():
    ep = episode("stochastic_solver", seed=4, params={"deviation_prob": 0.3})
    validate_episode(ep, LEVEL)
    info = ep.metadata["policy_info"]
    assert info["deviations"] > 0 and info["replans"] >= 2


def test_stochastic_policy_records_deadlock():
    # Pushing the box left twice puts it against the left wall (column 1), from where it can never reach the goal.
    level = lv("#######", "#     #", "#  $@ #", "#  .  #", "#######")
    saw = False
    for seed in range(150):
        ep = run_episode(level, build_policy("stochastic_solver", {"deviation_prob": 0.5}), seed, 40)
        validate_episode(ep, level)
        saw |= any(e[1] == "deadlocked" for e in ep.metadata["policy_info"]["events"])
    assert saw


def test_inefficient_policy_solves_and_records_mechanism():
    ep = episode("inefficient_solver", seed=1)
    validate_episode(ep, LEVEL)
    info = ep.metadata["policy_info"]
    assert ep.solved and info["mechanism"] in ("weighted", "noisy", "banned_push", "pushes") and info["attempts"]


def test_summary_counts_levels_with_multiple_returns():
    eps = [episode(), episode("inefficient_solver", seed=2), episode("random", seed=3, max_steps=20)]
    s = summarize(eps)
    assert s["n_levels"] == 1 and s["levels_with_success_and_failure"] == 1 and s["levels_with_multiple_returns"] == 1
