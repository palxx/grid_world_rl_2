"""JSONL (human readable, canonical raw store) and Parquet (per-step table) I/O."""
from __future__ import annotations

import json
from pathlib import Path

from ..environments.state import Level, State, render_state
from .schema import ACTION_NAMES, Episode


def write_jsonl(path, episodes) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_dict(), separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield Episode.from_dict(json.loads(line))


def find_episode(paths, episode_id: str) -> Episode:
    for p in paths:
        for ep in read_jsonl(p):
            if ep.episode_id == episode_id:
                return ep
    raise KeyError(episode_id)


def state_at(level: Level, ep: Episode, t: int) -> State:
    return State(tuple(ep.players[t]), frozenset(tuple(b) for b in ep.boxes[t]), t)


def step_rows(ep: Episode, level: Level) -> list[dict]:
    """One row per transition with the fields of the specification ('observation' is filled in the language phase)."""
    rtg, rows, T = ep.rtg_units, [], ep.length
    for t in range(T):
        rows.append(dict(
            episode_id=ep.episode_id, level_id=ep.level_id, level_hash=ep.level_hash, dataset_partition=ep.dataset_partition,
            policy_id=ep.policy_id, policy_params=json.dumps(ep.policy_params, sort_keys=True),
            environment_version=ep.environment_version, seed=ep.seed, time_step=t,
            state=render_state(level, state_at(level, ep, t)), observation=None, action=ACTION_NAMES[ep.actions[t]],
            reward=ep.reward_units[t] / 10, next_state=render_state(level, state_at(level, ep, t + 1)),
            terminated=bool(ep.terminated and t == T - 1), truncated=bool(ep.truncated and t == T - 1),
            episode_return=ep.episode_return, rtg=rtg[t] / 10, episode_length=T, solved=ep.solved,
            metadata=json.dumps(ep.metadata, sort_keys=True)))
    return rows


def write_parquet(path, episodes, levels_by_id) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r for ep in episodes for r in step_rows(ep, levels_by_id[ep.level_id])]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return len(rows)
