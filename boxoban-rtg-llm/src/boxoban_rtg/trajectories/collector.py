"""Unified trajectory collector: runs any Policy in the real environment and validates by replay."""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ..environments.boxoban import BoxobanEnv
from ..environments.level_loader import parse_level_id
from ..environments.state import Level
from ..policies import build_policy
from ..policies.base import Policy
from .schema import ENV_VERSION, Episode, make_episode_id
from .statistics import episode_quality
from .storage import write_jsonl
from .validators import ValidationError, validate_episode


def derive_seed(base: int, *parts) -> int:
    return random.Random(f"{base}|" + "|".join(map(str, parts))).randrange(1 << 31)


def run_episode(level: Level, policy: Policy, seed: int, max_steps: int, policy_id: str | None = None) -> Episode:
    rng = random.Random(seed)
    env = BoxobanEnv(max_steps)
    s = env.reset(level)
    policy.start(level, env, rng)
    actions, rewards, mp, mb = [], [], [], []
    players, boxes = [list(s.player)], [sorted(map(list, s.boxes))]
    while True:
        a = policy.act(env.state)
        r = env.step(a)
        policy.observe(a, r)
        actions.append(int(a))
        rewards.append(r.reward_units)
        mp.append(r.moved_player)
        mb.append(r.moved_box)
        players.append(list(r.state.player))
        boxes.append(sorted(map(list, r.state.boxes)))
        if r.done:
            break
    pid = policy_id or policy.policy_id
    try:
        d, sp, _, _ = parse_level_id(level.level_id)
        partition = f"{d}/{sp}" if sp else d
    except ValueError:  # hand-written test boards
        partition = "adhoc"
    return Episode(make_episode_id(pid, level.level_id, seed), level.level_id, level.level_hash,
                   partition, pid, policy.params, ENV_VERSION, seed, max_steps, actions, rewards, mp, mb,
                   players, boxes, r.terminated, r.truncated, dict(policy_info=policy.info()))


@dataclass
class CollectionReport:
    attempted: int = 0
    accepted: int = 0
    rejected: Counter = field(default_factory=Counter)
    seconds: float = 0.0


def _job(args):
    level, name, params, seed, max_steps, policy_id = args
    return run_episode(level, build_policy(name, params), seed, max_steps, policy_id)


def collect(jobs, levels_by_id, out_path, max_steps, workers=1, quality_limits=None, rejected_path=None, dedupe=True):
    """jobs: list of (level_id, policy_name, params, seed, policy_id).

    Every episode is replay-validated; invalid or over-repetitive episodes are rejected (and written to
    rejected_path) with a reason, never silently dropped. quality_limits: {policy_id: {"repeated_state_frac": x, ...}}.
    """
    t0 = time.time()
    rep = CollectionReport()
    tasks = [(levels_by_id[l], n, p, s, max_steps, pid) for (l, n, p, s, pid) in jobs]
    results = (ProcessPoolExecutor(workers).map(_job, tasks, chunksize=4) if workers > 1 else map(_job, tasks))
    accepted, rejected, seen_traj = [], [], set()
    for ep in results:
        rep.attempted += 1
        try:
            validate_episode(ep, levels_by_id[ep.level_id])
        except ValidationError as e:
            rep.rejected["replay_validation_failed"] += 1
            ep.metadata["rejection"] = str(e)
            rejected.append(ep)
            continue
        lim = (quality_limits or {}).get(ep.policy_id)
        if lim:
            q = episode_quality(ep)
            bad = [k for k, v in lim.items() if q[k] > v]
            if bad:
                rep.rejected["quality:" + ",".join(bad)] += 1
                ep.metadata["rejection"] = f"quality {bad}"
                rejected.append(ep)
                continue
        if dedupe:
            h = (ep.level_id, tuple(ep.actions))
            if h in seen_traj:
                rep.rejected["duplicate_trajectory"] += 1
                ep.metadata["rejection"] = "duplicate of an accepted trajectory"
                rejected.append(ep)
                continue
            seen_traj.add(h)
        accepted.append(ep)
        rep.accepted += 1
    write_jsonl(out_path, accepted)
    if rejected_path and rejected:
        write_jsonl(rejected_path, rejected)
    rep.seconds = time.time() - t0
    return accepted, rejected, rep
