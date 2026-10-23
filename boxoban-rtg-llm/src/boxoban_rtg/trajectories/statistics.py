"""Dataset statistics and episode quality measures. Plain Python, no pandas required."""
from __future__ import annotations

import hashlib
import json
import statistics as st
from collections import defaultdict

from .schema import Episode

_OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}


def episode_quality(ep: Episode) -> dict:
    """Measures used to tell genuine route variation from padding with repeated or reversed moves."""
    T = ep.length
    seen, repeated = set(), 0
    seen.add((tuple(ep.players[0]), tuple(map(tuple, ep.boxes[0]))))
    for t in range(1, T + 1):
        key = (tuple(ep.players[t]), tuple(map(tuple, ep.boxes[t])))
        repeated += key in seen
        seen.add(key)
    reversals = sum(1 for t in range(1, T) if ep.actions[t] == _OPPOSITE[ep.actions[t - 1]]
                    and not ep.moved_box[t] and not ep.moved_box[t - 1] and ep.moved_player[t - 1] and ep.moved_player[t])
    return dict(repeated_state_frac=repeated / T, reversal_frac=reversals / T,
                blocked_frac=sum(1 for m in ep.moved_player if not m) / T, pushes=sum(ep.moved_box))


def trajectory_hash(ep: Episode) -> str:
    return hashlib.sha1(json.dumps([ep.level_id, ep.actions]).encode()).hexdigest()


def _q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def summarize(episodes) -> dict:
    eps = list(episodes)
    by_policy = defaultdict(list)
    for e in eps:
        by_policy[e.policy_id].append(e)
    out = {"n_episodes": len(eps), "n_levels": len({e.level_id for e in eps}), "policies": {}}
    for pid, es in sorted(by_policy.items()):
        rets, lens = [e.episode_return for e in es], [e.length for e in es]
        quality = [episode_quality(e) for e in es]
        out["policies"][pid] = dict(
            n=len(es), success_rate=sum(e.solved for e in es) / len(es),
            return_mean=st.mean(rets), return_std=st.pstdev(rets), return_q=[_q(rets, p) for p in (0.05, 0.5, 0.95)],
            length_mean=st.mean(lens), length_q=[_q(lens, p) for p in (0.05, 0.5, 0.95)],
            repeated_state_frac=st.mean(q["repeated_state_frac"] for q in quality),
            reversal_frac=st.mean(q["reversal_frac"] for q in quality),
            blocked_frac=st.mean(q["blocked_frac"] for q in quality))
    hashes = [trajectory_hash(e) for e in eps]
    out["duplicate_trajectories"] = len(hashes) - len(set(hashes))
    by_level = defaultdict(list)
    for e in eps:
        by_level[e.level_id].append(e)
    out["levels_with_multiple_returns"] = sum(len({e.return_units for e in v}) >= 2 for v in by_level.values())
    out["levels_with_multiple_successful_returns"] = sum(len({e.return_units for e in v if e.solved}) >= 2 for v in by_level.values())
    out["levels_with_success_and_failure"] = sum(any(e.solved for e in v) and any(not e.solved for e in v) for v in by_level.values())
    out["overall_success_rate"] = sum(e.solved for e in eps) / len(eps) if eps else None
    return out
