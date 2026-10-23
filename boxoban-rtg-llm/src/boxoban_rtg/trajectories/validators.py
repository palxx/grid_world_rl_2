"""Replay-based verification. Nothing is accepted into a dataset without passing validate_episode."""
from __future__ import annotations

from ..environments.boxoban import BoxobanEnv
from ..environments.state import Level
from .rtg import rtg_units
from .schema import Episode


class ValidationError(Exception):
    pass


def validate_episode(ep: Episode, level: Level) -> None:
    errs = []
    if level.level_id != ep.level_id or level.level_hash != ep.level_hash:
        raise ValidationError(f"{ep.episode_id}: level id/hash mismatch")
    T = len(ep.actions)
    if not (len(ep.reward_units) == len(ep.moved_player) == len(ep.moved_box) == T and len(ep.players) == len(ep.boxes) == T + 1):
        raise ValidationError(f"{ep.episode_id}: inconsistent array lengths")
    if T == 0:
        raise ValidationError(f"{ep.episode_id}: empty episode")
    env = BoxobanEnv(ep.max_steps)
    s = env.reset(level)
    if [list(s.player)] != [ep.players[0]] or sorted(map(list, s.boxes)) != ep.boxes[0]:
        errs.append("initial state differs from the level")
    last = None
    for t, a in enumerate(ep.actions):
        if last is not None and last.done:
            errs.append(f"action {t} after the episode ended")
            break
        r = env.step(a)
        last = r
        if r.reward_units != ep.reward_units[t]:
            errs.append(f"t={t}: reward {r.reward_units} != recorded {ep.reward_units[t]}")
        if (r.moved_player, r.moved_box) != (ep.moved_player[t], ep.moved_box[t]):
            errs.append(f"t={t}: moved flags differ")
        if [list(r.state.player)] != [ep.players[t + 1]] or sorted(map(list, r.state.boxes)) != ep.boxes[t + 1]:
            errs.append(f"t={t}: next state differs")
        if len(errs) > 5:
            break
    if last is not None and not last.done:
        errs.append("episode ends before termination or truncation")
    if last is not None and (last.terminated, last.truncated) != (ep.terminated, ep.truncated):
        errs.append("terminated/truncated flags differ from replay")
    rtg = rtg_units(ep.reward_units)
    if rtg[0] != sum(ep.reward_units):
        errs.append("RTG(0) != episode return")
    if ep.solved and ep.return_units != -T + 10 * (len(level.goals) - len(level.init_boxes & level.goals)) + 100:
        errs.append("solved-episode return identity violated")
    if errs:
        raise ValidationError(f"{ep.episode_id} ({ep.policy_id}, {ep.level_id}): " + "; ".join(errs))
