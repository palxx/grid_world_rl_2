"""boxoban-rtg command line. Only subcommands listed here are implemented."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .environments.boxoban import BoxobanEnv, replay
from .environments.level_loader import DEFAULT_ROOT, load_level
from .environments.state import parse_level, render_state

LEVELS_URL = "https://github.com/google-deepmind/boxoban-levels"


def cmd_hardware_check(a):
    from .utils.hardware import format_report, hardware_report

    rep = hardware_report(a.expect_gpus, run_nccl=not a.no_nccl)
    print(format_report(rep))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(format_report(rep))
    if rep["problems"]:
        print("\nPROBLEMS (stop before large-scale GPU work):", *rep["problems"], sep="\n  - ", file=sys.stderr)
        return 2
    return 0


def cmd_download_levels(a):
    dest = Path(a.dest)
    if dest.exists():
        print(f"{dest} already exists; not touching it")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
        subprocess.run(["git", "clone", "--depth", "1", LEVELS_URL, str(dest)], check=True, env=env)
    commit = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    print(f"boxoban-levels commit: {commit}")
    return 0


def cmd_inspect_level(a):
    lv = load_level(a.level_id, a.root)
    print(f"{lv.level_id}  sha256={lv.level_hash}")
    print(render_state(lv, lv.initial_state()))
    print(f"player={lv.init_player} boxes={sorted(lv.init_boxes)} goals={sorted(lv.goals)}")
    return 0


def cmd_replay(a):
    lv = load_level(a.level_id, a.root) if a.level_id else parse_level(["#####", "#@$.#", "#####"], "toy")
    actions = a.actions.split(",") if a.actions else ["UP", "RIGHT"]
    env = BoxobanEnv(a.max_steps)
    s = env.reset(lv)
    print(f"{lv.level_id}\n{env.render()}\n")
    total = 0
    for act in actions:
        r = env.step(act)
        total += r.reward_units
        print(f"t={r.state.t} {act:5s} reward={r.reward:+.1f} moved_player={r.moved_player} moved_box={r.moved_box}"
              f" boxes_on_goal={r.boxes_on_goal} return={total / 10:.1f}\n{env.render()}\n")
        if r.done:
            print("terminated (solved)" if r.terminated else "truncated")
            break
    assert total == replay(lv, actions[: env.state.t], a.max_steps).return_units
    return 0


def cmd_smoke_test(a):
    """Phase-1 smoke test: toy puzzle end to end plus (if present) an official level."""
    toy = parse_level(["#######", "#@ $ .#", "#     #", "#######"], "toy")
    res = replay(toy, ["RIGHT", "RIGHT", "RIGHT"])  # move, push, push onto the goal
    ok = res.solved and res.return_units == -1 - 1 + (-1 + 10 + 100)  # 107 units = 10.7
    print(f"toy replay solved={res.solved} return={res.episode_return:.1f}")
    try:
        lv = load_level("unfiltered/test/000/0", a.root)
        r = replay(lv, ["UP", "DOWN", "LEFT", "RIGHT"])
        print(f"official {lv.level_id} hash={lv.level_hash[:12]} 4-step return={r.episode_return:.1f}")
    except FileNotFoundError as e:
        print(f"official levels not available ({e}); skipped")
    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _levels_for(ids, root):
    from .environments.splits import load_levels

    return {l.level_id: l for l in load_levels(ids, root)}


def cmd_solve(a):
    from .solvers.search import SolverConfig, solve_and_validate

    lv = load_level(a.level_id, a.root)
    cfg = SolverConfig(objective=a.objective, weight=a.weight, max_nodes=a.max_nodes, time_limit_s=a.time_limit)
    r = solve_and_validate(lv, cfg, a.max_steps)
    print(render_state(lv, lv.initial_state()))
    print(f"status={r.status} optimality={r.optimality} objective={r.objective} steps={r.steps} pushes={r.pushes} "
          f"nodes={r.nodes_expanded} seconds={r.seconds:.2f}")
    if r.found:
        print("actions:", ",".join(x.name for x in r.actions))
        print(f"replay_ok={r.replay_ok} replay_return={r.replay_return_units / 10:.1f}")
    return 0 if r.found and r.replay_ok else 1


def _load_cfg(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def cmd_collect(a):
    import hashlib
    import platform
    import subprocess as sp
    from datetime import datetime, timezone

    from .environments.splits import sample_level_ids
    from .trajectories.collector import collect, derive_seed
    from .trajectories.statistics import summarize

    cfg = _load_cfg(a.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    L = cfg["levels"]
    ids = sample_level_ids(L["difficulty"], L["split"], L["n"], L["sampling_seed"], root=a.root)
    by = _levels_for(ids, a.root)
    jobs = []
    for spec in cfg["policies"]:
        name, params, pid = spec["name"], dict(spec.get("params", {})), spec["id"]
        if name in ("drc", "drc_final", "drc_intermediate"):
            raise NotImplementedError("DRC rollouts need the learned-planner stack and Hugging Face checkpoints; not implemented yet")
        if name == "astar_csv":
            from .trajectories.astar_csv import detect_digit_mapping, digits_to_actions, read_astar_csv

            plans = read_astar_csv(params.pop("csv_path"), L["difficulty"], L["split"])
            have = [(by[i], d) for i, d in plans.items() if i in by]
            if not have:
                raise ValueError("A* CSV has no plans for the sampled levels")
            mapping, frac, _ = detect_digit_mapping(have[: min(len(have), 50)])
            params = {"policy_id": pid, "plans": {i: [int(x) for x in digits_to_actions(d, mapping)] for i, d in plans.items() if i in by},
                      "digit_mapping": [m.name for m in mapping]}
            name = "plan"
        per_level = spec.get("per_level")
        count = spec.get("count")
        seq = [(lid, k) for lid in ids for k in range(per_level)] if per_level else [(ids[i % len(ids)], i // len(ids)) for i in range(count)]
        for lid, k in seq:
            jobs.append((lid, name, params, derive_seed(cfg["seed"], pid, lid, k), pid))
    acc, rej, rep = collect(jobs, by, out / "episodes.jsonl", cfg["max_steps"], cfg.get("workers", 1),
                            cfg.get("quality_limits"), out / "rejected.jsonl")
    stats = summarize(acc) if acc else {}
    try:
        commit = sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    manifest = dict(config=cfg, config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                    created=datetime.now(timezone.utc).isoformat(), python=platform.python_version(), git_commit=commit,
                    levels=[dict(level_id=i, level_hash=by[i].level_hash) for i in ids],
                    attempted=rep.attempted, accepted=rep.accepted, rejected=dict(rep.rejected), seconds=round(rep.seconds, 1),
                    stats=stats)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"attempted={rep.attempted} accepted={rep.accepted} rejected={dict(rep.rejected)} in {rep.seconds:.1f}s -> {out}")
    for pid, v in stats.get("policies", {}).items():
        print(f"  {pid:20s} n={v['n']:4d} success={v['success_rate']:.2f} return={v['return_mean']:6.2f}+-{v['return_std']:.2f} "
              f"len={v['length_mean']:.1f} repeated={v['repeated_state_frac']:.2f} reversal={v['reversal_frac']:.2f}")
    for k in ("duplicate_trajectories", "levels_with_multiple_returns", "levels_with_multiple_successful_returns", "levels_with_success_and_failure"):
        if k in stats:
            print(f"  {k}: {stats[k]}")
    return 0


def _episode_paths(a):
    if a.path:
        return [Path(a.path)]
    return sorted(Path("data/trajectories").glob("*/episodes.jsonl"))


def cmd_validate_data(a):
    from .trajectories.storage import read_jsonl
    from .trajectories.validators import ValidationError, validate_episode

    paths = [Path(_load_cfg(a.config)["output_dir"]) / "episodes.jsonl"] if a.config else _episode_paths(a)
    eps = [e for p in paths for e in read_jsonl(p)]
    if not eps:
        print("no episodes found", file=sys.stderr)
        return 1
    by = _levels_for(sorted({e.level_id for e in eps}), a.root)
    bad = 0
    for e in eps:
        try:
            validate_episode(e, by[e.level_id])
        except ValidationError as err:
            bad += 1
            print("INVALID:", err, file=sys.stderr)
    ids = [e.episode_id for e in eps]
    if len(ids) != len(set(ids)):
        bad += 1
        print("duplicate episode ids", file=sys.stderr)
    print(f"validated {len(eps)} episodes from {len(paths)} file(s): {bad} problem(s)")
    return 1 if bad else 0


def cmd_inspect_trajectory(a):
    from .environments.state import State
    from .trajectories.schema import ACTION_NAMES
    from .trajectories.storage import find_episode, state_at

    ep = find_episode(_episode_paths(a), a.episode_id)
    lv = _levels_for([ep.level_id], a.root)[ep.level_id]
    print(f"{ep.episode_id} policy={ep.policy_id} level={ep.level_id} seed={ep.seed} length={ep.length} solved={ep.solved} "
          f"return={ep.episode_return:.1f}\npolicy_info={json.dumps(ep.metadata.get('policy_info'))}")
    print(render_state(lv, state_at(lv, ep, 0)))
    for t in range(ep.length):
        print(f"\nt={t} action={ACTION_NAMES[ep.actions[t]]} reward={ep.reward_units[t] / 10:+.1f} rtg_before={ep.rtg_units[t] / 10:.1f} "
              f"moved_player={ep.moved_player[t]} moved_box={ep.moved_box[t]}")
        if a.boards:
            print(render_state(lv, state_at(lv, ep, t + 1)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="boxoban-rtg")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("hardware-check")
    s.add_argument("--expect-gpus", type=int, default=4)
    s.add_argument("--no-nccl", action="store_true")
    s.add_argument("--out", default="logs/hardware.json")
    s.set_defaults(fn=cmd_hardware_check)
    s = sub.add_parser("download-levels")
    s.add_argument("--dest", default=str(DEFAULT_ROOT))
    s.set_defaults(fn=cmd_download_levels)
    s = sub.add_parser("inspect-level")
    s.add_argument("--level-id", required=True)
    s.add_argument("--root")
    s.set_defaults(fn=cmd_inspect_level)
    s = sub.add_parser("replay", help="step an action list through a level and print the board, rewards and return")
    s.add_argument("--level-id")
    s.add_argument("--actions", help="comma-separated UP,DOWN,LEFT,RIGHT")
    s.add_argument("--max-steps", type=int, default=120)
    s.add_argument("--root")
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("solve")
    s.add_argument("--level-id", required=True)
    s.add_argument("--objective", choices=["steps", "pushes"], default="steps")
    s.add_argument("--weight", type=float, default=1.0)
    s.add_argument("--max-nodes", type=int, default=200_000)
    s.add_argument("--time-limit", type=float, default=20.0)
    s.add_argument("--max-steps", type=int, default=120)
    s.add_argument("--root")
    s.set_defaults(fn=cmd_solve)
    s = sub.add_parser("collect")
    s.add_argument("--config", required=True)
    s.add_argument("--root")
    s.set_defaults(fn=cmd_collect)
    s = sub.add_parser("validate-data")
    s.add_argument("--config")
    s.add_argument("--path")
    s.add_argument("--root")
    s.set_defaults(fn=cmd_validate_data)
    s = sub.add_parser("inspect-trajectory")
    s.add_argument("--episode-id", required=True)
    s.add_argument("--path")
    s.add_argument("--boards", action="store_true", help="print the board after every step")
    s.add_argument("--root")
    s.set_defaults(fn=cmd_inspect_trajectory)
    s = sub.add_parser("smoke-test")
    s.add_argument("--root")
    s.set_defaults(fn=cmd_smoke_test)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
