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
    s = sub.add_parser("smoke-test")
    s.add_argument("--root")
    s.set_defaults(fn=cmd_smoke_test)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
