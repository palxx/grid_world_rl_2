"""Deterministic loader for the official google-deepmind/boxoban-levels files.

Layout: <root>/<difficulty>/<split>/NNN.txt (hard has no split directory), each
file holding "; <index>" headers followed by 10 board lines. Level ids are
"<difficulty>/<split>/<file>/<index>" (hard: "hard/<file>/<index>").
"""
from __future__ import annotations

import os
from pathlib import Path

from .state import Level, parse_level

ENV_VAR = "BOXOBAN_LEVELS_DIR"
DEFAULT_ROOT = Path("data/raw/boxoban-levels")


def levels_root(root=None) -> Path:
    p = Path(root or os.environ.get(ENV_VAR) or DEFAULT_ROOT)
    if not p.is_dir():
        raise FileNotFoundError(f"Boxoban levels not found at {p}; run `boxoban-rtg download-levels` or set {ENV_VAR}")
    return p


def split_dir(difficulty: str, split: str | None) -> str:
    return difficulty if difficulty == "hard" else f"{difficulty}/{split}"


def make_level_id(difficulty: str, split: str | None, file_stem: str, index: int) -> str:
    return f"{split_dir(difficulty, split)}/{file_stem}/{index}"


def parse_level_id(level_id: str) -> tuple[str, str | None, str, int]:
    parts = level_id.split("/")
    if len(parts) == 3 and parts[0] == "hard":
        return parts[0], None, parts[1], int(parts[2])
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], int(parts[3])
    raise ValueError(f"bad level id: {level_id!r}")


def read_level_file(path, difficulty: str, split: str | None) -> list[Level]:
    levels, cur, idx = [], [], None

    def flush():
        if idx is not None and cur:
            lid = make_level_id(difficulty, split, Path(path).stem, idx)
            levels.append(parse_level(cur, lid))

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(";"):
                flush()
                cur, idx = [], int(line[1:].strip())
            elif line:
                cur.append(line)
    flush()
    return levels


def load_level(level_id: str, root=None) -> Level:
    difficulty, split, stem, index = parse_level_id(level_id)
    path = levels_root(root) / split_dir(difficulty, split) / f"{stem}.txt"
    for lv in read_level_file(path, difficulty, split):
        if lv.level_id == level_id:
            return lv
    raise KeyError(f"{level_id} not in {path}")


def list_level_files(difficulty: str, split: str | None, root=None) -> list[Path]:
    return sorted((levels_root(root) / split_dir(difficulty, split)).glob("*.txt"))


def iter_levels(difficulty: str, split: str | None, root=None):
    """All levels of a partition in a fixed order (sorted file names, file order)."""
    for p in list_level_files(difficulty, split, root):
        yield from read_level_file(p, difficulty, split)
