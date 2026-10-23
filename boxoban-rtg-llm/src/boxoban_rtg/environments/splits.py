"""Deterministic level sampling with exact-duplicate (hash) exclusion across partitions.

Measured on the official files: unfiltered train/valid/test and medium/hard contain a few exact duplicate boards
across partitions (e.g. unfiltered train & medium train: 118). Splits therefore exclude by level hash, not
only by partition name.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .level_loader import levels_root, make_level_id, read_level_file, split_dir
from .state import Level

PARTITIONS = [("unfiltered", "train"), ("unfiltered", "valid"), ("unfiltered", "test"),
              ("medium", "train"), ("medium", "valid"), ("hard", None)]
LEVELS_PER_FILE = 1000


def partition_name(difficulty, split) -> str:
    return split_dir(difficulty, split)


def n_files(difficulty, split, root=None) -> int:
    return len(list((levels_root(root) / split_dir(difficulty, split)).glob("*.txt")))


def load_levels(level_ids, root=None) -> list[Level]:
    """Load specific levels, reading each file once; result keeps the order of level_ids."""
    from .level_loader import parse_level_id

    by_file = {}
    for lid in level_ids:
        d, s, stem, idx = parse_level_id(lid)
        by_file.setdefault((d, s, stem), set()).add(lid)
    got = {}
    for (d, s, stem), wanted in by_file.items():
        for lv in read_level_file(levels_root(root) / split_dir(d, s) / f"{stem}.txt", d, s):
            if lv.level_id in wanted:
                got[lv.level_id] = lv
    missing = [i for i in level_ids if i not in got]
    if missing:
        raise KeyError(f"levels not found: {missing[:3]}...")
    return [got[i] for i in level_ids]


def compute_overlaps(root=None) -> dict:
    """Hash every official partition and list boards present in more than one (slow, run once)."""
    seen = {}  # hash -> list of level ids
    for d, s in PARTITIONS:
        for f in sorted((levels_root(root) / split_dir(d, s)).glob("*.txt")):
            for lv in read_level_file(f, d, s):
                seen.setdefault(lv.level_hash, []).append(lv.level_id)
    return {h: ids for h, ids in seen.items() if len({partition_name(*parse_part(i)) for i in ids}) > 1 or len(ids) > 1}


def parse_part(level_id):
    from .level_loader import parse_level_id

    d, s, _, _ = parse_level_id(level_id)
    return d, s


def sample_level_ids(difficulty, split, n, seed, exclude_ids=(), root=None) -> list[str]:
    """Uniform sample without replacement, deterministic in (difficulty, split, n, seed)."""
    nf = n_files(difficulty, split, root)
    rng = random.Random(f"{seed}|{difficulty}|{split}")
    exclude = set(exclude_ids)
    out = []
    for flat in rng.sample(range(nf * LEVELS_PER_FILE), min(nf * LEVELS_PER_FILE, n + len(exclude) + 10)):
        lid = make_level_id(difficulty, split, f"{flat // LEVELS_PER_FILE:03d}", flat % LEVELS_PER_FILE)
        if lid not in exclude:
            out.append(lid)
        if len(out) == n:
            return out
    raise ValueError("not enough levels after exclusion")


def write_split(path, name, level_ids, levels, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(dict(name=name, meta=meta or {}, levels=[
        dict(level_id=l.level_id, level_hash=l.level_hash) for l in levels]), indent=1))


def assert_disjoint_by_hash(*groups):
    """groups: iterables of Level. Raises if any board appears in two groups."""
    seen = {}
    for gi, g in enumerate(groups):
        for lv in g:
            if seen.setdefault(lv.level_hash, gi) != gi:
                raise AssertionError(f"board {lv.level_id} appears in groups {seen[lv.level_hash]} and {gi}")
