"""Raw trajectory record. One Episode is one complete environment episode; LLM text is built elsewhere."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field

from .rtg import rtg_units

ENV_VERSION = "boxoban-rtg/0.1.0 (gym-sokoban 0.0.6 semantics; actions UP/DOWN/LEFT/RIGHT = gym 1-4)"
ACTION_NAMES = ("UP", "DOWN", "LEFT", "RIGHT")


def make_episode_id(policy_id: str, level_id: str, seed: int) -> str:
    return "ep-" + hashlib.sha1(f"{policy_id}|{level_id}|{seed}".encode()).hexdigest()[:16]


@dataclass
class Episode:
    episode_id: str
    level_id: str
    level_hash: str
    dataset_partition: str  # e.g. "unfiltered/train"
    policy_id: str  # exact generating source, e.g. "solver", "random", "drc33_final"
    policy_params: dict
    environment_version: str
    seed: int
    max_steps: int
    actions: list  # ints 0-3 (UP, DOWN, LEFT, RIGHT)
    reward_units: list  # executed rewards in units of 0.1
    moved_player: list
    moved_box: list
    players: list  # [[r, c]] length T+1 (state before each action, then final)
    boxes: list  # [[[r, c], ...]] length T+1, each sorted
    terminated: bool
    truncated: bool
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.actions)

    @property
    def solved(self) -> bool:
        return self.terminated

    @property
    def return_units(self) -> int:
        return sum(self.reward_units)

    @property
    def episode_return(self) -> float:
        return self.return_units / 10

    @property
    def rtg_units(self) -> list:
        return rtg_units(self.reward_units)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(length=self.length, solved=self.solved, episode_return_units=self.return_units,
                 episode_return=self.episode_return, rtg_units=self.rtg_units)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        names = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in names})
