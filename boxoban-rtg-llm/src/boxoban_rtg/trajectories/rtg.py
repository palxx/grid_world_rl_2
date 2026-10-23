"""Return-to-go helpers. All arithmetic is in integer reward units (0.1), so sums are exact.

RTG(t) = sum of rewards from step t through T-1; the terminal RTG (after the last step) is 0.
Closed loop: remaining(t+1) = remaining(t) - reward(t), with rewards taken from the executed environment.
"""
from __future__ import annotations


def rtg_units(reward_units) -> list[int]:
    out, acc = [], 0
    for r in reversed(list(reward_units)):
        acc += r
        out.append(acc)
    return out[::-1]


def update_remaining(remaining_units: int, executed_reward_units: int) -> int:
    return remaining_units - executed_reward_units


def units_to_float(u: int) -> float:
    return u / 10


def float_to_units(x: float) -> int:
    """Requested returns are given in reward units of 0.1; refuse values that are not on that grid."""
    u = round(x * 10)
    if abs(u - x * 10) > 1e-6:
        raise ValueError(f"{x} is not a multiple of 0.1")
    return int(u)
