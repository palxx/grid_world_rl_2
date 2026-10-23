"""Admissible lower bound on remaining pushes (and hence on remaining steps)."""
from __future__ import annotations


def box_push_lower_bound(boxes, push_dist: dict) -> int | None:
    """Sum over boxes of the single-box push distance to the nearest goal; None if some box is dead."""
    total = 0
    for b in boxes:
        d = push_dist.get(b)
        if d is None:
            return None
        total += d
    return total
