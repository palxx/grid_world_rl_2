import os
from pathlib import Path

import pytest

from boxoban_rtg.environments.state import parse_level

CANDIDATES = [os.environ.get("BOXOBAN_LEVELS_DIR"), "data/raw/boxoban-levels",
              "/home/user/google-deepmind/boxoban-levels"]


@pytest.fixture(scope="session")
def levels_root():
    for c in CANDIDATES:
        if c and Path(c, "unfiltered", "test", "000.txt").exists():
            return Path(c)
    pytest.skip("official boxoban-levels not available (set BOXOBAN_LEVELS_DIR)")


def lv(*rows, level_id="toy"):
    return parse_level(rows, level_id)


@pytest.fixture
def one_box():
    return lv("#####", "#@$.#", "#####")
