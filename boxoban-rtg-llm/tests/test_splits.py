import pytest

from boxoban_rtg.environments.splits import assert_disjoint_by_hash, load_levels, sample_level_ids, write_split
from conftest import lv


def test_assert_disjoint_detects_shared_board():
    a = lv("#####", "#@$.#", "#####", level_id="a")
    b = lv("#####", "#@$.#", "#####", level_id="b")  # same board, different id
    c = lv("######", "#@ $.#", "######", level_id="c")
    with pytest.raises(AssertionError):
        assert_disjoint_by_hash([a], [b])
    assert_disjoint_by_hash([a], [c])


@pytest.mark.levels
def test_sampling_is_deterministic_and_respects_exclusion(levels_root):
    a = sample_level_ids("unfiltered", "train", 20, seed=1, root=levels_root)
    assert a == sample_level_ids("unfiltered", "train", 20, seed=1, root=levels_root)
    assert a != sample_level_ids("unfiltered", "train", 20, seed=2, root=levels_root)
    b = sample_level_ids("unfiltered", "train", 20, seed=1, exclude_ids=a[:5], root=levels_root)
    assert not set(a[:5]) & set(b) and len(set(b)) == 20


@pytest.mark.levels
def test_partition_samples_are_hash_disjoint(levels_root, tmp_path):
    tr = load_levels(sample_level_ids("unfiltered", "train", 30, 0, root=levels_root), levels_root)
    va = load_levels(sample_level_ids("unfiltered", "valid", 30, 0, root=levels_root), levels_root)
    te = load_levels(sample_level_ids("unfiltered", "test", 30, 0, root=levels_root), levels_root)
    assert_disjoint_by_hash(tr, va, te)
    write_split(tmp_path / "s.json", "t", [l.level_id for l in tr], tr, {"seed": 0})
    assert (tmp_path / "s.json").exists()
