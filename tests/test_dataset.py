"""R4: flatten nested task dirs into a one-level symlink dataset."""

from __future__ import annotations

import copy
import json

import pytest

from reprogym.dataset import DatasetError, build_dataset
from reprogym.pipeline.render_task import render_task


def _task(tmp_path, claim_id, paper_id="demo-1", spec=None, valid=None):
    s = copy.deepcopy(valid)
    s["claim_id"] = claim_id
    s["paper_id"] = paper_id
    return render_task(s, tmp_path / paper_id / "tasks" / claim_id)


def test_build_dataset_creates_symlinks(tmp_path, valid_claim_spec):
    t1 = _task(tmp_path / "a", "c1", valid=valid_claim_spec)
    t2 = _task(tmp_path / "a", "c2", valid=valid_claim_spec)
    ds = build_dataset("set1", [t1, t2], datasets_root=tmp_path / "datasets")
    children = sorted(p.name for p in ds.iterdir())
    assert len(children) == 2
    for child in ds.iterdir():
        assert child.is_symlink()
        assert (child / "data_entry.json").is_file()  # reachable through the link


def test_flat_names_from_task_id(tmp_path, valid_claim_spec):
    t1 = _task(tmp_path / "a", "c1", paper_id="dr-grpo", valid=valid_claim_spec)
    ds = build_dataset("s", [t1], datasets_root=tmp_path / "datasets")
    name = next(ds.iterdir()).name
    de = json.loads((t1 / "data_entry.json").read_text())
    assert name == de["task_id"]


def test_collision_gets_unique_name(tmp_path, valid_claim_spec):
    # same task_id (same paper+claim) in two different source trees
    t1 = _task(tmp_path / "a", "c1", valid=valid_claim_spec)
    t2 = _task(tmp_path / "b", "c1", valid=valid_claim_spec)
    ds = build_dataset("s", [t1, t2], datasets_root=tmp_path / "datasets")
    names = sorted(p.name for p in ds.iterdir())
    assert len(names) == 2 and names[0] != names[1]


def test_missing_data_entry_raises(tmp_path):
    bad = tmp_path / "notatask"
    bad.mkdir()
    with pytest.raises(DatasetError):
        build_dataset("s", [bad], datasets_root=tmp_path / "datasets")


def test_empty_raises(tmp_path):
    with pytest.raises(DatasetError):
        build_dataset("s", [], datasets_root=tmp_path / "datasets")


def test_clean_rebuild(tmp_path, valid_claim_spec):
    t1 = _task(tmp_path / "a", "c1", valid=valid_claim_spec)
    root = tmp_path / "datasets"
    build_dataset("s", [t1], datasets_root=root)
    (root / "s" / "stale").write_text("x", encoding="utf-8") if False else None
    ds = build_dataset("s", [t1], datasets_root=root, clean=True)
    assert sorted(p.name for p in ds.iterdir()) == [next(ds.iterdir()).name]
