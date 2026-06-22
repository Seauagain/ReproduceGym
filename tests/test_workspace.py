"""M3: workspace preparation (input_files/ in, reward/ out)."""

from __future__ import annotations

import pytest

from reproducegym.pipeline.render_task import render_task
from reproducegym.sandbox.workspace import prepare_workspace


@pytest.fixture
def task_dir(tmp_path, valid_claim_spec):
    return render_task(valid_claim_spec, tmp_path / "task")


def test_input_files_copied_to_workspace_root(tmp_path, task_dir):
    ws = prepare_workspace(task_dir, tmp_path / "ws")
    assert (ws / "task.md").is_file()
    assert (ws / "params.yaml").is_file()
    assert (ws / "expected.json").is_file()


def test_output_dir_created(tmp_path, task_dir):
    ws = prepare_workspace(task_dir, tmp_path / "ws")
    assert (ws / "output").is_dir()


def test_reward_and_secrets_not_copied(tmp_path, task_dir):
    ws = prepare_workspace(task_dir, tmp_path / "ws")
    assert not (ws / "reward").exists()
    assert not (ws / "targets.yaml").exists()
    assert not (ws / "data_entry.json").exists()


def test_missing_input_files_raises(tmp_path):
    empty = tmp_path / "bad"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        prepare_workspace(empty, tmp_path / "ws")


def test_clean_rerun(tmp_path, task_dir):
    ws = prepare_workspace(task_dir, tmp_path / "ws")
    (ws / "stale.txt").write_text("x", encoding="utf-8")
    ws = prepare_workspace(task_dir, tmp_path / "ws", clean=True)
    assert not (ws / "stale.txt").exists()
    assert (ws / "task.md").is_file()
