from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import run as run_entry


def _task(root, claim_id, spec_hash):
    task = root / "runs" / "paper" / "03-task" / claim_id / spec_hash
    task.mkdir(parents=True)
    (task / "data_entry.json").write_text(
        json.dumps({"metadata": {"claim_id": claim_id, "spec_hash": spec_hash}}),
        encoding="utf-8",
    )
    return task


def test_resolve_task_requires_hash_for_ambiguous_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(run_entry, "REPO", tmp_path)
    _task(tmp_path, "c001_demo", "aaaaaaaaaaaa")
    wanted = _task(tmp_path, "c001_demo", "bbbbbbbbbbbb")

    with pytest.raises(SystemExit):
        run_entry.resolve_task("c001_demo")

    assert run_entry.resolve_task("c001_demo", "bbbbbbbbbbbb") == wanted


def test_resolve_existing_task_exact_hash_does_not_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(run_entry, "REPO", tmp_path)
    _task(tmp_path, "c001_demo", "aaaaaaaaaaaa")

    with pytest.raises(SystemExit):
        run_entry.resolve_existing_task(
            task_dir=None,
            claim_id="c001_demo",
            spec_hash="bbbbbbbbbbbb",
        )


def test_run_py_rejects_paper_build(monkeypatch):
    monkeypatch.setattr(run_entry, "force_env_provider", lambda: None)
    with pytest.raises(SystemExit, match="no longer builds from paper"):
        run_entry.main(["--claim_id", "c001_demo", "--server", "node", "--paper", "paper.md"])


def test_record_run_token_usage_from_captures(tmp_path):
    task = _task(tmp_path, "c001_demo", "aaaaaaaaaaaa")
    run_dir = tmp_path / "runs" / "paper" / "04-run" / "c001_demo" / "aaaaaaaaaaaa" / "001"
    comp = SimpleNamespace(
        api_type="anthropic",
        completion_id="msg_1",
        request={"model": "claude"},
        response={"usage": {"input_tokens": 10, "output_tokens": 4}},
    )

    run_entry._record_run_token_usage(
        task=task,
        run_dir=run_dir,
        session="c001_demo",
        capture_enabled=True,
        completions=[comp],
    )

    summary = json.loads((tmp_path / "runs" / "paper" / "token_usage.summary.json").read_text())
    assert summary["totals"]["usage_records"] == 1
    assert summary["totals"]["input_tokens"] == 10
    assert summary["totals"]["output_tokens"] == 4
