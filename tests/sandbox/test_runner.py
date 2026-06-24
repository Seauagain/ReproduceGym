"""M3: launch + run + resume, end-to-end with a fake agent."""

from __future__ import annotations

import json

import pytest

from reproducegym.pipeline.render_task import render_task
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.retry import resume
from reproducegym.sandbox.runner import run
from reproducegym.sandbox.sandbox import LocalSandbox
from tests.helpers import STREAM, FakeBackend, RecordingSandbox


@pytest.fixture
def task_dir(tmp_path, valid_claim_spec):
    return render_task(valid_claim_spec, tmp_path / "task")


def test_launch_prepares_runtime(tmp_path, task_dir):
    rt = launch(task_dir, tmp_path / "run", backend=FakeBackend(STREAM), sandbox=LocalSandbox())
    assert (rt.workspace / "task.md").is_file()
    assert not (rt.workspace / "reward").exists()
    assert rt.user_query  # taken from data_entry.json
    assert rt.metadata["claim_id"] == "c1_demo"


def test_run_records_trajectory(tmp_path, task_dir):
    rt = launch(task_dir, tmp_path / "run", backend=FakeBackend(STREAM), sandbox=LocalSandbox())
    result = run(rt)
    assert result.returncode == 0
    assert result.session_id == "sess-fake"  # lifted from the stream
    assert result.trajectory_path.is_file()
    assert len(result.trajectory.of_type("tool_use")) == 1
    # trajectory file is valid jsonl
    lines = result.trajectory_path.read_text().strip().splitlines()
    assert all(json.loads(ln) for ln in lines)


def test_resume_uses_prev_session_and_new_file(tmp_path, task_dir):
    backend = FakeBackend(STREAM)
    rt = launch(task_dir, tmp_path / "run", backend=backend, sandbox=LocalSandbox())
    first = run(rt)
    second = resume(rt, first)
    assert backend.calls[1]["resume"] is True
    assert backend.calls[1]["session_id"] == "sess-fake"
    assert second.trajectory_path != first.trajectory_path
    assert second.trajectory_path.is_file()


def test_metax_inventory_forwarded_into_sandbox_env(tmp_path, task_dir):
    rec = RecordingSandbox(STREAM)
    rt = launch(
        task_dir,
        tmp_path / "run",
        backend=FakeBackend(STREAM),
        sandbox=rec,
        metax_nodes={"verl": {"host": "10.0.0.9", "user": "root"}},
    )
    run(rt)
    assert "REPRODUCEGYM_METAX_NODES" in rec.env
    forwarded = json.loads(rec.env["REPRODUCEGYM_METAX_NODES"])
    assert forwarded["verl"]["host"] == "10.0.0.9"
    assert str(rec.cwd) == str(rt.workspace)
    assert rec.argv[0] == "bash"


def test_launch_installs_compute_access_when_nodes_given(tmp_path, task_dir):
    rt = launch(
        task_dir,
        tmp_path / "run",
        backend=FakeBackend(STREAM),
        sandbox=LocalSandbox(),
        metax_nodes={"verl": {"host": "10.0.0.9", "user": "root"}},
    )
    assert (rt.workspace / "metax_nodes.json").is_file()
    assert (rt.workspace / "metax_ssh.py").is_file()
    assert (rt.workspace / "compute_access.md").is_file()
    # task.md references the compute doc rather than inlining it.
    task_md = (rt.workspace / "task.md").read_text()
    assert "compute_access.md" in task_md
    assert "Compute access" in (rt.workspace / "compute_access.md").read_text()


def test_launch_no_compute_access_without_nodes(tmp_path, task_dir, monkeypatch):
    monkeypatch.delenv("REPRODUCEGYM_METAX_NODES", raising=False)
    monkeypatch.setenv("REPRODUCEGYM_METAX_CONFIG", str(tmp_path / "absent.yaml"))
    rt = launch(task_dir, tmp_path / "run", backend=FakeBackend(STREAM), sandbox=LocalSandbox())
    assert not (rt.workspace / "metax_ssh.py").exists()
    assert not (rt.workspace / "compute_access.md").exists()
