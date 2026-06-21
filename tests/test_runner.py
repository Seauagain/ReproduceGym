"""M3: launch + run + resume, end-to-end with a fake agent."""

from __future__ import annotations

import json

import pytest

from reprogym.pipeline.render_task import render_task
from reprogym.sandbox.backends import AgentBackend
from reprogym.sandbox.launcher import launch
from reprogym.sandbox.retry import resume
from reprogym.sandbox.runner import run
from reprogym.sandbox.sandbox import LocalSandbox, Sandbox, SandboxResult

STREAM = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "system", "subtype": "init", "session_id": "sess-fake", "model": "m"},
        {
            "type": "assistant",
            "session_id": "sess-fake",
            "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]},
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "done",
            "session_id": "sess-fake",
        },
    ]
)


class FakeBackend(AgentBackend):
    name = "fake"

    def __init__(self, stream: str):
        self.stream = stream
        self.calls: list[dict] = []

    def build_command(self, prompt, *, session_id=None, resume=False):
        self.calls.append({"prompt": prompt, "session_id": session_id, "resume": resume})
        script = "cat <<'REPRO_STREAM_EOF'\n" + self.stream + "\nREPRO_STREAM_EOF\n"
        return ["bash", "-c", script]

    def build_env(self, base):
        return dict(base)


class RecordingSandbox(Sandbox):
    name = "recording"

    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.argv = None
        self.cwd = None
        self.env = None

    def run(self, argv, *, cwd, env=None, timeout=None):
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(env or {})
        return SandboxResult(self.returncode, self.stdout, "")


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
    assert "REPROGYM_METAX_NODES" in rec.env
    forwarded = json.loads(rec.env["REPROGYM_METAX_NODES"])
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
    assert "Compute access" in (rt.workspace / "task.md").read_text()


def test_launch_no_compute_access_without_nodes(tmp_path, task_dir, monkeypatch):
    monkeypatch.delenv("REPROGYM_METAX_NODES", raising=False)
    monkeypatch.setenv("REPROGYM_METAX_CONFIG", str(tmp_path / "absent.yaml"))
    rt = launch(task_dir, tmp_path / "run", backend=FakeBackend(STREAM), sandbox=LocalSandbox())
    assert not (rt.workspace / "metax_ssh.py").exists()
    assert not (rt.workspace / "compute_access.md").exists()
