"""S2: Bohrium (lbg) compute provider -- access card + env/run-tag injection."""

from __future__ import annotations

import json

import pytest

from reprogym.compute.providers import LbgProvider, render_lbg_card
from reprogym.pipeline.render_task import render_task
from reprogym.sandbox.launcher import launch
from reprogym.sandbox.runner import run
from reprogym.sandbox.sandbox import Sandbox, SandboxResult
from tests.test_runner import STREAM, FakeBackend, RecordingSandbox


def test_from_spec_parses_params():
    p = LbgProvider.from_spec("project=4449832,gpu=5090,timeout=21600,template=verl-vllm")
    assert p.project_id == "4449832"
    assert p.gpu == "5090"
    assert p.timeout == 21600
    assert p.template == "verl-vllm"


def test_from_spec_defaults():
    p = LbgProvider.from_spec("")
    assert p.gpu == "4090"
    assert p.timeout == 43200
    assert p.template == ""


def test_card_encodes_iron_rules_and_run_tag():
    card = render_lbg_card(LbgProvider(project_id="4449832"), run_tag="rg-demo-001")
    # naming convention drives host teardown
    assert "rg-demo-001" in card
    assert "--name rg-demo-001" in card
    # billing + lifecycle guardrails
    assert "kill --force" in card
    assert "--timeout 43200" in card
    assert "--project-id 4449832" in card
    # credential is referenced by env name, never a literal value
    assert "BOHRIUM_ACCESS_KEY" in card


def test_env_injects_run_tag_and_ak(monkeypatch):
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", "bohr-ak-secret-123456")
    env = LbgProvider(project_id="4449832").env(run_tag="rg-demo-001")
    assert env["REPROGYM_RUN_TAG"] == "rg-demo-001"
    assert env["BOHRIUM_ACCESS_KEY"] == "bohr-ak-secret-123456"
    assert env["BOHRIUM_PROJECT_ID"] == "4449832"


@pytest.fixture
def task_dir(tmp_path, valid_claim_spec):
    return render_task(valid_claim_spec, tmp_path / "task")


def test_launch_installs_bohrium_card(tmp_path, task_dir):
    rt = launch(
        task_dir,
        tmp_path / "run",
        backend=FakeBackend(STREAM),
        compute="lbg:project=4449832",
    )
    assert rt.providers and rt.providers[0].name == "lbg"
    assert rt.run_tag
    assert (rt.workspace / "bohrium_access.md").is_file()
    assert "Bohrium" in (rt.workspace / "task.md").read_text()
    # ssh card is NOT installed when compute is lbg
    assert not (rt.workspace / "metax_ssh.py").exists()


def test_runner_forwards_ak_and_run_tag_then_redacts(tmp_path, task_dir, monkeypatch):
    ak = "bohr-ak-secret-7777aa"
    monkeypatch.setenv("BOHRIUM_ACCESS_KEY", ak)
    # an agent that echoes the AK back in a tool result (the leak we must scrub)
    leaky_stream = "\n".join(
        json.dumps(o)
        for o in [
            {"type": "system", "subtype": "init", "session_id": "s", "model": "m"},
            {
                "type": "user",
                "session_id": "s",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t",
                                          "content": f"logged in with {ak}"}]},
            },
            {"type": "result", "subtype": "success", "is_error": False,
             "result": f"ok {ak}", "session_id": "s"},
        ]
    )
    rec = RecordingSandbox(leaky_stream)
    rt = launch(
        task_dir,
        tmp_path / "run",
        backend=FakeBackend(leaky_stream),
        sandbox=rec,
        compute="lbg:project=4449832",
    )
    result = run(rt)
    # AK + run tag reach the sandbox env
    assert rec.env["BOHRIUM_ACCESS_KEY"] == ak
    assert rec.env["REPROGYM_RUN_TAG"] == rt.run_tag
    # ...but the persisted trajectory never contains the AK value
    assert ak not in result.trajectory_path.read_text()
    assert "\u00abREDACTED\u00bb" in result.trajectory_path.read_text()
