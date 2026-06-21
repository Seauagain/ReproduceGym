"""R6: training rollout adapter (dataset source + on-policy rollout)."""

from __future__ import annotations

import copy
import json

import pytest

from reprogym.pipeline.render_check import write_baseline_check
from reprogym.pipeline.render_task import render_task
from reprogym.sandbox.backends import AgentBackend
from reprogym.sandbox.sandbox import LocalSandbox
from train.rollout_adapter import as_rollout_source, rollout

AGENT_STREAM = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "system", "subtype": "init", "session_id": "sess-roll", "model": "m"},
        {"type": "result", "subtype": "success", "is_error": False, "result": "ok", "session_id": "sess-roll"},
    ]
)


class FakeAgentBackend(AgentBackend):
    name = "fake-agent"

    def build_command(self, prompt, *, session_id=None, resume=False):
        script = (
            "set -e\nmkdir -p output\n"
            'printf \'{"verdict":"reproduced"}\' > output/result.json\n'
            "printf 'a\\n' > output/metrics.csv\n"
            "cat <<'REPRO_EOF'\n" + AGENT_STREAM + "\nREPRO_EOF\n"
        )
        return ["bash", "-c", script]

    def build_env(self, base):
        return dict(base)


def _built_task(tmp_path, valid_claim_spec, claim_id="c1"):
    spec = copy.deepcopy(valid_claim_spec)
    spec["claim_id"] = claim_id
    task_dir = render_task(spec, tmp_path / "paper" / "tasks" / claim_id)
    write_baseline_check(spec, task_dir / "reward")
    return task_dir


def test_as_rollout_source_builds_dataset(tmp_path, valid_claim_spec):
    t1 = _built_task(tmp_path, valid_claim_spec, "c1")
    t2 = _built_task(tmp_path, valid_claim_spec, "c2")
    src = as_rollout_source("train1", [t1, t2], datasets_root=tmp_path / "datasets")
    assert sum(1 for _ in src.iterdir()) == 2
    for child in src.iterdir():
        assert (child / "data_entry.json").is_file()


def test_rollout_produces_reward_and_trajectory(tmp_path, valid_claim_spec):
    task_dir = _built_task(tmp_path, valid_claim_spec)
    out = rollout(
        task_dir,
        backend=FakeAgentBackend(),
        sandbox=LocalSandbox(),
        run_dir=tmp_path / "run",
    )
    assert out["reward"] == 0.8
    assert out["session_id"] == "sess-roll"
    assert out["trajectory_path"].is_file()
    assert out["returncode"] == 0


def test_rollout_no_score(tmp_path, valid_claim_spec):
    task_dir = _built_task(tmp_path, valid_claim_spec)
    out = rollout(
        task_dir,
        backend=FakeAgentBackend(),
        sandbox=LocalSandbox(),
        run_dir=tmp_path / "run",
        do_score=False,
    )
    assert out["reward"] is None
    assert out["trajectory_path"].is_file()
