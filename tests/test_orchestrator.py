"""M6: end-to-end orchestration (MD -> claims -> task -> run -> reward)."""

from __future__ import annotations

import json

import pytest

from reprogym.orchestrator import ReproduceError, reproduce
from reprogym.sandbox.backends import AgentBackend
from reprogym.sandbox.sandbox import LocalSandbox

PAPER_MD = """# A Critical Perspective on R1-Zero Training

We show that removing std normalization removes the length bias.
"""

CLAIMS_JSON = json.dumps(
    [
        {
            "claim_id": "c1_std_bias",
            "statement": "Removing std normalization removes the length bias.",
            "claim_type": "mechanism",
            "verifiability": "high",
            "requires_training": True,
            "cost": "M",
            "metrics": [
                {"name": "len_ratio", "formula": "mean(a)/mean(b)", "direction": "lower_is_better"}
            ],
        }
    ]
)

AGENT_STREAM = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "system", "subtype": "init", "session_id": "sess-orch", "model": "m"},
        {
            "type": "assistant",
            "session_id": "sess-orch",
            "message": {"content": [{"type": "text", "text": "Running the comparison."}]},
        },
        {"type": "result", "subtype": "success", "is_error": False, "result": "done", "session_id": "sess-orch"},
    ]
)


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete(self, prompt, **_):
        return self.response


class FakeAgentBackend(AgentBackend):
    """Writes the required outputs (verdict=reproduced) then emits a stream."""

    name = "fake-agent"

    def build_command(self, prompt, *, session_id=None, resume=False):
        script = (
            "set -e\n"
            "mkdir -p output\n"
            'printf \'{"claim_id":"c1_std_bias","verdict":"reproduced","strict_reproduction":false}\' '
            "> output/result.json\n"
            "printf 'a\\n' > output/metrics.csv\n"
            "cat <<'REPRO_EOF'\n" + AGENT_STREAM + "\nREPRO_EOF\n"
        )
        return ["bash", "-c", script]

    def build_env(self, base):
        return dict(base)


def test_reproduce_end_to_end(tmp_path):
    result = reproduce(
        PAPER_MD,
        client=FakeLLM(CLAIMS_JSON),
        backend=FakeAgentBackend(),
        sandbox=LocalSandbox(),
        paper_id="dr-grpo-demo",
        work_dir=tmp_path / "build",
        run_dir=tmp_path / "run",
    )
    assert result.claim_id == "c1_std_bias"
    assert result.paper_id == "dr-grpo-demo"
    assert result.validation == []
    assert result.reward == 0.8
    # artifacts exist
    assert result.claim_spec_path.is_file()
    assert (result.task_dir / "data_entry.json").is_file()
    assert (result.task_dir / "reward" / "check.py").is_file()
    # trajectory recorded
    assert result.trajectory_path.is_file()
    assert result.run_result.session_id == "sess-orch"
    assert len(result.run_result.trajectory.of_type("assistant_text")) == 1


def test_reproduce_unknown_claim_id_raises(tmp_path):
    with pytest.raises(ReproduceError):
        reproduce(
            PAPER_MD,
            claim_id="does_not_exist",
            client=FakeLLM(CLAIMS_JSON),
            backend=FakeAgentBackend(),
            sandbox=LocalSandbox(),
            work_dir=tmp_path / "build",
            run_dir=tmp_path / "run",
        )


def test_reproduce_paper_from_file(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text(PAPER_MD, encoding="utf-8")
    result = reproduce(
        md,
        client=FakeLLM(CLAIMS_JSON),
        backend=FakeAgentBackend(),
        sandbox=LocalSandbox(),
        work_dir=tmp_path / "build",
        run_dir=tmp_path / "run",
    )
    assert result.paper_id == "paper"  # derived from filename stem
    assert result.reward == 0.8
