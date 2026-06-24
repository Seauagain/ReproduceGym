from __future__ import annotations

import json
from types import SimpleNamespace

from reproducegym.pipeline.token_usage import (
    RecordingLLMClient,
    RecordingVLClient,
    TokenUsageRecorder,
    normalize_usage,
)


class UsageLLM:
    provider = "anthropic"
    model = "claude-test"

    def __init__(self):
        self.last_usage = None

    def complete(self, prompt, **_):
        self.last_usage = SimpleNamespace(input_tokens=11, output_tokens=7)
        return '[{"statement":"x","claim_type":"mechanism"}]'


class UsageVL:
    provider = "openai-compatible"
    model = "qwen-vl-test"

    def __init__(self):
        self.last_usage = None

    def read_figure(self, image_path, prompt, **_):
        self.last_usage = SimpleNamespace(prompt_tokens=13, completion_tokens=5, total_tokens=18)
        return json.dumps({"params": [{"name": "steps", "value": 150}]})


class NoUsageLLM:
    model = "fake"

    def complete(self, prompt, **_):
        return "[]"


def test_normalize_usage_supports_anthropic_and_openai_shapes():
    assert normalize_usage(SimpleNamespace(input_tokens=3, output_tokens=4)) == {
        "input_tokens": 3,
        "output_tokens": 4,
        "total_tokens": 7,
    }
    assert normalize_usage({"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11}) == {
        "input_tokens": 5,
        "output_tokens": 6,
        "total_tokens": 11,
    }


def test_recording_wrappers_capture_real_usage(tmp_path):
    recorder = TokenUsageRecorder(tmp_path, paper_id="p")
    llm = RecordingLLMClient(UsageLLM(), recorder, step="extract")
    vl = RecordingVLClient(UsageVL(), recorder, step="figure", metadata={"claim_id": "c1"})

    llm.complete("prompt")
    vl.read_figure(tmp_path / "fig.png", "look")
    summary_path = recorder.write_summary()

    summary = json.loads(summary_path.read_text())
    assert summary["totals"]["usage_records"] == 2
    assert summary["totals"]["input_tokens"] == 24
    assert summary["totals"]["output_tokens"] == 12
    assert summary["totals"]["total_tokens"] == 36


def test_no_usage_client_is_marked_unavailable(tmp_path):
    recorder = TokenUsageRecorder(tmp_path, paper_id="p")
    RecordingLLMClient(NoUsageLLM(), recorder, step="extract").complete("prompt")
    rec = recorder.records()[0]
    assert rec["usage_available"] is False
    assert rec["prompt_chars"] == len("prompt")
    assert rec["input_tokens"] is None
