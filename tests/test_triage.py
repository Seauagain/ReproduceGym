"""R3: claim triage (Claude) + deterministic resource profile."""

from __future__ import annotations

import json

import pytest
import yaml

from reprogym.pipeline.triage import (
    TriageError,
    parse_triage_json,
    triage,
    write_resource_profile,
)

CLAIMS = [
    {"claim_id": "c1", "statement": "s1", "claim_type": "mechanism", "cost": "M",
     "requires_training": True, "verifiability": "high"},
    {"claim_id": "c2", "statement": "s2", "claim_type": "eval_only", "cost": "S",
     "requires_training": False, "verifiability": "high"},
    {"claim_id": "c3", "statement": "s3", "claim_type": "headline", "cost": "XL",
     "requires_training": True, "verifiability": "medium"},
]

TRIAGE_JSON = json.dumps(
    {
        "build": ["c1", "c2"],
        "defer": [{"claim_id": "c3", "reason": "too costly for v0"}],
        "v0": "c2",
        "rationale": "c2 is cheap and cleanly verifiable; c3 headline is XL.",
        "scores": {"c2": {"cost": "S", "verifiability": "high"}},
    }
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, prompt, **_):
        self.calls.append(prompt)
        return self.response


def test_parse_triage_ok():
    res = parse_triage_json(TRIAGE_JSON, ["c1", "c2", "c3"])
    assert res["build"] == ["c1", "c2"]
    assert res["v0"] == "c2"
    assert res["defer"][0] == {"claim_id": "c3", "reason": "too costly for v0"}


def test_parse_defer_string_form():
    raw = json.dumps({"build": ["c1"], "defer": ["c2"], "v0": "c1"})
    res = parse_triage_json(raw, ["c1", "c2"])
    assert res["defer"] == [{"claim_id": "c2", "reason": ""}]


def test_parse_unknown_build_id_raises():
    raw = json.dumps({"build": ["zzz"], "v0": "zzz"})
    with pytest.raises(TriageError):
        parse_triage_json(raw, ["c1"])


def test_parse_v0_not_in_build_raises():
    raw = json.dumps({"build": ["c1"], "v0": "c2"})
    with pytest.raises(TriageError):
        parse_triage_json(raw, ["c1", "c2"])


def test_triage_writes_yaml(tmp_path):
    llm = FakeLLM(TRIAGE_JSON)
    res = triage(CLAIMS, client=llm, out_dir=tmp_path)
    assert res["v0"] == "c2"
    assert "CLAIMS" in llm.calls[0]
    loaded = yaml.safe_load((tmp_path / "paper_triage.yaml").read_text())
    assert loaded["build"] == ["c1", "c2"]


def test_triage_empty_raises():
    with pytest.raises(TriageError):
        triage([], client=FakeLLM("{}"))


def test_resource_profile_deterministic(tmp_path):
    path = write_resource_profile(CLAIMS, tmp_path)
    profile = yaml.safe_load(path.read_text())
    assert profile["totals"]["n_claims"] == 3
    assert profile["totals"]["requires_training"] == 2
    assert profile["totals"]["by_cost"] == {"M": 1, "S": 1, "XL": 1}
    assert profile["claims"]["c1"]["requires_training"] is True


def test_resource_profile_defaults_unknown(tmp_path):
    claims = [{"claim_id": "c9", "statement": "x", "claim_type": "mechanism"}]
    profile = yaml.safe_load(write_resource_profile(claims, tmp_path).read_text())
    assert profile["claims"]["c9"]["cost"] == "unknown"
    assert profile["claims"]["c9"]["requires_training"] is False
