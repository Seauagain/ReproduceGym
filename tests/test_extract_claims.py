"""M1: extract reproducible claims from paper markdown via an injected LLM."""

from __future__ import annotations

import json

import pytest

from reprogym.pipeline.extract_claims import (
    ExtractError,
    build_prompt,
    extract_claims,
    parse_claims_json,
)

GOOD_CLAIMS = [
    {
        "claim_id": "c1_std_bias",
        "statement": "Removing the std normalization removes the length bias.",
        "claim_type": "mechanism",
        "anchors": [{"kind": "figure", "ref": "Fig. 4"}],
        "metrics": [
            {"name": "len_ratio", "formula": "mean(a)/mean(b)", "direction": "lower_is_better"}
        ],
        "requires_training": True,
        "cost": "M",
        "verifiability": "high",
        "params": [{"name": "lr", "value": 1e-6, "source": "Sec 3", "status": "paper_specified"}],
        "notes": "seed unspecified",
    },
    {
        "claim_id": "c2_eval",
        "statement": "The model scores 43% on AIME.",
        "claim_type": "eval_only",
        "metrics": [{"name": "acc", "formula": "correct/total", "direction": "higher_is_better"}],
        "requires_training": False,
        "cost": "S",
        "verifiability": "high",
    },
]


def test_parse_plain_json_list():
    claims = parse_claims_json(json.dumps(GOOD_CLAIMS))
    assert [c["claim_id"] for c in claims] == ["c1_std_bias", "c2_eval"]


def test_parse_strips_code_fence():
    raw = "```json\n" + json.dumps(GOOD_CLAIMS) + "\n```"
    claims = parse_claims_json(raw)
    assert len(claims) == 2


def test_parse_unwraps_claims_object():
    raw = json.dumps({"claims": GOOD_CLAIMS})
    assert len(parse_claims_json(raw)) == 2


def test_parse_malformed_json_raises():
    with pytest.raises(ExtractError):
        parse_claims_json("not json at all {")


def test_parse_non_list_raises():
    with pytest.raises(ExtractError):
        parse_claims_json(json.dumps({"foo": "bar"}))


def test_parse_empty_list_raises():
    with pytest.raises(ExtractError):
        parse_claims_json("[]")


def test_parse_missing_required_key_raises():
    bad = [{"claim_id": "c1", "claim_type": "mechanism"}]  # no statement
    with pytest.raises(ExtractError) as exc:
        parse_claims_json(json.dumps(bad))
    assert "statement" in str(exc.value)


def test_parse_bad_claim_type_raises():
    bad = [{"claim_id": "c1", "statement": "x", "claim_type": "bogus"}]
    with pytest.raises(ExtractError):
        parse_claims_json(json.dumps(bad))


def test_parse_bad_claim_id_pattern_raises():
    bad = [{"claim_id": "C1 Bad", "statement": "x", "claim_type": "mechanism"}]
    with pytest.raises(ExtractError):
        parse_claims_json(json.dumps(bad))


def test_parse_duplicate_id_raises():
    dup = [
        {"claim_id": "c1", "statement": "a", "claim_type": "mechanism"},
        {"claim_id": "c1", "statement": "b", "claim_type": "eval_only"},
    ]
    with pytest.raises(ExtractError):
        parse_claims_json(json.dumps(dup))


def test_build_prompt_includes_paper_and_instructions():
    prompt = build_prompt("# My Paper\nbody text")
    assert "My Paper" in prompt
    assert "reproducible claims" in prompt.lower()


def test_extract_claims_with_text_input(make_llm):
    llm = make_llm(json.dumps(GOOD_CLAIMS))
    claims = extract_claims("# Paper\nsome body", client=llm)
    assert len(claims) == 2
    assert "some body" in llm.last_prompt


def test_extract_claims_with_path_input(make_llm, tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("# Paper From File\ncontent", encoding="utf-8")
    llm = make_llm(json.dumps(GOOD_CLAIMS))
    claims = extract_claims(md, client=llm)
    assert len(claims) == 2
    assert "Paper From File" in llm.last_prompt


def test_extract_claims_propagates_parse_error(make_llm):
    llm = make_llm("garbage")
    with pytest.raises(ExtractError):
        extract_claims("paper", client=llm)
