"""M1: extract reproducible claims from paper markdown via an injected LLM."""

from __future__ import annotations

import json

import pytest

from reproducegym.pipeline.extract_claims import (
    ExtractError,
    build_prompt,
    compact_figure_inventory,
    dedup_claim_candidates,
    extract_claims,
    extract_claim_candidates,
    finalize_claims,
    parse_claims_json,
    refine_claim_with_figure_evidence,
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


class SeqLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[str] = []

    def complete(self, prompt: str, **_):
        self.calls.append(prompt)
        if len(self.responses) == 1:
            return self.responses[0]
        return self.responses.pop(0)


def test_parse_plain_json_list():
    claims = parse_claims_json(json.dumps(GOOD_CLAIMS))
    assert [c["claim_id"] for c in claims] == ["c001_c1_std_bias", "c002_c2_eval"]
    assert [c["claim_num"] for c in claims] == [1, 2]


def test_parse_strips_code_fence():
    raw = "```json\n" + json.dumps(GOOD_CLAIMS) + "\n```"
    claims = parse_claims_json(raw)
    assert len(claims) == 2


def test_parse_extracts_json_from_wrapped_text():
    raw = "Here are the claims:\n\n" + json.dumps(GOOD_CLAIMS) + "\n\nDone."
    claims = parse_claims_json(raw)
    assert len(claims) == 2


def test_parse_extracts_fenced_json_from_wrapped_text():
    raw = "Sure.\n\n```json\n" + json.dumps({"claims": GOOD_CLAIMS}) + "\n```\n"
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


def test_parse_duplicate_id_gets_stable_suffix():
    dup = [
        {"claim_id": "c1", "statement": "a", "claim_type": "mechanism"},
        {"claim_id": "c1", "statement": "b", "claim_type": "eval_only"},
    ]
    claims = parse_claims_json(json.dumps(dup))
    assert [c["claim_id"] for c in claims] == ["c001_c1", "c002_c1_2"]


def test_build_prompt_includes_paper_and_instructions():
    prompt = build_prompt("# My Paper\nbody text", figure_inventory="- Fig. 1: fig1.png")
    assert "My Paper" in prompt
    assert "reproducible claims" in prompt.lower()
    assert "FIGURE INVENTORY" in prompt


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


def test_compact_figure_inventory_omits_nearby_context():
    inv = compact_figure_inventory([
        {
            "figure_ref": "Fig. 1",
            "caption": "Figure 1: useful caption",
            "context": "HUGE_NEARBY_CONTEXT_SHOULD_NOT_APPEAR",
        }
    ])
    assert "Figure 1: useful caption" in inv
    assert "HUGE_NEARBY_CONTEXT" not in inv


def test_extract_claim_candidates_chunks_and_uses_compact_inventory():
    claim = json.dumps([
        {
            "statement": "A claim from this chunk.",
            "claim_type": "mechanism",
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
        }
    ])
    llm = SeqLLM([claim])
    paper = "# A\n" + ("a " * 120) + "\n\n# B\n" + ("b " * 120)
    out = extract_claim_candidates(
        paper,
        client=llm,
        figures=[
            {
                "figure_ref": "Fig. 1",
                "caption": "Figure 1: compact only",
                "context": "FULL_CONTEXT_NOT_ALLOWED",
            }
        ],
        max_chunk_chars=160,
    )
    assert len(out) >= 2
    assert len(llm.calls) >= 2
    assert all("FULL_CONTEXT_NOT_ALLOWED" not in prompt for prompt in llm.calls)
    assert all("PAPER CHUNK" in prompt for prompt in llm.calls)


def test_dedup_claim_candidates_local_merge_without_client():
    dup = [
        {
            "statement": "Same claim.",
            "claim_type": "mechanism",
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
            "metrics": [{"name": "a", "formula": "a", "direction": "higher_is_better"}],
        },
        {
            "statement": "Same claim.",
            "claim_type": "mechanism",
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
            "params": [{"name": "steps", "value": 100, "status": "paper_specified"}],
        },
    ]
    out = dedup_claim_candidates(dup)
    assert len(out) == 1
    assert out[0]["claim_id"].startswith("c001_")
    assert out[0]["metrics"][0]["name"] == "a"
    assert out[0]["params"][0]["name"] == "steps"


def test_refine_claim_with_figure_evidence_preserves_claim_id():
    claim = {
        "claim_id": "c001_demo",
        "statement": "Original claim.",
        "claim_type": "mechanism",
        "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
    }
    refined_raw = json.dumps(
        {
            "statement": "Original claim with figure-derived metric.",
            "claim_type": "mechanism",
            "anchors": [{"kind": "figure", "ref": "Fig. 1"}],
            "metrics": [{"name": "m", "formula": "m", "direction": "higher_is_better"}],
        }
    )
    out = refine_claim_with_figure_evidence(
        claim,
        [{"figure_ref": "Fig. 1", "params": []}],
        client=SeqLLM([refined_raw]),
    )
    assert out["claim_id"] == "c001_demo"
    assert out["metrics"][0]["name"] == "m"


def test_finalize_claims_orders_by_importance():
    out = finalize_claims(
        [
            {"statement": "Less important", "claim_type": "eval_only", "importance_rank": 2},
            {"statement": "Most important", "claim_type": "mechanism", "importance_rank": 1},
        ]
    )
    assert [c["claim_id"] for c in out] == ["c001_most_important", "c002_less_important"]
