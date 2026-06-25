from __future__ import annotations

import copy

from reproducegym.pipeline.rlvr_task_contract import (
    assign_final_claim_ids,
    build_claim_verification_report,
    compute_claim_uid,
    compute_contract_hash,
    select_claims_for_build,
)


def _refined_claim(**overrides):
    claim = {
        "statement": "Method A improves pass@1 over Method B.",
        "claim_type": "headline",
        "importance_rank": 1,
        "source_mode": "global",
        "anchors": [{"kind": "figure", "ref": "Fig. 2"}],
        "evidence_anchors": [{"kind": "figure", "ref": "Fig. 2"}],
        "reproduction_protocol": {
            "summary": "Run both methods on the same evaluation set and write metrics.csv."
        },
        "verification_contract": {
            "type": "numeric_threshold",
            "conditions": [
                {"label": "method_a", "description": "Method A"},
                {"label": "method_b", "description": "Method B"},
            ],
            "metrics": [
                {
                    "name": "pass1_gap",
                    "formula": "mean(method_a.pass1) - mean(method_b.pass1)",
                    "direction": "higher_is_better",
                }
            ],
            "params": [],
            "thresholds": [
                {
                    "metric": "pass1_gap",
                    "pass_threshold": 0.05,
                    "target_value": 0.1,
                    "tolerance_abs": 0.05,
                    "source": "Table 1",
                    "target_evidence": {"source": "Table 1"},
                }
            ],
            "verdict_rules": {},
        },
        "likely_pool": "rlvr",
    }
    claim.update(overrides)
    return claim


def test_claim_uid_is_deterministic_and_order_independent():
    a = {
        "statement": "A improves B.",
        "evidence_anchors": [
            {"kind": "figure", "ref": "Fig. 2"},
            {"kind": "section", "ref": "Sec. 4"},
        ],
    }
    b = {
        "statement": "  A improves   B. ",
        "evidence_anchors": [
            {"kind": "section", "ref": "Sec. 4"},
            {"kind": "figure", "ref": "Fig. 2"},
        ],
    }

    assert compute_claim_uid(a) == compute_claim_uid(b)


def test_contract_hash_ignores_protocol_prose_but_tracks_thresholds():
    claim = _refined_claim()
    changed_protocol = copy.deepcopy(claim)
    changed_protocol["reproduction_protocol"]["summary"] = "Same verifier, clearer prose."
    changed_contract = copy.deepcopy(claim)
    changed_contract["verification_contract"]["thresholds"] = [
        {"metric": "pass1_gap", "pass_threshold": 0.1, "exposure": "visible"}
    ]

    assert compute_contract_hash(claim) == compute_contract_hash(changed_protocol)
    assert compute_contract_hash(claim) != compute_contract_hash(changed_contract)


def test_verification_report_covers_every_refined_claim_and_hashes_contract():
    refined = [
        _refined_claim(),
        _refined_claim(
            statement="Qualitative claim.",
            verification_contract={
                "type": "artifact_metric",
                "conditions": [],
                "metrics": [],
                "params": [],
                "thresholds": [],
                "verdict_rules": {},
            },
        ),
    ]

    report = build_claim_verification_report(refined)

    assert len(report) == 2
    assert all(item["claim_uid"] for item in report)
    assert all(item["contract_hash"] for item in report)
    assert report[0]["verification"]["pool"] == "rlvr"
    assert report[1]["verification"]["pool"] == "exploration"
    assert "reason" in report[1]["verification"]


def test_verification_report_marks_unsupported_formula_unbuildable():
    refined = [
        _refined_claim(
            statement="Natural language formula.",
            verification_contract={
                "type": "numeric_threshold",
                "conditions": [],
                "metrics": [
                    {
                        "name": "pass1",
                        "formula": "num_correct / num_total * 100 on AIME 2024",
                        "direction": "higher_is_better",
                    }
                ],
                "params": [],
                "thresholds": [{"metric": "pass1", "pass_threshold": 70.0}],
                "verdict_rules": {},
            },
        )
    ]

    report = build_claim_verification_report(refined)
    selected = select_claims_for_build(refined, report, max_claims=1)

    assert report[0]["buildable"] is False
    assert report[0]["verification"]["pool"] == "exploration"
    assert selected == []


def test_select_claims_for_build_prefers_rlvr_then_exploration():
    rlvr = _refined_claim(statement="RLVR claim", importance_rank=2)
    exploration = _refined_claim(
        statement="Exploration claim",
        importance_rank=1,
        verification_contract={
            "type": "artifact_metric",
            "conditions": [],
            "metrics": [],
            "params": [],
            "thresholds": [],
            "verdict_rules": {},
        },
    )
    report = build_claim_verification_report([exploration, rlvr])

    selected = select_claims_for_build([exploration, rlvr], report, max_claims=2)

    assert [item["statement"] for item in selected] == ["RLVR claim"]
    assert selected[0]["verification"]["pool"] == "rlvr"
    assert selected[0]["contract_hash"]


def test_directional_only_claim_is_not_selected_for_rlvr():
    claim = _refined_claim(
        verification_contract={
            "type": "directional_comparison",
            "conditions": [
                {"label": "method_a", "description": "Method A"},
                {"label": "method_b", "description": "Method B"},
            ],
            "metrics": [
                {
                    "name": "pass1_gap",
                    "formula": "mean(method_a.pass1) - mean(method_b.pass1)",
                    "direction": "higher_is_better",
                }
            ],
            "params": [],
            "thresholds": [],
            "verdict_rules": {},
        }
    )

    report = build_claim_verification_report([claim])
    selected = select_claims_for_build([claim], report, max_claims=1)

    assert report[0]["verification"]["pool"] == "exploration"
    assert "target_value" in report[0]["verification"]["reason"]
    assert selected == []


def test_direction_target_conflict_is_not_selected_for_rlvr():
    claim = _refined_claim(
        statement="Baseline hallucination rate should match a reported value.",
        verification_contract={
            "type": "numeric_threshold",
            "conditions": [{"label": "baseline", "description": "baseline"}],
            "metrics": [
                {
                    "name": "hallucination_rate",
                    "formula": "mean(baseline.hallucinated)",
                    "direction": "lower_is_better",
                }
            ],
            "params": [],
            "thresholds": [
                {
                    "metric": "hallucination_rate",
                    "pass_threshold": 0.3,
                    "target_value": 0.41,
                    "tolerance_abs": 0.11,
                    "source": "paper text",
                    "target_evidence": {"source": "paper text"},
                }
            ],
            "verdict_rules": {},
        },
    )

    report = build_claim_verification_report([claim])

    assert report[0]["verification"]["pool"] == "exploration"
    assert "reward curve" in report[0]["verification"]["reason"]


def test_near_zero_delta_claim_uses_synthesized_lower_direction():
    claim = _refined_claim(
        statement="ArXiv papers produce no notable improvement.",
        verification_contract={
            "type": "numeric_threshold",
            "conditions": [{"label": "arxiv", "description": "arxiv data"}],
            "metrics": [
                {
                    "name": "max_delta_across_benchmarks",
                    "formula": "max(delta)",
                    "direction": "higher_is_better",
                }
            ],
            "params": [],
            "thresholds": [
                {
                    "metric": "max_delta_across_benchmarks",
                    "pass_threshold": 1.0,
                    "target_value": 0.0,
                    "tolerance_abs": 1.0,
                    "rationale": "No notable improvement: maximum delta should be <= 1 percentage point.",
                    "source": "Table 8",
                    "target_evidence": {"source": "Table 8"},
                }
            ],
            "verdict_rules": {},
        },
    )

    report = build_claim_verification_report([claim])

    assert report[0]["verification"]["pool"] == "rlvr"
    curve = report[0]["reward_curves"]["max_delta_across_benchmarks"]
    assert curve["direction"] == "lower_is_better"
    assert curve["points"][-1] == {"value": 0.0, "reward": 1.0}


def test_final_claim_id_assignment_preserves_claim_uid_and_contract_hash():
    refined = [_refined_claim(statement="Z claim")]
    report = build_claim_verification_report(refined)
    selected = select_claims_for_build(refined, report, max_claims=1)

    assigned = assign_final_claim_ids(selected)

    assert assigned[0]["claim_id"].startswith("c001_")
    assert assigned[0]["claim_uid"] == selected[0]["claim_uid"]
    assert assigned[0]["contract_hash"] == selected[0]["contract_hash"]
