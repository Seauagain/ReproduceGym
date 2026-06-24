from __future__ import annotations

import copy

from reproducegym.pipeline.synthesize_contract import apply_verification_contract


def test_hidden_target_param_becomes_threshold(valid_claim_spec):
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["params"].append(
        {
            "name": "length_ratio_target",
            "value": 0.8,
            "source": "Fig. 4",
            "status": "paper_specified",
            "use": "target",
            "metric": "length_ratio",
            "exposure": "hidden",
            "confidence": 0.7,
        }
    )

    out = apply_verification_contract(spec)

    assert out["verification"]["mode"] == "numeric_threshold"
    assert out["verification"]["pool"] == "rlvr"
    assert out["thresholds"][0]["metric"] == "length_ratio"
    # Figure-read targets get conservative tolerance; lower_is_better relaxes upward.
    assert out["thresholds"][0]["pass_threshold"] == 1.04
    assert out["thresholds"][0]["exposure"] == "hidden"
    assert out["thresholds"][0]["target_evidence"]["source"] == "Fig. 4"
    assert "reproduced" in out["verdict_rules"]


def test_directional_metric_gets_neutral_threshold(valid_claim_spec):
    # length_ratio = mean(treatment.len) / mean(baseline.len) compares two
    # conditions, so we still expose a diagnostic no-effect point of 1.0. Without
    # an absolute paper target it is not a strong RLVR task.
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["params"] = []

    out = apply_verification_contract(spec)

    assert out["verification"]["mode"] == "unverifiable"
    assert out["verification"]["pool"] == "exploration"
    assert out["thresholds"][0]["metric"] == "length_ratio"
    assert out["thresholds"][0]["pass_threshold"] == 1.0
    assert "target_value" not in out["thresholds"][0]
    # the neutral point is structural (implied by the public claim), so it is a
    # visible pass criterion, not a hidden answer-key number.
    assert out["thresholds"][0]["exposure"] == "visible"


def test_directional_difference_neutral_point_is_zero(valid_claim_spec):
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["params"] = []
    spec["metrics"] = [
        {
            "name": "len_gap",
            "formula": "mean(baseline.len) - mean(treatment.len)",
            "direction": "higher_is_better",
        }
    ]

    out = apply_verification_contract(spec)

    assert out["verification"]["mode"] == "unverifiable"
    assert out["verification"]["pool"] == "exploration"
    assert out["thresholds"][0]["pass_threshold"] == 0.0


def test_noncomparative_metric_without_target_routes_to_exploration(valid_claim_spec):
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["params"] = []
    spec["metrics"] = [
        {"name": "accuracy", "formula": "mean(acc)", "direction": "higher_is_better"}
    ]

    out = apply_verification_contract(spec)

    assert out["verification"]["mode"] == "unverifiable"
    assert out["verification"]["pool"] == "exploration"
    assert "accuracy" in out["verification"]["reason"]


def test_relative_target_rejected_for_absolute_metric(valid_claim_spec):
    # A ratio-flavoured target must not be force-bound to an absolute-magnitude
    # metric (the bug that produced semantically-wrong thresholds).
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["metrics"] = [
        {"name": "response_length", "formula": "mean(len)", "direction": "lower_is_better"}
    ]
    spec["params"] = [
        {
            "name": "response_length_ratio",
            "value": 0.85,
            "source": "Fig. 8",
            "status": "paper_specified",
            "use": "target",
            "metric": "response_length",
            "exposure": "hidden",
        }
    ]

    out = apply_verification_contract(spec)

    assert out["verification"]["pool"] == "exploration"
    assert "mismatch" in out["verification"]["reason"]
    assert out["thresholds"] == []


def test_ambiguous_targets_not_bound(valid_claim_spec):
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"] = []
    spec["verdict_rules"] = {}
    spec["metrics"] = [
        {"name": "accuracy", "formula": "mean(acc)", "direction": "higher_is_better"}
    ]
    spec["params"] = [
        {"name": "accuracy_peak", "value": 25, "use": "target", "source": "Fig. 8"},
        {"name": "accuracy_final", "value": 30, "use": "target", "source": "Fig. 8"},
    ]

    out = apply_verification_contract(spec)

    assert out["verification"]["pool"] == "exploration"
    assert "ambiguous" in out["verification"]["reason"]
    assert out["thresholds"] == []
