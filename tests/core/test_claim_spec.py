"""M1 foundation: claim-spec schema validation."""

from __future__ import annotations

import json

import pytest

from reproducegym.claim_spec import (
    ClaimSpecError,
    dump_claim_spec,
    enum_values,
    iter_validation_errors,
    load_claim_spec,
    validate_claim_spec,
)


def test_valid_spec_passes(valid_claim_spec):
    assert iter_validation_errors(valid_claim_spec) == []
    validate_claim_spec(valid_claim_spec)  # must not raise


def test_schema_accepts_diagnostic_metrics_and_sanitizer_provenance(valid_claim_spec):
    valid_claim_spec["conditions"][0]["source_label"] = "Oat-Zero-7B"
    valid_claim_spec["metrics"][0]["source_name"] = "4shot_avg"
    valid_claim_spec["diagnostic_metrics"] = [
        {
            "name": "ungrounded_gap",
            "formula": "mean(treatment.len) - mean(baseline.len)",
            "direction": "higher_is_better",
            "source_name": "Ungrounded Gap",
        }
    ]
    valid_claim_spec["verification"] = {
        "mode": "numeric_threshold",
        "pool": "rlvr",
        "targets_bound": ["length_ratio"],
        "diagnostic_reason": "ungrounded metric(s) moved to diagnostics: ungrounded_gap",
    }

    validate_claim_spec(valid_claim_spec)


def test_missing_required_field_raises(valid_claim_spec):
    del valid_claim_spec["metrics"]
    with pytest.raises(ClaimSpecError) as exc:
        validate_claim_spec(valid_claim_spec)
    assert "metrics" in str(exc.value)


def test_bad_claim_type_enum_raises(valid_claim_spec):
    valid_claim_spec["claim_type"] = "not_a_type"
    with pytest.raises(ClaimSpecError):
        validate_claim_spec(valid_claim_spec)


def test_unknown_top_level_key_rejected(valid_claim_spec):
    valid_claim_spec["surprise"] = 1
    errors = iter_validation_errors(valid_claim_spec)
    assert errors, "additionalProperties:false should reject unknown keys"


def test_bad_claim_id_pattern_rejected(valid_claim_spec):
    valid_claim_spec["claim_id"] = "Has Spaces"
    assert iter_validation_errors(valid_claim_spec)


def test_enum_values_reads_schema():
    types = enum_values("claim_type")
    assert "mechanism" in types and "headline" in types
    with pytest.raises(KeyError):
        enum_values("statement")  # not an enum field


def test_load_claim_spec_yaml_roundtrip(tmp_path, valid_claim_spec):
    path = tmp_path / "c1_demo.yaml"
    dump_claim_spec(valid_claim_spec, path)
    loaded = load_claim_spec(path)
    assert loaded["claim_id"] == "c1_demo"
    assert loaded["metrics"][0]["name"] == "length_ratio"


def test_load_claim_spec_json(tmp_path, valid_claim_spec):
    path = tmp_path / "c1_demo.json"
    path.write_text(json.dumps(valid_claim_spec), encoding="utf-8")
    loaded = load_claim_spec(path)  # yaml.safe_load parses JSON too
    assert loaded["paper_id"] == "demo-0001"


def test_load_invalid_spec_raises(tmp_path, valid_claim_spec):
    del valid_claim_spec["statement"]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(valid_claim_spec), encoding="utf-8")
    with pytest.raises(ClaimSpecError):
        load_claim_spec(path)
