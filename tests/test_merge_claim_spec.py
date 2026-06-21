"""M6 bridge: extracted claim -> canonical claim spec."""

from __future__ import annotations

import pytest

from reprogym.claim_spec import load_claim_spec, validate_claim_spec
from reprogym.pipeline.merge_claim_spec import MergeError, merge_claim_spec

CLAIM = {
    "claim_id": "c1_std_bias",
    "statement": "Removing std normalization removes the length bias.",
    "claim_type": "mechanism",
    "verifiability": "high",
    "requires_training": True,
    "cost": "M",
    "metrics": [{"name": "len_ratio", "formula": "mean(a)/mean(b)", "direction": "lower_is_better"}],
    "params": [{"name": "lr", "value": 1e-6, "source": "Sec 3", "status": "paper_specified"}],
    # extractor-only fields that are NOT in the schema must be dropped:
    "required_experiments": "run A vs B",
    "notes": "seed unspecified",
}


def test_merge_produces_valid_spec():
    spec = merge_claim_spec(CLAIM, paper_id="demo-1")
    validate_claim_spec(spec)  # must not raise
    assert spec["paper_id"] == "demo-1"
    assert spec["tier"] and spec["exposure_policy"]
    assert "required_experiments" not in spec and "notes" not in spec


def test_merge_carries_optional_fields():
    spec = merge_claim_spec(CLAIM, paper_id="demo-1")
    assert spec["requires_training"] is True and spec["cost"] == "M"
    assert spec["params"][0]["name"] == "lr"


def test_merge_metric_direction_default():
    claim = dict(CLAIM, metrics=[{"name": "acc"}])
    spec = merge_claim_spec(claim, paper_id="p")
    assert spec["metrics"][0]["direction"] == "higher_is_better"
    assert spec["metrics"][0]["formula"]  # filled placeholder


def test_merge_missing_required_raises():
    with pytest.raises(MergeError):
        merge_claim_spec({"statement": "x", "claim_type": "mechanism"}, paper_id="p")


def test_merge_folds_figure_params():
    spec = merge_claim_spec(
        CLAIM,
        figure_params={"group_size": {"value": 8, "source": "Fig 5"}},
        paper_id="p",
    )
    names = {p["name"] for p in spec["params"]}
    assert "group_size" in names and "lr" in names


def test_merge_overrides_and_out_path(tmp_path):
    out = tmp_path / "claims" / "c1.yaml"
    spec = merge_claim_spec(
        CLAIM,
        paper_id="p",
        tier="T1_strict",
        thresholds=[{"metric": "len_ratio", "pass_threshold": 0.8}],
        out_path=out,
    )
    assert spec["tier"] == "T1_strict"
    assert out.is_file()
    assert load_claim_spec(out)["claim_id"] == "c1_std_bias"
