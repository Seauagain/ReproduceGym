"""M2: render a claim spec into a ClawGym-pure sandbox task."""

from __future__ import annotations

import copy
import json

import pytest
import yaml

from reproducegym.pipeline.render_task import (
    INPUT_MOUNT_DIR,
    derive_contract,
    render_task,
    task_id_for,
)


@pytest.fixture
def rendered(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    return task_dir, valid_claim_spec


def test_clawgym_pure_layout(rendered):
    task_dir, _ = rendered
    assert (task_dir / "data_entry.json").is_file()
    assert (task_dir / "input_files" / "task.md").is_file()
    assert (task_dir / "input_files" / "params.yaml").is_file()
    assert (task_dir / "input_files" / "protocol.yaml").is_file()
    assert (task_dir / "input_files" / "expected.json").is_file()
    assert (task_dir / "reward" / "reward.sh").is_file()
    assert (task_dir / "reward" / "targets.yaml").is_file()
    # ClawGym-pure: NO private/, and check.py is NOT auto-rendered.
    assert not (task_dir / "private").exists()
    assert not (task_dir / "reward" / "check.py").exists()


def test_data_entry_contract(rendered):
    task_dir, spec = rendered
    de = json.loads((task_dir / "data_entry.json").read_text())
    assert de["task_id"] == task_id_for(spec)
    assert de["input_mount_dir"] == INPUT_MOUNT_DIR
    assert de["metadata"]["claim_id"] == "c1_demo"
    assert de["metadata"]["spec_hash"] == "deadbeef1234"
    assert de["metadata"]["paper_id"] == "demo-0001"
    assert de["metadata"]["private_targets_hidden"] is True  # threshold is hidden


def test_reward_sh_is_executable_and_calls_check(rendered):
    task_dir, _ = rendered
    rsh = task_dir / "reward" / "reward.sh"
    assert "check.py" in rsh.read_text()
    assert rsh.stat().st_mode & 0o111  # some execute bit set


def test_hidden_threshold_only_in_reward(rendered):
    task_dir, _ = rendered
    targets = yaml.safe_load((task_dir / "reward" / "targets.yaml").read_text())
    assert targets["primary_thresholds"]["length_ratio"]["pass_threshold"] == 0.8
    # expected.json must NOT carry the hidden number
    exp = json.loads((task_dir / "input_files" / "expected.json").read_text())
    assert exp["thresholds_hidden"] is True
    for m in exp["primary_metrics"]:
        assert "pass_threshold" not in m
    # the literal value must not leak into any visible file
    for p in (task_dir / "input_files").rglob("*"):
        if p.is_file():
            assert "0.8" not in p.read_text(), f"hidden threshold leaked into {p.name}"


def test_visible_threshold_is_exposed(tmp_path, valid_claim_spec):
    spec = copy.deepcopy(valid_claim_spec)
    spec["thresholds"][0]["exposure"] = "visible"
    task_dir = render_task(spec, tmp_path / "task")
    exp = json.loads((task_dir / "input_files" / "expected.json").read_text())
    assert exp["primary_metrics"][0]["pass_threshold"] == 0.8
    assert "0.8" in (task_dir / "input_files" / "task.md").read_text()


def test_params_grouped_by_status(rendered):
    task_dir, _ = rendered
    params = yaml.safe_load((task_dir / "input_files" / "params.yaml").read_text())
    assert params["claim_id"] == "c1_demo"
    assert params["spec_hash"] == "deadbeef1234"
    assert params["paper_specified"]["learning_rate"]["value"] == 1e-6


def test_protocol_metrics_match_spec(rendered):
    task_dir, spec = rendered
    proto = yaml.safe_load((task_dir / "input_files" / "protocol.yaml").read_text())
    c = derive_contract(spec)
    assert set(proto["metric_computation"].keys()) == set(c["metric_names"])
    assert proto["workspace_contract"]["agent_must_write"] == c["required_files"]
    assert proto["spec_hash"] == "deadbeef1234"


def test_task_id_is_slugified():
    spec = {"paper_id": "dr-grpo-2503.20783", "claim_id": "c1_std_bias"}
    assert task_id_for(spec) == "dr-grpo-2503-20783-c1-std-bias"


def test_render_accepts_yaml_path(tmp_path, valid_claim_spec):
    from reproducegym.claim_spec import dump_claim_spec

    spec_path = tmp_path / "spec.yaml"
    dump_claim_spec(valid_claim_spec, spec_path)
    task_dir = render_task(spec_path, tmp_path / "task")
    assert (task_dir / "data_entry.json").is_file()
