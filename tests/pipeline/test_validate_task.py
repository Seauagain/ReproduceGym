"""M2: consistency gate over a rendered task."""

from __future__ import annotations

import json

import pytest

from reproducegym.pipeline.render_task import derive_contract, render_task
from reproducegym.pipeline.validate_task import _num_strings, _visible_text_for_leak_scan, validate_task


def _write_check_py(task_dir, contract_overrides=None):
    """Author a minimal check.py whose CONTRACT matches (or is tampered from) the spec."""
    base = {
        "claim_id": "c1_demo",
        "contract_hash": "cafebabe1234",
        "metrics": ["length_ratio"],
        "thresholds": {"length_ratio": 0.8},
        "required_files": ["output/result.json", "output/metrics.csv"],
        "verdicts": ["reproduced", "failed", "inconclusive", "invalid"],
    }
    if contract_overrides:
        base.update(contract_overrides)
    text = "CONTRACT = " + json.dumps(base) + "\n"
    (task_dir / "reward" / "check.py").write_text(text, encoding="utf-8")


@pytest.fixture
def task(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    return task_dir, valid_claim_spec


def test_rendered_task_only_missing_check_py(task):
    task_dir, spec = task
    problems = validate_task(task_dir, spec)
    assert len(problems) == 1
    assert "check.py" in problems[0]


def test_consistent_check_py_passes(task):
    task_dir, spec = task
    _write_check_py(task_dir)
    assert validate_task(task_dir, spec) == []


def test_check_py_threshold_drift_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir, {"thresholds": {"length_ratio": 0.9}})
    problems = validate_task(task_dir, spec)
    assert any("CONTRACT thresholds" in p for p in problems)


def test_check_py_contract_hash_drift_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir, {"contract_hash": "badbadbad"})
    problems = validate_task(task_dir, spec)
    assert any("contract_hash" in p for p in problems)


def test_unexecutable_metric_formula_detected(task):
    task_dir, spec = task
    spec = dict(spec)
    spec["metrics"] = [dict(spec["metrics"][0], formula="num_correct / num_total * 100 on AIME")]
    problems = validate_task(task_dir, spec)
    assert any("not executable by check.py" in p for p in problems)


def test_check_py_metric_drift_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir, {"metrics": ["wrong_metric"]})
    problems = validate_task(task_dir, spec)
    assert any("CONTRACT metrics" in p for p in problems)


def test_check_py_without_contract_detected(task):
    task_dir, spec = task
    (task_dir / "reward" / "check.py").write_text("print('hi')\n", encoding="utf-8")
    problems = validate_task(task_dir, spec)
    assert any("CONTRACT" in p for p in problems)


def test_expected_json_tamper_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir)
    exp_path = task_dir / "input_files" / "expected.json"
    exp = json.loads(exp_path.read_text())
    exp["primary_metrics"][0]["pass_threshold"] = 0.8  # leaking a hidden threshold
    exp_path.write_text(json.dumps(exp), encoding="utf-8")
    problems = validate_task(task_dir, spec)
    assert any("expected.json" in p for p in problems)


def test_exposure_leak_in_task_md_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir)
    md = task_dir / "input_files" / "task.md"
    md.write_text(md.read_text() + "\nNote: target is 0.8\n", encoding="utf-8")
    problems = validate_task(task_dir, spec)
    assert any("exposure leak" in p for p in problems)


def test_float_hidden_threshold_does_not_search_bare_integer():
    assert "5" not in _num_strings(5.0)
    assert "5.0" in _num_strings(5.0)


def test_leak_scan_ignores_renderer_protocol_version():
    text = _visible_text_for_leak_scan("protocol_version: 0.1\nthreshold hint: 0.1\n")
    assert "protocol_version" not in text
    assert "threshold hint: 0.1" in text


def test_missing_data_entry_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir)
    (task_dir / "data_entry.json").unlink()
    problems = validate_task(task_dir, spec)
    assert any("data_entry.json missing" in p for p in problems)


def test_protocol_required_files_drift_detected(task):
    task_dir, spec = task
    _write_check_py(task_dir)
    proto = task_dir / "input_files" / "protocol.yaml"
    text = proto.read_text().replace("output/result.json", "output/oops.json")
    proto.write_text(text, encoding="utf-8")
    problems = validate_task(task_dir, spec)
    assert any("agent_must_write" in p for p in problems)


def test_rlvr_task_without_threshold_is_rejected(tmp_path, valid_claim_spec):
    spec = dict(valid_claim_spec)
    spec["thresholds"] = []
    spec["verification"] = {"mode": "numeric_threshold", "pool": "rlvr"}
    task_dir = render_task(spec, tmp_path / "task")
    _write_check_py(task_dir, {"thresholds": {}})

    problems = validate_task(task_dir, spec)

    assert any("no executable threshold" in p for p in problems)


def test_rlvr_threshold_without_evidence_is_rejected(tmp_path, valid_claim_spec):
    spec = dict(valid_claim_spec)
    spec["thresholds"] = [
        {"metric": "length_ratio", "pass_threshold": 0.8, "exposure": "hidden"}
    ]
    spec["verification"] = {"mode": "numeric_threshold", "pool": "rlvr"}
    task_dir = render_task(spec, tmp_path / "task")
    _write_check_py(task_dir)

    problems = validate_task(task_dir, spec)

    assert any("missing target evidence source" in p for p in problems)
