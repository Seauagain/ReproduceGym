"""M6: baseline verifier generation + integration with render/validate/score."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reprogym.pipeline.render_check import render_baseline_check_py, write_baseline_check
from reprogym.pipeline.render_task import render_task
from reprogym.pipeline.validate_task import validate_task
from reprogym.verify import score


def test_generated_check_is_valid_python(valid_claim_spec):
    text = render_baseline_check_py(valid_claim_spec)
    compile(text, "check.py", "exec")  # must parse
    assert "CONTRACT" in text


def test_render_then_baseline_check_passes_validation(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    write_baseline_check(valid_claim_spec, task_dir / "reward")
    assert validate_task(task_dir, valid_claim_spec) == []


def _run_check(check_path, workspace):
    proc = subprocess.run(
        [sys.executable, str(check_path), str(workspace), "--reward-only"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return float(proc.stdout.strip().splitlines()[-1])


def test_baseline_reward_for_verdicts(tmp_path, valid_claim_spec):
    check_path = write_baseline_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "metrics.csv").write_text("x\n", encoding="utf-8")

    (ws / "output" / "result.json").write_text('{"verdict": "reproduced"}', encoding="utf-8")
    assert _run_check(check_path, ws) == 0.8

    (ws / "output" / "result.json").write_text('{"verdict": "failed"}', encoding="utf-8")
    assert _run_check(check_path, ws) == 0.35


def test_baseline_strict_bonus(tmp_path, valid_claim_spec):
    check_path = write_baseline_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "metrics.csv").write_text("x\n", encoding="utf-8")
    (ws / "output" / "result.json").write_text(
        '{"verdict": "reproduced", "strict_reproduction": true}', encoding="utf-8"
    )
    assert _run_check(check_path, ws) == 0.9


def test_baseline_missing_files_is_invalid(tmp_path, valid_claim_spec):
    check_path = write_baseline_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _run_check(check_path, ws) == 0.0


def test_baseline_full_loop_via_score(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    write_baseline_check(valid_claim_spec, task_dir / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "metrics.csv").write_text("x\n", encoding="utf-8")
    (ws / "output" / "result.json").write_text('{"verdict": "reproduced"}', encoding="utf-8")
    assert score(task_dir, ws) == 0.8
