"""Generated reward/check.py: recompute verifier (no agent self-report trust)."""

from __future__ import annotations

import subprocess
import sys

import pytest

from reprogym.pipeline.render_check import render_check_py, write_check
from reprogym.pipeline.render_task import render_task
from reprogym.pipeline.validate_task import validate_task
from reprogym.verify import score


def _write_metrics_csv(ws, *, baseline_len, treatment_len, n=50):
    (ws / "output").mkdir(parents=True, exist_ok=True)
    lines = ["condition,step,len"]
    for i in range(n):
        lines.append(f"baseline,{i},{baseline_len}")
    for i in range(n):
        lines.append(f"treatment,{i},{treatment_len}")
    (ws / "output" / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_generated_check_is_valid_python(valid_claim_spec):
    text = render_check_py(valid_claim_spec)
    compile(text, "check.py", "exec")
    assert "CONTRACT" in text and "SPEC" in text
    assert "def recompute(" in text  # engine embedded verbatim
    # the verifier derives the verdict; it must not read one back out of result.json
    assert 'result.get("verdict")' not in text
    assert 'result["verdict"]' not in text


def test_render_then_check_passes_validation(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    write_check(valid_claim_spec, task_dir / "reward")
    assert validate_task(task_dir, valid_claim_spec) == []


def _run_check(check_path, workspace):
    proc = subprocess.run(
        [sys.executable, str(check_path), str(workspace), "--reward-only"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return float(proc.stdout.strip().splitlines()[-1])


def test_check_reward_reproduced_and_failed(tmp_path, valid_claim_spec):
    check_path = write_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "result.json").write_text("{}", encoding="utf-8")

    _write_metrics_csv(ws, baseline_len=100, treatment_len=70)  # ratio 0.7 <= 0.8
    assert _run_check(check_path, ws) == 0.8

    _write_metrics_csv(ws, baseline_len=100, treatment_len=95)  # ratio 0.95 > 0.8
    assert _run_check(check_path, ws) == 0.35


def test_check_ignores_self_reported_verdict(tmp_path, valid_claim_spec):
    check_path = write_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    # Agent claims success and a perfect reward, but the data misses the threshold.
    (ws / "output" / "result.json").write_text(
        '{"verdict": "reproduced", "reward": 1.0, "strict_reproduction": true}', encoding="utf-8"
    )
    _write_metrics_csv(ws, baseline_len=100, treatment_len=95)
    assert _run_check(check_path, ws) == 0.35  # recomputed -> failed, self-report ignored


def test_check_missing_files_is_invalid(tmp_path, valid_claim_spec):
    check_path = write_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    ws.mkdir()
    assert _run_check(check_path, ws) == 0.0


def test_full_loop_via_score(tmp_path, valid_claim_spec):
    task_dir = render_task(valid_claim_spec, tmp_path / "task")
    write_check(valid_claim_spec, task_dir / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "result.json").write_text("{}", encoding="utf-8")
    _write_metrics_csv(ws, baseline_len=100, treatment_len=70)
    assert score(task_dir, ws) == 0.8


def test_writes_verification_report(tmp_path, valid_claim_spec):
    check_path = write_check(valid_claim_spec, tmp_path / "reward")
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "result.json").write_text("{}", encoding="utf-8")
    _write_metrics_csv(ws, baseline_len=100, treatment_len=70)
    _run_check(check_path, ws)
    import json

    report = json.loads((ws / "output" / "verification_report.json").read_text())
    assert report["verdict"] == "reproduced"
    assert report["scored_by"] == "reprogym-recompute"
