"""M5: hidden verifier scoring (reward.sh -> float)."""

from __future__ import annotations

import os
import stat

import pytest

from reprogym.verify import ScoreError, parse_reward, score


def _make_task(tmp_path, reward_sh_body: str):
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    rsh = task_dir / "reward" / "reward.sh"
    rsh.write_text(reward_sh_body, encoding="utf-8")
    rsh.chmod(rsh.stat().st_mode | stat.S_IEXEC)
    ws = tmp_path / "workspace"
    ws.mkdir()
    return task_dir, ws


def test_parse_reward_picks_last_line():
    assert parse_reward("log line\nanother\n0.73\n") == 0.73


def test_parse_reward_no_output_raises():
    with pytest.raises(ScoreError):
        parse_reward("   \n\n")


def test_parse_reward_non_float_raises():
    with pytest.raises(ScoreError):
        parse_reward("diagnostics\nnot a number")


def test_score_reads_last_float(tmp_path):
    task_dir, ws = _make_task(tmp_path, "#!/usr/bin/env bash\necho debug\necho 0.5\n")
    assert score(task_dir, ws) == 0.5


def test_score_passes_workspace_path(tmp_path):
    body = '#!/usr/bin/env bash\necho "ws=$1" > "$1/seen.txt"\necho 1.0\n'
    task_dir, ws = _make_task(tmp_path, body)
    score(task_dir, ws)
    assert (ws / "seen.txt").read_text().strip() == f"ws={ws}"


def test_score_clamps_to_unit_interval(tmp_path):
    task_dir, ws = _make_task(tmp_path, "#!/usr/bin/env bash\necho 5.0\n")
    assert score(task_dir, ws) == 1.0
    task_dir2, ws2 = _make_task(tmp_path / "neg", "#!/usr/bin/env bash\necho -2\n")
    assert score(task_dir2, ws2) == 0.0


def test_score_no_clamp(tmp_path):
    task_dir, ws = _make_task(tmp_path, "#!/usr/bin/env bash\necho 2.5\n")
    assert score(task_dir, ws, clamp=False) == 2.5


def test_score_nonzero_exit_raises(tmp_path):
    task_dir, ws = _make_task(tmp_path, "#!/usr/bin/env bash\necho boom >&2\nexit 3\n")
    with pytest.raises(ScoreError) as exc:
        score(task_dir, ws)
    assert "exited 3" in str(exc.value)


def test_score_missing_reward_sh_raises(tmp_path):
    (tmp_path / "task").mkdir()
    (tmp_path / "workspace").mkdir()
    with pytest.raises(ScoreError):
        score(tmp_path / "task", tmp_path / "workspace")


def test_score_missing_workspace_raises(tmp_path):
    task_dir, _ = _make_task(tmp_path, "#!/usr/bin/env bash\necho 1\n")
    with pytest.raises(ScoreError):
        score(task_dir, tmp_path / "nope")


def test_score_via_check_py_integration(tmp_path):
    """reward.sh -> check.py recomputes a metric from the workspace output."""
    task_dir = tmp_path / "task"
    (task_dir / "reward").mkdir(parents=True)
    (task_dir / "reward" / "reward.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nWS="${1:-.}"\n'
        'DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'python3 "$DIR/check.py" "$WS" --reward-only\n',
        encoding="utf-8",
    )
    (task_dir / "reward" / "check.py").write_text(
        "import json, sys\n"
        "ws = sys.argv[1]\n"
        "data = json.load(open(ws + '/output/result.json'))\n"
        "print('recomputed', data['value'])\n"
        "print(1.0 if data['value'] >= 0.8 else 0.3)\n",
        encoding="utf-8",
    )
    ws = tmp_path / "workspace"
    (ws / "output").mkdir(parents=True)
    (ws / "output" / "result.json").write_text('{"value": 0.9}', encoding="utf-8")
    assert score(task_dir, ws) == 1.0
