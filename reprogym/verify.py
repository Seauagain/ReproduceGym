"""Step 7: hidden verifier scoring.

Runs the task's `reward/reward.sh <workspace>` and parses the LAST stdout line as
the scalar reward (the ClawGym reward contract). The reproduction agent never sees
reward/ -- it is invoked here, host-side, against the finished workspace. No LLM.

`reward.sh` resolves its own SCRIPT_DIR, so check.py and reward/targets.yaml are
read from the task's reward/ in place; we do not need to copy them next to the
agent's outputs.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Mapping


class ScoreError(RuntimeError):
    """Raised when scoring cannot produce a numeric reward."""


def parse_reward(stdout: str) -> float:
    """Return the last non-empty stdout line parsed as a float."""
    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    if not lines:
        raise ScoreError("reward.sh produced no output")
    last = lines[-1]
    try:
        return float(last)
    except ValueError as exc:
        raise ScoreError(f"last stdout line is not a float: {last!r}") from exc


def score(
    task_dir: str | Path,
    workspace_dir: str | Path,
    *,
    timeout: float = 1800.0,
    env: Mapping[str, str] | None = None,
    clamp: bool = True,
) -> float:
    """Score a finished workspace against a task's hidden reward/reward.sh."""
    reward_sh = Path(task_dir) / "reward" / "reward.sh"
    if not reward_sh.is_file():
        raise ScoreError(f"reward/reward.sh not found under {task_dir}")
    workspace_dir = Path(workspace_dir)
    if not workspace_dir.is_dir():
        raise ScoreError(f"workspace dir does not exist: {workspace_dir}")

    run_env = dict(os.environ if env is None else env)
    try:
        proc = subprocess.run(
            ["bash", str(reward_sh), str(workspace_dir)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(workspace_dir),
            env=run_env,
        )
    except subprocess.TimeoutExpired as exc:
        raise ScoreError(f"reward.sh timed out after {timeout}s") from exc

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        raise ScoreError(f"reward.sh exited {proc.returncode}: {tail}")

    reward = parse_reward(proc.stdout)
    if clamp:
        reward = max(0.0, min(1.0, reward))
    return reward
