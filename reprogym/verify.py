"""Step 7: hidden verifier scoring.

Copies the task's reward/ into the finished workspace, runs reward/reward.sh
with the workspace path, and parses the last stdout line as the scalar reward.
The reproduction agent never sees reward/. No LLM here. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def score(task_dir: Path, workspace_dir: Path) -> float:
    raise NotImplementedError("scaffold: run reward/reward.sh, parse float")
