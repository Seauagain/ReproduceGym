"""Prepare the agent workspace from a task's input_files/.

ClawGym mounts input_files/ AS the agent workspace (data_entry.input_mount_dir).
We mirror that on the host: copy input_files/* to the workspace root and create
output/. reward/ is deliberately NOT copied -- the verifier runs out-of-band in
reproducegym.verify and the agent must never see the hidden targets.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def prepare_workspace(
    task_dir: str | Path,
    workspace_dir: str | Path,
    *,
    clean: bool = False,
) -> Path:
    task_dir = Path(task_dir)
    input_dir = task_dir / "input_files"
    if not input_dir.is_dir():
        raise FileNotFoundError(f"task has no input_files/: {task_dir}")

    ws = Path(workspace_dir)
    if clean and ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True, exist_ok=True)

    for item in input_dir.iterdir():
        dst = ws / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    (ws / "output").mkdir(exist_ok=True)
    return ws
