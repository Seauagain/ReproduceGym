"""Step 4: launch a sandbox on the host.

Reuses the ClawGym sandbox backend (docker default, chroot/unshare when Docker
is restricted), copies the task's input_files/ into the agent workspace, and
injects the reproduction agent's API key from .env. The agent (Claude Code by
default) runs inside; the host keeps reward/ and secrets outside. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def launch(task_dir: Path) -> object:
    raise NotImplementedError("scaffold: start sandbox, mount input_files, inject key")
