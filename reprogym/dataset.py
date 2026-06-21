"""Flatten authored sandboxes/ into a rollout-ready dataset.

ClawGym's _discover_task_entries scans only ONE level and requires every child
to be a task dir with data_entry.json. The authoring layout is nested
(sandboxes/<paper>/tasks/<claim>/), so build a flat datasets/<name>/ of symlinks
pointing at the selected task dirs, and hand that path to the rollout as
source_path. datasets/ is a build artifact (gitignored). Stub only.
"""

from __future__ import annotations

from pathlib import Path


def build_dataset(name: str, task_dirs: list[Path]) -> Path:
    raise NotImplementedError("scaffold: symlink task dirs into datasets/<name>/")
