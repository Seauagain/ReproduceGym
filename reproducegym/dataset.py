"""Flatten rendered task dirs into a rollout-ready dataset.

The ClawGym rollout discovers tasks ONE level deep and requires every child to be
a task dir with data_entry.json. The pipeline layout is nested
(runs/<paper>/03-task/<claim>/<hash>/), so we build a flat datasets/<name>/ of symlinks
pointing at the selected task dirs and hand that path to the rollout as
source_path. datasets/ is a build artifact (gitignored, auto-created).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from reproducegym.config import REPO_ROOT

DATASETS_ROOT = REPO_ROOT / "datasets"


class DatasetError(ValueError):
    pass


def _flat_name(task_dir: Path) -> str:
    de = task_dir / "data_entry.json"
    if not de.is_file():
        raise DatasetError(f"not a task dir (no data_entry.json): {task_dir}")
    try:
        task_id = json.loads(de.read_text(encoding="utf-8")).get("task_id")
    except json.JSONDecodeError as exc:
        raise DatasetError(f"invalid data_entry.json in {task_dir}: {exc}") from exc
    raw = task_id or task_dir.name
    return re.sub(r"-{2,}", "-", re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw))).strip("-")


def build_dataset(
    name: str,
    task_dirs: list[str | Path],
    *,
    datasets_root: str | Path | None = None,
    clean: bool = False,
) -> Path:
    """Symlink each task dir into datasets/<name>/ with a unique flat name."""
    if not task_dirs:
        raise DatasetError("no task dirs given")
    root = Path(datasets_root) if datasets_root is not None else DATASETS_ROOT
    target = root / name
    if clean and target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    used: set[str] = set()
    for td in task_dirs:
        td = Path(td).resolve()
        base = _flat_name(td)
        flat = base
        i = 2
        while flat in used or (target / flat).exists():
            flat = f"{base}-{i}"
            i += 1
        used.add(flat)
        (target / flat).symlink_to(td, target_is_directory=True)
    return target
