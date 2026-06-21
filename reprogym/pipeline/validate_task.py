"""Step 3b: consistency gate over a rendered task.

Rejects the task unless metric names, threshold values, required output files /
metrics.csv columns, and the verdict label set AGREE across task.md,
params.yaml, protocol.yaml, expected.json and reward/check.py. Also enforces the
ClawGym contract (data_entry.json + metadata, input_files/ present, reward/
reward.sh present) and the exposure rule (no hidden value leaks into
input_files/). Stub only.
"""

from __future__ import annotations

from pathlib import Path


def validate_task(task_dir: Path, claim_spec_path: Path) -> list[str]:
    raise NotImplementedError("scaffold: return list of inconsistencies (empty = ok)")
