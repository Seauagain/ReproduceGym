"""Step 3a: deterministically render a claim spec into a sandbox task.

claim spec -> tasks/<claim_id>/{data_entry.json, input_files/(task.md,
params.yaml, protocol.yaml, expected.json, paper.md, starter/), reward/(reward.sh,
targets)}. ClawGym-pure: NO private/ (verifier-only data lives under reward/).
Exposure routing decides visible vs hidden per leaf (see schema/task_contract.md).
reward/check.py is NOT written here; it is authored by the build-task skill. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def render_task(claim_spec_path: Path, task_dir: Path) -> Path:
    raise NotImplementedError("scaffold: claim spec -> rendered task files")
