"""Training rollout mode: feed ReproGym tasks to the ClawGym RL rollout.

Builds a flat datasets/<name>/ (via reprogym.dataset) from selected sandbox
tasks and points ClawGym-Agents/RL/clawgym_rl_rollout.py at it as source_path.
Same task dir, zero changes -> rollouts produce trajectories + reward for policy
updates. Stub only.
"""

from __future__ import annotations

from pathlib import Path


def as_rollout_source(name: str, task_dirs: list[Path]) -> Path:
    raise NotImplementedError("scaffold: build dataset, return rollout source_path")
