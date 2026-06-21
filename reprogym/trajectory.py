"""Trajectory recording, shared by both modes.

Captures the reproduction agent's messages / tool calls / observations into a
serializable form (e.g. runs/<run_id>/trajectory.jsonl) for interactive runs,
and is reused as the rollout trajectory format for RL training. Stub only.
"""

from __future__ import annotations


class Trajectory:
    def append(self, event: dict) -> None:
        raise NotImplementedError("scaffold")

    def dump(self, path) -> None:
        raise NotImplementedError("scaffold")
