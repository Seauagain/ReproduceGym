"""Step 5: run the reproduction agent and record the trajectory.

Issues the task's user_query (from data_entry.json) to the in-sandbox agent. The
agent works locally and ssh's to verl/MetaX nodes for GPU when needed -- ops are
plain shell actions captured into the trajectory, not wrapped in submit/poll.
Noise distillation is offline, later. Stub only.
"""

from __future__ import annotations


def run(runtime: object, user_query: str) -> object:
    raise NotImplementedError("scaffold: drive agent, capture trajectory")
