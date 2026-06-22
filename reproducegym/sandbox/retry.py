"""Step 6: resume an interrupted reproduction.

If a run breaks mid-way (timeout, transient cluster failure), the host control
agent resumes the SAME agent session and continues to completion. Resume relies on
the backend's session id (captured into the prior RunResult / trajectory meta).
"""

from __future__ import annotations

from reproducegym.sandbox.launcher import Runtime
from reproducegym.sandbox.runner import RunResult, run

DEFAULT_FOLLOW_UP = (
    "Continue the reproduction task from where you left off. Re-check input_files/task.md, "
    "finish any remaining steps, and ensure every required output file under output/ exists."
)


def resume(
    runtime: Runtime,
    previous: RunResult,
    *,
    follow_up: str | None = None,
    timeout: float | None = None,
) -> RunResult:
    if not previous.session_id:
        raise ValueError("cannot resume: previous run has no session_id")
    return run(
        runtime,
        follow_up or DEFAULT_FOLLOW_UP,
        resume_session_id=previous.session_id,
        timeout=timeout,
    )
