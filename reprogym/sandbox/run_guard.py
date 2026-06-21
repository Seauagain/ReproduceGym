"""Host-side run guard: always reclaim provisioned compute after a run.

Bohrium sandboxes the agent provisions are billed by the hour and live outside
ReproGym's own sandbox, so the host MUST reclaim them even when the run errors or
times out. ``run_guarded`` wraps :func:`reprogym.sandbox.runner.run` in a
try/finally that sweeps every provider's teardown (matching the run_tag prefix).

Teardown is best-effort: a failure to reclaim must never mask the run's outcome,
but it is surfaced on stderr because an un-reclaimed sandbox keeps costing money.
"""

from __future__ import annotations

import sys
from typing import Sequence

from reprogym.compute.providers import CliRunner
from reprogym.sandbox.runner import RunResult, run


def reclaim(runtime, *, runner: CliRunner | None = None) -> dict[str, list[str]]:
    """Sweep all providers attached to the runtime; return {provider: killed_ids}."""
    killed: dict[str, list[str]] = {}
    run_tag = getattr(runtime, "run_tag", "")
    if not run_tag:
        return killed
    for provider in getattr(runtime, "providers", []):
        teardown = getattr(provider, "teardown", None)
        if teardown is None:
            continue
        try:
            ids = teardown(run_tag, runner=runner) or []
        except Exception as exc:  # noqa: BLE001 - guard must not raise
            sys.stderr.write(
                f"[reprogym] WARNING: teardown of {provider.name} failed: {exc!r}; "
                f"check for orphaned sandboxes (tag {run_tag}).\n"
            )
            ids = []
        killed[provider.name] = ids
    return killed


def run_guarded(
    runtime,
    user_query: str | None = None,
    *,
    runner: CliRunner | None = None,
    **kwargs,
) -> RunResult:
    """Run the agent, then reclaim provisioned compute -- even on error/timeout."""
    try:
        return run(runtime, user_query, **kwargs)
    finally:
        reclaim(runtime, runner=runner)
