"""agent_trace: a runtime-agnostic trajectory capture + build layer.

Sits between an agent (Claude Code / Codex / OpenCode / OpenClaw) and the model
API, forwards every request verbatim to the agent's real upstream, captures the
on-wire request + response, then stitches the multi-turn calls back into a
single prefix-matched trajectory and exports it as raw and/or SFT records.

Ported from Polar (ProRL-Agent-Server) but flipped from its serve-path
(self-hosted model, token-level) to a log-path (passthrough, message-level).

This package is intentionally self-contained so it can be split into its own
repository later via `git subtree split --prefix=agent_trace`.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.0.1"
