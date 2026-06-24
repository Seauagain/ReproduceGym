"""Passthrough capture proxy.

A dependency-free (stdlib-only) HTTP proxy that sits between an agent runtime
(Claude Code / Codex / openclaw / opencode) and a real upstream model API. It
forwards every request **verbatim** to the upstream and tees the (possibly
streamed) response back to the agent while persisting a native ``CompletionRecord``
to disk. Because the forward is verbatim, beta features the agent enables (e.g.
the ``context-1m`` header Claude Code sends for ``opus[1m]``) pass straight
through untouched.

The proxy never needs token ids or a self-hosted model: it is a logging tap, not
an inference server. Token-level builders remain possible later, but the primary
artifact is the native wire request + response, which message-level builders
stitch into trajectories.
"""

from __future__ import annotations

from agent_trace.proxy.capture_writer import CaptureWriter
from agent_trace.proxy.server import make_handler, serve

__all__ = ["CaptureWriter", "make_handler", "serve"]
