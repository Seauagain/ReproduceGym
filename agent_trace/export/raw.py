"""Raw export: the native on-wire request/response per captured completion.

This is the lossless view — the exact body the agent sent and the upstream
returned (response reconstructed from SSE when streamed). The verbatim byte
stream also lives next to each record as a ``.raw`` sidecar.
"""

from __future__ import annotations

from typing import Any

from agent_trace.store.models import CompletionSession


def session_to_raw(session: CompletionSession) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in session.sorted_completions():
        out.append(
            {
                "completion_id": rec.completion_id,
                "timestamp": rec.timestamp,
                "api_type": rec.api_type,
                "request": rec.original_request or rec.request,
                "response": rec.response,
                "metadata": rec.metadata,
            }
        )
    return out
