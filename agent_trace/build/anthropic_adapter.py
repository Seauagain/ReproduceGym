"""Normalize a native Anthropic ``CompletionRecord`` into the chat envelope.

The builders only understand the OpenAI-chat view (``request.messages`` +
``response.choices[0].message``). Rather than teach them the Anthropic wire
shape, we re-envelope each captured record:

- ``system`` (str or block list) becomes a leading ``{"role":"system",...}``
  message.
- Each Anthropic message maps to exactly **one** chat message (content blocks
  preserved verbatim). Keeping the 1:1 message count is what lets prefix-merging
  line up the echoed assistant turn and the interstitial tool results across
  steps.
- The reconstructed assistant response becomes ``choices[0].message`` with a
  mapped ``finish_reason``.

Content blocks are kept in Anthropic form (lossless); only the envelope changes,
so fingerprints stay stable turn-to-turn and nothing about the original payload
is dropped.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_trace.store.models import CompletionRecord

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _convert_tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list) or not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") and "function" in tool:  # already chat-shaped
            out.append(deepcopy(tool))
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return out or None


def _chat_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = request.get("system")
    if system:
        messages.append({"role": "system", "content": deepcopy(system)})
    for msg in request.get("messages", []) or []:
        if not isinstance(msg, dict):
            continue
        messages.append(
            {"role": msg.get("role"), "content": deepcopy(msg.get("content"))}
        )
    return messages


def anthropic_to_chat_record(record: CompletionRecord) -> CompletionRecord:
    """Return a new record in chat-envelope form (Anthropic blocks preserved)."""
    request = record.request if isinstance(record.request, dict) else {}
    response = record.response if isinstance(record.response, dict) else {}

    chat_request = {
        "model": request.get("model"),
        "messages": _chat_messages(request),
        "tools": _convert_tools(request.get("tools")),
    }

    assistant_message = {
        "role": response.get("role", "assistant"),
        "content": deepcopy(response.get("content", [])),
    }
    finish_reason = _STOP_REASON_MAP.get(
        response.get("stop_reason"), response.get("stop_reason")
    )
    chat_response = {
        "choices": [{"message": assistant_message, "finish_reason": finish_reason}],
        "usage": response.get("usage"),
    }

    return CompletionRecord(
        completion_id=record.completion_id,
        timestamp=record.timestamp,
        api_type="openai_chat",
        request=chat_request,
        original_request=record.original_request or request,
        response=chat_response,
        metadata={**deepcopy(record.metadata), "source_api_type": "anthropic"},
    )


def to_chat_record(record: CompletionRecord) -> CompletionRecord:
    """Dispatch on api_type; pass through records already in chat form."""
    if record.api_type == "anthropic":
        return anthropic_to_chat_record(record)
    return record


def to_chat_session(session: "CompletionSession") -> "CompletionSession":
    """Return a copy of the session with every record normalized to chat form."""
    from agent_trace.store.models import CompletionSession

    return CompletionSession(
        session_id=session.session_id,
        created_at=session.created_at,
        task_id=session.task_id,
        model_requested=session.model_requested,
        model_used=session.model_used,
        api_type="openai_chat",
        metadata=dict(session.metadata),
        completions=[to_chat_record(c) for c in session.completions],
    )
