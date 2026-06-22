"""Helpers to normalize a captured record into a message-level Trace.

Slice 1 handles the OpenAI Chat shape (``request.messages`` +
``response.choices[0].message``). Anthropic / OpenAI-Responses parsers land in a
later slice; they will normalize into this same chat view before extraction, so
the builders never learn the wire format.

The message *fingerprint* is the key primitive for prefix matching: two records
belong to the same chain when one record's messages are a fingerprint-prefix of
a later record's messages. Fingerprints drop known-volatile keys (e.g.
``cache_control``) so cosmetic re-serialization between turns does not break the
prefix relationship.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from agent_trace.store.models import CompletionRecord, Trace

# Keys that an agent/SDK may add or mutate between otherwise-identical turns.
_VOLATILE_KEYS = frozenset({"cache_control"})


def _canonical(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _canonical(v)
            for k, v in sorted(obj.items())
            if k not in _VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_canonical(v) for v in obj]
    return obj


def message_fingerprint(message: dict[str, Any]) -> str:
    """A stable, hashable identity for a chat message (volatile keys stripped)."""
    return json.dumps(_canonical(message), sort_keys=True, ensure_ascii=False)


def fingerprints(messages: list[dict[str, Any]]) -> list[str]:
    return [message_fingerprint(m) for m in messages]


def _extract_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(m) for m in messages if isinstance(m, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(t) for t in tools if isinstance(t, dict)]
    return extracted or None


def build_trace_from_record(record: CompletionRecord) -> Trace:
    """Normalize one captured (OpenAI-chat) record into a single-turn Trace."""
    request = record.request if isinstance(record.request, dict) else {}
    response = record.response if isinstance(record.response, dict) else {}

    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    return Trace(
        prompt_messages=_extract_messages(request),
        response_messages=[deepcopy(message)] if isinstance(message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        metadata=deepcopy(record.metadata),
    )
