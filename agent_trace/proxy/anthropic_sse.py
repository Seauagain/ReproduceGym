"""Reconstruct an Anthropic ``/v1/messages`` response from its SSE stream.

Claude Code streams responses as Server-Sent Events. To persist a record that
looks like the non-streaming Messages response (so downstream parsers have one
shape to handle), we replay the event sequence:

    message_start -> (content_block_start / _delta / _stop)* -> message_delta ->
    message_stop

and rebuild the final ``message`` object (content blocks, tool_use inputs,
thinking, stop_reason, usage). The raw SSE text is still persisted verbatim
alongside the record, so this reconstruction never loses fidelity — it is a
convenience view, not the source of truth.
"""

from __future__ import annotations

import json
from typing import Any


def parse_sse_events(raw: str) -> list[dict[str, Any]]:
    """Parse a raw SSE byte/str stream into the list of JSON ``data:`` payloads."""
    events: list[dict[str, Any]] = []
    for block in raw.replace("\r\n", "\n").split("\n\n"):
        data_lines = [
            line[len("data:") :].lstrip()
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        if not data_lines:
            continue
        payload = "\n".join(data_lines)
        if payload == "[DONE]":
            continue
        try:
            events.append(json.loads(payload))
        except json.JSONDecodeError:
            continue
    return events


def reconstruct_message(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold an Anthropic SSE event list into a single Messages-API message dict."""
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    for ev in events:
        et = ev.get("type")
        if et == "message_start":
            message = dict(ev.get("message") or {})
            message["content"] = []
        elif et == "content_block_start":
            idx = ev.get("index", len(order))
            cb = dict(ev.get("content_block") or {})
            ct = cb.get("type")
            if ct == "text":
                cb.setdefault("text", "")
            elif ct == "thinking":
                cb.setdefault("thinking", "")
            elif ct == "tool_use":
                cb["_partial_json"] = ""
            blocks[idx] = cb
            order.append(idx)
        elif et == "content_block_delta":
            idx = ev.get("index")
            cb = blocks.get(idx)
            if cb is None:
                continue
            delta = ev.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                cb["text"] = cb.get("text", "") + delta.get("text", "")
            elif dt == "thinking_delta":
                cb["thinking"] = cb.get("thinking", "") + delta.get("thinking", "")
            elif dt == "input_json_delta":
                cb["_partial_json"] = cb.get("_partial_json", "") + delta.get(
                    "partial_json", ""
                )
            elif dt == "signature_delta":
                cb["signature"] = cb.get("signature", "") + delta.get("signature", "")
        elif et == "content_block_stop":
            cb = blocks.get(ev.get("index"))
            if cb is not None and cb.get("type") == "tool_use":
                partial = cb.pop("_partial_json", "")
                try:
                    cb["input"] = json.loads(partial) if partial else {}
                except json.JSONDecodeError:
                    cb["input"] = {"_unparsed_partial_json": partial}
        elif et == "message_delta":
            delta = ev.get("delta") or {}
            for key, val in delta.items():
                message[key] = val
            if ev.get("usage"):
                message["usage"] = {**(message.get("usage") or {}), **ev["usage"]}

    content: list[dict[str, Any]] = []
    for idx in order:
        cb = blocks.get(idx)
        if cb is None:
            continue
        cb.pop("_partial_json", None)
        content.append(cb)
    if message:
        message["content"] = content
    return message


def reconstruct_from_raw(raw: str) -> dict[str, Any]:
    """Convenience: parse + reconstruct in one step."""
    return reconstruct_message(parse_sse_events(raw))
