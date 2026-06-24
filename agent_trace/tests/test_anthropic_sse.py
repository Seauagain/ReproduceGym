"""Reconstructing an Anthropic message from its SSE stream."""

from __future__ import annotations

import json

from agent_trace.proxy.anthropic_sse import parse_sse_events, reconstruct_from_raw


def _sse(*events: dict) -> str:
    return "".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events
    )


def test_parse_drops_done_and_non_data_lines():
    raw = "event: ping\ndata: {\"type\": \"ping\"}\n\n" "data: [DONE]\n\n"
    events = parse_sse_events(raw)
    assert events == [{"type": "ping"}]


def test_reconstruct_text_message():
    raw = _sse(
        {
            "type": "message_start",
            "message": {"id": "msg_1", "role": "assistant", "model": "opus",
                        "usage": {"input_tokens": 10}},
        },
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hello"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": " world"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
         "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    )
    msg = reconstruct_from_raw(raw)
    assert msg["id"] == "msg_1"
    assert msg["role"] == "assistant"
    assert msg["stop_reason"] == "end_turn"
    assert msg["content"] == [{"type": "text", "text": "Hello world"}]
    assert msg["usage"]["input_tokens"] == 10
    assert msg["usage"]["output_tokens"] == 5


def test_reconstruct_tool_use_assembles_partial_json():
    raw = _sse(
        {"type": "message_start",
         "message": {"id": "msg_2", "role": "assistant", "model": "opus"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_1", "name": "bash"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": "{\"cmd\":"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": " \"ls\"}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    )
    msg = reconstruct_from_raw(raw)
    block = msg["content"][0]
    assert block["type"] == "tool_use"
    assert block["name"] == "bash"
    assert block["input"] == {"cmd": "ls"}
    assert "_partial_json" not in block
    assert msg["stop_reason"] == "tool_use"


def test_reconstruct_text_then_tool_use_two_blocks():
    raw = _sse(
        {"type": "message_start",
         "message": {"id": "m", "role": "assistant", "model": "opus"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "let me run it"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "t1", "name": "bash"}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    )
    msg = reconstruct_from_raw(raw)
    assert [b["type"] for b in msg["content"]] == ["text", "tool_use"]
    assert msg["content"][0]["text"] == "let me run it"
    assert msg["content"][1]["input"] == {}
