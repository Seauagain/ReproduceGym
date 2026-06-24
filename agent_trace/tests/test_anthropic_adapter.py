"""Anthropic native records -> chat envelope -> merged trajectory."""

from __future__ import annotations

from agent_trace.build.anthropic_adapter import anthropic_to_chat_record, to_chat_session
from agent_trace.build.registry import get_builder
from agent_trace.store.models import CompletionRecord, CompletionSession

_SYS = "You are a coding agent."
_TOOLS = [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}]

_ASSISTANT_1 = [
    {"type": "text", "text": "running"},
    {"type": "tool_use", "id": "t1", "name": "bash", "input": {"cmd": "ls"}},
]
_TOOL_RESULT = [{"type": "tool_result", "tool_use_id": "t1", "content": "file.txt"}]


def _rec(cid, messages, response, ts):
    return CompletionRecord(
        completion_id=cid,
        timestamp=ts,
        api_type="anthropic",
        request={"model": "opus", "system": _SYS, "tools": _TOOLS, "messages": messages},
        original_request={"model": "opus", "system": _SYS, "messages": messages},
        response=response,
        metadata={"stream": True},
    )


def test_single_anthropic_record_envelope():
    rec = _rec(
        "c1",
        [{"role": "user", "content": "do it"}],
        {"role": "assistant", "content": _ASSISTANT_1, "stop_reason": "tool_use"},
        "t0",
    )
    chat = anthropic_to_chat_record(rec)
    assert chat.api_type == "openai_chat"
    assert chat.request["messages"][0] == {"role": "system", "content": _SYS}
    assert chat.request["messages"][1] == {"role": "user", "content": "do it"}
    assert chat.request["tools"][0]["type"] == "function"
    assert chat.request["tools"][0]["function"]["name"] == "bash"
    choice = chat.response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] == _ASSISTANT_1


def test_anthropic_tool_loop_merges_to_single_trace():
    step1_msgs = [{"role": "user", "content": "do it"}]
    step2_msgs = [
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": _ASSISTANT_1},
        {"role": "user", "content": _TOOL_RESULT},
    ]
    session = CompletionSession(
        session_id="s",
        completions=[
            _rec(
                "c1",
                step1_msgs,
                {"role": "assistant", "content": _ASSISTANT_1, "stop_reason": "tool_use"},
                "t0",
            ),
            _rec(
                "c2",
                step2_msgs,
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "done"}],
                    "stop_reason": "end_turn",
                },
                "t1",
            ),
        ],
    )

    chat_session = to_chat_session(session)
    trajectory = get_builder("message_prefix_merging").build(chat_session)

    assert trajectory.status == "COMPLETED"
    assert len(trajectory.traces) == 1
    trace = trajectory.traces[0]
    # prompt = [system, user]
    assert [m["role"] for m in trace.prompt_messages] == ["system", "user"]
    # response = assistant_1 + tool_result(user) + assistant_2
    assert [m["role"] for m in trace.response_messages] == ["assistant", "user", "assistant"]
    assert trace.metadata["merged_completion_count"] == 2
    assert trace.finish_reason == "stop"
