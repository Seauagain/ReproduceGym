"""Unit tests for the message-level prefix-merging builder.

Records are synthetic OpenAI-chat-shaped completions. Each record's
``request.messages`` is the full conversation the agent resent on that step, and
``response.choices[0].message`` is that step's assistant turn.
"""

from __future__ import annotations

from agent_trace.build import MessagePrefixMergingBuilder, PerRequestBuilder
from agent_trace.store.models import CompletionRecord, CompletionSession


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _tool_result(call_id: str, text: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _assistant_tool_call(call_id: str, name: str, args: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def _record(cid: str, ts: str, messages: list[dict], assistant: dict, finish: str = "stop", tools=None) -> CompletionRecord:
    request: dict = {"model": "m", "messages": messages}
    if tools is not None:
        request["tools"] = tools
    return CompletionRecord(
        completion_id=cid,
        timestamp=ts,
        api_type="openai_chat",
        request=request,
        original_request=request,
        response={"choices": [{"index": 0, "message": assistant, "finish_reason": finish}]},
        metadata={},
    )


def _session(records: list[CompletionRecord]) -> CompletionSession:
    return CompletionSession(session_id="s1", completions=records)


def _roles(messages: list[dict]) -> list[str]:
    return [m["role"] for m in messages]


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #

def test_empty_session_errors():
    traj = MessagePrefixMergingBuilder().build(_session([]))
    assert traj.status == "ERROR"
    assert traj.traces == []


def test_single_completion_one_trace():
    rec = _record("c1", "t1", [_user("hi")], _assistant_text("hello"))
    traj = MessagePrefixMergingBuilder().build(_session([rec]))

    assert traj.status == "COMPLETED"
    assert len(traj.traces) == 1
    trace = traj.traces[0]
    assert _roles(trace.prompt_messages) == ["user"]
    assert _roles(trace.response_messages) == ["assistant"]
    assert trace.finish_reason == "stop"


def test_two_step_tool_loop_merges():
    a1 = _assistant_tool_call("tc1", "calc", '{"x":1}')
    r1 = _record("c1", "t1", [_user("add")], a1, finish="tool_calls")
    r2 = _record(
        "c2",
        "t2",
        [_user("add"), a1, _tool_result("tc1", "42")],
        _assistant_text("done"),
        finish="stop",
    )

    traj = MessagePrefixMergingBuilder().build(_session([r1, r2]))

    assert len(traj.traces) == 1, "tool loop must merge into one trajectory"
    trace = traj.traces[0]
    assert _roles(trace.prompt_messages) == ["user"]
    # assistant(tool_call) -> tool result -> assistant(final)
    assert _roles(trace.response_messages) == ["assistant", "tool", "assistant"]
    assert trace.finish_reason == "stop"
    assert traj.metadata["reconstruction_stats"]["chains_reconstructed_full"] == 1


def test_three_step_chain_merges_in_order():
    a1 = _assistant_tool_call("tc1", "f", "{}")
    a2 = _assistant_tool_call("tc2", "g", "{}")
    a3 = _assistant_text("final")
    r1 = _record("c1", "t1", [_user("go")], a1, finish="tool_calls")
    r2 = _record("c2", "t2", [_user("go"), a1, _tool_result("tc1", "A")], a2, finish="tool_calls")
    r3 = _record(
        "c3",
        "t3",
        [_user("go"), a1, _tool_result("tc1", "A"), a2, _tool_result("tc2", "B")],
        a3,
        finish="stop",
    )

    traj = MessagePrefixMergingBuilder().build(_session([r1, r2, r3]))

    assert len(traj.traces) == 1
    trace = traj.traces[0]
    assert _roles(trace.prompt_messages) == ["user"]
    assert _roles(trace.response_messages) == [
        "assistant", "tool", "assistant", "tool", "assistant",
    ]
    contents = [m.get("content") for m in trace.response_messages]
    assert contents[1] == "A" and contents[3] == "B" and contents[4] == "final"


def test_tools_carried_from_first_record():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    a1 = _assistant_tool_call("tc1", "calc", "{}")
    r1 = _record("c1", "t1", [_user("x")], a1, finish="tool_calls", tools=tools)
    r2 = _record("c2", "t2", [_user("x"), a1, _tool_result("tc1", "1")], _assistant_text("ok"), tools=tools)

    traj = MessagePrefixMergingBuilder().build(_session([r1, r2]))
    assert traj.traces[0].tools == tools


def test_interleaved_parallel_sessions_split_into_two_chains():
    aa = _assistant_tool_call("a", "f", "{}")
    bb = _assistant_tool_call("b", "g", "{}")
    a1 = _record("a1", "t1", [_user("alpha")], aa, finish="tool_calls")
    b1 = _record("b1", "t2", [_user("beta")], bb, finish="tool_calls")
    a2 = _record("a2", "t3", [_user("alpha"), aa, _tool_result("a", "A")], _assistant_text("da"))
    b2 = _record("b2", "t4", [_user("beta"), bb, _tool_result("b", "B")], _assistant_text("db"))

    traj = MessagePrefixMergingBuilder().build(_session([a1, b1, a2, b2]))

    assert len(traj.traces) == 2
    prompts = sorted(t.prompt_messages[0]["content"] for t in traj.traces)
    assert prompts == ["alpha", "beta"]
    for trace in traj.traces:
        assert _roles(trace.response_messages) == ["assistant", "tool", "assistant"]


def test_compaction_break_starts_new_chain():
    a1 = _assistant_text("ans1")
    r1 = _record("c1", "t1", [_user("q1")], a1)
    # continues the chain: prev prompt [q1] is a prefix, then a new user turn.
    r2 = _record("c2", "t2", [_user("q1"), a1, _user("q2")], _assistant_text("ans2"))
    # compaction rewrote earlier turns -> not a prefix of r2 -> fresh chain.
    r3 = _record("c3", "t3", [_user("summary-of-history"), _user("q3")], _assistant_text("ans3"))

    traj = MessagePrefixMergingBuilder().build(_session([r1, r2, r3]))

    assert len(traj.traces) == 2
    merged = max(traj.traces, key=lambda t: len(t.response_messages))
    assert _roles(merged.prompt_messages) == ["user"]
    assert _roles(merged.response_messages) == ["assistant", "user", "assistant"]


def test_per_request_builder_one_trace_each():
    r1 = _record("c1", "t1", [_user("a")], _assistant_text("x"))
    r2 = _record("c2", "t2", [_user("a"), _assistant_text("x"), _user("b")], _assistant_text("y"))
    traj = PerRequestBuilder().build(_session([r1, r2]))
    assert len(traj.traces) == 2
