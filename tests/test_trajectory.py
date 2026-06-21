"""M4: trajectory recording (Claude stream-json -> normalized jsonl)."""

from __future__ import annotations

import json

import pytest

from reprogym.trajectory import Trajectory

STREAM_LINES = [
    {"type": "system", "subtype": "init", "session_id": "sess-1", "model": "claude-x"},
    {
        "type": "assistant",
        "session_id": "sess-1",
        "message": {"content": [{"type": "text", "text": "Let me inspect the task."}]},
    },
    {
        "type": "assistant",
        "session_id": "sess-1",
        "message": {
            "content": [
                {"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"command": "ls"}}
            ]
        },
    },
    {
        "type": "user",
        "session_id": "sess-1",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": "tu1", "is_error": False, "content": "a\nb\n"}
            ]
        },
    },
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "done",
        "num_turns": 2,
        "session_id": "sess-1",
        "total_cost_usd": 0.01,
    },
]


@pytest.fixture
def stream_text():
    return "\n".join(json.dumps(o) for o in STREAM_LINES)


def test_parse_event_types_in_order(stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    assert [e["type"] for e in traj.events] == [
        "system_init",
        "assistant_text",
        "tool_use",
        "tool_result",
        "result",
    ]


def test_session_id_lifted_to_meta(stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    assert traj.meta["session_id"] == "sess-1"
    assert traj.meta["model"] == "claude-x"


def test_tool_use_and_result_fields(stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    tu = traj.of_type("tool_use")[0]
    assert tu["tool"] == "Bash" and tu["input"]["command"] == "ls"
    tr = traj.of_type("tool_result")[0]
    assert tr["tool_use_id"] == "tu1" and tr["is_error"] is False
    assert tr["content"] == "a\nb\n"


def test_result_event(stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    res = traj.of_type("result")[0]
    assert res["is_error"] is False and res["result"] == "done"


def test_indices_are_sequential(stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    assert [e["i"] for e in traj.events] == list(range(len(traj.events)))


def test_tool_result_content_block_list():
    line = json.dumps(
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu9",
                        "content": [{"type": "text", "text": "hello "}, {"type": "text", "text": "world"}],
                    }
                ]
            },
        }
    )
    traj = Trajectory.from_claude_stream(line)
    assert traj.of_type("tool_result")[0]["content"] == "hello world"


def test_malformed_line_skipped_by_default(stream_text):
    traj = Trajectory.from_claude_stream(stream_text + "\nnot-json-here\n")
    assert len(traj.of_type("result")) == 1


def test_malformed_line_strict_raises(stream_text):
    with pytest.raises(json.JSONDecodeError):
        Trajectory.from_claude_stream("garbage{", strict=True)


def test_dump_and_reload_roundtrip(tmp_path, stream_text):
    traj = Trajectory.from_claude_stream(stream_text)
    path = traj.dump(tmp_path / "runs" / "trajectory.jsonl")
    assert path.is_file()
    reloaded = Trajectory.from_jsonl(path)
    assert reloaded.events == traj.events


def test_append_and_summary():
    traj = Trajectory(meta={"agent": "claude-code"})
    traj.append({"type": "tool_use", "tool": "Bash"})
    traj.append({"type": "tool_use", "tool": "Read"})
    s = traj.summary()
    assert s["n_events"] == 2 and s["counts"]["tool_use"] == 2
    assert s["meta"]["agent"] == "claude-code"
