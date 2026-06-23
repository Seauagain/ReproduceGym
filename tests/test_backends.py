"""M3: pluggable agent backends (command + env construction)."""

from __future__ import annotations

import pytest

from reproducegym.sandbox.backends import (
    AgentBackend,
    ClaudeCodeBackend,
    CodexBackend,
    OpenCodeBackend,
    get_backend,
)


def test_claude_command_fresh_session():
    be = ClaudeCodeBackend(model="m1", max_turns=7)
    cmd = be.build_command("reproduce it", session_id="sid-1")
    assert cmd[0] == "claude"
    assert "-p" in cmd and "reproduce it" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in cmd and "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--model") + 1] == "m1"
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert cmd[cmd.index("--session-id") + 1] == "sid-1"
    assert "--resume" not in cmd


def test_claude_uncapped_turns_drops_flag_and_env():
    # max_turns=0 means UNCAPPED: no --max-turns flag, and the env var must be
    # stripped so the claude CLI can't silently re-cap from a leftover value.
    be = ClaudeCodeBackend(model="m1", max_turns=0)
    cmd = be.build_command("go", session_id="s")
    assert "--max-turns" not in cmd
    env = be.build_env({"CLAUDE_CODE_MAX_TURNS": "100", "PATH": "/usr/bin"})
    assert "CLAUDE_CODE_MAX_TURNS" not in env
    assert env["PATH"] == "/usr/bin"


def test_claude_explicit_cap_sets_flag_and_env():
    be = ClaudeCodeBackend(model="m1", max_turns=7)
    cmd = be.build_command("go", session_id="s")
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    env = be.build_env({"PATH": "/usr/bin"})
    assert env["CLAUDE_CODE_MAX_TURNS"] == "7"


def test_claude_command_resume():
    be = ClaudeCodeBackend(model="m1")
    cmd = be.build_command("continue", session_id="sid-1", resume=True)
    assert cmd[cmd.index("--resume") + 1] == "sid-1"
    assert "--session-id" not in cmd


def test_claude_build_env_preserves_and_does_not_crash():
    be = ClaudeCodeBackend(model="m1")
    env = be.build_env({"ANTHROPIC_API_KEY": "preset", "PATH": "/usr/bin"})
    assert env["ANTHROPIC_API_KEY"] == "preset"
    assert env["PATH"] == "/usr/bin"


def test_codex_command_fresh_and_resume():
    be = CodexBackend(model="gpt")
    fresh = be.build_command("do", session_id="s")
    assert fresh[0] == "codex" and "exec" in fresh and "do" in fresh
    resumed = be.build_command("do", session_id="s", resume=True)
    assert "resume" in resumed and "s" in resumed


def test_opencode_command():
    be = OpenCodeBackend()
    cmd = be.build_command("task", session_id="s")
    assert cmd[0] == "opencode" and "run" in cmd and "task" in cmd


def test_get_backend_by_name_and_instance():
    assert isinstance(get_backend("claude-code"), ClaudeCodeBackend)
    assert isinstance(get_backend("claude"), ClaudeCodeBackend)
    assert isinstance(get_backend("codex"), CodexBackend)
    assert isinstance(get_backend("opencode"), OpenCodeBackend)
    inst = ClaudeCodeBackend()
    assert get_backend(inst) is inst


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError):
        get_backend("no-such-agent")
