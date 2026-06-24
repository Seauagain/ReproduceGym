"""M3: pluggable agent backends (command + env construction)."""

from __future__ import annotations

import pytest

import reproducegym.config as config
from reproducegym.sandbox.backends import (
    AgentBackend,
    ClaudeCodeBackend,
    CodexBackend,
    OpenCodeBackend,
    get_backend,
)


@pytest.fixture
def backend_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=from-file-key",
                "ANTHROPIC_BASE_URL=https://api.gpugeek.com/",
                "ANTHROPIC_DEFAULT_OPUS_MODEL=from-file-model",
                "CLAUDE_CODE_MAX_TURNS=5",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env)
    return env


def test_claude_command_fresh_session(backend_env):
    be = ClaudeCodeBackend(model="ignored", max_turns=7)
    cmd = be.build_command("reproduce it", session_id="sid-1")
    assert cmd[0] == "claude"
    assert "--bare" in cmd
    assert "-p" in cmd and "reproduce it" in cmd
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in cmd and "--dangerously-skip-permissions" in cmd
    assert cmd[cmd.index("--model") + 1] == "ignored"
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    assert cmd[cmd.index("--session-id") + 1] == "sid-1"
    assert "--resume" not in cmd


def test_claude_uncapped_turns_drops_flag_and_env(backend_env):
    # max_turns=0 means UNCAPPED: no --max-turns flag, and the env var must be
    # stripped so the claude CLI can't silently re-cap from a leftover value.
    be = ClaudeCodeBackend(model="ignored", max_turns=0)
    cmd = be.build_command("go", session_id="s")
    assert "--max-turns" not in cmd
    env = be.build_env({"CLAUDE_CODE_MAX_TURNS": "100", "PATH": "/usr/bin"})
    assert "CLAUDE_CODE_MAX_TURNS" not in env
    assert env["PATH"] == "/usr/bin"


def test_claude_explicit_cap_sets_flag_and_env(backend_env):
    be = ClaudeCodeBackend(model="ignored", max_turns=7)
    cmd = be.build_command("go", session_id="s")
    assert cmd[cmd.index("--max-turns") + 1] == "7"
    env = be.build_env({"PATH": "/usr/bin", "CLAUDE_CODE_MAX_TURNS": "100"})
    assert "CLAUDE_CODE_MAX_TURNS" not in env


def test_claude_default_vendor_model_adds_1m_suffix(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "ANTHROPIC_API_KEY=k\n"
        "ANTHROPIC_BASE_URL=https://api.gpugeek.com/\n"
        "ANTHROPIC_DEFAULT_OPUS_MODEL=Vendor2/Claude-4.6-opus\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env)
    cmd = ClaudeCodeBackend().build_command("go", session_id="s")
    assert cmd[cmd.index("--model") + 1] == "Vendor2/Claude-4.6-opus[1m]"


def test_claude_build_env_scrubs_every_proxy_variant(backend_env):
    # A leftover proxy is the ECONNRESET root cause; build_env must drop all variants.
    be = ClaudeCodeBackend(model="ignored")
    env = be.build_env({
        "http_proxy": "http://127.0.0.1:17890",
        "https_proxy": "http://127.0.0.1:17890",
        "all_proxy": "http://127.0.0.1:17890",
        "no_proxy": "localhost",
        "HTTP_PROXY": "http://127.0.0.1:17890",
        "HTTPS_PROXY": "http://127.0.0.1:17890",
        "ALL_PROXY": "http://127.0.0.1:17890",
        "NO_PROXY": "localhost",
        "PATH": "/usr/bin",
    })
    for key in (
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    ):
        assert key not in env
    assert env["PATH"] == "/usr/bin"


def test_claude_build_env_forwards_tuning_and_drops_auth_token(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "ANTHROPIC_API_KEY=k\n"
        "ANTHROPIC_BASE_URL=https://api.gpugeek.com/\n"
        "ANTHROPIC_DEFAULT_OPUS_MODEL=Vendor2/Claude-4.6-opus\n"
        "ANTHROPIC_DEFAULT_SONNET_MODEL=Vendor2/Claude-4.6-Sonnet\n"
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000\n"
        "CLAUDE_CODE_EFFORT_LEVEL=max\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env_path)
    be = ClaudeCodeBackend()
    env = be.build_env({"ANTHROPIC_AUTH_TOKEN": "stale", "PATH": "/usr/bin"})
    # tuning knobs forwarded from .env (parity with cc-ds)
    assert env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "128000"
    assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "max"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "Vendor2/Claude-4.6-Sonnet"
    # the resolved 1M main model also drives the opus default alias
    assert env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "Vendor2/Claude-4.6-opus[1m]"
    # --bare auths via API key only; stray AUTH_TOKEN is dropped
    assert "ANTHROPIC_AUTH_TOKEN" not in env


def test_claude_build_env_honors_capture_proxy_base_url(backend_env):
    # run.py redirects ANTHROPIC_BASE_URL to the local capture proxy; build_env must
    # keep it (not clobber back to the .env relay) so the proxy actually captures.
    be = ClaudeCodeBackend(model="ignored")
    env = be.build_env({
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:54321",
        "PATH": "/usr/bin",
    })
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:54321"
    # auth key still comes from .env
    assert env["ANTHROPIC_API_KEY"] == "from-file-key"


def test_claude_build_env_uses_dotenv_base_url_when_not_loopback(backend_env):
    be = ClaudeCodeBackend(model="ignored")
    env = be.build_env({
        "ANTHROPIC_BASE_URL": "https://some-other-relay.example",
        "PATH": "/usr/bin",
    })
    assert env["ANTHROPIC_BASE_URL"] == "https://api.gpugeek.com/"


def test_claude_command_resume(backend_env):
    be = ClaudeCodeBackend(model="ignored")
    cmd = be.build_command("continue", session_id="sid-1", resume=True)
    assert cmd[cmd.index("--resume") + 1] == "sid-1"
    assert "--session-id" not in cmd


def test_claude_build_env_overrides_model_api_from_dotenv(backend_env):
    be = ClaudeCodeBackend(model="ignored")
    env = be.build_env({
        "ANTHROPIC_API_KEY": "stale-shell-key",
        "ANTHROPIC_BASE_URL": "https://stale.example",
        "PATH": "/usr/bin",
    })
    assert env["ANTHROPIC_API_KEY"] == "from-file-key"
    assert env["ANTHROPIC_BASE_URL"] == "https://api.gpugeek.com/"
    assert env["PATH"] == "/usr/bin"


def test_backend_env_removes_absent_model_api_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=from-file-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env_path)
    env = ClaudeCodeBackend().build_env({
        "ANTHROPIC_BASE_URL": "stale",
        "OPENAI_API_KEY": "stale-openai",
        "PATH": "/usr/bin",
    })
    assert env["ANTHROPIC_API_KEY"] == "from-file-key"
    assert "ANTHROPIC_BASE_URL" not in env


def test_opencode_env_removes_absent_model_api_keys(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=from-file-key\n", encoding="utf-8")
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env_path)
    env = OpenCodeBackend().build_env({
        "ANTHROPIC_API_KEY": "stale-anthropic",
        "ANTHROPIC_BASE_URL": "stale-base",
        "OPENAI_API_KEY": "stale-openai",
        "OPENAI_BASE_URL": "stale-openai-base",
        "PATH": "/usr/bin",
    })
    assert env["ANTHROPIC_API_KEY"] == "from-file-key"
    assert "ANTHROPIC_BASE_URL" not in env
    assert "OPENAI_API_KEY" not in env
    assert "OPENAI_BASE_URL" not in env


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
