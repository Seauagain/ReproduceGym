"""Pluggable reproduction-agent backends.

A backend knows how to (a) build the agent CLI argv for a prompt (incl. resume),
(b) build the env that carries the agent's API key/model from .env, and (c) parse
the agent's stdout into a Trajectory. Claude Code is the default; opencode and
codex are supported as alternatives. All three are headless CLIs that emit a JSON
event stream and support resuming a prior session, which is what lets the host
control agent retry an interrupted run.

Command construction is pure (no network), so it is unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Mapping

from reproducegym.config import get_env, load_dotenv
from reproducegym.trajectory import Trajectory


class AgentBackend:
    name = "base"
    env_keys: tuple[str, ...] = ()

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        raise NotImplementedError

    def build_env(self, base: Mapping[str, str]) -> dict[str, str]:
        """Copy base env and ensure the backend's keys (from .env) are present."""
        load_dotenv()
        env = dict(base)
        for key in self.env_keys:
            value = get_env(key)
            if value and not env.get(key):
                env[key] = value
        return env

    def parse(self, stdout: str, *, meta: dict[str, Any] | None = None) -> Trajectory:
        return Trajectory.from_claude_stream(stdout, meta=meta)


class ClaudeCodeBackend(AgentBackend):
    name = "claude-code"
    env_keys = (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
        "CLAUDE_CODE_MAX_TURNS",
    )

    def __init__(
        self,
        *,
        binary: str = "claude",
        model: str | None = None,
        max_turns: int | None = None,
    ):
        load_dotenv()
        self.binary = binary
        self.model = model or get_env("ANTHROPIC_DEFAULT_OPUS_MODEL")
        mt = max_turns if max_turns is not None else get_env("CLAUDE_CODE_MAX_TURNS")
        self.max_turns = int(mt) if mt else None

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.model:
            cmd += ["--model", self.model]
        if self.max_turns:
            cmd += ["--max-turns", str(self.max_turns)]
        if resume and session_id:
            cmd += ["--resume", session_id]
        elif session_id:
            cmd += ["--session-id", session_id]
        return cmd


class OpenCodeBackend(AgentBackend):
    name = "opencode"
    env_keys = ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL")

    def __init__(self, *, binary: str = "opencode", model: str | None = None):
        self.binary = binary
        self.model = model

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        cmd = [self.binary, "run", prompt, "--print-logs"]
        if self.model:
            cmd += ["--model", self.model]
        if resume and session_id:
            cmd += ["--session", session_id]
        return cmd


class CodexBackend(AgentBackend):
    name = "codex"
    env_keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL")

    def __init__(self, *, binary: str = "codex", model: str | None = None):
        self.binary = binary
        self.model = model

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        if resume and session_id:
            cmd = [self.binary, "exec", "resume", session_id, "--json"]
        else:
            cmd = [self.binary, "exec", prompt, "--json", "--dangerously-bypass-approvals-and-sandbox"]
        if self.model:
            cmd += ["-m", self.model]
        return cmd


_REGISTRY = {
    "claude-code": ClaudeCodeBackend,
    "claude": ClaudeCodeBackend,
    "opencode": OpenCodeBackend,
    "codex": CodexBackend,
}


def get_backend(name: str | AgentBackend, **kwargs: Any) -> AgentBackend:
    if isinstance(name, AgentBackend):
        return name
    try:
        return _REGISTRY[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown agent backend {name!r}; have {sorted(_REGISTRY)}") from exc
