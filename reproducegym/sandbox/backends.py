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
from urllib.parse import urlsplit

from reproducegym.config import dotenv_values
from reproducegym.trajectory import Trajectory

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})


def _is_loopback_url(url: str) -> bool:
    """True for a local capture-proxy URL (http://127.0.0.1:PORT). Used so build_env
    honors run.py's capture redirect instead of clobbering it back to the relay."""
    if not url:
        return False
    try:
        return (urlsplit(url).hostname or "") in _LOOPBACK_HOSTS
    except ValueError:
        return False


class AgentBackend:
    name = "base"
    env_keys: tuple[str, ...] = ()

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        raise NotImplementedError

    def build_env(self, base: Mapping[str, str]) -> dict[str, str]:
        """Copy base env and ensure the backend's keys (from .env) are present."""
        file_env = dotenv_values()
        env = dict(base)
        for key in self.env_keys:
            value = file_env.get(key)
            if value:
                env[key] = value
            else:
                env.pop(key, None)
        return env

    def parse(self, stdout: str, *, meta: dict[str, Any] | None = None) -> Trajectory:
        return Trajectory.from_claude_stream(stdout, meta=meta)


class ClaudeCodeBackend(AgentBackend):
    name = "claude-code"
    # Endpoint + auth are the only secrets the redactor must scrub from trajectories.
    env_keys = (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_API_KEY",
    )
    # Claude Code model-routing / tuning knobs forwarded verbatim from .env, matching
    # the interactive `cc-ds` baseline. All of these are PROVEN safe against the
    # gpugeek relay (a `claude --bare -p` smoke with them set returns normally and
    # reports contextWindow=1_000_000); they are NOT the cause of the connection
    # resets (a leftover *_PROXY in the agent env is — see build_env).
    _CLAUDE_TUNING_KEYS = (
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_EFFORT_LEVEL",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    )
    # Any proxy in the sandbox env routes Claude Code's relay calls through it; the
    # gpugeek relay then resets the connection (the CLI retries 11x and dies with
    # "ECONNRESET"). The model API is reachable directly, so strip every proxy
    # variant. Capture-mode upstream uses http.client (proxy-agnostic) anyway.
    _PROXY_KEYS = (
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    )

    def __init__(
        self,
        *,
        binary: str = "claude",
        model: str | None = None,
        max_turns: int | None = None,
    ):
        env = dotenv_values()
        self.binary = binary
        self.model = model or env.get("CLAUDE_CODE_MODEL") or env.get("ANTHROPIC_DEFAULT_OPUS_MODEL")
        if self.model and self.model.startswith("Vendor2/") and "[" not in self.model:
            self.model += "[1m]"
        # Precedence: an explicit arg wins; 0 means UNCAPPED (no cap), which is
        # distinct from None=unset (fall back to the CLAUDE_CODE_MAX_TURNS env
        # default). A long training run polled turn-by-turn always hits a finite
        # cap, so callers pass 0 to bound the run by wall-clock instead.
        if max_turns is None:
            env_mt = env.get("CLAUDE_CODE_MAX_TURNS")
            self.max_turns = int(env_mt) if env_mt else None
        else:
            self.max_turns = max_turns or None

    def build_env(self, base: Mapping[str, str]) -> dict[str, str]:
        # super().build_env injects ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY from .env.
        incoming_base_url = base.get("ANTHROPIC_BASE_URL", "")
        env = super().build_env(base)
        file_env = dotenv_values()
        # CAPTURE: when run.py points the agent at the local capture proxy
        # (http://127.0.0.1:PORT), keep it. super() would otherwise overwrite it with
        # the .env relay URL, so the agent would skip the proxy and capture nothing.
        if _is_loopback_url(incoming_base_url):
            env["ANTHROPIC_BASE_URL"] = incoming_base_url
        # Forward model-routing / tuning knobs from .env (parity with cc-ds): keeps
        # subagents on the right model and preserves the 1M context + effort level.
        for key in self._CLAUDE_TUNING_KEYS:
            value = file_env.get(key)
            if value:
                env[key] = value
        # The resolved main model (with its [1m] suffix) must also drive the `opus`
        # alias so the default-model path gets the 1M context window, not just the
        # explicit --model flag.
        if self.model:
            env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = self.model
        # ROOT-CAUSE FIX for the ECONNRESET storms: scrub every proxy variant so the
        # agent talks to the relay (or the local capture proxy) directly.
        for key in self._PROXY_KEYS:
            env.pop(key, None)
        # --bare authenticates strictly via ANTHROPIC_API_KEY; a stray AUTH_TOKEN
        # (e.g. from the deepseek cc-ds flow) makes auth ambiguous.
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        # Turns are bounded by --max-turns (CLI) + wall-clock timeout, never by a
        # leftover env cap that would silently truncate a long polled training run.
        env.pop("CLAUDE_CODE_MAX_TURNS", None)
        return env

    def build_command(
        self, prompt: str, *, session_id: str | None = None, resume: bool = False
    ) -> list[str]:
        cmd = [
            self.binary,
            "--bare",
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
