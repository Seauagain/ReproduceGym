"""Host-side sandbox runtime (steps 4-6).

  launcher  start sandbox on the HOST (reuse ClawGym chroot/docker backend),
            mount input_files/, inject the API key from .env
  runner    issue the task user_query to the in-sandbox reproduction agent,
            let it reach remote GPUs via plain ssh when needed, record trajectory
  retry     resume the conversation after an interruption

reward/ is never mounted here; scoring happens out-of-band in reprogym.verify.
"""

from __future__ import annotations

from reprogym.sandbox.backends import (
    AgentBackend,
    ClaudeCodeBackend,
    CodexBackend,
    OpenCodeBackend,
    get_backend,
)
from reprogym.sandbox.launcher import Runtime, launch
from reprogym.sandbox.retry import resume
from reprogym.sandbox.runner import RunResult, run
from reprogym.sandbox.sandbox import DockerSandbox, LocalSandbox, Sandbox, SandboxResult
from reprogym.sandbox.workspace import prepare_workspace

__all__ = [
    "AgentBackend",
    "ClaudeCodeBackend",
    "CodexBackend",
    "OpenCodeBackend",
    "get_backend",
    "Runtime",
    "launch",
    "resume",
    "RunResult",
    "run",
    "Sandbox",
    "SandboxResult",
    "LocalSandbox",
    "DockerSandbox",
    "prepare_workspace",
]
