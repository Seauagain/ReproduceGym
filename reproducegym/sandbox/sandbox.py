"""Host sandboxes that execute the agent argv.

The sandbox runs on the HOST. LocalSandbox just runs the argv in the prepared
workspace (lightest isolation; the workspace only contains input_files/, never
reward/ or secrets beyond the injected key). DockerSandbox wraps the same argv in
`docker run` mounting the workspace, for stronger isolation -- env var NAMES are
forwarded with `-e KEY` so secret VALUES never appear in the process list.
"""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

# Sentinel returncode when the agent subprocess is killed for exceeding its
# wall-clock budget (mirrors GNU timeout's exit code).
TIMEOUT_RETURNCODE = 124


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    name = "base"

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: str | Path,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> SandboxResult:
        raise NotImplementedError


class LocalSandbox(Sandbox):
    name = "local"

    def run(self, argv, *, cwd, env=None, timeout=None) -> SandboxResult:
        # stdin=DEVNULL mirrors the proven headless invocation (`claude ... < /dev/null`):
        # without it `claude -p` can block waiting on an inherited stdin.
        # start_new_session puts the agent in its own process group so a wall-clock
        # timeout can kill the WHOLE tree (claude + any ssh/poll children), not just
        # the parent -- this is what stops orphaned agents from lingering.
        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return SandboxResult(proc.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            # Budget hit: tear down the process group, then collect partial output so
            # the captured trajectory (and stream-json stdout) is still persisted.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                proc.kill()
            stdout, stderr = proc.communicate()
            return SandboxResult(TIMEOUT_RETURNCODE, stdout or "", stderr or "")


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(
        self,
        image: str,
        *,
        mount: str = "/workspace",
        docker_bin: str = "docker",
        mount_ssh: bool = True,
        ssh_dir: str = "~/.ssh",
    ):
        self.image = image
        self.mount = mount
        self.docker_bin = docker_bin
        self.mount_ssh = mount_ssh
        self.ssh_dir = ssh_dir

    def build_argv(
        self, argv: Sequence[str], *, cwd: str | Path, env_keys: Sequence[str] = ()
    ) -> list[str]:
        docker = [
            self.docker_bin,
            "run",
            "--rm",
            "-v",
            f"{Path(cwd).resolve()}:{self.mount}",
            "-w",
            self.mount,
        ]
        if self.mount_ssh:
            # read-only ssh creds so the in-container agent can ssh to MetaX
            docker += ["-v", f"{Path(self.ssh_dir).expanduser()}:/root/.ssh:ro"]
        for key in env_keys:
            docker += ["-e", key]  # forward NAME only; value supplied via env
        docker.append(self.image)
        docker += list(argv)
        return docker

    def run(self, argv, *, cwd, env=None, timeout=None) -> SandboxResult:
        env = dict(env) if env is not None else {}
        docker_argv = self.build_argv(argv, cwd=cwd, env_keys=list(env.keys()))
        proc = subprocess.run(
            docker_argv,
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxResult(proc.returncode, proc.stdout, proc.stderr)
