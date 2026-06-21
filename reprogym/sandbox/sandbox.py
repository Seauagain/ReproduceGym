"""Host sandboxes that execute the agent argv.

The sandbox runs on the HOST. LocalSandbox just runs the argv in the prepared
workspace (lightest isolation; the workspace only contains input_files/, never
reward/ or secrets beyond the injected key). DockerSandbox wraps the same argv in
`docker run` mounting the workspace, for stronger isolation -- env var NAMES are
forwarded with `-e KEY` so secret VALUES never appear in the process list.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


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
        proc = subprocess.run(
            list(argv),
            cwd=str(cwd),
            env=dict(env) if env is not None else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxResult(proc.returncode, proc.stdout, proc.stderr)


class DockerSandbox(Sandbox):
    name = "docker"

    def __init__(self, image: str, *, mount: str = "/workspace", docker_bin: str = "docker"):
        self.image = image
        self.mount = mount
        self.docker_bin = docker_bin

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
