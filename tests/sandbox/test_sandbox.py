"""M3: host sandboxes (local exec + docker argv construction)."""

from __future__ import annotations

from pathlib import Path

from reproducegym.sandbox.sandbox import DockerSandbox, LocalSandbox


def test_local_sandbox_runs_and_captures(tmp_path):
    res = LocalSandbox().run(["bash", "-c", "echo hi"], cwd=tmp_path)
    assert res.returncode == 0
    assert res.stdout.strip() == "hi"


def test_local_sandbox_cwd(tmp_path):
    res = LocalSandbox().run(["bash", "-c", "pwd"], cwd=tmp_path)
    assert res.stdout.strip() == str(tmp_path)


def test_local_sandbox_env(tmp_path):
    res = LocalSandbox().run(
        ["bash", "-c", "echo $FOO"], cwd=tmp_path, env={"FOO": "bar", "PATH": "/usr/bin:/bin"}
    )
    assert res.stdout.strip() == "bar"


def test_local_sandbox_nonzero(tmp_path):
    res = LocalSandbox().run(["bash", "-c", "exit 2"], cwd=tmp_path)
    assert res.returncode == 2


def test_local_sandbox_stdin_is_devnull(tmp_path):
    # `cat` reads stdin; with stdin=DEVNULL it must get EOF immediately, not hang.
    res = LocalSandbox().run(["bash", "-c", "cat"], cwd=tmp_path, timeout=10)
    assert res.returncode == 0
    assert res.stdout == ""


def test_local_sandbox_timeout_kills_tree_and_returns_124(tmp_path):
    from reproducegym.sandbox.sandbox import TIMEOUT_RETURNCODE

    res = LocalSandbox().run(["bash", "-c", "sleep 30"], cwd=tmp_path, timeout=1)
    assert res.returncode == TIMEOUT_RETURNCODE


def test_docker_build_argv(tmp_path):
    sb = DockerSandbox(image="repro:latest")
    argv = sb.build_argv(["claude", "-p", "x"], cwd=tmp_path, env_keys=["ANTHROPIC_API_KEY"])
    assert argv[:4] == ["docker", "run", "--rm", "-v"]
    assert argv[4] == f"{Path(tmp_path).resolve()}:/workspace"
    assert "-w" in argv and "/workspace" in argv
    assert argv[argv.index("-e") + 1] == "ANTHROPIC_API_KEY"
    # image precedes the agent argv
    img_idx = argv.index("repro:latest")
    assert argv[img_idx + 1 :] == ["claude", "-p", "x"]


def test_docker_build_argv_mounts_ssh_by_default(tmp_path):
    sb = DockerSandbox(image="repro:latest", ssh_dir=str(tmp_path / ".ssh"))
    argv = sb.build_argv(["claude"], cwd=tmp_path)
    assert f"{(tmp_path / '.ssh')}:/root/.ssh:ro" in argv


def test_docker_build_argv_ssh_mount_can_be_disabled(tmp_path):
    sb = DockerSandbox(image="repro:latest", mount_ssh=False)
    argv = sb.build_argv(["claude"], cwd=tmp_path)
    assert not any(":/root/.ssh:ro" in a for a in argv)
