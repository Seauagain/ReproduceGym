"""S3: host run-guard -- always reclaim provisioned compute (fake lbg)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from reproducegym.compute.providers import LbgProvider
from reproducegym.pipeline.render_task import render_task
from reproducegym.sandbox.launcher import launch
from reproducegym.sandbox.run_guard import reclaim, run_guarded
from reproducegym.sandbox.sandbox import Sandbox
from tests.helpers import STREAM, FakeBackend, RecordingSandbox


class FakeLbg:
    """Records argv; answers `sdbx list --json` with the configured sandboxes."""

    def __init__(self, sandboxes, *, wrap_key=None):
        self.sandboxes = sandboxes
        self.wrap_key = wrap_key
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:3] == ["lbg", "sdbx", "list"]:
            payload = {self.wrap_key: self.sandboxes} if self.wrap_key else self.sandboxes
            return json.dumps(payload)
        return ""

    def kills(self):
        return [c[-1] for c in self.calls if c[:3] == ["lbg", "sdbx", "kill"]]


@pytest.fixture
def task_dir(tmp_path, valid_claim_spec):
    return render_task(valid_claim_spec, tmp_path / "task")


def test_teardown_kills_only_matching_prefix():
    fake = FakeLbg([
        {"id": "s1", "name": "myrun-train"},
        {"id": "s2", "name": "myrun-eval"},
        {"id": "s3", "name": "other-x"},
    ])
    killed = LbgProvider().teardown("myrun", runner=fake)
    assert set(killed) == {"s1", "s2"}
    assert set(fake.kills()) == {"s1", "s2"}


def test_teardown_handles_wrapped_list_and_alt_id_keys():
    fake = FakeLbg([{"sandboxId": "z9", "name": "myrun-a"}], wrap_key="sandboxes")
    assert LbgProvider().teardown("myrun", runner=fake) == ["z9"]


def test_teardown_best_effort_on_failures():
    def boom(argv):
        raise RuntimeError("network")

    assert LbgProvider().teardown("x", runner=boom) == []
    assert LbgProvider().teardown("x", runner=lambda a: "not-json") == []


def test_reclaim_sweeps_providers():
    rt = SimpleNamespace(run_tag="myrun", providers=[LbgProvider()])
    fake = FakeLbg([{"id": "s1", "name": "myrun-train"}])
    assert reclaim(rt, runner=fake) == {"lbg": ["s1"]}


def test_reclaim_noop_without_run_tag():
    rt = SimpleNamespace(run_tag="", providers=[LbgProvider()])
    assert reclaim(rt) == {}


def test_run_guarded_reclaims_after_run(tmp_path, task_dir):
    rec = RecordingSandbox(STREAM)
    rt = launch(
        task_dir, tmp_path / "myrun", backend=FakeBackend(STREAM),
        sandbox=rec, compute="lbg:project=4449832",
    )
    fake = FakeLbg([{"id": "s1", "name": rt.run_tag + "-train"}])
    result = run_guarded(rt, runner=fake)
    assert result.returncode == 0
    assert ["lbg", "sdbx", "list", "--json"] in fake.calls
    assert "s1" in fake.kills()


def test_run_guarded_reclaims_on_error(tmp_path, task_dir):
    class BoomSandbox(Sandbox):
        def run(self, argv, *, cwd, env=None, timeout=None):
            raise RuntimeError("agent crashed")

    rt = launch(
        task_dir, tmp_path / "myrun2", backend=FakeBackend(STREAM),
        sandbox=BoomSandbox(), compute="lbg:project=4449832",
    )
    fake = FakeLbg([{"id": "s9", "name": rt.run_tag + "-train"}])
    with pytest.raises(RuntimeError):
        run_guarded(rt, runner=fake)
    assert "s9" in fake.kills()  # swept despite the crash
