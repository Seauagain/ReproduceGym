"""S1: secret redaction before a trajectory is persisted / 回流训练."""

from __future__ import annotations

from reproducegym.redact import collect_secrets, redact_text, redact_trajectory
from reproducegym.trajectory import Trajectory


def test_redact_text_masks_secret():
    out = redact_text("token is sk-ABCDEF123456 ok", ["sk-ABCDEF123456"])
    assert "sk-ABCDEF123456" not in out
    assert "token is" in out and "ok" in out


def test_redact_text_ignores_short_secrets():
    # too-short values must NOT be masked (would nuke ordinary text)
    assert redact_text("a b c", ["a", "b"]) == "a b c"


def test_redact_text_longest_first_on_overlap():
    out = redact_text("AKLONGSECRET", ["AKLONG", "AKLONGSECRET"])
    assert out.count("«REDACTED»") == 1
    assert "AKLONG" not in out


def test_collect_secrets_filters_missing_and_short():
    env = {"ANTHROPIC_API_KEY": "sk-realkey-123456", "EMPTY": "", "SHORT": "x"}
    got = collect_secrets(env, ["ANTHROPIC_API_KEY", "EMPTY", "SHORT", "MISSING"])
    assert got == ["sk-realkey-123456"]


def test_redact_trajectory_scrubs_events_and_meta():
    ak = "BOHRIUM-AK-SECRET-001"
    traj = Trajectory(meta={"session_id": "s1", "note": f"login {ak}"})
    traj.append({"type": "tool_use", "tool": "bash", "input": {"command": f"lbg login --ak {ak}"}})
    traj.append({"type": "tool_result", "content": f"using {ak} done"})

    redact_trajectory(traj, [ak])

    blob = str(traj.events) + str(traj.meta)
    assert ak not in blob
    assert "«REDACTED»" in blob
    # non-secret structure preserved
    assert traj.events[0]["tool"] == "bash"
    assert traj.meta["session_id"] == "s1"


def test_redact_trajectory_noop_without_secrets():
    traj = Trajectory(meta={})
    traj.append({"type": "assistant_text", "text": "hello"})
    redact_trajectory(traj, [])
    assert traj.events[0]["text"] == "hello"
