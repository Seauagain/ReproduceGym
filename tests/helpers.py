"""Shared lightweight test doubles and fixture builders."""

from __future__ import annotations

import json
from pathlib import Path

from reproducegym.sandbox.backends import AgentBackend
from reproducegym.sandbox.sandbox import Sandbox, SandboxResult

STREAM = "\n".join(
    json.dumps(o)
    for o in [
        {"type": "system", "subtype": "init", "session_id": "sess-fake", "model": "m"},
        {
            "type": "assistant",
            "session_id": "sess-fake",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "done",
            "session_id": "sess-fake",
        },
    ]
)


class FakeBackend(AgentBackend):
    name = "fake"

    def __init__(self, stream: str):
        self.stream = stream
        self.calls: list[dict] = []

    def build_command(self, prompt, *, session_id=None, resume=False):
        self.calls.append({"prompt": prompt, "session_id": session_id, "resume": resume})
        script = "cat <<'REPRO_STREAM_EOF'\n" + self.stream + "\nREPRO_STREAM_EOF\n"
        return ["bash", "-c", script]

    def build_env(self, base):
        return dict(base)


class RecordingSandbox(Sandbox):
    name = "recording"

    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode
        self.argv = None
        self.cwd = None
        self.env = None

    def run(self, argv, *, cwd, env=None, timeout=None):
        self.argv = list(argv)
        self.cwd = cwd
        self.env = dict(env or {})
        return SandboxResult(self.returncode, self.stdout, "")


class FakeVL:
    """Multimodal client fake keyed by image filename."""

    def __init__(self, by_name: dict[str, str], *, default: str = "[]"):
        self.by_name = by_name
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def read_figure(self, image_path, prompt):
        name = Path(image_path).name
        self.calls.append((name, prompt))
        return self.by_name.get(name, self.default)


def make_figures(tmp_path: Path, names: list[str]) -> Path:
    d = tmp_path / "figures"
    d.mkdir()
    for name in names:
        (d / name).write_bytes(b"img")
    return d


def make_parse_bundle(root: Path) -> Path:
    """Minimal parsed paper bundle with one figure."""
    pdir = root / "runs" / "paper1" / "00-parse"
    (pdir / "figures").mkdir(parents=True)
    (pdir / "figures" / "fig1.jpg").write_bytes(b"jpeg")
    (pdir / "paper.md").write_text(
        "# P\n\n![](images/fig1.jpg)\n\nFigure 1: length over step.\n",
        encoding="utf-8",
    )
    (pdir / "figures.index.json").write_text(
        json.dumps(
            [
                {
                    "figure_ref": "Fig. 1",
                    "image_file": "fig1.jpg",
                    "source_path": str(pdir / "figures" / "fig1.jpg"),
                    "caption": "Figure 1: length over step.",
                    "context": "Figure 1: length over step.",
                }
            ]
        ),
        encoding="utf-8",
    )
    return root / "runs" / "paper1"
