"""Trajectory recording, shared by both modes.

Captures the reproduction agent's messages / tool calls / observations into a
normalized, serializable form (`runs/<run_id>/trajectory.jsonl`). The same format
is reused as the RL rollout trajectory so interactive (off-policy) and training
(on-policy) runs are directly comparable.

The primary source is a Claude Code (or compatible) `--output-format stream-json`
stream: one JSON object per line. `Trajectory.from_claude_stream` flattens those
into a stable event schema:

    {"i": <int>, "type": "system_init"|"assistant_text"|"tool_use"
                          |"tool_result"|"result", ...}

`session_id` (needed for resume) is lifted into `meta`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class Trajectory:
    def __init__(self, *, meta: dict[str, Any] | None = None):
        self.events: list[dict[str, Any]] = []
        self.meta: dict[str, Any] = dict(meta or {})

    # -- building -------------------------------------------------------- #
    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        event = dict(event)
        event.setdefault("i", len(self.events))
        self.events.append(event)
        return event

    def extend(self, events: Iterable[dict[str, Any]]) -> None:
        for e in events:
            self.append(e)

    # -- introspection --------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.events)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("type") == event_type]

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.get("type", "?")] = counts.get(e.get("type", "?"), 0) + 1
        return {"meta": self.meta, "n_events": len(self.events), "counts": counts}

    # -- io -------------------------------------------------------------- #
    def dump(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for e in self.events:
                fh.write(json.dumps(e, ensure_ascii=False) + "\n")
        return out

    @classmethod
    def from_jsonl(cls, path: str | Path, *, meta: dict | None = None) -> "Trajectory":
        traj = cls(meta=meta)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                traj.events.append(json.loads(line))
        return traj

    # -- claude code stream-json ---------------------------------------- #
    @classmethod
    def from_claude_stream(
        cls,
        stream: str | Iterable[str],
        *,
        strict: bool = False,
        meta: dict | None = None,
    ) -> "Trajectory":
        traj = cls(meta=meta)
        lines = stream.splitlines() if isinstance(stream, str) else stream
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                if strict:
                    raise
                continue
            traj._ingest_claude_event(obj)
        return traj

    def _ingest_claude_event(self, obj: dict[str, Any]) -> None:
        etype = obj.get("type")
        sid = obj.get("session_id")
        if sid and "session_id" not in self.meta:
            self.meta["session_id"] = sid

        if etype == "system":
            if obj.get("subtype") == "init" and "model" in obj:
                self.meta.setdefault("model", obj["model"])
            self.append({"type": "system_init", "subtype": obj.get("subtype")})

        elif etype == "assistant":
            for block in obj.get("message", {}).get("content", []):
                btype = block.get("type")
                if btype == "text":
                    self.append({"type": "assistant_text", "text": block.get("text", "")})
                elif btype == "tool_use":
                    self.append(
                        {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "tool": block.get("name"),
                            "input": block.get("input", {}),
                        }
                    )

        elif etype == "user":
            for block in obj.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    self.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.get("tool_use_id"),
                            "is_error": bool(block.get("is_error", False)),
                            "content": _flatten_tool_result(block.get("content")),
                        }
                    )

        elif etype == "result":
            self.append(
                {
                    "type": "result",
                    "subtype": obj.get("subtype"),
                    "is_error": bool(obj.get("is_error", False)),
                    "result": obj.get("result"),
                    "num_turns": obj.get("num_turns"),
                    "total_cost_usd": obj.get("total_cost_usd"),
                }
            )


def _flatten_tool_result(content: Any) -> str:
    """Claude tool_result content can be a string or a list of content blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
