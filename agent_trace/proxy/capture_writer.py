"""Persist captured completions to disk in a builder-readable layout.

Layout (one dir per session)::

    <save_dir>/sessions/<session_id>/completions/0001-<id>.json   # native record
    <save_dir>/sessions/<session_id>/completions/0001-<id>.raw    # verbatim wire bytes
    <save_dir>/sessions/<session_id>/session.json                 # session metadata

The ``.json`` is the structured ``CompletionRecord`` (native request +
reconstructed response); the ``.raw`` sidecar holds the literal upstream bytes
(SSE text or JSON) so nothing about the original format is lost.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class CaptureWriter:
    def __init__(self, save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self._lock = threading.Lock()
        self._seq: dict[str, int] = {}

    def _session_dir(self, session_id: str) -> Path:
        return self.save_dir / "sessions" / session_id

    def _next_seq(self, session_id: str) -> int:
        with self._lock:
            n = self._seq.get(session_id, 0) + 1
            self._seq[session_id] = n
            return n

    def write_session_meta(self, session_id: str, meta: dict[str, Any]) -> None:
        sdir = self._session_dir(session_id)
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "session.json").write_text(
            json.dumps({"session_id": session_id, **meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write(
        self,
        session_id: str,
        record: dict[str, Any],
        raw_bytes: bytes | None = None,
    ) -> str:
        seq = self._next_seq(session_id)
        cdir = self._session_dir(session_id) / "completions"
        cdir.mkdir(parents=True, exist_ok=True)

        cid = str(record.get("completion_id") or "rec")
        stem = f"{seq:04d}-{cid}"
        if raw_bytes is not None:
            raw_path = cdir / f"{stem}.raw"
            raw_path.write_bytes(raw_bytes)
            record.setdefault("metadata", {})["raw_path"] = raw_path.name

        json_path = cdir / f"{stem}.json"
        json_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(json_path)
