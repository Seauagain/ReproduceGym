"""Load captured completions from the on-disk layout into a CompletionSession."""

from __future__ import annotations

import json
from pathlib import Path

from agent_trace.store.models import CompletionRecord, CompletionSession


def load_session(save_dir: str | Path, session_id: str) -> CompletionSession:
    """Read ``<save_dir>/sessions/<session_id>`` into a CompletionSession.

    Completions are ordered by their filename sequence prefix (the order they
    were captured), which is the natural chronological order for a single agent.
    """
    sdir = Path(save_dir) / "sessions" / session_id
    comp_dir = sdir / "completions"

    meta: dict = {}
    meta_path = sdir / "session.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            meta = {}

    records: list[CompletionRecord] = []
    if comp_dir.is_dir():
        for path in sorted(comp_dir.glob("*.json")):
            try:
                records.append(CompletionRecord.from_dict(json.loads(path.read_text())))
            except json.JSONDecodeError:
                continue

    return CompletionSession(
        session_id=session_id,
        created_at=meta.get("created_at"),
        metadata=meta,
        api_type=records[0].api_type if records else None,
        completions=records,
    )
