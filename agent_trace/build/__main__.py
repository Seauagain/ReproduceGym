"""CLI: build a trajectory from a captured session and (optionally) export it.

Example::

    python -m agent_trace.build \
        --save-dir runs/agent-trace-live/captures \
        --session metax-001 \
        --out-dir runs/agent-trace-live/built \
        --raw --sft
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from agent_trace.build.anthropic_adapter import to_chat_session
from agent_trace.build.registry import get_builder
from agent_trace.export.raw import session_to_raw
from agent_trace.export.sft import trajectory_to_sft
from agent_trace.store.loader import load_session


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(trajectory) -> str:
    lines = [f"status={trajectory.status}  traces={len(trajectory.traces)}"]
    stats = trajectory.metadata.get("reconstruction_stats", {})
    if stats:
        lines.append(
            "  completions={completions_total} merged={completions_merged} "
            "chains={chains_total} full={chains_reconstructed_full} "
            "truncated={chains_reconstructed_truncated}".format(**stats)
        )
    for i, tr in enumerate(trajectory.traces):
        roles = [m.get("role") for m in tr.response_messages]
        lines.append(
            f"  trace[{i}]: prompt_msgs={len(tr.prompt_messages)} "
            f"response_msgs={len(tr.response_messages)} "
            f"merged={tr.metadata.get('merged_completion_count')} "
            f"finish={tr.finish_reason} roles={roles}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_trace.build")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--builder", default="message_prefix_merging")
    parser.add_argument("--out-dir")
    parser.add_argument("--raw", action="store_true", help="write raw native records")
    parser.add_argument("--sft", action="store_true", help="write SFT chat samples")
    args = parser.parse_args(argv)

    native = load_session(args.save_dir, args.session)
    if not native.completions:
        print(f"no completions under {args.save_dir}/sessions/{args.session}")
        return 1

    chat = to_chat_session(native)
    builder = get_builder(args.builder)
    trajectory = builder.build(chat)

    print(_summarize(trajectory))

    if args.out_dir:
        out = Path(args.out_dir)
        _write_json(out / "trajectory.json", asdict(trajectory))
        if args.raw:
            _write_json(out / "raw.json", session_to_raw(native))
        if args.sft:
            _write_json(out / "sft.json", trajectory_to_sft(trajectory))
        print(f"wrote -> {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
