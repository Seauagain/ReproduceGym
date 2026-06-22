"""SFT export: one chat sample per reconstructed trace.

A trace's prompt is the static context; its response is the stitched multi-turn
tail (assistant turns + interstitial tool results). For SFT we emit the full
message list plus the tool schema, with a ``loss`` hint marking which messages
are assistant-generated (the supervised targets).
"""

from __future__ import annotations

from typing import Any

from agent_trace.store.models import Trajectory


def _is_assistant(message: dict[str, Any]) -> bool:
    return isinstance(message, dict) and message.get("role") == "assistant"


def trajectory_to_sft(trajectory: Trajectory) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for trace in trajectory.traces:
        messages = [*trace.prompt_messages, *trace.response_messages]
        loss = [_is_assistant(m) for m in messages]
        samples.append(
            {
                "messages": messages,
                "tools": trace.tools,
                "loss": loss,
                "finish_reason": trace.finish_reason,
                "metadata": trace.metadata,
            }
        )
    return samples
