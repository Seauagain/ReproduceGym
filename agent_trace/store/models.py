"""Data shapes for the capture -> build -> export pipeline.

Stdlib-dataclass schemas (the repo avoids pydantic). The proxy captures
``CompletionRecord``s into a ``CompletionSession``; a builder turns that into a
``Trajectory`` of trainable ``Trace``s; an exporter serializes traces into raw /
SFT records.

These mirror Polar's ``trajectory/models.py`` but the message fields are the
primary payload (Polar leaned on token ids, which only exist with a self-hosted
model). Token-level fields are kept as optional defaults so a future serve-mode
builder can fill them without a schema change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Json = dict[str, Any]

VALID_STATUS = ("COMPLETED", "TIMEOUT", "ERROR")


@dataclass
class CompletionRecord:
    """One captured model call.

    In passthrough/logging mode the request is forwarded verbatim, so
    ``request`` and ``original_request`` are the same native body the agent
    sent; ``response`` is the native upstream response. ``api_type`` is one of
    ``anthropic`` / ``openai_chat`` / ``openai_responses`` / ``google``.
    """

    completion_id: str
    timestamp: str | None = None
    api_type: str | None = None
    request: Json = field(default_factory=dict)
    original_request: Json = field(default_factory=dict)
    response: Json = field(default_factory=dict)
    metadata: Json = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Json) -> "CompletionRecord":
        """Load a persisted record, tolerating Polar's on-disk field names.

        Polar writes ``original_request`` + ``transformed_request`` + ``response``;
        passthrough writes a single verbatim request. We prefer the native
        (original) request for trajectory building.
        """
        original = d.get("original_request") or {}
        request = original or d.get("request") or d.get("transformed_request") or {}
        return cls(
            completion_id=str(d.get("completion_id") or d.get("id") or ""),
            timestamp=d.get("timestamp"),
            api_type=d.get("api_type"),
            request=request,
            original_request=original or request,
            response=d.get("response") or {},
            metadata=d.get("metadata") or {},
        )


@dataclass
class CompletionSession:
    """Every record captured during one agent run."""

    session_id: str
    created_at: str | None = None
    task_id: str | None = None
    model_requested: str | None = None
    model_used: str | None = None
    api_type: str | None = None
    metadata: Json = field(default_factory=dict)
    completions: list[CompletionRecord] = field(default_factory=list)

    def sorted_completions(self) -> list[CompletionRecord]:
        """Completions in deterministic (timestamp, completion_id) order."""
        return sorted(
            self.completions,
            key=lambda c: (c.timestamp or "", c.completion_id),
        )

    @classmethod
    def from_dict(cls, d: Json) -> "CompletionSession":
        comps = d.get("completions") or []
        return cls(
            session_id=str(d.get("session_id") or ""),
            created_at=d.get("created_at"),
            task_id=d.get("task_id"),
            model_requested=d.get("model_requested"),
            model_used=d.get("model_used"),
            api_type=d.get("api_type"),
            metadata=d.get("metadata") or {},
            completions=[CompletionRecord.from_dict(c) for c in comps],
        )


@dataclass
class Trace:
    """One reconstructed interaction (a trainable example).

    ``prompt_messages`` is the static context; ``response_messages`` is the
    stitched assistant turns + interstitial tool results / user messages that
    follow it.
    """

    prompt_messages: list[Json] = field(default_factory=list)
    response_messages: list[Json] = field(default_factory=list)
    tools: list[Json] | None = None
    finish_reason: str | None = None
    reward: float | None = None
    # Optional token-level fields — only a serve-mode builder fills these.
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    loss_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    metadata: Json = field(default_factory=dict)


@dataclass
class Trajectory:
    """A terminal status plus the reconstructed traces."""

    status: str
    traces: list[Trace] = field(default_factory=list)
    metadata: Json = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(f"invalid trajectory status: {self.status!r}")
