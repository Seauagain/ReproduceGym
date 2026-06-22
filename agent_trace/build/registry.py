"""Builder name -> class registry."""

from __future__ import annotations

from typing import Any

from agent_trace.build.base import TrajectoryBuilder
from agent_trace.build.message_prefix_merging import MessagePrefixMergingBuilder
from agent_trace.build.per_request import PerRequestBuilder
from agent_trace.store.models import CompletionSession, Trajectory

_BUILDERS: dict[str, type[TrajectoryBuilder]] = {
    PerRequestBuilder.name: PerRequestBuilder,
    MessagePrefixMergingBuilder.name: MessagePrefixMergingBuilder,
}


def get_builder(name: str, **config: Any) -> TrajectoryBuilder:
    """Construct a registered builder by name."""
    try:
        cls = _BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_BUILDERS))
        raise KeyError(f"unknown builder {name!r}; known: {known}") from exc
    return cls(**config)


def build_with(name: str, session: CompletionSession, **config: Any) -> Trajectory:
    """Convenience: construct a builder and run it on a session."""
    return get_builder(name, **config).build(session)
