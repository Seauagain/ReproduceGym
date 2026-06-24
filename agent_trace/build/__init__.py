"""Trajectory builders: turn a CompletionSession into a Trajectory of Traces."""

from __future__ import annotations

from agent_trace.build.base import TrajectoryBuilder
from agent_trace.build.message_prefix_merging import MessagePrefixMergingBuilder
from agent_trace.build.per_request import PerRequestBuilder
from agent_trace.build.registry import build_with, get_builder

__all__ = [
    "TrajectoryBuilder",
    "MessagePrefixMergingBuilder",
    "PerRequestBuilder",
    "get_builder",
    "build_with",
]
