"""The builder contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_trace.store.models import CompletionSession, Trajectory


class TrajectoryBuilder(ABC):
    """Convert a captured session into a Trajectory of trainable traces."""

    name: str = "base"

    @abstractmethod
    def build(self, session: CompletionSession) -> Trajectory:  # pragma: no cover
        raise NotImplementedError
