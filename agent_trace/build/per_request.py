"""per_request builder: one trace per captured completion."""

from __future__ import annotations

from agent_trace.build.base import TrajectoryBuilder
from agent_trace.build.record_utils import build_trace_from_record
from agent_trace.store.models import CompletionSession, Trajectory


class PerRequestBuilder(TrajectoryBuilder):
    """Each completion becomes its own independent trace."""

    name = "per_request"

    def build(self, session: CompletionSession) -> Trajectory:
        records = session.sorted_completions()
        if not records:
            return Trajectory(
                status="ERROR",
                error="no completions",
                metadata={"builder": self.name, "session_id": session.session_id},
            )
        traces = [build_trace_from_record(record) for record in records]
        return Trajectory(
            status="COMPLETED",
            traces=traces,
            metadata={
                "builder": self.name,
                "session_id": session.session_id,
                "task_id": session.task_id,
                "record_count": len(records),
                "trace_count": len(traces),
            },
        )
