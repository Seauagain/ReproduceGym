"""Estimate a reproduction's wall-clock budget and a sane polling cadence from
the claim's nature, BEFORE launching it.

Why: a GRPO training reproduction runs for many hours (sometimes >1 day), so the
host must size the agent's wall-clock timeout up front instead of using one flat
number, and must tell the agent how often to poll. Polling cadence scales with
expected runtime: a 36h training should be checked every ~30min, not every turn.

Inputs come from the task's data_entry.json metadata (``cost`` tier S/M/L and
``requires_training``); no LLM, no network -- pure and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

# cost tier -> (wall-clock hours, poll interval seconds)
_TRAIN = {"S": (8, 600), "M": (18, 900), "L": (36, 1800)}
_LIGHT = {"S": (1, 120), "M": (3, 300), "L": (8, 600)}


@dataclass(frozen=True)
class RuntimeEstimate:
    timeout_s: int
    poll_s: int
    hours: float
    requires_training: bool
    cost: str

    @property
    def label(self) -> str:
        kind = "training" if self.requires_training else "light"
        return (f"~{self.hours:g}h wall-clock ({kind}, cost={self.cost}); "
                f"poll every ~{self.poll_s // 60}min")


def estimate_runtime(*, requires_training: bool, cost: str | None) -> RuntimeEstimate:
    """Map (requires_training, cost tier) -> a timeout + poll cadence.

    Training tiers allow long runs (L = 36h, i.e. > 1 day) because real RL
    training legitimately takes that long; light (eval/analysis) tiers stay short.
    Unknown cost falls back to the M tier.
    """
    tier = (cost or "M").strip().upper()[:1] or "M"
    table = _TRAIN if requires_training else _LIGHT
    hours, poll = table.get(tier, table["M"])
    return RuntimeEstimate(
        timeout_s=hours * 3600,
        poll_s=poll,
        hours=hours,
        requires_training=requires_training,
        cost=tier,
    )
