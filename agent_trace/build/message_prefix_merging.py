"""Message-level prefix-merging trajectory builder.

A multi-turn agent resends the whole growing conversation on every step, so
consecutive requests share a common message prefix. This builder stitches that
append-only chain back into a single trajectory:

    prompt_messages = first request's messages
    response_messages = assistant_1
                        + interstitial (tool results / user turns) before step 2
                        + assistant_2
                        + interstitial before step 3
                        + assistant_3 ...

It is the message-level analogue of Polar's ``prefix_merging`` (which works on
token ids and therefore needs a self-hosted model). Here we compare *message
fingerprints* instead of token ids, so it works for any frontier-API agent
logged in passthrough mode.

Two stages, mirroring Polar:

1. Grouping — route each completion to the chain it append-extends, tested on
   message fingerprints: a completion joins the chain whose last request's
   messages are a fingerprint-prefix of it. Interleaved parallel sessions each
   carry a distinct prefix, so they route to distinct chains. On overlap the
   longest matching tip wins.

2. Finalization — walk each chain. The previous assistant turn is echoed back
   into the next request's history; everything *after* that echo is the
   interstitial (tool results, injected user turns). When the prefix relation
   breaks (e.g. context compaction rewrote earlier turns) the chain is
   truncated at the break and the remainder starts a fresh chain.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from agent_trace.build.base import TrajectoryBuilder
from agent_trace.build.record_utils import build_trace_from_record, fingerprints
from agent_trace.store.models import CompletionRecord, CompletionSession, Trace, Trajectory


class MessagePrefixMergingBuilder(TrajectoryBuilder):
    name = "message_prefix_merging"

    def build(self, session: CompletionSession) -> Trajectory:
        records = session.sorted_completions()
        if not records:
            return Trajectory(
                status="ERROR",
                error="no completions",
                metadata={"builder": self.name, "session_id": session.session_id},
            )

        chains = self._group(records)

        stats = {
            "chains_total": len(chains),
            "chains_reconstructed_full": 0,
            "chains_reconstructed_truncated": 0,
            "completions_total": len(records),
            "completions_merged": 0,
        }
        traces = [self._finalize_chain(chain, stats) for chain in chains]

        return Trajectory(
            status="COMPLETED",
            traces=traces,
            metadata={
                "builder": self.name,
                "session_id": session.session_id,
                "task_id": session.task_id,
                "api_type": session.api_type,
                "record_count": len(records),
                "trace_count": len(chains),
                "reconstruction_stats": stats,
            },
        )

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group(self, records: list[CompletionRecord]) -> list[list[CompletionRecord]]:
        chains: list[list[CompletionRecord]] = []
        chain_tips: list[list[str]] = []  # fingerprints of each chain's last prompt

        for record in records:
            prompt_fp = fingerprints(build_trace_from_record(record).prompt_messages)
            idx = self._find_extendable_chain(prompt_fp, chain_tips)
            if idx is None:
                idx = len(chains)
                chains.append([])
                chain_tips.append([])
            chains[idx].append(record)
            chain_tips[idx] = prompt_fp
        return chains

    @staticmethod
    def _find_extendable_chain(
        prompt_fp: list[str],
        chain_tips: list[list[str]],
    ) -> int | None:
        """Return the open chain this record append-extends, else None.

        A record continues a chain iff its prompt fingerprints begin with that
        chain's last prompt fingerprints. Longest matching tip wins.
        """
        best_idx: int | None = None
        best_len = -1
        for idx, tip in enumerate(chain_tips):
            n = len(tip)
            if n > best_len and 0 < n <= len(prompt_fp) and prompt_fp[:n] == tip:
                best_idx, best_len = idx, n
        return best_idx

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def _finalize_chain(
        self,
        chain: list[CompletionRecord],
        stats: dict[str, int],
    ) -> Trace:
        first = build_trace_from_record(chain[0])
        prompt_messages = [deepcopy(m) for m in first.prompt_messages]
        response_messages: list[dict[str, Any]] = [
            deepcopy(m) for m in first.response_messages
        ]

        prev_prompt_fp = fingerprints(first.prompt_messages)
        prev_prompt_len = len(first.prompt_messages)
        prev_resp_len = len(first.response_messages)
        kept = 1

        for i in range(1, len(chain)):
            current = build_trace_from_record(chain[i])
            cur_fp = fingerprints(current.prompt_messages)

            # Prefix break: prior turns were rewritten (e.g. compaction).
            if len(cur_fp) < prev_prompt_len or cur_fp[:prev_prompt_len] != prev_prompt_fp:
                break

            # The prev assistant turn is echoed back into history right after the
            # prev prompt; the interstitial is everything after that echo.
            interstitial_start = prev_prompt_len + prev_resp_len
            if interstitial_start > len(current.prompt_messages):
                break
            interstitial = current.prompt_messages[interstitial_start:]

            response_messages.extend(deepcopy(m) for m in interstitial)
            response_messages.extend(deepcopy(m) for m in current.response_messages)

            prev_prompt_fp = cur_fp
            prev_prompt_len = len(current.prompt_messages)
            prev_resp_len = len(current.response_messages)
            kept += 1

        stats["completions_merged"] += kept
        if kept == len(chain):
            stats["chains_reconstructed_full"] += 1
        else:
            stats["chains_reconstructed_truncated"] += 1

        last_kept = build_trace_from_record(chain[kept - 1])
        return Trace(
            prompt_messages=prompt_messages,
            response_messages=response_messages,
            tools=deepcopy(first.tools),
            finish_reason=last_kept.finish_reason,
            metadata=self._chain_metadata(chain[:kept]),
        )

    @staticmethod
    def _chain_metadata(chain: list[CompletionRecord]) -> dict[str, Any]:
        return {
            "session_completion_ids": [c.completion_id for c in chain],
            "merged_completion_count": len(chain),
            "completion_metadata": [dict(c.metadata) for c in chain],
        }
