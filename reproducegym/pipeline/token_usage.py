"""Token usage accounting for parse/build/run pipeline stages.

The recorder stores provider-reported token usage when available. It never
pretends estimates are real tokens; calls without provider usage are recorded
with ``usage_available=false`` plus prompt/output character counts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def normalize_usage(raw: Any) -> dict[str, int | None]:
    """Normalize Anthropic/OpenAI usage objects into common token fields."""
    input_tokens = _get(raw, "input_tokens")
    output_tokens = _get(raw, "output_tokens")
    prompt_tokens = _get(raw, "prompt_tokens")
    completion_tokens = _get(raw, "completion_tokens")
    total_tokens = _get(raw, "total_tokens")
    in_tok = input_tokens if input_tokens is not None else prompt_tokens
    out_tok = output_tokens if output_tokens is not None else completion_tokens
    if total_tokens is None and (in_tok is not None or out_tok is not None):
        total_tokens = int(in_tok or 0) + int(out_tok or 0)
    return {
        "input_tokens": int(in_tok) if in_tok is not None else None,
        "output_tokens": int(out_tok) if out_tok is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


@dataclass
class TokenUsageRecord:
    paper_id: str
    stage: str
    step: str
    call_id: str
    provider: str | None = None
    model: str | None = None
    usage_available: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    prompt_chars: int | None = None
    completion_chars: int | None = None
    elapsed_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict[str, Any]:
        claim_uid = self.metadata.get("claim_uid")
        source_mode = self.metadata.get("source_mode")
        fallback_reason = self.metadata.get("fallback_reason")
        return {
            "created_at": self.created_at,
            "paper_id": self.paper_id,
            "stage": self.stage,
            "step": self.step,
            "claim_uid": claim_uid,
            "source_mode": source_mode,
            "call_id": self.call_id,
            "provider": self.provider,
            "model": self.model,
            "usage_available": self.usage_available,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "prompt_chars": self.prompt_chars,
            "completion_chars": self.completion_chars,
            "elapsed_ms": self.elapsed_ms,
            "wall_ms": self.elapsed_ms,
            "fallback_reason": fallback_reason,
            "metadata": self.metadata,
        }


class TokenUsageRecorder:
    """Append token usage records and maintain an aggregate summary."""

    def __init__(self, root: str | Path, *, paper_id: str):
        self.root = Path(root)
        self.paper_id = paper_id
        self.jsonl_path = self.root / "token_usage.jsonl"
        self.summary_path = self.root / "token_usage.summary.json"
        self._counter = 0

    def _next_call_id(self, stage: str, step: str) -> str:
        self._counter += 1
        return f"{stage}.{step}.{self._counter:04d}"

    def record(
        self,
        *,
        stage: str,
        step: str,
        provider: str | None = None,
        model: str | None = None,
        usage: Any = None,
        prompt: str | None = None,
        completion: str | None = None,
        elapsed_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TokenUsageRecord:
        norm = normalize_usage(usage)
        usage_available = any(v is not None for v in norm.values())
        rec = TokenUsageRecord(
            paper_id=self.paper_id,
            stage=stage,
            step=step,
            call_id=self._next_call_id(stage, step),
            provider=provider,
            model=model,
            usage_available=usage_available,
            input_tokens=norm["input_tokens"],
            output_tokens=norm["output_tokens"],
            total_tokens=norm["total_tokens"],
            prompt_chars=len(prompt) if prompt is not None else None,
            completion_chars=len(completion) if completion is not None else None,
            elapsed_ms=elapsed_ms,
            metadata=metadata or {},
        )
        self.root.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        return rec

    def record_event(
        self,
        *,
        stage: str,
        step: str,
        elapsed_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TokenUsageRecord:
        return self.record(
            stage=stage,
            step=step,
            elapsed_ms=elapsed_ms,
            metadata={"event": True, **(metadata or {})},
        )

    def records(self) -> list[dict[str, Any]]:
        if not self.jsonl_path.is_file():
            return []
        out = []
        for line in self.jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out.append(json.loads(line))
        return out

    def summary(self) -> dict[str, Any]:
        records = self.records()
        groups: dict[str, dict[str, Any]] = {}
        totals = {
            "records": len(records),
            "usage_records": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "elapsed_ms": 0,
            "usage_unavailable_records": 0,
        }
        for rec in records:
            key = "|".join(
                str(rec.get(k) or "") for k in ("stage", "step", "provider", "model")
            )
            group = groups.setdefault(
                key,
                {
                    "stage": rec.get("stage"),
                    "step": rec.get("step"),
                    "provider": rec.get("provider"),
                    "model": rec.get("model"),
                    "records": 0,
                    "usage_records": 0,
                    "usage_unavailable_records": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "elapsed_ms": 0,
                },
            )
            group["records"] += 1
            totals["elapsed_ms"] += int(rec.get("elapsed_ms") or 0)
            group["elapsed_ms"] += int(rec.get("elapsed_ms") or 0)
            if rec.get("usage_available"):
                totals["usage_records"] += 1
                group["usage_records"] += 1
                for field_name in ("input_tokens", "output_tokens", "total_tokens"):
                    val = int(rec.get(field_name) or 0)
                    totals[field_name] += val
                    group[field_name] += val
            else:
                totals["usage_unavailable_records"] += 1
                group["usage_unavailable_records"] += 1
        return {
            "paper_id": self.paper_id,
            "generated_at": _now_iso(),
            "totals": totals,
            "by_stage_step_provider_model": list(groups.values()),
        }

    def write_summary(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.summary_path


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


class RecordingLLMClient:
    """LLM client wrapper that records `.complete(...)` calls."""

    def __init__(
        self,
        client: Any,
        recorder: TokenUsageRecorder,
        *,
        step: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.client = client
        self.recorder = recorder
        self.step = step
        self.metadata = metadata or {}

    def complete(self, prompt: str, **kwargs: Any) -> str:
        start = time.perf_counter()
        result = self.client.complete(prompt, **kwargs)
        self.recorder.record(
            stage="build",
            step=self.step,
            provider=getattr(self.client, "provider", "anthropic"),
            model=kwargs.get("model", getattr(self.client, "model", None)),
            usage=getattr(self.client, "last_usage", None),
            prompt=prompt,
            completion=result,
            elapsed_ms=_elapsed_ms(start),
            metadata=self.metadata,
        )
        return result


class RecordingVLClient:
    """Multimodal client wrapper that records `.read_figure(...)` calls."""

    def __init__(
        self,
        client: Any,
        recorder: TokenUsageRecorder,
        *,
        step: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.client = client
        self.recorder = recorder
        self.step = step
        self.metadata = metadata or {}

    def read_figure(self, image_path: str | Path, prompt: str, **kwargs: Any) -> str:
        start = time.perf_counter()
        result = self.client.read_figure(image_path, prompt, **kwargs)
        self.recorder.record(
            stage="build",
            step=self.step,
            provider=getattr(self.client, "provider", "openai-compatible"),
            model=kwargs.get("model", getattr(self.client, "model", None)),
            usage=getattr(self.client, "last_usage", None),
            prompt=prompt,
            completion=result,
            elapsed_ms=_elapsed_ms(start),
            metadata={**self.metadata, "image_path": str(image_path)},
        )
        return result
