"""Thin clients for the cloud-API backends. No local model / GPU.

Keys/endpoints come from .env (see .env.example), loaded via reproducegym.config:
  - claude      : claim reasoning + triage   (ANTHROPIC_*, relay base_url; no /v1)
  - multimodal : figure -> experimental evidence (MULTIMODAL_* or QWEN_* fallback)
  - mineru      : PDF -> Markdown + figures   (mineru-open-api CLI; see pipeline.parse)

SDKs are imported lazily so importing this module (and running unit tests with
fake clients) never requires the network or the optional dependencies.
"""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from reproducegym.config import dotenv_values, require_dotenv


class ClaudeClient:
    """Anthropic-API client (works against the gpugeek relay base_url)."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 16384,
    ):
        env = dotenv_values()
        # Model API settings are read from the repo .env file only. Never fall
        # back to process environment variables; stale shell ANTHROPIC_* values
        # can silently route requests to the wrong relay/key.
        self.api_key = env.get("ANTHROPIC_API_KEY") or require_dotenv("ANTHROPIC_API_KEY")
        self.base_url = env.get("ANTHROPIC_BASE_URL")
        self.model = env.get("ANTHROPIC_DEFAULT_OPUS_MODEL") or env.get(
            "ANTHROPIC_DEFAULT_SONNET_MODEL"
        )
        # Claim extraction over a full paper can emit ~8-12k tokens of JSON; an
        # 8k cap silently truncates -> invalid JSON. Allow an env override.
        env_cap = env.get("REPRODUCEGYM_MAX_OUTPUT_TOKENS")
        self.max_tokens = int(env_cap) if env_cap else max_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic  # lazy

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, prompt: str, *, system: str | None = None, **kwargs: Any) -> str:
        client = self._ensure_client()
        resp = client.messages.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system or "You are a careful research-reproduction assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )


def multimodal_figure_configured() -> bool:
    """Whether an OpenAI-compatible vision model is configured."""

    env = dotenv_values()
    api_key = env.get("MULTIMODAL_API_KEY") or env.get("VISION_API_KEY") or env.get("QWEN_API_KEY")
    base_url = env.get("MULTIMODAL_BASE_URL") or env.get("VISION_BASE_URL") or env.get("QWEN_BASE_URL")
    model = (
        env.get("MULTIMODAL_VISION_MODEL")
        or env.get("VISION_MODEL")
        or env.get("QWEN_VL_MODEL")
    )
    return bool(api_key and base_url and model)


class MultimodalFigureClient:
    """OpenAI-compatible multimodal client for figure evidence extraction."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int | None = None,
    ):
        env = dotenv_values()
        self.api_key = (
            env.get("MULTIMODAL_API_KEY") or env.get("VISION_API_KEY") or require_dotenv("QWEN_API_KEY")
        )
        self.base_url = (
            env.get("MULTIMODAL_BASE_URL") or env.get("VISION_BASE_URL") or env.get("QWEN_BASE_URL")
        )
        self.model = (
            env.get("MULTIMODAL_VISION_MODEL") or env.get("VISION_MODEL") or env.get("QWEN_VL_MODEL")
        )
        if not self.model:
            raise RuntimeError(
                "multimodal vision model is not set (MULTIMODAL_VISION_MODEL, VISION_MODEL, or QWEN_VL_MODEL)"
            )
        env_max = env.get("MULTIMODAL_MAX_TOKENS") or env.get("VISION_MAX_TOKENS") or env.get("QWEN_VL_MAX_TOKENS")
        self.max_tokens = max_tokens or (int(env_max) if env_max else 16384)
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @staticmethod
    def _data_uri(image_path: str | Path) -> str:
        path = Path(image_path)
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{b64}"

    def read_figure(self, image_path: str | Path, prompt: str, **kwargs: Any) -> str:
        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=kwargs.get("model", self.model),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": self._data_uri(image_path)}},
                    ],
                }
            ],
        )
        return resp.choices[0].message.content or ""


QwenVLClient = MultimodalFigureClient


class MinerUClient:
    """PDF -> Markdown via the mineru-open-api CLI (see pipeline.parse)."""

    def to_markdown(self, pdf_path) -> object:
        raise NotImplementedError("use pipeline.parse (mineru-open-api CLI)")
