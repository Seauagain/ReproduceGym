"""Thin clients for the three cloud-API backends. No local model / GPU.

Keys/endpoints come from .env (see .env.example):
  - mineru    : PDF -> Markdown + figures        (MINERU_*)
  - qwen_vl   : figure -> experimental params     (QWEN_*, OpenAI-compatible)
  - claude    : claim reasoning + build-task      (ANTHROPIC_*, relay; no /v1)

qwen_vl can reuse the openai SDK pointed at QWEN_BASE_URL. Stub only.
"""

from __future__ import annotations


class MinerUClient:
    def to_markdown(self, pdf_path) -> object:
        raise NotImplementedError("scaffold")


class QwenVLClient:
    def read_figure(self, image_path, prompt: str) -> object:
        raise NotImplementedError("scaffold")


class ClaudeClient:
    def complete(self, prompt: str, **kw) -> object:
        raise NotImplementedError("scaffold")
