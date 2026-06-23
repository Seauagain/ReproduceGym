"""Model clients must read API settings from .env only."""

from __future__ import annotations

import reproducegym.config as config
from reproducegym.models import ClaudeClient, MultimodalFigureClient, multimodal_figure_configured


def test_claude_client_uses_dotenv_not_process_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "ANTHROPIC_API_KEY=file-key",
                "ANTHROPIC_BASE_URL=https://api.gpugeek.com/",
                "ANTHROPIC_DEFAULT_OPUS_MODEL=Vendor2/Claude-4.6-opus",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-shell-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://wrong.example")
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "wrong-model")

    client = ClaudeClient(api_key="ignored", base_url="ignored", model="ignored")
    assert client.api_key == "file-key"
    assert client.base_url == "https://api.gpugeek.com/"
    assert client.model == "Vendor2/Claude-4.6-opus"


def test_multimodal_client_uses_dotenv_not_process_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "QWEN_API_KEY=file-qwen-key",
                "QWEN_BASE_URL=https://dashscope.example/v1",
                "QWEN_VL_MODEL=qwen-vl-file",
                "QWEN_VL_MAX_TOKENS=1234",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "DEFAULT_ENV_PATH", env)
    monkeypatch.setenv("QWEN_API_KEY", "stale-qwen-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://wrong.example/v1")
    monkeypatch.setenv("QWEN_VL_MODEL", "wrong-qwen-model")

    assert multimodal_figure_configured() is True
    client = MultimodalFigureClient(api_key="ignored", base_url="ignored", model="ignored")
    assert client.api_key == "file-qwen-key"
    assert client.base_url == "https://dashscope.example/v1"
    assert client.model == "qwen-vl-file"
    assert client.max_tokens == 1234
