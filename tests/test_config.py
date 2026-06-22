"""Unit tests for the .env loader (offline)."""

from __future__ import annotations

import os

from reproducegym.config import load_dotenv, parse_env_text


def test_parse_basic_and_quotes_and_comments():
    text = """
# a comment
ANTHROPIC_API_KEY=abc123
QWEN_BASE_URL="https://api.example.com/v1"
export FOO=bar  # trailing comment
EMPTY=
"""
    parsed = parse_env_text(text)
    assert parsed["ANTHROPIC_API_KEY"] == "abc123"
    assert parsed["QWEN_BASE_URL"] == "https://api.example.com/v1"
    assert parsed["FOO"] == "bar"
    assert parsed["EMPTY"] == ""


def test_parse_resolves_var_references():
    text = "MINERU_TOKEN=tok123\nMINERU_API_KEY=${MINERU_TOKEN}\n"
    parsed = parse_env_text(text)
    assert parsed["MINERU_API_KEY"] == "tok123"


def test_load_dotenv_missing_file_returns_empty(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_load_dotenv_no_override_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_TEST_KEY", "original")
    env = tmp_path / ".env"
    env.write_text("REPRO_TEST_KEY=fromfile\nREPRO_NEW_KEY=new\n", encoding="utf-8")
    load_dotenv(env)
    assert os.environ["REPRO_TEST_KEY"] == "original"
    assert os.environ["REPRO_NEW_KEY"] == "new"


def test_load_dotenv_override(tmp_path, monkeypatch):
    monkeypatch.setenv("REPRO_TEST_KEY2", "original")
    env = tmp_path / ".env"
    env.write_text("REPRO_TEST_KEY2=fromfile\n", encoding="utf-8")
    load_dotenv(env, override=True)
    assert os.environ["REPRO_TEST_KEY2"] == "fromfile"
