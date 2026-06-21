"""Shared fixtures for ReproGym tests."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def valid_claim_spec() -> dict:
    """A minimal-but-complete claim spec that passes schema validation."""
    return copy.deepcopy(
        {
            "claim_id": "c1_demo",
            "paper_id": "demo-0001",
            "claim_type": "mechanism",
            "requires_training": True,
            "cost": "M",
            "verifiability": "high",
            "tier": "T2_proxy",
            "exposure_policy": "v0_full_paper_public",
            "statement": "Removing the bias term reduces the response-length growth.",
            "anchors": [
                {"kind": "figure", "ref": "Fig. 4", "note": "length-vs-step curves"},
            ],
            "conditions": [
                {"label": "baseline", "description": "with bias term"},
                {"label": "treatment", "description": "bias term removed"},
            ],
            "matched_variables": ["seed", "batch_size"],
            "params": [
                {
                    "name": "learning_rate",
                    "value": 1e-6,
                    "source": "Sec. 3.1",
                    "status": "paper_specified",
                    "exposure": "visible",
                },
            ],
            "metrics": [
                {
                    "name": "length_ratio",
                    "formula": "mean(treatment.len) / mean(baseline.len)",
                    "direction": "lower_is_better",
                    "window": "last_50_steps",
                },
            ],
            "thresholds": [
                {"metric": "length_ratio", "pass_threshold": 0.8, "exposure": "hidden"},
            ],
            "required_outputs": {
                "files": ["output/result.json", "output/metrics.csv"],
                "metrics_csv_columns": ["condition", "step", "len"],
                "min_rows_per_condition": 50,
            },
            "verdict_rules": {
                "reproduced": ["length_ratio meets pass_threshold"],
                "failed": ["length_ratio misses pass_threshold"],
            },
            "reward": {
                "base_by_verdict": {"reproduced": 0.8, "failed": 0.35, "invalid": 0.0},
                "evidence_bonus_cap": 0.2,
            },
        }
    )


class FakeLLM:
    """A deterministic LLM stand-in. Records the last prompt it was given."""

    def __init__(self, response: str):
        self._response = response
        self.calls: list[str] = []

    def complete(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        return self._response

    @property
    def last_prompt(self) -> str:
        return self.calls[-1]


@pytest.fixture
def make_llm():
    def _make(response: str) -> FakeLLM:
        return FakeLLM(response)

    return _make
