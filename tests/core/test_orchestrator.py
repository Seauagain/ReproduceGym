"""Legacy orchestrator entrypoints are disabled in the two-stage workflow."""

from __future__ import annotations

import pytest

from reproducegym.orchestrator import ReproduceError, build_task, reproduce

PAPER_MD = """# A Critical Perspective on R1-Zero Training

We show that removing std normalization removes the length bias.
"""


def test_legacy_build_task_is_disabled():
    with pytest.raises(ReproduceError, match="build_claim_tasks"):
        build_task(PAPER_MD)


def test_legacy_reproduce_is_disabled():
    with pytest.raises(ReproduceError, match="Build tasks first"):
        reproduce(PAPER_MD)
