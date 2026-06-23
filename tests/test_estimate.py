"""Runtime estimation: nature of the claim -> wall-clock budget + poll cadence."""

from __future__ import annotations

from reproducegym.estimate import estimate_runtime


def test_training_L_allows_more_than_a_day():
    e = estimate_runtime(requires_training=True, cost="L")
    assert e.timeout_s > 24 * 3600  # real RL training can exceed a day
    assert e.poll_s >= 1800         # poll sparsely for very long runs


def test_training_scales_with_cost():
    s = estimate_runtime(requires_training=True, cost="S")
    m = estimate_runtime(requires_training=True, cost="M")
    l = estimate_runtime(requires_training=True, cost="L")
    assert s.timeout_s < m.timeout_s < l.timeout_s
    assert s.poll_s <= m.poll_s <= l.poll_s


def test_light_tasks_are_short_and_poll_often():
    light = estimate_runtime(requires_training=False, cost="M")
    train = estimate_runtime(requires_training=True, cost="M")
    assert light.timeout_s < train.timeout_s
    assert light.poll_s < train.poll_s


def test_unknown_cost_falls_back_to_M():
    assert estimate_runtime(requires_training=True, cost=None).timeout_s == \
        estimate_runtime(requires_training=True, cost="M").timeout_s
    assert estimate_runtime(requires_training=True, cost="weird").timeout_s == \
        estimate_runtime(requires_training=True, cost="M").timeout_s


def test_label_is_human_readable():
    e = estimate_runtime(requires_training=True, cost="L")
    assert "training" in e.label and "36h" in e.label and "min" in e.label
