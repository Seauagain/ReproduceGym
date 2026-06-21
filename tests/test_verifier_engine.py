"""Recompute verifier engine: safe formula eval + artifact-only verdict derivation.

The whole point of this module is that scoring is RECOMPUTED from artifacts and
never trusts an agent's self-reported verdict/score.
"""

from __future__ import annotations

import json

import pytest

from reprogym.verifier.engine import (
    VerifierError,
    parse_window,
    recompute,
    resolve_series,
    safe_eval,
)


# --------------------------------------------------------------------------- #
# series resolution + windowing
# --------------------------------------------------------------------------- #
def _rows():
    rows = []
    for i in range(3):
        rows.append({"condition": "baseline", "step": str(i), "len": str(100 + i)})
    for i in range(3):
        rows.append({"condition": "treatment", "step": str(i), "len": str(70 + i)})
    return rows


def test_resolve_series_filters_by_condition():
    assert resolve_series(_rows(), "treatment.len") == [70.0, 71.0, 72.0]
    assert resolve_series(_rows(), "baseline.len") == [100.0, 101.0, 102.0]


def test_resolve_series_bare_column_uses_all_rows():
    assert resolve_series(_rows(), "step") == [0, 1, 2, 0, 1, 2]


def test_resolve_series_window_last():
    assert resolve_series(_rows(), "baseline.len", window="last_2_steps") == [101.0, 102.0]


def test_resolve_series_unknown_condition_raises():
    with pytest.raises(VerifierError):
        resolve_series(_rows(), "missing.len")


def test_resolve_series_unknown_column_raises():
    with pytest.raises(VerifierError):
        resolve_series(_rows(), "treatment.nope")


def test_parse_window():
    assert parse_window("last_50_steps") == ("last", 50)
    assert parse_window("first_20") == ("first", 20)
    assert parse_window(None) is None
    assert parse_window("all") is None


# --------------------------------------------------------------------------- #
# formula evaluation
# --------------------------------------------------------------------------- #
def test_safe_eval_ratio_of_means():
    val = safe_eval("mean(treatment.len) / mean(baseline.len)", _rows())
    assert val == pytest.approx(71.0 / 101.0)


def test_safe_eval_aggregations():
    assert safe_eval("max(baseline.len)", _rows()) == 102.0
    assert safe_eval("min(treatment.len)", _rows()) == 70.0
    assert safe_eval("last(baseline.len)", _rows()) == 102.0
    assert safe_eval("abs(mean(treatment.len) - mean(baseline.len))", _rows()) == pytest.approx(30.0)


def test_safe_eval_rejects_bare_series_in_arithmetic():
    with pytest.raises(VerifierError):
        safe_eval("treatment.len / baseline.len", _rows())


def test_safe_eval_rejects_unknown_function():
    with pytest.raises(VerifierError):
        safe_eval("median(baseline.len) + sqrt(treatment.len)", _rows())


def test_safe_eval_rejects_attribute_injection():
    for evil in [
        "__import__('os').system('echo hi')",
        "().__class__.__bases__",
        "mean(os.len)",  # os is just a condition label, but no rows -> error
    ]:
        with pytest.raises(VerifierError):
            safe_eval(evil, _rows())


def test_safe_eval_division_by_zero():
    rows = [{"condition": "a", "len": "0"}, {"condition": "b", "len": "5"}]
    with pytest.raises(VerifierError):
        safe_eval("mean(b.len) / mean(a.len)", rows)


# --------------------------------------------------------------------------- #
# recompute: verdict derived from artifacts only
# --------------------------------------------------------------------------- #
SPEC = {
    "claim_id": "c1_demo",
    "metrics": ["length_ratio"],
    "formulas": {"length_ratio": "mean(treatment.len) / mean(baseline.len)"},
    "directions": {"length_ratio": "lower_is_better"},
    "windows": {"length_ratio": "last_50_steps"},
    "thresholds": {"length_ratio": 0.8},
    "required_files": ["output/result.json", "output/metrics.csv"],
    "metrics_csv": "output/metrics.csv",
    "metrics_csv_columns": ["condition", "step", "len"],
    "condition_col": "condition",
    "conditions": ["baseline", "treatment"],
    "min_rows_per_condition": 50,
    "verdicts": ["reproduced", "failed", "inconclusive", "invalid"],
    "reward_by_verdict": {"reproduced": 0.8, "failed": 0.35, "inconclusive": 0.2, "invalid": 0.0},
}


def _make_workspace(tmp_path, *, baseline_len, treatment_len, n=50, result_verdict=None):
    ws = tmp_path / "ws"
    (ws / "output").mkdir(parents=True)
    lines = ["condition,step,len"]
    for i in range(n):
        lines.append(f"baseline,{i},{baseline_len}")
    for i in range(n):
        lines.append(f"treatment,{i},{treatment_len}")
    (ws / "output" / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {} if result_verdict is None else {"verdict": result_verdict, "reward": 1.0}
    (ws / "output" / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    return ws


def test_recompute_reproduced_when_metric_passes(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=70)  # ratio 0.7 <= 0.8
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "reproduced"
    assert rep["reward"] == 0.8
    assert rep["metrics"]["length_ratio"]["passed"] is True
    assert rep["scored_by"] == "reprogym-recompute"


def test_recompute_failed_when_metric_misses(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=95)  # ratio 0.95 > 0.8
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "failed"
    assert rep["reward"] == 0.35


def test_recompute_ignores_agent_self_reported_verdict(tmp_path):
    # Agent LIES: result.json claims reproduced+reward 1.0, but the data fails.
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=95, result_verdict="reproduced")
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "failed"  # derived from artifacts, not from result.json
    assert rep["reward"] == 0.35


def test_recompute_invalid_when_required_file_missing(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=70)
    (ws / "output" / "result.json").unlink()
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "invalid"
    assert rep["reward"] == 0.0


def test_recompute_invalid_when_column_missing(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=70)
    (ws / "output" / "metrics.csv").write_text("condition,step\nbaseline,0\n", encoding="utf-8")
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "invalid"


def test_recompute_inconclusive_when_too_few_rows(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=70, n=10)  # < 50
    rep = recompute(ws, SPEC)
    assert rep["verdict"] == "inconclusive"
    assert rep["reward"] == 0.2


def test_recompute_inconclusive_when_formula_unevaluable(tmp_path):
    ws = _make_workspace(tmp_path, baseline_len=100, treatment_len=70)
    bad = dict(SPEC, formulas={"length_ratio": "mean(treatment.len) / mean(ghost.len)"})
    rep = recompute(ws, bad)
    assert rep["verdict"] == "inconclusive"
    assert any("length_ratio" in e for e in rep["errors"])
