"""Synthesize verifier grounding from extracted claim specs.

The extractor can recover paper targets as hidden params before it knows how the
verifier will score them. This module performs the narrow deterministic bridge:

1. hidden numeric target params that can be *safely* bound to a metric become
   numeric thresholds (with audit evidence + tolerance);
2. metrics whose formula is a cross-condition comparison (a ratio/difference of
   two or more conditions) but that carry no absolute paper number may get a
   public no-effect threshold for diagnostics, but they stay out of the RLVR pool
   because reward curves require paper-grounded target values;
3. anything that still cannot be bound stays in the ``exploration`` pool with an
   explicit reason, instead of silently producing a degenerate RLVR task.

Two guardrails keep the heuristic binding honest:

- relative/absolute class guard: a target that reads as a ratio/reduction must
  not be bound to an absolute-magnitude metric (and vice versa);
- ambiguity guard: if several distinct target values would map to one metric by
  name heuristic, none are bound -- guessing is worse than routing to
  exploration.
"""

from __future__ import annotations

import ast
import copy
import re
from typing import Any

from reproducegym.verifier.engine import VerifierError, safe_eval

DEFAULT_TEXT_TARGET_REL_TOLERANCE = 0.15
DEFAULT_FIGURE_TARGET_REL_TOLERANCE = 0.30
RLVR_MODES = {"numeric_threshold", "directional", "structural"}

# Tokens that mark a value as a *relative/derived* quantity (a ratio between
# conditions, a reduction, a delta) rather than an absolute metric magnitude.
RELATIVE_TOKENS = ("ratio", "relative", "reduction", "delta", "speedup")


def _norm_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _has_relative_token(text: Any) -> bool:
    low = str(text or "").lower()
    return any(tok in low for tok in RELATIVE_TOKENS)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "")
    if text.endswith("%"):
        text = text[:-1].strip()
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text):
        return None
    return float(text)


def _formula_refs(formula: Any) -> tuple[set[tuple[str, str]], set[str]]:
    """Return ({(condition, column)}, {bare_column}) referenced by a metric formula.

    Mirrors the verifier grammar: series must be wrapped in an aggregation, so we
    only look at the single argument of each call. Unparseable formulas yield
    empty sets (treated as non-comparative).
    """
    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError:
        return set(), set()
    attr_refs: set[tuple[str, str]] = set()
    bare: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            arg = node.args[0]
            if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
                attr_refs.add((arg.value.id, arg.attr))
            elif isinstance(arg, ast.Name):
                bare.add(arg.id)
    return attr_refs, bare


def _metric_is_relative(metric: dict[str, Any]) -> bool:
    """A metric is relative/derived if its name says so or its formula compares
    two or more conditions (a ratio/difference is not an absolute magnitude)."""
    if _has_relative_token(metric.get("name")):
        return True
    attr_refs, _ = _formula_refs(metric.get("formula"))
    return len({label for label, _ in attr_refs}) >= 2


def _target_is_relative(param: dict[str, Any]) -> bool:
    return _has_relative_token(param.get("name")) or _has_relative_token(param.get("read_from"))


def _tokens(value: Any) -> set[str]:
    text = str(value or "").lower()
    raw = [tok for tok in re.split(r"[^a-z0-9]+", text) if tok]
    synonyms = {
        "output": "response",
        "outputs": "response",
        "avg": "average",
        "benchmark": "score",
        "pass1": "accuracy",
        "pass": "accuracy",
    }
    out = {synonyms.get(tok, tok) for tok in raw}
    if "length" in out:
        out.add("response_length")
    if "incorrect" in out:
        out.add("wrong")
    if "correct" in out:
        out.add("right")
    return out


def _text_blob(*values: Any) -> str:
    return " ".join(str(v or "") for v in values).lower()


def _metric_blob(metric: dict[str, Any]) -> str:
    return _text_blob(metric.get("name"), metric.get("formula"), metric.get("direction"))


def _semantic_bind_score(param: dict[str, Any], metric: dict[str, Any]) -> int:
    p_blob = _text_blob(
        param.get("name"),
        param.get("metric"),
        param.get("condition"),
        param.get("read_from"),
        param.get("source"),
    )
    m_blob = _metric_blob(metric)
    p_tokens = _tokens(p_blob)
    m_tokens = _tokens(m_blob)
    score = len(p_tokens & m_tokens)
    if "length" in p_tokens and "length" in m_tokens and "ratio" in m_tokens:
        score += 3
    if "incorrect" in p_tokens and "incorrect" in m_tokens:
        score += 5
    elif "incorrect" in p_tokens and "incorrect" not in m_tokens:
        score -= 4
    if "correct" in p_tokens and "correct" in m_tokens:
        score += 3
    elif "correct" in p_tokens and "incorrect" in m_tokens:
        score -= 4
    if ("panel 2" in p_blob or "plot 2" in p_blob or "subplot 2" in p_blob) and (
        "overall" in m_tokens or "mean_response_length" in m_blob
    ):
        score += 6
    if ("panel 4" in p_blob or "plot 4" in p_blob or "subplot 4" in p_blob) and "incorrect" in m_tokens:
        score += 6
    if {"accuracy", "score"} & p_tokens and {"accuracy", "score", "difference"} & m_tokens:
        score += 2
    return score


def _bind_metric(param: dict[str, Any], metrics: list[dict[str, Any]]) -> tuple[str | None, bool, int]:
    """Return (metric_name, is_explicit, score). Explicit means the param named the metric."""
    metric_names = [str(m["name"]) for m in metrics if m.get("name")]
    explicit = param.get("metric")
    if explicit in metric_names:
        return str(explicit), True, 100
    if len(metric_names) == 1:
        return metric_names[0], False, 1

    target_name = _norm_name(str(param.get("name", "")))
    for name in metric_names:
        norm = _norm_name(name)
        if norm and (norm in target_name or target_name in norm):
            return name, False, 20

    scored = [
        (str(metric["name"]), _semantic_bind_score(param, metric))
        for metric in metrics
        if metric.get("name")
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    if scored and scored[0][1] >= 5 and (len(scored) == 1 or scored[0][1] > scored[1][1]):
        return scored[0][0], False, scored[0][1]
    return None, False, 0


def _rel_tolerance(param: dict[str, Any]) -> float:
    tol = param.get("tolerance")
    if isinstance(tol, dict):
        rel = _numeric(tol.get("rel"))
        if rel is not None and rel >= 0:
            return rel
    else:
        rel = _numeric(tol)
        if rel is not None and rel >= 0:
            return rel

    source = str(param.get("source") or "").lower()
    read_from = str(param.get("read_from") or "").lower()
    confidence = _numeric(param.get("confidence"))
    figure_like = "fig" in source or "visual" in read_from or (
        confidence is not None and confidence < 0.85
    )
    return DEFAULT_FIGURE_TARGET_REL_TOLERANCE if figure_like else DEFAULT_TEXT_TARGET_REL_TOLERANCE


def _pass_threshold(target: float, direction: str, rel_tol: float) -> float:
    margin = abs(target) * rel_tol
    if direction == "lower_is_better":
        value = target + margin
    else:
        value = target - margin
    return float(f"{value:.12g}")


def _threshold_for_target(
    param: dict[str, Any],
    *,
    metric: dict[str, Any],
) -> dict[str, Any] | None:
    target = _target_value_for_metric(param, metric)
    if target is None:
        return None
    rel_tol = _rel_tolerance(param)
    direction = str(metric.get("direction") or "higher_is_better")
    evidence = {
        k: v
        for k, v in {
            "param_name": param.get("name"),
            "source": param.get("source"),
            "read_from": param.get("read_from"),
            "confidence": param.get("confidence"),
        }.items()
        if v is not None
    }
    threshold = {
        "metric": metric["name"],
        "pass_threshold": _pass_threshold(target, direction, rel_tol),
        "target_value": target,
        "tolerance_abs": abs(target) * rel_tol,
        "exposure": param.get("exposure") or "hidden",
        "target_evidence": evidence,
        "rationale": (
            f"paper target {target:g} from {param.get('source', 'unknown source')} "
            f"with rel_tolerance={rel_tol:g}"
        ),
        "source": param.get("source"),
        "confidence": param.get("confidence"),
        "tolerance": {"rel": rel_tol},
    }
    return {k: v for k, v in threshold.items() if v is not None}


def _target_value_for_metric(param: dict[str, Any], metric: dict[str, Any]) -> float | None:
    target = _numeric(param.get("value"))
    if _metric_is_relative(metric):
        text = _text_blob(param.get("read_from"), param.get("name"), param.get("metric"))
        matches = re.findall(r"(?:ratio|relative(?:\\s+height)?|dr\\.?\\s*grpo\\s*/\\s*grpo)[^0-9]{0,20}([0-9]+(?:\\.[0-9]+)?)", text)
        candidates = [_numeric(m) for m in matches]
        candidates = [c for c in candidates if c is not None and 0 < c <= 5]
        if candidates:
            return float(candidates[-1])
    return target


def _neutral_point(metric: dict[str, Any]) -> float | None:
    """The metric value when all compared conditions are equal (no effect).

    Returns None unless the formula compares >=2 distinct conditions. By probing
    the formula with identical per-condition data we get the comparison's neutral
    point uniformly: ratio -> 1.0, difference -> 0.0, relative reduction -> 0.0.
    """
    attr_refs, bare = _formula_refs(metric.get("formula"))
    labels = {label for label, _ in attr_refs}
    if len(labels) < 2:
        return None
    rows: list[dict[str, Any]] = []
    constant = 1.0
    for label, column in attr_refs:
        rows.append({"condition": label, column: constant})
    for row in rows:
        for column in bare:
            row.setdefault(column, constant)
    try:
        return safe_eval(str(metric.get("formula")), rows, "condition", metric.get("window"))
    except VerifierError:
        return None


def _directional_threshold(metric: dict[str, Any]) -> dict[str, Any] | None:
    neutral = _neutral_point(metric)
    if neutral is None:
        return None
    direction = str(metric.get("direction") or "higher_is_better")
    relation = "<=" if direction == "lower_is_better" else ">="
    # The neutral point (ratio 1.0 / difference 0.0) is structural -- it follows
    # from the public claim direction and reveals no paper answer-key number, so
    # it is exposed as a visible pass criterion rather than a hidden target.
    return {
        "metric": metric["name"],
        "pass_threshold": float(f"{neutral:.12g}"),
        "exposure": "visible",
        "source": "directional comparison across conditions",
        "rationale": (
            f"directional claim: recomputed {metric['name']} must be {relation} {neutral:g} "
            "(the no-effect point where the compared conditions are equal)"
        ),
    }


def _bind_targets(
    params: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    metric_by_name: dict[str, dict[str, Any]],
    already_bound: set[str],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Bind hidden numeric target params to metrics, with safety guardrails.

    Returns (metric -> chosen param, rejection_reasons).
    """
    per_metric: dict[str, list[tuple[dict[str, Any], bool, int]]] = {}
    rejected: list[str] = []

    for param in params:
        if param.get("use") != "target":
            continue
        if _numeric(param.get("value")) is None:
            continue
        metric_name, explicit, score = _bind_metric(param, metrics)
        if not metric_name:
            rejected.append(f"target {param.get('name')!r} did not bind to any metric")
            continue
        if metric_name in already_bound:
            continue
        if _target_is_relative(param) != _metric_is_relative(metric_by_name[metric_name]):
            rejected.append(
                f"target {param.get('name')!r} rejected: relative/absolute mismatch with "
                f"metric {metric_name!r}"
            )
            continue
        per_metric.setdefault(metric_name, []).append((param, explicit, score))

    bindings: dict[str, dict[str, Any]] = {}
    for metric_name, candidates in per_metric.items():
        candidates = sorted(candidates, key=lambda item: item[2], reverse=True)
        explicit = [p for p, is_explicit, _ in candidates if is_explicit]
        if explicit:
            bindings[metric_name] = explicit[0]
            continue
        best_score = candidates[0][2]
        best = [p for p, _, score in candidates if score == best_score]
        distinct = {_target_value_for_metric(p, metric_by_name[metric_name]) for p in best}
        if len(distinct) > 1:
            rejected.append(
                f"metric {metric_name!r} has ambiguous targets {sorted(distinct)}; not bound"
            )
            continue
        bindings[metric_name] = best[0]
    return bindings, rejected


def _default_verdict_rules(threshold_metrics: list[str]) -> dict[str, list[str]]:
    if not threshold_metrics:
        return {}
    joined = ", ".join(threshold_metrics)
    return {
        "reproduced": [f"all primary metrics meet their thresholds: {joined}"],
        "failed": [f"at least one primary metric misses its threshold: {joined}"],
        "inconclusive": ["required outputs exist but a metric cannot be recomputed"],
        "invalid": ["required output files are missing or malformed"],
    }


def apply_verification_contract(spec: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with synthesized thresholds + verification/pool metadata."""
    out = copy.deepcopy(spec)
    metrics = list(out.get("metrics") or [])
    metric_by_name = {str(m["name"]): m for m in metrics if m.get("name")}
    metric_names = list(metric_by_name)

    thresholds = [dict(t) for t in out.get("thresholds") or []]
    seen = {t.get("metric") for t in thresholds}

    # 1) Bind hidden numeric target params -> numeric thresholds (guarded).
    bindings, rejected = _bind_targets(out.get("params") or [], metrics, metric_by_name, seen)
    for metric_name, param in bindings.items():
        threshold = _threshold_for_target(param, metric=metric_by_name[metric_name])
        if threshold is None:
            continue
        thresholds.append(threshold)
        seen.add(metric_name)

    # 2) Directional fallback: comparative metrics with no absolute number get a
    #    no-effect threshold so the qualitative claim is still scorable.
    directional_metrics: set[str] = set()
    for metric_name, metric in metric_by_name.items():
        if metric_name in seen:
            continue
        threshold = _directional_threshold(metric)
        if threshold is None:
            continue
        thresholds.append(threshold)
        seen.add(metric_name)
        directional_metrics.add(metric_name)

    out["thresholds"] = thresholds

    threshold_metrics = [str(t["metric"]) for t in thresholds if t.get("metric") in metric_by_name]
    covers_all_metrics = bool(metric_names) and set(metric_names).issubset(set(threshold_metrics))

    if not out.get("verdict_rules"):
        out["verdict_rules"] = _default_verdict_rules(threshold_metrics)

    verification = dict(out.get("verification") or {})
    target_metrics = {
        str(t["metric"])
        for t in thresholds
        if t.get("metric") in metric_by_name and t.get("target_value") is not None
    }
    covers_all_targets = bool(metric_names) and set(metric_names).issubset(target_metrics)

    if covers_all_metrics and covers_all_targets:
        verification.setdefault("mode", "numeric_threshold")
        verification.setdefault("pool", "rlvr")
    elif target_metrics:
        kept = [metric for metric in metrics if metric.get("name") in target_metrics]
        kept_names = {str(metric["name"]) for metric in kept if metric.get("name")}
        diagnostic = [metric for metric in metrics if metric.get("name") not in kept_names]
        out["metrics"] = kept
        out["diagnostic_metrics"] = diagnostic
        thresholds = [
            threshold
            for threshold in thresholds
            if threshold.get("metric") in kept_names and threshold.get("target_value") is not None
        ]
        out["thresholds"] = thresholds
        threshold_metrics = [str(t["metric"]) for t in thresholds if t.get("metric") in kept_names]
        out["verdict_rules"] = _default_verdict_rules(threshold_metrics)
        verification.setdefault("mode", "numeric_threshold")
        verification.setdefault("pool", "rlvr")
        if diagnostic:
            verification["diagnostic_reason"] = (
                "ungrounded metric(s) moved to diagnostics: "
                + ", ".join(str(metric.get("name")) for metric in diagnostic if metric.get("name"))
            )
    else:
        verification["mode"] = verification.get("mode") or "unverifiable"
        verification["pool"] = "exploration"
        reason_bits: list[str] = []
        missing = sorted(set(metric_names) - set(threshold_metrics))
        missing_targets = sorted(set(metric_names) - target_metrics)
        if missing:
            reason_bits.append(
                "missing executable thresholds for primary metrics: " + ", ".join(missing)
            )
        if missing_targets:
            reason_bits.append(
                "missing paper-grounded target_value for primary metrics: "
                + ", ".join(missing_targets)
            )
        elif not metric_names:
            reason_bits.append("claim has no primary metrics")
        if directional_metrics:
            reason_bits.append(
                "directional-only threshold(s) are diagnostic, not strong RLVR targets: "
                + ", ".join(sorted(directional_metrics))
            )
        reason_bits.extend(rejected)
        if reason_bits:
            verification["reason"] = "; ".join(reason_bits)

    verification.setdefault("targets_bound", threshold_metrics)
    out["verification"] = verification
    return out
