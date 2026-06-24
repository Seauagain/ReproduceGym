"""ReproduceGym recompute verifier engine (stdlib-only, embeddable).

Core principle: a claim is scored ONLY by recomputing its metrics from the
agent's raw reproduction artifact (``output/metrics.csv``) and comparing the
recomputed values against the hidden thresholds. The engine NEVER reads an
agent-declared verdict, score, ``strict_reproduction`` flag, or any other
self-reported judgement -- the verdict is derived here from the numbers.

The source between the ``REPRODUCEGYM ENGINE BEGIN/END`` markers is embedded verbatim
into each task's ``reward/check.py`` (which runs inside the sandbox-free scoring
step with no reproducegym import), so this block must stay stdlib-only and self
contained. ``render_check.py`` slices it out by the markers.

Formula grammar (safe AST subset over metrics.csv):
    aggregations: mean, sum, min, max, median, std, var, last, first, count
    abs(<scalar expr>); + - * / % ** ; numeric constants
    series refs: ``<condition>.<column>`` (rows where condition==label)
                 or bare ``<column>`` (all rows). A series MUST be wrapped in an
                 aggregation; bare series in arithmetic is rejected.
"""

from __future__ import annotations

import ast
import csv
import math
import statistics
from pathlib import Path

# === REPRODUCEGYM ENGINE BEGIN ===
_AGG = {
    "mean": lambda s: statistics.fmean(s),
    "sum": lambda s: float(sum(s)),
    "min": lambda s: float(min(s)),
    "max": lambda s: float(max(s)),
    "median": lambda s: float(statistics.median(s)),
    "std": lambda s: float(statistics.pstdev(s)),
    "var": lambda s: float(statistics.pvariance(s)),
    "last": lambda s: float(s[-1]),
    "first": lambda s: float(s[0]),
    "count": lambda s: float(len(s)),
}


class VerifierError(Exception):
    """A formula/data problem that makes a metric impossible to recompute."""


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def parse_window(window):
    """Return ('last'|'first', N) from a window string like 'last_50_steps', else None."""
    if not window:
        return None
    kind = "first" if "first" in str(window) else "last"
    num = ""
    for ch in str(window):
        if ch.isdigit():
            num += ch
        elif num:
            break
    return (kind, int(num)) if num else None


def _apply_window(values, window):
    w = parse_window(window)
    if not w:
        return values
    kind, n = w
    if n <= 0 or n >= len(values):
        return values
    return values[-n:] if kind == "last" else values[:n]


def resolve_series(rows, ref, condition_col="condition", window=None):
    """Resolve 'label.column' (filtered by condition) or 'column' (all rows) to floats."""
    if "." in ref:
        label, col = ref.split(".", 1)
        selected = [r for r in rows if (r.get(condition_col) or "").strip() == label]
        if not selected:
            raise VerifierError("no rows for condition %r" % label)
    else:
        col, selected = ref, rows
    out = []
    for r in selected:
        if col not in r:
            raise VerifierError("unknown column %r" % col)
        raw = r.get(col)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            raise VerifierError("non-numeric value %r in column %r" % (raw, col))
    out = _apply_window(out, window)
    if not out:
        raise VerifierError("empty series for %r" % ref)
    return out


def _ref_name(node):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return "%s.%s" % (node.value.id, node.attr)
    if isinstance(node, ast.Name):
        return node.id
    return None


def _eval_node(node, rows, condition_col, window):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, rows, condition_col, window)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise VerifierError("only numeric constants allowed")
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval_node(node.operand, rows, condition_col, window)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, rows, condition_col, window)
        right = _eval_node(node.right, rows, condition_col, window)
        op = node.op
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            if right == 0:
                raise VerifierError("division by zero in formula")
            return left / right
        if isinstance(op, ast.Mod):
            if right == 0:
                raise VerifierError("modulo by zero in formula")
            return left % right
        if isinstance(op, ast.Pow):
            if abs(right) > 8:
                raise VerifierError("exponent too large")
            return left ** right
        raise VerifierError("operator not allowed: %s" % type(op).__name__)
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            raise VerifierError("unsupported function call")
        fname = node.func.id
        if len(node.args) != 1:
            raise VerifierError("%s() takes exactly one argument" % fname)
        arg = node.args[0]
        if fname == "abs":
            return abs(_eval_node(arg, rows, condition_col, window))
        if fname not in _AGG:
            raise VerifierError("function not allowed: %s" % fname)
        ref = _ref_name(arg)
        if ref is None:
            raise VerifierError("%s() argument must be 'column' or 'condition.column'" % fname)
        return _AGG[fname](resolve_series(rows, ref, condition_col, window))
    if isinstance(node, (ast.Name, ast.Attribute)):
        raise VerifierError("series %r must be wrapped in an aggregation" % (_ref_name(node),))
    raise VerifierError("unsupported expression: %s" % type(node).__name__)


def safe_eval(formula, rows, condition_col="condition", window=None):
    """Evaluate a metric formula over metrics.csv rows. Raises VerifierError on any
    parse/data problem. Only the whitelisted AST subset above is permitted."""
    try:
        tree = ast.parse(str(formula), mode="eval")
    except SyntaxError as exc:
        raise VerifierError("cannot parse formula %r: %s" % (formula, exc))
    value = _eval_node(tree, rows, condition_col, window)
    if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
        raise VerifierError("formula did not yield a finite number")
    return float(value)


def _passes(value, threshold, direction):
    if direction == "higher_is_better":
        return value >= threshold
    return value <= threshold


def _as_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _continuous_score(value, details, direction):
    """Return a smooth [0,1] score from paper-target distance, if available.

    Verdicts still use pass_threshold. This score is the RL signal: exact-or-better
    matches and values within tolerance get 1.0; worse results decay as
    0.9 ** (error_beyond_tolerance / tolerance).
    """
    target = _as_float(details.get("target_value"))
    if target is None:
        return None
    tol = _as_float(details.get("tolerance_abs"))
    if tol is None:
        pass_threshold = _as_float(details.get("pass_threshold"))
        if pass_threshold is not None:
            tol = abs(pass_threshold - target)
    if tol is None or tol <= 0:
        tol = max(abs(target) * 0.15, 1e-12)

    if direction == "lower_is_better":
        err = max(0.0, float(value) - (target + tol))
    else:
        err = max(0.0, (target - tol) - float(value))
    return max(0.0, min(1.0, 0.9 ** (err / tol)))


def _curve_score(value, curve):
    """Piecewise-linear score from explicit reward curve points."""
    points = curve.get("points") or []
    parsed = []
    for point in points:
        x = _as_float(point.get("value"))
        y = _as_float(point.get("reward"))
        if x is None or y is None:
            continue
        parsed.append((float(x), max(0.0, min(1.0, float(y)))))
    if len(parsed) < 2:
        raise VerifierError("reward curve requires at least two numeric points")
    parsed.sort(key=lambda item: item[0])
    for (x0, _), (x1, _) in zip(parsed, parsed[1:]):
        if x0 == x1:
            raise VerifierError("reward curve has duplicate value point %r" % x0)
    value = float(value)
    if value <= parsed[0][0]:
        return parsed[0][1]
    if value >= parsed[-1][0]:
        return parsed[-1][1]
    for (x0, y0), (x1, y1) in zip(parsed, parsed[1:]):
        if x0 <= value <= x1:
            frac = (value - x0) / (x1 - x0)
            return max(0.0, min(1.0, y0 + frac * (y1 - y0)))
    raise VerifierError("reward curve interpolation failed")


def _aggregate_scores(scores, spec):
    if not scores:
        return None
    reward_cfg = spec.get("reward") or {}
    aggregation = reward_cfg.get("aggregation") or "min"
    if aggregation == "weighted_mean":
        weights = reward_cfg.get("weights") or {}
        total_w = 0.0
        total = 0.0
        for name, score in scores.items():
            w = _as_float(weights.get(name), 1.0)
            if w is None or w <= 0:
                continue
            total_w += w
            total += w * score
        return (total / total_w) if total_w else None
    if aggregation == "mean":
        return sum(scores.values()) / len(scores)
    return min(scores.values())


def recompute(workspace, spec):
    """Score a finished workspace by recomputing metrics from artifacts.

    Returns {reward, verdict, metrics, errors, scored_by}. The verdict is derived
    from recomputed numbers only; no agent self-report is consulted.
    """
    workspace = Path(workspace)
    rbv = spec.get("reward_by_verdict", {})

    def out(verdict, metrics=None, errors=None):
        reward = max(0.0, min(1.0, float(rbv.get(verdict, 0.0))))
        return {
            "reward": round(reward, 6),
            "verdict": verdict,
            "metrics": metrics or {},
            "errors": errors or [],
            "scored_by": "reproducegym-recompute",
        }

    missing = [f for f in spec.get("required_files", []) if not (workspace / f).exists()]
    if missing:
        return out("invalid", errors=["missing required file: " + m for m in missing])

    csv_rel = spec.get("metrics_csv", "output/metrics.csv")
    try:
        rows = load_rows(workspace / csv_rel)
    except Exception as exc:  # noqa: BLE001
        return out("invalid", errors=["unreadable %s: %s" % (csv_rel, exc)])
    if not rows:
        return out("inconclusive", errors=["%s has no data rows" % csv_rel])

    header = set(rows[0].keys())
    miss_cols = [c for c in spec.get("metrics_csv_columns", []) if c not in header]
    if miss_cols:
        return out("invalid", errors=["metrics.csv missing columns: " + ", ".join(miss_cols)])

    condition_col = spec.get("condition_col", "condition")
    min_rows = spec.get("min_rows_per_condition")
    errors = []
    if min_rows:
        for label in spec.get("conditions", []):
            n = sum(1 for r in rows if (r.get(condition_col) or "").strip() == label)
            if n < min_rows:
                errors.append("condition %r has %d rows (< %d required)" % (label, n, min_rows))
    if errors:
        return out("inconclusive", errors=errors)

    formulas = spec.get("formulas", {})
    directions = spec.get("directions", {})
    windows = spec.get("windows", {})
    thresholds = spec.get("thresholds", {})
    threshold_details = spec.get("threshold_details", {})
    reward_curves = spec.get("reward_curves", {}) or {}
    metrics = spec.get("metrics", [])
    if not metrics:
        return out("inconclusive", errors=["no primary metrics configured"])

    report = {}
    all_pass = True
    metric_scores = []
    curve_scores = {}
    for name in metrics:
        if name not in thresholds:
            errors.append("no threshold for metric %r" % name)
            continue
        try:
            value = safe_eval(formulas.get(name, ""), rows, condition_col, windows.get(name))
        except VerifierError as exc:
            errors.append("metric %r: %s" % (name, exc))
            continue
        direction = directions.get(name, "higher_is_better")
        passed = _passes(value, thresholds[name], direction)
        all_pass = all_pass and passed
        if reward_curves:
            curve = reward_curves.get(name)
            if not curve:
                errors.append("no reward curve for metric %r" % name)
                continue
            try:
                score = _curve_score(value, curve)
            except VerifierError as exc:
                errors.append("metric %r reward curve: %s" % (name, exc))
                continue
            curve_scores[name] = score
        else:
            score = _continuous_score(
                value,
                threshold_details.get(name, {"pass_threshold": thresholds[name]}),
                direction,
            )
            if score is not None:
                metric_scores.append(score)
        report[name] = {
            "value": round(value, 6),
            "pass_threshold": thresholds[name],
            "direction": direction,
            "passed": passed,
        }
        if score is not None:
            report[name]["reward"] = round(score, 6)

    if errors:
        return out("inconclusive", metrics=report, errors=errors)
    verdict = "reproduced" if all_pass else "failed"
    if reward_curves:
        shaped = _aggregate_scores(curve_scores, spec)
        if shaped is None:
            return out("inconclusive", metrics=report, errors=["no reward curves scored"])
        return {
            "reward": round(max(0.0, min(1.0, shaped)), 6),
            "verdict": verdict,
            "metrics": report,
            "errors": [],
            "scored_by": "reproducegym-recompute",
        }
    if metric_scores:
        shaped = max(0.0, min(1.0, sum(metric_scores) / len(metric_scores)))
        reward = min(shaped, max(0.0, min(1.0, float(rbv.get(verdict, 0.0)))))
        return {
            "reward": round(reward, 6),
            "verdict": verdict,
            "metrics": report,
            "errors": [],
            "scored_by": "reproducegym-recompute",
        }
    return out(verdict, metrics=report)
# === REPRODUCEGYM ENGINE END ===
