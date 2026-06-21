"""ReproGym recompute verifier engine (stdlib-only, embeddable).

Core principle: a claim is scored ONLY by recomputing its metrics from the
agent's raw reproduction artifact (``output/metrics.csv``) and comparing the
recomputed values against the hidden thresholds. The engine NEVER reads an
agent-declared verdict, score, ``strict_reproduction`` flag, or any other
self-reported judgement -- the verdict is derived here from the numbers.

The source between the ``REPROGYM ENGINE BEGIN/END`` markers is embedded verbatim
into each task's ``reward/check.py`` (which runs inside the sandbox-free scoring
step with no reprogym import), so this block must stay stdlib-only and self
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

# === REPROGYM ENGINE BEGIN ===
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
            "scored_by": "reprogym-recompute",
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

    report = {}
    all_pass = True
    for name in spec.get("metrics", []):
        if name not in thresholds:
            errors.append("no threshold for metric %r" % name)
            continue
        try:
            value = safe_eval(formulas.get(name, ""), rows, condition_col, windows.get(name))
        except VerifierError as exc:
            errors.append("metric %r: %s" % (name, exc))
            continue
        passed = _passes(value, thresholds[name], directions.get(name, "higher_is_better"))
        all_pass = all_pass and passed
        report[name] = {
            "value": round(value, 6),
            "pass_threshold": thresholds[name],
            "direction": directions.get(name),
            "passed": passed,
        }

    if errors:
        return out("inconclusive", metrics=report, errors=errors)
    return out("reproduced" if all_pass else "failed", metrics=report)
# === REPROGYM ENGINE END ===
