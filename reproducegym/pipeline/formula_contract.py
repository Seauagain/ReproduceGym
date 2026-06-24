"""Static checks for formulas supported by the generated verifier."""

from __future__ import annotations

import ast
from typing import Any

AGG_FUNCS = {"mean", "sum", "min", "max", "median", "std", "var", "last", "first", "count"}


def _is_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return True
    return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)


def _check(node: ast.AST) -> str | None:
    if isinstance(node, ast.Expression):
        return _check(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            return "only numeric constants are allowed"
        return None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _check(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)
    ):
        return _check(node.left) or _check(node.right)
    if isinstance(node, ast.Call):
        if node.keywords or not isinstance(node.func, ast.Name):
            return "unsupported function call"
        fname = node.func.id
        if len(node.args) != 1:
            return f"{fname}() must take exactly one argument"
        if fname == "abs":
            return _check(node.args[0])
        if fname not in AGG_FUNCS:
            return f"function not allowed: {fname}"
        if not _is_ref(node.args[0]):
            return f"{fname}() argument must be a column or condition.column reference"
        return None
    if isinstance(node, (ast.Name, ast.Attribute)):
        return "series references must be wrapped in an aggregation"
    return f"unsupported expression: {type(node).__name__}"


def formula_problem(formula: Any) -> str | None:
    try:
        tree = ast.parse(str(formula or ""), mode="eval")
    except SyntaxError as exc:
        return f"cannot parse formula: {exc.msg}"
    return _check(tree)


def formulas_problem(metrics: list[dict[str, Any]]) -> dict[str, str]:
    problems: dict[str, str] = {}
    for metric in metrics or []:
        if not isinstance(metric, dict):
            continue
        name = str(metric.get("name") or "")
        problem = formula_problem(metric.get("formula"))
        if problem:
            problems[name or "<unnamed>"] = problem
    return problems
