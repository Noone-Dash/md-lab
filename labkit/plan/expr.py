"""A tiny safe expression evaluator for the rules engine.

No eval(). An AST whitelist only — rules are data, and data must never be able to
execute arbitrary code.
"""

from __future__ import annotations

import ast
import math

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.BinOp, ast.Compare, ast.IfExp,
    ast.Name, ast.Load, ast.Constant, ast.And, ast.Or, ast.Not, ast.USub, ast.UAdd,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Call, ast.Tuple, ast.List, ast.Attribute, ast.Subscript, ast.Index,
)

_FUNCS = {
    "min": min, "max": max, "abs": abs, "round": round, "len": len,
    "any": any, "all": all, "float": float, "int": int, "str": str,
    "sqrt": math.sqrt, "floor": math.floor, "ceil": math.ceil,
    "startswith": lambda s, p: str(s).startswith(p),
    "contains": lambda s, p: p in (s or ""),
    "isnum": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}


class ExprError(ValueError):
    pass


def compile_expr(src: str):
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"cannot parse {src!r}: {e}") from e
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"disallowed syntax {type(node).__name__} in {src!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
                raise ExprError(f"disallowed call in {src!r}")
    return compile(tree, "<rule>", "eval")


def eval_expr(code, ctx: dict):
    env = dict(_FUNCS)
    env.update(ctx)
    try:
        return eval(code, {"__builtins__": {}}, env)  # noqa: S307 - AST-whitelisted above
    except (TypeError, KeyError, ZeroDivisionError, AttributeError):
        return None          # an unevaluable rule is skipped, never a false pass
