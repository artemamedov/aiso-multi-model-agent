import ast
import math
from typing import Any


_BINARY_OPS: dict[type[ast.operator], Any] = {
    ast.Add: lambda x, y: x + y,
    ast.Sub: lambda x, y: x - y,
    ast.Mult: lambda x, y: x * y,
    ast.Div: lambda x, y: x / y,
    ast.Pow: lambda x, y: x**y,
    ast.Mod: lambda x, y: x % y,
}
_UNARY_OPS: dict[type[ast.unaryop], Any] = {
    ast.UAdd: lambda x: +x,
    ast.USub: lambda x: -x,
}
_ALLOWED_FUNCS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
}
_ALLOWED_CONSTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed in expressions.")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return float(_BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTS:
            return float(_ALLOWED_CONSTS[node.id])
        raise ValueError(f"Unsupported name in expression: {node.id}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call in expression.")
        func_name = node.func.id
        if func_name not in _ALLOWED_FUNCS:
            raise ValueError(f"Unsupported function: {func_name}")
        args = [_eval_node(arg) for arg in node.args]
        # round() needs int for ndigits
        if func_name == "round" and len(args) == 2:
            args[1] = int(args[1])
        return float(_ALLOWED_FUNCS[func_name](*args))
    raise ValueError("Unsupported expression.")


def _safe_eval_expression(expression: str) -> float:
    normalized = expression.replace("^", "**").strip()
    parsed = ast.parse(normalized, mode="eval")
    return _eval_node(parsed)


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return str(value)


def calculator(expression: str, ndigits: int = -1) -> str:
    """Evaluate a full arithmetic expression and return the numeric result.

    Use this for any numeric computation. Always pass the complete expression,
    including every operation required by the question. Supported helpers:
    `sqrt(...)`, `abs(...)`, `round(...)`, `pi`, and `e`.

    Args:
        expression: Full math expression to evaluate.
        ndigits: Optional final rounding digits to apply to the result.

    Returns:
        Numeric result as text, formatted for direct answering.
    """
    result = 0.0
    try:
        result = _safe_eval_expression(expression)
    except Exception as e:
        import sys
        print(f"[CALCULATOR ERROR] expression={expression!r} error={e}", file=sys.stderr)
        result = 0.0

    if ndigits is not None and ndigits > 0:
        result = round(float(result), ndigits)

    return _format_number(float(result))
