"""Safe-ish SymPy calculator for STEM study questions.

Only a restricted set of SymPy names is exposed; ``eval`` / imports / attribute
dunders are rejected.  Timeout is enforced with a worker thread (fail-open).
"""
from __future__ import annotations

import ast
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app.config import settings

logger = logging.getLogger(__name__)

_MATH_HINT = re.compile(
    r"(?:"
    r"\b(?:solve|calculate|compute|evaluate|simplify|derivative|integral|factor)\b|"
    r"tính|giải|đạo hàm|tích phân|phương trình|rút gọn|"
    r"[=+\-*/^√∫∑π]|\\frac|\\sqrt|\d+\s*[+\-*/^]\s*\d+"
    r")",
    re.IGNORECASE,
)

# Expressions we try to pull out of a natural-language question.
_EXPR_PATTERNS = [
    re.compile(r"(?:tính|compute|calculate|evaluate|simplify)\s*:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:giải|solve)\s*(?:(?:phương trình|equation)\s*)?:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:đạo hàm|derivative)\s*(?:của|of)?\s*:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:tích phân|integral)\s*(?:của|of)?\s*:?\s*(.+)$", re.I | re.S),
]

_ALLOWED_FUNCS = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "ln", "sqrt", "Abs", "abs",
    "factorial", "gamma", "floor", "ceiling", "sign", "re", "im",
    "simplify", "expand", "factor", "collect", "cancel", "apart",
    "diff", "integrate", "limit", "solve", "nsolve", "Eq",
    "Matrix", "det", "transpose", "inv", "eye", "zeros", "ones",
    "Sum", "Product", "oo", "pi", "E", "I", "Rational", "Integer", "Float",
    "N", "Pow", "Add", "Mul",
})

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Tuple, ast.List, ast.keyword, ast.Attribute,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)


@dataclass
class MathResult:
    ok: bool
    expression: str = ""
    result: str = ""
    error: str = ""

    def as_prompt_block(self) -> str:
        if self.ok:
            return f"Calculator result for `{self.expression}`:\n{self.result}"
        return f"Calculator could not evaluate `{self.expression}`: {self.error}"


def looks_like_math(question: str) -> bool:
    return bool(_MATH_HINT.search(question or ""))


def _extract_expression(question: str) -> str:
    q = (question or "").strip()
    for pat in _EXPR_PATTERNS:
        m = pat.search(q)
        if m:
            return m.group(1).strip().rstrip("?.!")
    # Fallback: longest stretch that looks algebraic.
    m = re.search(r"([0-9A-Za-zπ√(][0-9A-Za-zπ√+\-*/^=().\s\\]{2,})", q)
    return (m.group(1).strip() if m else q).rstrip("?.!")


def _validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("Attribute access denied")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Name denied")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _ALLOWED_FUNCS and func.id not in {"x", "y", "z", "t", "n", "a", "b", "c"}:
                    # Allow unknown names only as symbols via Name nodes, not as calls.
                    if func.id not in _ALLOWED_FUNCS:
                        raise ValueError(f"Function not allowed: {func.id}")
            elif isinstance(func, ast.Attribute):
                if func.attr not in _ALLOWED_FUNCS and not func.attr.startswith("is_"):
                    raise ValueError(f"Method not allowed: {func.attr}")


def _sympy_namespace() -> Dict[str, Any]:
    import sympy as sp

    ns: Dict[str, Any] = {name: getattr(sp, name) for name in _ALLOWED_FUNCS if hasattr(sp, name)}
    ns["ln"] = sp.log
    ns["abs"] = sp.Abs
    ns["pi"] = sp.pi
    ns["E"] = sp.E
    ns["I"] = sp.I
    ns["oo"] = sp.oo
    # Common free symbols.
    for sym in ("x", "y", "z", "t", "n", "a", "b", "c"):
        ns[sym] = sp.symbols(sym)
    return ns


def _prepare_expr(expr: str) -> str:
    text = expr.strip().replace("^", "**").replace("π", "pi")
    # Bare equation → solve(Eq(left, right)) when not already wrapped.
    if "=" in text and "==" not in text and "Eq(" not in text and "solve(" not in text:
        left, right = text.split("=", 1)
        if left.strip() and right.strip():
            text = f"solve(Eq({left.strip()}, {right.strip()}))"
    return text


def evaluate(expression: str, timeout_s: Optional[float] = None) -> MathResult:
    """Evaluate a restricted SymPy expression. Never raises to the caller."""
    if not settings.math_tool_enabled:
        return MathResult(ok=False, expression=expression, error="Math tool disabled")
    timeout = settings.math_tool_timeout_s if timeout_s is None else timeout_s
    expr = _prepare_expr(expression)
    if not expr or len(expr) > 500:
        return MathResult(ok=False, expression=expr, error="Expression empty or too long")

    box: Dict[str, Any] = {}

    def _run() -> None:
        try:
            tree = ast.parse(expr, mode="eval")
            _validate_ast(tree)
            ns = _sympy_namespace()
            # Use SymPy's parser for friendlier math syntax when AST is a plain call/name.
            import sympy as sp
            try:
                value = sp.sympify(expr, locals=ns, evaluate=True)
            except Exception:
                value = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
            if hasattr(value, "doit"):
                try:
                    value = value.doit()
                except Exception:  # noqa: BLE001
                    pass
            box["result"] = str(value)
            box["ok"] = True
        except Exception as exc:  # noqa: BLE001
            box["ok"] = False
            box["error"] = str(exc)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    if thread.is_alive():
        return MathResult(ok=False, expression=expr, error=f"Timed out after {timeout}s")
    if box.get("ok"):
        return MathResult(ok=True, expression=expr, result=str(box.get("result", "")))
    return MathResult(ok=False, expression=expr, error=str(box.get("error", "unknown error")))


def try_answer(question: str) -> Optional[MathResult]:
    """Extract an expression from a question and evaluate it. ``None`` if not math-like."""
    if not looks_like_math(question):
        return None
    expr = _extract_expression(question)
    result = evaluate(expr)
    return result
