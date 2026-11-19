from __future__ import annotations

import ast
import json
import sys
from typing import Any, Dict

ALLOWED_FUNCS = frozenset({
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "ln", "sqrt", "Abs", "abs",
    "factorial", "gamma", "floor", "ceiling", "sign", "re", "im",
    "simplify", "expand", "factor", "collect", "cancel", "apart",
    "diff", "integrate", "limit", "solve", "nsolve", "Eq",
    "Matrix", "det", "transpose", "inv", "eye", "zeros", "ones",
    "Sum", "Product", "oo", "pi", "E", "I", "Rational", "Integer", "Float",
    "N", "Pow", "Add", "Mul",
})

ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Tuple, ast.List, ast.keyword, ast.Attribute,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UAdd, ast.USub,
)

MAX_EXPR_LEN = 500
MAX_EXPONENT = 10_000          # largest literal exponent accepted
MAX_FACTORIAL_ARG = 1_000      # factorial(n) / gamma(n) literal argument
MAX_LITERAL = 10 ** 30         # any single numeric literal
MAX_RESULT_CHARS = 4_000
_SIZE_SENSITIVE = {"factorial", "gamma", "Pow", "Matrix", "eye", "zeros", "ones"}


def _const_value(node: ast.AST):
    """Numeric value of a literal (possibly with a unary sign), else None."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        inner = _const_value(node.operand)
        return None if inner is None else (inner if isinstance(node.op, ast.UAdd) else -inner)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def validate_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError(f"Disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("Attribute access denied")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Name denied")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                raise ValueError("String literals are not allowed")
            if isinstance(node.value, (int, float)) and abs(node.value) > MAX_LITERAL:
                raise ValueError("Number too large")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in ALLOWED_FUNCS:
                    raise ValueError(f"Function not allowed: {func.id}")
                if func.id in ("factorial", "gamma"):
                    for arg in node.args:
                        val = _const_value(arg)
                        if val is not None and abs(val) > MAX_FACTORIAL_ARG:
                            raise ValueError("Argument too large")
            elif isinstance(func, ast.Attribute):
                if func.attr not in ALLOWED_FUNCS and not func.attr.startswith("is_"):
                    raise ValueError(f"Method not allowed: {func.attr}")
            else:
                raise ValueError("Unsupported call target")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp = _const_value(node.right)
            if exp is not None and abs(exp) > MAX_EXPONENT:
                raise ValueError("Exponent too large")
            # a ** (b ** c) / a ** factorial(n): the classic way to ask for an astronomically big number
            if exp is None and any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) or
                                   (isinstance(n, ast.Call) and getattr(n.func, "id", "") in _SIZE_SENSITIVE)
                                   for n in ast.walk(node.right)):
                raise ValueError("Nested power is not allowed")


def build_namespace() -> Dict[str, Any]:
    import sympy as sp

    ns: Dict[str, Any] = {name: getattr(sp, name) for name in ALLOWED_FUNCS if hasattr(sp, name)}
    ns["ln"] = sp.log
    ns["abs"] = sp.Abs
    ns["pi"] = sp.pi
    ns["E"] = sp.E
    ns["I"] = sp.I
    ns["oo"] = sp.oo
    for sym in ("x", "y", "z", "t", "n", "a", "b", "c"):
        ns[sym] = sp.symbols(sym)
    return ns


def evaluate_expression(expr: str) -> Dict[str, Any]:
    try:
        if not expr or len(expr) > MAX_EXPR_LEN:
            return {"ok": False, "error": "Expression empty or too long"}
        tree = ast.parse(expr, mode="eval")
        validate_ast(tree)
        ns = build_namespace()
        import sympy as sp
        try:
            value = sp.sympify(expr, locals=ns, evaluate=True)
        except Exception:  # noqa: BLE001
            value = eval(compile(tree, "<math>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307
        if hasattr(value, "doit"):
            try:
                value = value.doit()
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "result": str(value)[:MAX_RESULT_CHARS]}
    except MemoryError:
        return {"ok": False, "error": "Calculation needs too much memory"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)[:300]}


def _apply_limits(timeout: float) -> None:
    """Best effort (POSIX only): cap memory and CPU time of this process."""
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))                 # 1 GiB address space
        cpu = int(timeout) + 2
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:  # noqa: BLE001 - not available on Windows
        pass


def main() -> int:
    try:
        request = json.loads(sys.stdin.read() or "{}")
        expr = str(request.get("expr", ""))
        timeout = float(request.get("timeout", 3.0))
    except Exception:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": "Bad request"}))
        return 0
    _apply_limits(timeout)
    print(json.dumps(evaluate_expression(expr), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
