"""Safe-ish SymPy calculator for STEM study questions.

Only a restricted set of SymPy names is exposed; ``eval`` / imports / attribute
dunders are rejected.  The expression is evaluated in a separate, resource-limited process
(see ``_math_worker.py``) that is killed on timeout, so a hostile expression cannot freeze the API.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from typing import Optional

from app.agent.tools import _math_worker
from app.config import settings

logger = logging.getLogger(__name__)

# --- Math detection -------------------------------------------------------------------------------------------
# A question is "math" only when it carries a real expression AND (a compute verb or a calculus/algebra noun), or
# when it is (almost) nothing but an arithmetic expression.  Bare operators / words are NOT enough: "client-server",
# "TCP/UDP", "a = b trong Java", "Giải thích ...", "Tính chất của stack" are ordinary study questions.
_VERB = re.compile(
    r"\b(?:solve|calculate|compute|evaluate|simplify|differentiate|integrate|factorize|expand)\b|"
    r"(?<!\w)(?:tính|rút gọn|tìm nghiệm|giải)(?!\s*(?:thích|pháp|quyết|mã|thuật|trí|đáp))(?!\w)",
    re.IGNORECASE,
)
_NOUN = re.compile(
    r"\b(?:derivative|integral|equation|determinant|limit|matrix|inverse)\b|"
    r"(?<!\w)(?:đạo hàm|tích phân|phương trình|định thức|ma trận|giới hạn|nghịch đảo)(?!\w)",
    re.IGNORECASE,
)
_EXPR = re.compile(
    r"\d[a-zA-Z]?\s*[-+*/^×÷]\s*[\w(]"             # 2+2, 3x - 5, 12 * 12
    r"|[a-zA-Z)]\s*(?:\^|\*\*)\s*[\w(]"            # x^2, (a+b)**2
    r"|[a-zA-Z]\s*[-+*/]\s*\d"                       # x + 3
    r"|\d\s*[a-zA-Z]\s*(?:=|\b)"                    # 2x = 6
    r"|[√∫∑π]|\\(?:frac|sqrt|int|sum)\b"
    r"|\b(?:sin|cos|tan|cot|log|ln|sqrt|exp)\b\s*[(\dx-z]"
    r"|\bx\s*=\s*[-\d(]",
    re.IGNORECASE,
)
_LEAD_IN = re.compile(r"\b(?:what is|what's|bằng bao nhiêu|bao nhiêu|bằng|là)\b|[?=]", re.IGNORECASE)
_ARITH_CHARS = re.compile(r"[\d\s.,+\-*/^()×÷xyz]")


def _mostly_expression(question: str) -> bool:
    """True when the text is essentially a bare arithmetic expression ("12 * 12", "What is 3/4 + 1/2?")."""
    text = _LEAD_IN.sub(" ", question or "").strip()
    compact = "".join(text.split())
    if len(compact) < 3 or not _EXPR.search(text):
        return False
    return sum(1 for ch in compact if _ARITH_CHARS.fullmatch(ch)) / len(compact) >= 0.8


# Expressions we try to pull out of a natural-language question.
_EXPR_PATTERNS = [
    re.compile(r"(?:tính|compute|calculate|evaluate|simplify)\s*:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:giải|solve)\s*(?:(?:phương trình|equation)\s*)?:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:đạo hàm|derivative)\s*(?:của|of)?\s*:?\s*(.+)$", re.I | re.S),
    re.compile(r"(?:tích phân|integral)\s*(?:của|of)?\s*:?\s*(.+)$", re.I | re.S),
]

# Kept for backwards compatibility; the single source of truth lives in the worker module.
_ALLOWED_FUNCS = _math_worker.ALLOWED_FUNCS
_ALLOWED_NODES = _math_worker.ALLOWED_NODES
_validate_ast = _math_worker.validate_ast
_sympy_namespace = _math_worker.build_namespace


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
    q = question or ""
    if not _EXPR.search(q):
        return False
    return bool(_VERB.search(q) or _NOUN.search(q) or _mostly_expression(q))


def _extract_expression(question: str) -> str:
    q = (question or "").strip()
    for pat in _EXPR_PATTERNS:
        m = pat.search(q)
        if m:
            return m.group(1).strip().rstrip("?.!")
    # Fallback: longest stretch that looks algebraic.
    m = re.search(r"([0-9A-Za-zπ√(][0-9A-Za-zπ√+\-*/^=().\s\\]{2,})", q)
    return (m.group(1).strip() if m else q).rstrip("?.!")


def _prepare_expr(expr: str) -> str:
    text = expr.strip().replace("^", "**").replace("π", "pi")
    # Bare equation → solve(Eq(left, right)) when not already wrapped.
    if "=" in text and "==" not in text and "Eq(" not in text and "solve(" not in text:
        left, right = text.split("=", 1)
        if left.strip() and right.strip():
            text = f"solve(Eq({left.strip()}, {right.strip()}))"
    return text


_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_math_worker.py")
_SLOTS = threading.BoundedSemaphore(4)   # at most 4 calculator processes at the same time


def evaluate(expression: str, timeout_s: Optional[float] = None) -> MathResult:
    """Evaluate a restricted SymPy expression in a killable sandbox process. Never raises to the caller."""
    if not settings.math_tool_enabled:
        return MathResult(ok=False, expression=expression, error="Math tool disabled")
    timeout = settings.math_tool_timeout_s if timeout_s is None else timeout_s
    expr = _prepare_expr(expression)
    if not expr or len(expr) > _math_worker.MAX_EXPR_LEN:
        return MathResult(ok=False, expression=expr, error="Expression empty or too long")

    # Cheap rejection in-process (syntax / size rules) before paying for a subprocess.
    try:
        _math_worker.validate_ast(ast.parse(expr, mode="eval"))
    except Exception as exc:  # noqa: BLE001
        return MathResult(ok=False, expression=expr, error=str(exc)[:300])

    if not _SLOTS.acquire(timeout=1.0):
        return MathResult(ok=False, expression=expr, error="Calculator is busy, please retry")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", _WORKER],
            input=json.dumps({"expr": expr, "timeout": timeout}),
            capture_output=True, text=True,
            timeout=timeout + 1.5,            # + interpreter / SymPy start-up; the child is killed on expiry
            check=False,
        )
    except subprocess.TimeoutExpired:
        return MathResult(ok=False, expression=expr, error=f"Timed out after {timeout}s")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Math sandbox failed to start: {exc}")
        return MathResult(ok=False, expression=expr, error="Calculator unavailable")
    finally:
        _SLOTS.release()

    try:
        data = json.loads((proc.stdout or "").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - killed by the CPU/memory limit, or crashed
        return MathResult(ok=False, expression=expr, error="Calculation exceeded its resource limits")
    if data.get("ok"):
        return MathResult(ok=True, expression=expr, result=str(data.get("result", "")))
    return MathResult(ok=False, expression=expr, error=str(data.get("error", "unknown error")))


def try_answer(question: str) -> Optional[MathResult]:
    """Extract an expression from a question and evaluate it. ``None`` if not math-like."""
    if not looks_like_math(question):
        return None
    expr = _extract_expression(question)
    result = evaluate(expr)
    return result
