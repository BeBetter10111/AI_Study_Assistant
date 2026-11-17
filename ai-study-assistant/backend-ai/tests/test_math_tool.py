"""SymPy calculator: correct results + sandbox rejects unsafe input."""
import pytest

from app.agent.tools import math_tool


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)


def test_arithmetic_and_calculus():
    assert math_tool.evaluate("2 + 3*4").result in ("14", "14.0")
    assert math_tool.evaluate("diff(x**3, x)").result.replace(" ", "") == "3*x**2"
    assert math_tool.evaluate("integrate(2*x, x)").result.replace(" ", "") == "x**2"


def test_matrix_determinant():
    res = math_tool.evaluate("Matrix([[1,2],[3,4]]).det()")
    assert res.ok and res.result == "-2"


@pytest.mark.parametrize("bad", [
    "__import__('os').system('echo hi')",
    "open('/etc/passwd').read()",
    "().__class__.__bases__",
    "exec('1')",
])
def test_unsafe_expressions_rejected(bad):
    assert not math_tool.evaluate(bad).ok


def test_try_answer_from_natural_language():
    res = math_tool.try_answer("Tính 12 * 12")
    assert res is not None and res.ok and res.result in ("144", "144.0")
    assert math_tool.try_answer("Hãy kể về lịch sử Việt Nam") is None
