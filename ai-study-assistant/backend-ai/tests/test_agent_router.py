"""Unit tests for the agentic router + SymPy math tool."""
import pytest

from app.agent.router import classify_intent_keywords, classify_route, run_route_tools
from app.agent.tools import math_tool


def test_classify_math_intent():
    assert classify_intent_keywords("Tính 2 + 2") == "math"
    assert classify_intent_keywords("Solve x**2 - 1 = 0") == "math"


def test_classify_language_intent():
    assert classify_intent_keywords('Correct my grammar: "I goes to school"') == "language"
    assert classify_intent_keywords("Sửa lỗi ngữ pháp giúp tôi") == "language"


def test_classify_legacy_intents():
    assert classify_intent_keywords("Who are you?") == "identity"
    assert classify_intent_keywords("Tóm tắt bài học này") == "summarization"
    assert classify_intent_keywords("Explain photosynthesis") == "explanation"
    assert classify_intent_keywords("Generate a quiz") == "quiz_generation"


def test_math_evaluate_simple(monkeypatch):
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)
    result = math_tool.evaluate("2 + 2")
    assert result.ok
    assert result.result in ("4", "4.0")


def test_math_evaluate_equation(monkeypatch):
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)
    result = math_tool.evaluate("x**2 - 1 = 0")
    assert result.ok
    assert "-1" in result.result or "1" in result.result


def test_math_rejects_import(monkeypatch):
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)
    result = math_tool.evaluate("__import__('os').system('echo hi')")
    assert not result.ok


def test_run_route_tools_math(monkeypatch):
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)
    monkeypatch.setattr("app.agent.router.settings.router_enabled", True)
    monkeypatch.setattr("app.agent.router.settings.math_tool_enabled", True)
    route = run_route_tools("Calculate 3 * 7")
    assert route.intent == "math"
    assert route.domain == "stem"
    assert route.tools and route.tools[0].ok
    assert "21" in route.tools[0].display


def test_classify_route_respects_disabled_router(monkeypatch):
    monkeypatch.setattr("app.agent.router.settings.router_enabled", False)
    route = classify_route("Calculate 1 + 1")
    # Legacy path: no math specialization when router is off.
    assert route.intent == "qa"
    assert not route.use_math_tool
