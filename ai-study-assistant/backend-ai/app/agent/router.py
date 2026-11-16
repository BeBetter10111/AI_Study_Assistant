"""Domain-aware agentic router for chat.

Extends the keyword intent classifier with STEM / language domains and optional
tool execution (SymPy calculator, grammar correction + TTS text).  When
``router_llm_enabled`` is true and keywords are ambiguous, one structured LLM
call disambiguates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel, Field

from app.agent.tools import language_tool, math_tool
from app.config import settings

logger = logging.getLogger(__name__)

# Re-export the same keyword sets chat.py historically used (kept here so both
# paths stay aligned if chat imports from the router).
IDENTITY_KEYWORDS = (
    "bạn là ai", "bạn là gì", "bạn tên gì", "tên bạn là gì", "giới thiệu về bạn",
    "bạn làm được gì", "bạn có thể làm gì", "bạn giúp được gì", "bạn có khả năng gì",
    "who are you", "what are you", "what is your name", "introduce yourself",
    "what can you do", "what are your capabilities", "who r u", "what r u",
)
EXPLAIN_INTENT_KEYWORDS = (
    "explain", "giải thích", "please explain", "can you explain",
    "tại sao", "vì sao", "lý do là gì", "explain why", "explain how",
)
QUIZ_KEYWORDS = (
    "quiz", "trắc nghiệm", "câu hỏi ôn tập", "làm bài kiểm tra",
    "create a quiz", "generate quiz", "make a quiz", "test me", "quiz me",
)
SUMMARY_KEYWORDS = (
    "tóm tắt", "summary", "summarize", "summarise",
    "give me a summary", "please summarize", "overview of",
    "tổng quan", "nội dung chính", "tóm tắt nội dung", "tóm tắt bài học",
)

_BASE_INTENTS = frozenset({
    "identity", "quiz_generation", "summarization", "explanation", "qa",
    "math", "language",
})


class _LLMRoute(BaseModel):
    intent: str = Field(description="One of: identity, quiz_generation, summarization, explanation, qa, math, language")
    domain: str = Field(default="general", description="general | stem | language | legal | medical")
    use_math_tool: bool = False
    use_language_tool: bool = False
    confidence: float = 0.5


@dataclass
class ToolOutcome:
    name: str
    ok: bool
    prompt_block: str = ""
    display: str = ""          # short text safe to show / append
    tts_text: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class RouteDecision:
    intent: str                  # chat pipeline intent (qa/explanation/.../math/language)
    domain: str = "general"      # stem | language | general | ...
    use_math_tool: bool = False
    use_language_tool: bool = False
    source: str = "keyword"      # keyword | llm
    tools: List[ToolOutcome] = field(default_factory=list)

    @property
    def tool_context(self) -> str:
        blocks = [t.prompt_block for t in self.tools if t.prompt_block]
        return "\n\n".join(blocks)

    @property
    def tts_text(self) -> str:
        for t in self.tools:
            if t.tts_text:
                return t.tts_text
        return ""


def _is_identity_question(question: str) -> bool:
    lowered = question.lower().strip()
    return len(lowered) <= 80 and any(kw in lowered for kw in IDENTITY_KEYWORDS)


def classify_intent_keywords(question: str) -> str:
    """Keyword-only intent (zero LLM). Includes math / language domains."""
    lowered = (question or "").lower()
    if _is_identity_question(question):
        return "identity"
    if any(kw in lowered for kw in QUIZ_KEYWORDS):
        return "quiz_generation"
    if any(kw in lowered for kw in SUMMARY_KEYWORDS):
        return "summarization"
    if math_tool.looks_like_math(question) and not language_tool.looks_like_language_task(question):
        return "math"
    if language_tool.looks_like_language_task(question):
        return "language"
    if any(kw in lowered for kw in EXPLAIN_INTENT_KEYWORDS):
        return "explanation"
    return "qa"


def _keyword_route(question: str) -> RouteDecision:
    intent = classify_intent_keywords(question)
    domain = "general"
    use_math = False
    use_lang = False
    if intent == "math":
        domain, use_math = "stem", True
    elif intent == "language":
        domain, use_lang = "language", True
    elif math_tool.looks_like_math(question):
        domain, use_math = "stem", True
    elif language_tool.looks_like_language_task(question):
        domain, use_lang = "language", True
    return RouteDecision(
        intent=intent, domain=domain,
        use_math_tool=use_math, use_language_tool=use_lang, source="keyword",
    )


def _llm_route(question: str, fallback: RouteDecision) -> RouteDecision:
    from app.core.llm_engine import llm_engine

    prompt = (
        "Classify the study question for a tutoring assistant.\n"
        f"Question: {question}\n\n"
        "Pick intent and whether specialized tools are needed."
    )
    try:
        data = llm_engine.generate_structured_output(prompt, _LLMRoute, temperature=0.0)
        intent = (data.get("intent") or fallback.intent).strip().lower()
        if intent not in _BASE_INTENTS:
            intent = fallback.intent
        domain = (data.get("domain") or fallback.domain).strip().lower() or "general"
        return RouteDecision(
            intent=intent,
            domain=domain,
            use_math_tool=bool(data.get("use_math_tool")) or intent == "math",
            use_language_tool=bool(data.get("use_language_tool")) or intent == "language",
            source="llm",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM router failed, using keywords: %s", exc)
        return fallback


def classify_route(question: str) -> RouteDecision:
    """Return a routing decision. Respects ``router_enabled`` / ``router_llm_enabled``."""
    if not settings.router_enabled:
        # Preserve legacy keyword intents only (no math/language specialization).
        lowered = (question or "").lower()
        if _is_identity_question(question):
            intent = "identity"
        elif any(kw in lowered for kw in QUIZ_KEYWORDS):
            intent = "quiz_generation"
        elif any(kw in lowered for kw in SUMMARY_KEYWORDS):
            intent = "summarization"
        elif any(kw in lowered for kw in EXPLAIN_INTENT_KEYWORDS):
            intent = "explanation"
        else:
            intent = "qa"
        return RouteDecision(intent=intent, source="keyword")

    decision = _keyword_route(question)
    ambiguous = decision.intent == "qa" and (
        math_tool.looks_like_math(question) or language_tool.looks_like_language_task(question)
        or ("?" in question and len(question.split()) >= 8)
    )
    if settings.router_llm_enabled and ambiguous:
        decision = _llm_route(question, decision)
    return decision


def run_route_tools(question: str, decision: Optional[RouteDecision] = None) -> RouteDecision:
    """Classify (if needed) and execute enabled tools, attaching outcomes to the decision."""
    route = decision or classify_route(question)
    if route.use_math_tool and settings.math_tool_enabled:
        result = math_tool.try_answer(question)
        if result is not None:
            display = (
                f"**Result:** `{result.result}`\n\n*(expression: `{result.expression}`)*"
                if result.ok else
                f"Could not compute `{result.expression}`: {result.error}"
            )
            route.tools.append(ToolOutcome(
                name="math",
                ok=result.ok,
                prompt_block=result.as_prompt_block(),
                display=display,
                data={"expression": result.expression, "result": result.result, "error": result.error},
            ))
    if route.use_language_tool and settings.language_tool_enabled:
        result = language_tool.try_correct(question)
        if result is not None:
            if result.ok:
                display = f"**Corrected:** {result.corrected}"
                if result.explanations:
                    display += "\n\n" + "\n".join(f"- {e}" for e in result.explanations[:6])
            else:
                display = result.error or "Language tool failed."
            route.tools.append(ToolOutcome(
                name="language",
                ok=result.ok,
                prompt_block=result.as_prompt_block(),
                display=display,
                tts_text=result.tts_text if result.want_tts else "",
                data={
                    "original": result.original,
                    "corrected": result.corrected,
                    "explanations": result.explanations,
                    "want_tts": result.want_tts,
                },
            ))
    return route
