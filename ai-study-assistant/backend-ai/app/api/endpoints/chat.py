import json
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from app.config import settings
from app.core.llm_engine import llm_engine
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

IDENTITY_KEYWORDS = (
    "bạn là ai", "bạn là gì", "bạn tên gì", "tên bạn là gì", "giới thiệu về bạn",
    "bạn làm được gì", "bạn có thể làm gì", "bạn giúp được gì", "bạn có khả năng gì",
    "who are you", "what are you", "what is your name", "introduce yourself",
    "what can you do", "what are your capabilities", "your capabilities",
    "what are your functions", "what are your features", "what are your skills",
    "what are your abilities", "what are your strengths", "what are your weaknesses",
    "what are your limitations", "what are your restrictions", "what are your constraints",
    "what are your boundaries", "what are your rules", "what are your guidelines",
    "what are your policies", "what are your procedures", "what are your protocols",
    "what are your standards", "what are your requirements", "what are your expectations", 
    "who r u", "what r u", "ur name", "ur capabilities", "ur functions", "ur features",
    "ur skills", "ur abilities", "ur strengths", "ur weaknesses", "ur limitations", "ur restrictions", "ur constraints", "ur boundaries", "ur rules",
    "ur guidelines", "ur policies", "ur procedures", "ur protocols", "ur standards", "ur requirements", "ur expectations"
)
EXPLAIN_INTENT_KEYWORDS = (
    "explain", "giải thích", "please explain", "can you explain",
    "tại sao", "vì sao", "lý do là gì", "vì lý do gì", "giải thích giúp tôi",
    "explain why", "explain how", "explain the reason", "explain the cause",
    "explain the meaning", "explain the concept", "explain the idea",
    "explain the principle", "explain the mechanism", "explain the process",
    "explain the function", "explain the purpose", "explain the significance",
    "explain the importance", "explain the relevance", "explain the implications",
    "explain the consequences", "explain the effects", "explain the impact",
    "explain the outcome", "explain the result", "explain the conclusion",
    "explain the inference", "explain the deduction", "explain the reasoning",
    "explain the logic", "explain the rationale", "explain the justification",
    "explain the argument", "explain the evidence", "explain the data",
    "explain the statistics", "explain the research", "explain the study",
    "explain the experiment", "explain the observation", "explain the phenomenon",
    "explain the theory", "explain the hypothesis", "explain the model",
    "explain the framework", "explain the approach", "explain the method",
    "explain the technique", "explain the strategy", "explain the plan",
    "explain the procedure", "explain the protocol", "explain the guideline",
)

QUIZ_KEYWORDS = (
    "quiz", "trắc nghiệm", "câu hỏi ôn tập", "làm bài kiểm tra",
    "create a quiz", "generate quiz", "make a quiz", "test me", "quiz me",
)

SUMMARY_KEYWORDS = (
    "tóm tắt", "summary", "summarize", "summarise",
    "give me a summary", "please summarize", "give a summary",
    "what is the main idea", "overview of", "briefly explain",
    "tổng quan", "nội dung chính",
    "tóm tắt nội dung", "tóm tắt ý chính", "tóm tắt bài học",
    "tóm tắt chương", "tóm tắt bài giảng", "tóm tắt giáo trình", "tóm tắt tài liệu", "tóm tắt slide",
    "tóm tắt bài báo", "tóm tắt nghiên cứu", "tóm tắt báo cáo", "tóm tắt luận văn", "tóm tắt luận án", "tóm tắt sách", "tóm tắt tài liệu học thuật",
)

# Pydantic schemas
class HistoryMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    document_id: Optional[str] = None
    top_k: int = Field(default=3, ge=1, le=20)
    history: List[HistoryMessage] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    document_id: str
    chunk_index: int
    score: float
    content: str
    page_number: Optional[int] = None
    slide_number: Optional[int] = None
    slide_title: Optional[str] = None
    section_title: Optional[str] = None
    is_summary: bool = False


class QueryResponse(BaseModel):
    intent: str
    answer: str
    citations: List[Citation]
    standalone_query: str  
    latency_ms: float

def _is_identity_question(question: str) -> bool:
    lowered = question.lower().strip()
    return len(lowered) <= 80 and any(kw in lowered for kw in IDENTITY_KEYWORDS)

def _identity_answer(question: str) -> str:
    return IDENTITY_SYSTEM + "\n\n" + (
        "I am an AI study assistant that can help you understand and review your uploaded documents. "
        "You can ask me questions about the content, request summaries, explanations, or quizzes. "
        "Please upload a document and ask me anything about it."
    )

# Intent classification
def _classify_intent(question: str) -> str:
    """Keyword-based intent router — zero extra LLM calls."""
    lowered = question.lower()
    if _is_identity_question(question):
        return "identity"
    if any(kw in lowered for kw in QUIZ_KEYWORDS):
        return "quiz_generation"
    if any(kw in lowered for kw in SUMMARY_KEYWORDS):
        return "summarization"
    if any(kw in lowered for kw in EXPLAIN_INTENT_KEYWORDS):
        return "explanation"
    return "qa"


# Conversational query rewriter

_REWRITE_SYSTEM = (
    "You are a query rewriting assistant. "
    "Given a conversation history and a follow-up question, rewrite the follow-up "
    "question into a complete, standalone question that can be understood without "
    "the conversation history. "
    "Output ONLY the rewritten question — no explanation, no quotation marks."
)


def _rewrite_query(question: str, history: List[HistoryMessage]) -> str:
    """Rewrite a follow-up question into a standalone query.

    If there is no history, or the question already looks self-contained
    (contains explicit references like 'this document', 'slide 3', etc.),
    we skip the rewrite to save latency.
    """
    if not history:
        return question

    # Simple heuristic: if the question has no pronouns / vague references,
    # it is likely already standalone.
    vague_tokens = ("it", "this", "that", "they", "them", "he", "she",
                    "nó", "đó", "họ", "điều này", "vấn đề này")
    lowered = question.lower()
    needs_rewrite = any(tok in lowered for tok in vague_tokens)
    if not needs_rewrite:
        return question

    # Build a short history string (last 6 turns max to keep prompt short)
    recent = history[-6:]
    history_text = "\n".join(f"{m.role.capitalize()}: {m.content}" for m in recent)
    prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Rewritten standalone question:"
    )

    try:
        rewritten = llm_engine.generate_rag_response(
            prompt=prompt,
            system_prompt=_REWRITE_SYSTEM,
            temperature=0.0,
            max_tokens=128,
        ).strip()
        if rewritten:
            logger.debug(f"Query rewritten: '{question}' → '{rewritten}'")
            return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed, using original: {exc}")

    return question

def _citation_label(hit: Dict[str, Any], idx: int) -> str:
    """Human-readable source label for a hit, e.g. 'Slide 12 – Intro'."""
    if hit.get("slide_number") is not None:
        label = f"Slide {hit['slide_number']}"
        if hit.get("slide_title"):
            label += f" – {hit['slide_title']}"
        return label
    if hit.get("page_number") is not None:
        return f"Page {hit['page_number']}"
    if hit.get("section_title"):
        return f"Section: {hit['section_title']}"
    return f"Source {idx + 1}"

_QA_SYSTEM = (
    "You are a multilingual study assistant. "
    "IMPORTANT: Always reply in the SAME LANGUAGE as the user's question. "
    "Answer ONLY based on the provided context snippets. "
    "If the context is insufficient, say so clearly — do NOT hallucinate. "
    "When referencing information, include inline citation numbers like [1] or [2] "
    "that correspond to the numbered context snippets below."
)

_SUMMARY_SYSTEM = (
    "You are a multilingual study assistant. "
    "IMPORTANT: Always reply in the SAME LANGUAGE as the user's question. "
    "Your task is to produce a concise, accurate summary of the provided context. "
    "First, identify the key point of each numbered snippet. "
    "Then combine them into a coherent paragraph. "
    "Include inline citations like [1], [2] to indicate which snippet supports each point. "
    "Do NOT add information outside the provided context."
)

_EXPLAIN_SYSTEM = (
    "You are a multilingual study assistant. "
    "IMPORTANT: Always reply in the SAME LANGUAGE as the user's question. "
    "Your task is to provide a clear, step-by-step explanation of the provided context. "
    "Use simple language and examples where appropriate. "
    "Include inline citations like [1], [2] to indicate which snippet supports each point. "
    "Do NOT add information outside the provided context."
)

IDENTITY_SYSTEM = (
    "I'm an AI study assistant. I answer based on the documents you upload "
    "(PDF, DOCX, PPTX, Markdown) and do not browse the web.\n\n"
    "I can:\n"
    "- Answer questions from your documents, with page/slide/section citations.\n"
    "- Summarize and explain the content.\n"
    "- Generate multiple-choice quizzes and flashcards by topic.\n"
    "- Adjust exercise difficulty based on your quiz results.\n\n"
    "Upload a document and ask me anything about it."
)


# Shared helpers (used by both /query and /query/stream)
def _build_prompts(intent: str, hits: List[Dict[str, Any]], standalone_query: str) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for the given retrieval hits."""
    context = "\n\n".join(
        f"[{i + 1}] ({_citation_label(hit, i)})\n{hit['content']}" for i, hit in enumerate(hits)
    )
    if intent == "summarization":
        return _SUMMARY_SYSTEM, f"Context snippets:\n{context}\n\nRequest: {standalone_query}"
    if intent == "explanation":
        return _EXPLAIN_SYSTEM, f"Context snippets:\n{context}\n\nRequest: {standalone_query}"
    if intent == "identity":
        return _IDENTITY_SYSTEM, f"Context snippets:\n{context}\n\nRequest: {standalone_query}"
    return _QA_SYSTEM, f"Context snippets:\n{context}\n\nQuestion: {standalone_query}"


def _build_citations(hits: List[Dict[str, Any]]) -> List[Citation]:
    return [
        Citation(
            document_id=hit["document_id"],
            chunk_index=hit["chunk_index"],
            score=hit.get("score", 0.0),
            content=hit["content"],
            page_number=hit.get("page_number"),
            slide_number=hit.get("slide_number"),
            slide_title=hit.get("slide_title"),
            section_title=hit.get("section_title"),
            is_summary=hit.get("is_summary", False),
        )
        for hit in hits
    ]


# Main endpoint
@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty.",
        )

    start_time = time.perf_counter()
    intent = _classify_intent(question)

    # Fast-path for quiz intent — redirect user to dedicated feature
    if intent == "quiz_generation":
        latency_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            intent=intent,
            answer=(
                "To create a quiz, please use the 'Create Quiz' feature "
                "and select the topic you want to review."
            ),
            citations=[],
            standalone_query=question,
            latency_ms=latency_ms,
        )

    if intent == "identity":
        return QueryResponse(
            intent=intent, answer=_identity_answer(question), citations=[],
            standalone_query=question,
            latency_ms=(time.perf_counter() - start_time) * 1000,
        )

    # 1. Conversational query rewrite (run in threadpool — LLM call)
    standalone_query = await run_in_threadpool(
        _rewrite_query, question, payload.history
    )

    # 2. Hybrid (dense + BM25) search; fetch_k/alpha come from settings
    hits = vector_store.search_similar_chunks(
        query=standalone_query,
        limit=payload.top_k,
        document_id=payload.document_id,
    )

    if not hits:
        latency_ms = (time.perf_counter() - start_time) * 1000
        return QueryResponse(
            intent=intent,
            answer=(
                "No relevant content found in the uploaded document. "
                "Please upload study materials before asking questions."
            ),
            citations=[],
            standalone_query=standalone_query,
            latency_ms=latency_ms,
        )

    system_prompt, user_prompt = _build_prompts(intent, hits, standalone_query)

    answer = await run_in_threadpool(
        llm_engine.generate_rag_response,
        prompt=user_prompt,
        system_prompt=system_prompt,
    )

    latency_ms = (time.perf_counter() - start_time) * 1000
    if latency_ms > settings.max_response_latency_ms:
        logger.warning(
            f"Query latency {latency_ms:.0f}ms exceeds threshold "
            f"{settings.max_response_latency_ms}ms"
        )

    citations = _build_citations(hits)

    return QueryResponse(
        intent=intent,
        answer=answer,
        citations=citations,
        standalone_query=standalone_query,
        latency_ms=latency_ms,
    )


# Streaming endpoint — reports each real pipeline step as it happens (SSE)
def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _active_model_name() -> str:
    return settings.ollama_model_name if settings.llm_backend == "ollama" else settings.llm_model_name


_INTENT_LABELS = {
    "qa": "Question answering",
    "summarization": "Summarization",
    "explanation": "Explanation",
    "quiz_generation": "Quiz request",
    "identity": "About the assistant",
}


@router.post("/query/stream")
async def query_stream(payload: QueryRequest):
    """Same pipeline as ``/query`` but streams Server-Sent Events:

    * ``{"type": "step", "id", "label", "status": "running"|"done", "detail"?, "ms"?}``
    * ``{"type": "token", "text"}``  (one per generated token/chunk, before the result)
    * ``{"type": "result", "data": <QueryResponse>}``  (always last on success)
    * ``{"type": "error", "detail"}``

    All intents that need the LLM (qa, explanation, summarization) share the same
    path: rewrite -> search -> streamed generation.
    """
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    async def events():
        start_time = time.perf_counter()
        step_started = time.perf_counter()

        def running(step_id: str, label: str) -> str:
            nonlocal step_started
            step_started = time.perf_counter()
            return _sse({"type": "step", "id": step_id, "label": label, "status": "running"})

        def done(step_id: str, label: str, detail: Optional[str] = None) -> str:
            ms = round((time.perf_counter() - step_started) * 1000)
            return _sse({"type": "step", "id": step_id, "label": label,
                         "status": "done", "detail": detail, "ms": ms})

        def final(intent: str, answer: str, citations: List[Citation], standalone_query: str) -> str:
            response = QueryResponse(
                intent=intent, answer=answer, citations=citations,
                standalone_query=standalone_query,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
            return _sse({"type": "result", "data": response.model_dump()})

        try:
            intent = _classify_intent(question)
            yield running("intent", "Understanding your question")
            yield done("intent", "Understanding your question", _INTENT_LABELS.get(intent, intent))

            # Fast paths: no retrieval, no LLM call.
            if intent == "quiz_generation":
                yield final(
                    intent,
                    "To create a quiz, please use the 'Create Quiz' feature "
                    "and select the topic you want to review.",
                    [], question,
                )
                return

            if intent == "identity":
                yield final(intent, _identity_answer(question), [], question)
                return

            # 1. Rewrite follow-up question (only when there is history)
            standalone_query = question
            if payload.history:
                yield running("rewrite", "Rewriting your follow-up as a standalone question")
                standalone_query = await run_in_threadpool(_rewrite_query, question, payload.history)
                yield done("rewrite", "Rewriting your follow-up as a standalone question",
                           standalone_query if standalone_query != question else "No rewrite needed")

            # 2. Hybrid search + re-rank
            yield running("search", "Searching your document")
            hits = await run_in_threadpool(
                vector_store.search_similar_chunks,
                query=standalone_query, limit=payload.top_k,
                document_id=payload.document_id,
            )
            if not hits:
                yield done("search", "Searching your document", "No matching passages found")
                yield final(
                    intent,
                    "No relevant content found in the uploaded document. "
                    "Please upload study materials before asking questions.",
                    [], standalone_query,
                )
                return

            labels = ", ".join(_citation_label(h, i) for i, h in enumerate(hits))
            yield done("search", "Searching your document",
                       f"Found {len(hits)} relevant passage(s): {labels}")

            # 3. Streamed generation (qa / explanation / summarization)
            system_prompt, user_prompt = _build_prompts(intent, hits, standalone_query)
            gen_label = f"Writing the answer with {_active_model_name()}"
            yield running("generate", gen_label)
            parts: List[str] = []
            async for ev in iterate_in_threadpool(
                llm_engine.stream_rag_response(user_prompt, system_prompt)
            ):
                if "token" in ev:
                    parts.append(ev["token"])
                    yield _sse({"type": "token", "text": ev["token"]})
            answer = "".join(parts).strip()
            yield done("generate", gen_label)

            latency_ms = (time.perf_counter() - start_time) * 1000
            if latency_ms > settings.max_response_latency_ms:
                logger.warning(f"Query latency {latency_ms:.0f}ms exceeds threshold "
                               f"{settings.max_response_latency_ms}ms")
            yield final(intent, answer, _build_citations(hits), standalone_query)
        except Exception as exc:
            logger.exception("Streaming query failed")
            yield _sse({"type": "error", "detail": f"Something went wrong while answering: {exc}"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )