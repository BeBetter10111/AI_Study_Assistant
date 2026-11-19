import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.security.rate_limiter import limiter
from app.core.concurrency import llm_slot, user_request
from app.core.llm_engine import llm_engine
from app.core.quiz_tokens import issue_quiz_id, verify_quiz_id
from app.models.models import Document, QuizAttempt, User
from app.rag.vector_store import vector_store
from app.rl import agent as rl_agent

logger = logging.getLogger(__name__)
router = APIRouter()

# Schemas
class QuizQuestion(BaseModel):
    question: str
    options: List[str] = Field(min_length=2, max_length=8)
    correct_index: int
    explanation: str


class QuizSchema(BaseModel):
    topic: str
    questions: List[QuizQuestion]


class GenerateQuizRequest(BaseModel):
    topic: str = Field(min_length=1, max_length=200)
    num_questions: int = Field(default=10, ge=1, le=50)
    document_id: Optional[str] = None
    # The client may suggest a difficulty, but the backend always prioritizes
    # the value read from the RL state. This field is only a fallback when the user has no
    # history (default "medium").
    difficulty: Optional[str] = Field(default=None, pattern=r"^(easy|medium|hard)$")


class GenerateQuizResponse(BaseModel):
    quiz_id: str
    topic: str
    questions: List[QuizQuestion]
    difficulty: str  # Applied difficulty — displayed by the frontend.


class SubmitQuizRequest(BaseModel):
    document_id: Optional[str] = None
    is_correct: bool
    response_time_s: float = Field(ge=0)
    hints_used: int = Field(default=0, ge=0)


class SubmitQuizResponse(BaseModel):
    attempt_id: str
    next_difficulty: str  # Difficulty for the next quiz — displayed by the frontend.


# Difficulty prompt fragments
_DIFFICULTY_INSTRUCTION: dict[str, str] = {
    "easy": (
        "The questions should be straightforward and test basic recall. "
        "Use simple language. Avoid trick questions or advanced analysis."
    ),
    "medium": (
        "The questions should require understanding, not just memorisation. "
        "Include some application and analysis questions alongside recall."
    ),
    "hard": (
        "The questions should be challenging: focus on deep understanding, "
        "critical thinking, edge cases, and synthesis across concepts. "
        "Avoid purely factual recall questions."
    ),
}


# Endpoints
@router.post("/generate", response_model=GenerateQuizResponse)
@limiter.limit("20/minute")
async def generate_quiz(
    request: Request,
    payload: GenerateQuizRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Sinh bộ câu hỏi theo độ khó phù hợp với trình độ hiện tại của user."""
    # Limit the number of concurrent resource-intensive requests per user (see app/core/concurrency.py).
    with user_request(str(current_user.id)):
        return await _generate_quiz(payload, db, current_user)


async def _generate_quiz(payload: GenerateQuizRequest, db: AsyncSession, current_user: User):
    # 1) Get the user's difficulty from the RL state (backend-authoritative)
    rl_difficulty = await rl_agent.get_current_difficulty(
        user_id=str(current_user.id), db=db
    )
    # If the client suggests a difficulty and the user has no history (index=0, default easy),
    # prioritize the RL state — do not let the client override it and game the difficulty.
    difficulty = rl_difficulty

    # 2) Get context from the vector store — scoped to the authenticated user's documents
    #    Use a thread pool to avoid blocking the event loop with embedding/Qdrant/re-ranker work (see chat.py).
    hits = await run_in_threadpool(
        vector_store.search_similar_chunks,
        query=payload.topic, limit=8,
        document_id=payload.document_id,
        owner_id=str(current_user.id),
    )
    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Không tìm thấy nội dung liên quan đến chủ đề này. "
                "Hãy upload tài liệu liên quan trước khi tạo quiz."
            ),
        )

    context = "\n\n".join(hit["content"] for hit in hits)
    difficulty_instruction = _DIFFICULTY_INSTRUCTION[difficulty]

    prompt = (
        f"Based on the following context, please create {payload.num_questions} "
        f"multiple-choice questions about the topic: '{payload.topic}'.\n\n"
        f"Difficulty level: {difficulty.upper()}\n"
        f"{difficulty_instruction}\n\n"
        f"Each question must have at least 2 options, with only 1 correct answer "
        f"(the correct_index is a 0-based index in the options array), along with "
        f"a brief explanation.\n\n"
        f"Context:\n{context}"
    )

    try:
        with llm_slot():  # No slot available => 429 "busy" (HTTPException passes through the `except Exception` below).
            result = await run_in_threadpool(
                llm_engine.generate_structured_output, prompt=prompt, schema_class=QuizSchema
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error generating quiz: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Không thể tạo quiz từ mô hình ngôn ngữ.",
        ) from exc

    questions = result.get("questions") or []
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Mô hình ngôn ngữ không trả về câu hỏi hợp lệ.",
        )

    valid_questions = [
        q for q in questions if 0 <= q.get("correct_index", -1) < len(q.get("options", []))
    ]
    if not valid_questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Không có câu hỏi nào hợp lệ được tạo ra.",
        )

    return GenerateQuizResponse(
        quiz_id=issue_quiz_id(str(current_user.id), len(valid_questions)),
        topic=result.get("topic", payload.topic),
        questions=valid_questions,
        difficulty=difficulty,
    )


@router.post("/{quiz_id}/submit", response_model=SubmitQuizResponse)
@limiter.limit("60/minute")
async def submit_quiz_result(
    request: Request,
    quiz_id: str,
    payload: SubmitQuizRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Nộp kết quả một câu, cập nhật RL state và trả về difficulty tiếp theo."""
    try:
        quiz_uuid = uuid.UUID(quiz_id)
        document_uuid = uuid.UUID(payload.document_id) if payload.document_id else None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="quiz_id hoặc document_id không hợp lệ.",
        ) from exc

    allowed = verify_quiz_id(str(current_user.id), quiz_uuid)
    if allowed is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="quiz_id không hợp lệ.")
    submitted = (await db.execute(
        select(func.count(QuizAttempt.id)).where(
            QuizAttempt.user_id == current_user.id, QuizAttempt.quiz_id == quiz_uuid
        )
    )).scalar_one()
    if submitted >= allowed:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Quiz này đã nộp đủ kết quả.")

    # document_id must refer to this user's document. Previously, an unknown ID violated the Postgres foreign key
    # -> HTTP 500, and could associate an attempt with another user's document.
    if document_uuid is not None:
        owned = await db.execute(
            select(Document.id).where(
                Document.id == document_uuid, Document.owner_id == current_user.id
            )
        )
        if owned.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
            )

    # Save the attempt to the DB
    attempt = QuizAttempt(
        user_id=current_user.id,
        document_id=document_uuid,
        quiz_id=quiz_uuid,
        is_correct=payload.is_correct,
        response_time_s=payload.response_time_s,
        hints_used=payload.hints_used,
    )
    db.add(attempt)
    await db.commit()
    await db.refresh(attempt)

    # Update the RL state (async, persistent)
    next_difficulty = await rl_agent.update_state(
        user_id=str(current_user.id),
        quiz_result={
            "is_correct": payload.is_correct,
            "response_time_s": payload.response_time_s,
            "hints_used": payload.hints_used,
        },
        db=db,
    )

    return SubmitQuizResponse(
        attempt_id=str(attempt.id),
        next_difficulty=next_difficulty,
    )
