import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.core.llm_engine import llm_engine
from app.models.models import QuizAttempt, User
from app.rag.vector_store import vector_store
from app.rl import agent as rl_agent

logger = logging.getLogger(__name__)
router = APIRouter()

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


class GenerateQuizResponse(BaseModel):
    quiz_id: str
    topic: str
    questions: List[QuizQuestion]


class SubmitQuizRequest(BaseModel):
    document_id: Optional[str] = None
    is_correct: bool
    response_time_s: float = Field(ge=0)
    hints_used: int = Field(default=0, ge=0)


@router.post("/generate", response_model=GenerateQuizResponse)
async def generate_quiz(payload: GenerateQuizRequest):
    hits = vector_store.search_similar_chunks(
        query=payload.topic, limit=8, document_id=payload.document_id
    )
    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Can not find relevant context for the topic. "
            "Please upload relevant documents before creating a quiz.",
        )

    context = "\n\n".join(hit["content"] for hit in hits)
    prompt = (
        f"Based on the following context, please create {payload.num_questions} multiple-choice questions "
        f"about the topic: '{payload.topic}'. Each question must have at least 2 options, with only 1 correct answer "
        f"(the correct_index is a 0-based index in the options array), along with a brief explanation.\n\n"
        f"Context:\n{context}"
    )

    try:
        result = await run_in_threadpool(
            llm_engine.generate_structured_output, prompt=prompt, schema_class=QuizSchema
        )
    except Exception as exc:
        logger.error(f"Error generating quiz: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Can not generate quiz from the language model.",
        ) from exc

    questions = result.get("questions") or []
    if not questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail="Can not generate valid questions from the language model."
        )

    valid_questions = [
        q for q in questions if 0 <= q.get("correct_index", -1) < len(q.get("options", []))
    ]
    if not valid_questions:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Can not generate valid questions from the language model.",
        )

    return GenerateQuizResponse(
        quiz_id=str(uuid.uuid4()),
        topic=result.get("topic", payload.topic),
        questions=valid_questions,
    )


@router.post("/{quiz_id}/submit")
async def submit_quiz_result(
    quiz_id: str,
    payload: SubmitQuizRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        quiz_uuid = uuid.UUID(quiz_id)
        document_uuid = uuid.UUID(payload.document_id) if payload.document_id else None
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="quiz_id or document_id is invalid.",
        ) from exc

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

    next_difficulty = rl_agent.update_state(
        user_id=str(current_user.id),
        quiz_result={
            "is_correct": payload.is_correct,
            "response_time_s": payload.response_time_s,
            "hints_used": payload.hints_used,
        },
    )

    return {"attempt_id": str(attempt.id), "next_difficulty": next_difficulty}
