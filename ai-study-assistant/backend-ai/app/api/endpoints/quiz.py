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

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

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
    # difficulty có thể được client gợi ý, nhưng backend luôn ưu tiên
    # giá trị đọc từ RL state. Trường này chỉ là fallback khi user chưa có
    # lịch sử (mặc định "medium").
    difficulty: Optional[str] = Field(default=None, pattern=r"^(easy|medium|hard)$")


class GenerateQuizResponse(BaseModel):
    quiz_id: str
    topic: str
    questions: List[QuizQuestion]
    difficulty: str  # độ khó đã được áp dụng — để frontend hiển thị


class SubmitQuizRequest(BaseModel):
    document_id: Optional[str] = None
    is_correct: bool
    response_time_s: float = Field(ge=0)
    hints_used: int = Field(default=0, ge=0)


class SubmitQuizResponse(BaseModel):
    attempt_id: str
    next_difficulty: str  # độ khó cho bài tiếp theo — frontend dùng để hiển thị


# ---------------------------------------------------------------------------
# Difficulty prompt fragments
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/generate", response_model=GenerateQuizResponse)
async def generate_quiz(
    payload: GenerateQuizRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Sinh bộ câu hỏi theo độ khó phù hợp với trình độ hiện tại của user."""
    # 1) Lấy difficulty từ RL state của user (backend-authoritative)
    rl_difficulty = await rl_agent.get_current_difficulty(
        user_id=str(current_user.id), db=db
    )
    # Nếu client gợi ý difficulty và user chưa có lịch sử (index=0, default easy),
    # ưu tiên RL state — không cho client override để tránh gian lận độ khó.
    difficulty = rl_difficulty

    # 2) Lấy context từ vector store
    hits = vector_store.search_similar_chunks(
        query=payload.topic, limit=8, document_id=payload.document_id
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
        result = await run_in_threadpool(
            llm_engine.generate_structured_output, prompt=prompt, schema_class=QuizSchema
        )
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
        quiz_id=str(uuid.uuid4()),
        topic=result.get("topic", payload.topic),
        questions=valid_questions,
        difficulty=difficulty,
    )


@router.post("/{quiz_id}/submit", response_model=SubmitQuizResponse)
async def submit_quiz_result(
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

    # Lưu attempt vào DB
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

    # Cập nhật RL state (async, persistent)
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
