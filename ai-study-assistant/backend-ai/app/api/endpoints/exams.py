"""Exam-prep mode: generate an exam from a syllabus, grade it on the server, report weak topics.

The answer key is stored server-side and only revealed after a submission, so (unlike the quiz endpoint whose
``is_correct`` is client-reported) the score and per-topic statistics here cannot be forged by the client.
"""
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.endpoints.quiz import QuizSchema
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.concurrency import llm_slot, user_request
from app.core.llm_engine import llm_engine
from app.exams import logic
from app.models.models import Document, Exam, ExamAttempt, ExamQuestion, User
from app.rag.vector_store import vector_store

logger = logging.getLogger(__name__)
router = APIRouter()

_MIX_INSTRUCTION = (
    "Mix the difficulty: about 30% easy (recall), 50% medium (understanding and application) and 20% hard "
    "(analysis, edge cases). Make the wrong options plausible; never use 'all of the above' or 'none of the above'."
)


# Schemas
class GenerateExamRequest(BaseModel):
    title: str = Field(default="", max_length=200)
    syllabus: str = Field(min_length=3, max_length=6000)
    num_questions: int = Field(default=10, ge=1, le=100)
    duration_min: Optional[int] = Field(default=None, ge=1, le=600)
    document_id: Optional[str] = None


class ExamQuestionPublic(BaseModel):
    index: int
    topic: str
    question: str
    options: List[str]


class ExamPublic(BaseModel):
    id: str
    title: str
    document_id: Optional[str] = None
    duration_min: Optional[int] = None
    num_questions: int
    topics: List[str]
    skipped_topics: List[str] = []
    questions: List[ExamQuestionPublic]


class ExamSummary(BaseModel):
    id: str
    title: str
    num_questions: int
    duration_min: Optional[int] = None
    created_at: Optional[str] = None
    attempts: int = 0
    best_score: Optional[float] = None
    last_score: Optional[float] = None


class AnswerIn(BaseModel):
    index: int = Field(ge=0)
    choice: Optional[int] = Field(default=None, ge=0)


class SubmitExamRequest(BaseModel):
    answers: List[AnswerIn] = Field(max_length=200)
    duration_s: Optional[float] = Field(default=None, ge=0, le=86400)


# Helpers
def _uuid(value: str, what: str = "id") -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{what} không hợp lệ.") from None


async def _owned_exam(db: AsyncSession, exam_id: str, user: User) -> Exam:
    exam = (await db.execute(
        select(Exam).where(Exam.id == _uuid(exam_id, "exam_id"), Exam.owner_id == user.id)
    )).scalar_one_or_none()
    if exam is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy đề thi.")
    return exam


async def _questions(db: AsyncSession, exam_id: uuid.UUID) -> List[ExamQuestion]:
    return list((await db.execute(
        select(ExamQuestion).where(ExamQuestion.exam_id == exam_id).order_by(ExamQuestion.idx)
    )).scalars())


def _public(exam: Exam, questions: List[ExamQuestion], skipped: Optional[List[str]] = None) -> ExamPublic:
    topics: List[str] = []
    for q in questions:
        if q.topic not in topics:
            topics.append(q.topic)
    return ExamPublic(
        id=str(exam.id), title=exam.title, document_id=str(exam.document_id) if exam.document_id else None,
        duration_min=exam.duration_min, num_questions=len(questions), topics=topics,
        skipped_topics=skipped or [],
        questions=[ExamQuestionPublic(index=q.idx, topic=q.topic, question=q.question,
                                      options=json.loads(q.options_json or "[]")) for q in questions],
    )


def _generate_topic_questions(topic: str, count: int, document_id: Optional[str], owner_id: str) -> List[Dict[str, Any]]:
    """Blocking: retrieve context for one syllabus topic and ask the LLM for ``count`` valid MCQs."""
    hits = vector_store.search_similar_chunks(query=topic, limit=6, document_id=document_id, owner_id=owner_id)
    if not hits:
        return []
    context = "\n\n".join(h["content"] for h in hits)
    prompt = (
        f"Based ONLY on the following context, create {count} multiple-choice questions about the topic: "
        f"'{topic}'.\n{_MIX_INSTRUCTION}\n"
        "Each question needs at least 2 options with exactly one correct answer (correct_index is 0-based) and a "
        "brief explanation. Write in the same language as the context.\n\n"
        f"Context:\n{context}"
    )
    with llm_slot():
        result = llm_engine.generate_structured_output(prompt=prompt, schema_class=QuizSchema)
    cleaned = [logic.clean_mcq(q) for q in (result.get("questions") or [])]
    return [q for q in cleaned if q][:count]


# Endpoints
@router.post("/generate", response_model=ExamPublic)
@limiter.limit("6/minute")
async def generate_exam(
    request: Request, payload: GenerateExamRequest,
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Generate an exam from a syllabus: topics are parsed, questions are allocated evenly and generated per topic."""
    topics = logic.parse_syllabus(payload.syllabus, settings.exam_max_topics)
    if not topics:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Đề cương không có chủ đề hợp lệ.")
    total = min(payload.num_questions, settings.exam_max_questions)
    document_uuid = None
    if payload.document_id:
        document_uuid = _uuid(payload.document_id, "document_id")
        owned = (await db.execute(
            select(Document.id).where(Document.id == document_uuid, Document.owner_id == current_user.id)
        )).first()
        if owned is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy tài liệu.")

    count = (await db.execute(select(func.count(Exam.id)).where(Exam.owner_id == current_user.id))).scalar_one()
    if count >= settings.max_exams_per_user:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Bạn đã đạt số đề thi tối đa. Hãy xoá bớt đề cũ.")

    owner_id = str(current_user.id)
    generated: List[Dict[str, Any]] = []
    skipped: List[str] = []
    seen = set()
    with user_request(owner_id):
        for topic, n in logic.allocate_questions(total, topics):
            try:
                items = await run_in_threadpool(_generate_topic_questions, topic, n, payload.document_id, owner_id)
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad topic must not kill the whole exam
                logger.warning("Exam topic %r failed: %s", topic, exc)
                items = []
            fresh = []
            for q in items:
                key = q["question"].lower()
                if key not in seen:
                    seen.add(key)
                    fresh.append({**q, "topic": topic})
            if fresh:
                generated += fresh
            else:
                skipped.append(topic)

    if not generated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Can not generate any questions from the syllabus topics. Please check your syllabus or try again later.",
        )

    exam = Exam(owner_id=current_user.id, document_id=document_uuid,
                title=(payload.title.strip() or f"Đề thi {topics[0]}")[:200], syllabus=payload.syllabus,
                num_questions=len(generated), duration_min=payload.duration_min)
    db.add(exam)
    await db.flush()
    rows = [ExamQuestion(exam_id=exam.id, idx=i, topic=q["topic"][:255], question=q["question"],
                         options_json=json.dumps(q["options"], ensure_ascii=False),
                         correct_index=q["correct_index"], explanation=q.get("explanation", ""))
            for i, q in enumerate(generated)]
    db.add_all(rows)
    await db.commit()
    return _public(exam, rows, skipped)


@router.get("", response_model=List[ExamSummary])
@limiter.limit("120/minute")
async def list_exams(request: Request, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    exams = list((await db.execute(
        select(Exam).where(Exam.owner_id == current_user.id).order_by(Exam.created_at.desc()).limit(200)
    )).scalars())
    attempts = list((await db.execute(
        select(ExamAttempt).where(ExamAttempt.owner_id == current_user.id).order_by(ExamAttempt.created_at)
    )).scalars())
    by_exam: Dict[uuid.UUID, List[ExamAttempt]] = {}
    for a in attempts:
        by_exam.setdefault(a.exam_id, []).append(a)
    out = []
    for e in exams:
        mine = by_exam.get(e.id, [])
        out.append(ExamSummary(
            id=str(e.id), title=e.title, num_questions=e.num_questions, duration_min=e.duration_min,
            created_at=e.created_at.isoformat() if e.created_at else None, attempts=len(mine),
            best_score=max((a.score for a in mine), default=None), last_score=mine[-1].score if mine else None,
        ))
    return out


@router.get("/report/weak-topics")
@limiter.limit("60/minute")
async def weak_topics_report(
    request: Request, exam_id: Optional[str] = None, document_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Per-topic accuracy over the user's attempts, weakest first (Beta-smoothed, with Wilson interval)."""
    query = select(ExamAttempt).where(ExamAttempt.owner_id == current_user.id)
    if exam_id:
        query = query.where(ExamAttempt.exam_id == _uuid(exam_id, "exam_id"))
    if document_id:
        query = query.join(Exam, Exam.id == ExamAttempt.exam_id).where(Exam.document_id == _uuid(document_id, "document_id"))
    attempts = list((await db.execute(query.order_by(ExamAttempt.created_at.desc()).limit(200))).scalars())
    attempts.reverse()  # oldest -> newest
    rows = logic.topic_report([json.loads(a.per_topic_json or "{}") for a in attempts],
                              settings.exam_weak_threshold, settings.exam_weak_min_questions)
    weak = [r["topic"] for r in rows if r["weak"]]
    return {
        "attempts_considered": len(attempts),
        "threshold": settings.exam_weak_threshold,
        "topics": rows,
        "weak_topics": weak,
        "recommendations": [
            {"topic": t, "action": "quiz", "message": f"Ôn lại '{t}': tạo quiz luyện tập theo chủ đề này."} for t in weak
        ],
    }


@router.get("/{exam_id}", response_model=ExamPublic)
@limiter.limit("120/minute")
async def get_exam(request: Request, exam_id: str, db: AsyncSession = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    exam = await _owned_exam(db, exam_id, current_user)
    return _public(exam, await _questions(db, exam.id))


@router.post("/{exam_id}/submit")
@limiter.limit("30/minute")
async def submit_exam(
    request: Request, exam_id: str, payload: SubmitExamRequest,
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Grade on the server; returns the score, per-topic breakdown, weak topics and the answer review."""
    exam = await _owned_exam(db, exam_id, current_user)
    rows = await _questions(db, exam.id)
    by_index = {q.idx: q for q in rows}

    answers: Dict[int, Optional[int]] = {}
    for a in payload.answers:
        q = by_index.get(a.index)
        if q is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Câu hỏi {a.index} không tồn tại.")
        if a.index in answers:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Câu hỏi {a.index} bị gửi trùng.")
        if a.choice is not None and a.choice >= len(json.loads(q.options_json or "[]")):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Đáp án của câu {a.index} không hợp lệ.")
        answers[a.index] = a.choice

    graded = logic.grade_attempt(
        [{"index": q.idx, "topic": q.topic, "question": q.question, "options": json.loads(q.options_json or "[]"),
          "correct_index": q.correct_index, "explanation": q.explanation} for q in rows],
        answers,
    )
    attempt = ExamAttempt(
        exam_id=exam.id, owner_id=current_user.id,
        answers_json=json.dumps([{"index": i, "choice": c} for i, c in sorted(answers.items())]),
        correct=graded["correct"], total=graded["total"], score=graded["score"],
        per_topic_json=json.dumps(graded["per_topic"], ensure_ascii=False), duration_s=payload.duration_s,
    )
    db.add(attempt)
    await db.commit()

    topics = []
    for topic, res in graded["per_topic"].items():
        acc = res["correct"] / res["total"] if res["total"] else 0.0
        topics.append({"topic": topic, "correct": res["correct"], "total": res["total"], "accuracy": round(acc, 3),
                       "weak": acc < settings.exam_weak_threshold})
    topics.sort(key=lambda t: t["accuracy"])
    overtime = bool(exam.duration_min and payload.duration_s and payload.duration_s > exam.duration_min * 60)
    return {
        "attempt_id": str(attempt.id), "score": graded["score"], "correct": graded["correct"],
        "total": graded["total"], "unanswered": graded["unanswered"], "overtime": overtime,
        "per_topic": topics, "weak_topics": [t["topic"] for t in topics if t["weak"]],
        "review": graded["review"],
    }


@router.delete("/{exam_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_exam(request: Request, exam_id: str, db: AsyncSession = Depends(get_db),
                      current_user: User = Depends(get_current_user)):
    exam = await _owned_exam(db, exam_id, current_user)
    # Explicit child deletes keep this correct on databases without FK enforcement (SQLite in tests).
    await db.execute(delete(ExamAttempt).where(ExamAttempt.exam_id == exam.id))
    await db.execute(delete(ExamQuestion).where(ExamQuestion.exam_id == exam.id))
    await db.execute(delete(Exam).where(Exam.id == exam.id))
    await db.commit()
