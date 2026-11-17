import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import delete, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.conversation_store import DEFAULT_TITLE, parse_uuid, purge_stale_empty
from app.models.models import ChatMessage, Conversation, Document, User

logger = logging.getLogger(__name__)
router = APIRouter()

LIST_LIMIT = 100


class ConversationOut(BaseModel):
    id: str
    title: str
    document_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: List[Dict[str, Any]] = []
    intent: Optional[str] = None
    cached: bool = False
    created_at: Optional[datetime] = None


class ConversationDetail(ConversationOut):
    messages: List[MessageOut]


class CreateConversationRequest(BaseModel):
    document_id: Optional[str] = None


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


def _out(conv: Conversation) -> ConversationOut:
    return ConversationOut(
        id=str(conv.id), title=conv.title or DEFAULT_TITLE,
        document_id=str(conv.document_id) if conv.document_id else None,
        created_at=conv.created_at, updated_at=conv.updated_at,
    )


def _message_out(msg: ChatMessage) -> MessageOut:
    try:
        citations = json.loads(msg.citations_json) if msg.citations_json else []
    except ValueError:
        citations = []
    return MessageOut(id=str(msg.id), role=msg.role, content=msg.content, citations=citations,
                      intent=msg.intent, cached=bool(msg.cached), created_at=msg.created_at)


async def _owned(db: AsyncSession, conversation_id: str, user: User) -> Conversation:
    cid = parse_uuid(conversation_id)
    conv = None
    if cid is not None:
        conv = (await db.execute(
            select(Conversation).where(Conversation.id == cid, Conversation.owner_id == user.id)
        )).scalar_one_or_none()
    if conv is None:  # Return 404 both when it does not exist and when it belongs to someone else.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found.")
    return conv


@router.get("", response_model=List[ConversationOut])
@limiter.limit("120/minute")
async def list_conversations(
    request: Request, db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    """Mới nhất trước. Hội thoại chưa có tin nhắn nào (vừa tạo, chưa gửi) được ẩn đi."""
    has_messages = exists().where(ChatMessage.conversation_id == Conversation.id)
    rows = (await db.execute(
        select(Conversation)
        .where(Conversation.owner_id == current_user.id, has_messages)
        .order_by(Conversation.updated_at.desc())
        .limit(LIST_LIMIT)
    )).scalars().all()
    return [_out(c) for c in rows]


@router.post("", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("60/minute")
async def create_conversation(
    request: Request,
    payload: CreateConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document_id = None
    if payload.document_id:
        doc_uuid = parse_uuid(payload.document_id)
        owned = None
        if doc_uuid is not None:
            owned = (await db.execute(
                select(Document.id).where(Document.id == doc_uuid, Document.owner_id == current_user.id)
            )).first()
        if owned is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
        document_id = doc_uuid

    # 1) Clean up this user's old empty conversations (no background job required).
    await purge_stale_empty(db, settings.empty_conversation_ttl_s, current_user.id)

    # 2) Reuse an existing empty conversation instead of creating another. This prevents repeated create calls from
    #    bloating the table: each user can have at most one empty conversation at a time.
    no_messages = ~exists().where(ChatMessage.conversation_id == Conversation.id)
    reusable = (await db.execute(
        select(Conversation)
        .where(Conversation.owner_id == current_user.id, no_messages)
        .order_by(Conversation.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if reusable is not None:
        reusable.document_id = document_id
        await db.commit()
        await db.refresh(reusable)
        return _out(reusable)

    # 3) Per-user conversation limit.
    cap = settings.max_conversations_per_user
    if cap > 0:
        total = (await db.execute(
            select(func.count(Conversation.id)).where(Conversation.owner_id == current_user.id)
        )).scalar_one()
        if int(total) >= cap:
            await db.commit()  # Persist the cleanup performed in step 1.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Conversation limit reached ({cap}). Delete some old conversations to start a new one.",
            )

    conv = Conversation(id=uuid.uuid4(), owner_id=current_user.id, title="", document_id=document_id)
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    return _out(conv)


@router.get("/{conversation_id}", response_model=ConversationDetail)
@limiter.limit("120/minute")
async def get_conversation(
    request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _owned(db, conversation_id, current_user)
    messages = (await db.execute(
        select(ChatMessage).where(ChatMessage.conversation_id == conv.id).order_by(ChatMessage.created_at)
    )).scalars().all()
    return ConversationDetail(**_out(conv).model_dump(), messages=[_message_out(m) for m in messages])


@router.patch("/{conversation_id}", response_model=ConversationOut)
@limiter.limit("60/minute")
async def rename_conversation(
    request: Request,
    conversation_id: str,
    payload: RenameConversationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _owned(db, conversation_id, current_user)
    title = " ".join(payload.title.split())
    if not title:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Title cannot be empty.")
    conv.title = title
    await db.commit()
    await db.refresh(conv)
    return _out(conv)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("60/minute")
async def delete_conversation(
    request: Request,
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    conv = await _owned(db, conversation_id, current_user)
    # Delete messages explicitly (do not rely on ON DELETE CASCADE) for consistent behavior across databases.
    await db.execute(delete(ChatMessage).where(ChatMessage.conversation_id == conv.id))
    await db.execute(delete(Conversation).where(Conversation.id == conv.id))
    await db.commit()
    return None
