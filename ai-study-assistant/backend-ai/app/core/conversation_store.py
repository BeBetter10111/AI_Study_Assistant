"""Lưu / đọc lịch sử hội thoại hỏi đáp (bảng conversations, chat_messages).

Dùng phiên DB RIÊNG, ngắn hạn (``coredb.async_session()``) thay vì phiên của request: endpoint
``/query/stream`` ghi tin nhắn từ bên trong generator đang stream, lúc đó phiên của request có thể đã
được dependency đóng. Mọi hàm đều lọc theo ``owner_id`` nên không đọc/ghi được hội thoại của người khác.

Việc lưu lịch sử KHÔNG được làm hỏng chat: các hàm ghi bắt lỗi, ghi log và trả False.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, exists, func, select, update

from app.core import database as coredb
from app.models.models import ChatMessage, Conversation

logger = logging.getLogger(__name__)

DEFAULT_TITLE = "Cuộc trò chuyện mới"
TITLE_MAX = 60


def parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError):
        return None


def make_title(question: str) -> str:
    text = " ".join(question.split())
    return text if len(text) <= TITLE_MAX else text[: TITLE_MAX - 1].rstrip() + "…"


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def get_owned(conversation_id: str, owner_id: str) -> Optional[Dict[str, Any]]:
    """Trả về {id, title, document_id, message_count} nếu hội thoại thuộc về user, ngược lại None."""
    cid, oid = parse_uuid(conversation_id), parse_uuid(owner_id)
    if cid is None or oid is None:
        return None
    async with coredb.async_session() as session:
        row = (await session.execute(
            select(Conversation.id, Conversation.title, Conversation.document_id)
            .where(Conversation.id == cid, Conversation.owner_id == oid)
        )).first()
        if row is None:
            return None
        count = (await session.execute(
            select(func.count(ChatMessage.id)).where(ChatMessage.conversation_id == cid)
        )).scalar_one()
    return {"id": row.id, "title": row.title, "document_id": row.document_id, "message_count": int(count)}


async def recent_history(conversation_id: str, owner_id: str, limit: int = 6) -> List[Dict[str, str]]:
    """``limit`` tin gần nhất (cũ -> mới) dạng {role, content}; dùng để viết lại câu hỏi nối tiếp."""
    cid, oid = parse_uuid(conversation_id), parse_uuid(owner_id)
    if cid is None or oid is None:
        return []
    try:
        async with coredb.async_session() as session:
            rows = (await session.execute(
                select(ChatMessage.role, ChatMessage.content)
                .join(Conversation, Conversation.id == ChatMessage.conversation_id)
                .where(ChatMessage.conversation_id == cid, Conversation.owner_id == oid)
                .order_by(ChatMessage.created_at.desc())
                .limit(limit)
            )).all()
    except Exception:  # noqa: BLE001
        logger.exception("Could not load conversation history")
        return []
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


async def append_message(
    conversation_id: str,
    owner_id: str,
    role: str,
    content: str,
    *,
    citations: Optional[List[Dict[str, Any]]] = None,
    intent: Optional[str] = None,
    cached: bool = False,
    title_from: Optional[str] = None,
) -> bool:
    """Thêm một tin nhắn; cập nhật ``updated_at`` và (nếu còn trống) đặt tiêu đề từ ``title_from``."""
    cid, oid = parse_uuid(conversation_id), parse_uuid(owner_id)
    if cid is None or oid is None:
        return False
    try:
        async with coredb.async_session() as session:
            conv = (await session.execute(
                select(Conversation).where(Conversation.id == cid, Conversation.owner_id == oid)
            )).scalar_one_or_none()
            if conv is None:
                return False
            session.add(ChatMessage(
                conversation_id=cid, role=role, content=content,
                citations_json=json.dumps(citations, ensure_ascii=False) if citations else None,
                intent=intent, cached=bool(cached), created_at=_now(),
            ))
            values: Dict[str, Any] = {"updated_at": _now()}
            if title_from and not (conv.title or "").strip():
                values["title"] = make_title(title_from)
            await session.execute(update(Conversation).where(Conversation.id == cid).values(**values))
            await session.commit()
        return True
    except Exception:  # noqa: BLE001 - Saving history must not break chat.
        logger.exception("Could not save chat message")
        return False


async def purge_stale_empty(session, ttl_s: int, owner_id=None) -> int:
    """Xóa hội thoại RỖNG (chưa từng có tin nhắn) tạo cách đây quá ``ttl_s`` giây. Trả về số dòng đã xóa.

    Hội thoại rỗng sinh ra khi người dùng mở trò chuyện mới nhưng chưa gửi gì (hoặc gửi lỗi trước khi lưu). Chúng
    bị ẩn khỏi danh sách nên không ai dọn; nếu không xóa thì chỉ là rác tích lũy. Không tự commit: caller commit.
    """
    cutoff = _now() - timedelta(seconds=max(0, ttl_s))
    stmt = delete(Conversation).where(
        ~exists().where(ChatMessage.conversation_id == Conversation.id),
        Conversation.created_at < cutoff,
    )
    if owner_id is not None:
        stmt = stmt.where(Conversation.owner_id == owner_id)
    result = await session.execute(stmt)
    return int(result.rowcount or 0)
