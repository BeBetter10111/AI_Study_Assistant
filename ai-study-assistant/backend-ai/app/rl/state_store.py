"""Persistent RL state store — Redis (L1 cache) + PostgreSQL (L2 ground truth).

Kiến trúc 2 lớp:
  L1 Redis:      đọc/ghi nhanh trong request cycle, TTL 120 ngày.
  L2 PostgreSQL: bền vững qua restart / multi-instance, là source of truth.

Nếu Redis không khả dụng, module tự fallback đọc/ghi thẳng DB — không
crash service. Nếu DB cũng lỗi, trả về state mặc định (easy, history=[]).
"""
from __future__ import annotations

import json
import logging
import uuid as _uuid_mod
from typing import Any

import redis.asyncio as aioredis
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.models import UserRLState

logger = logging.getLogger(__name__)

_REDIS_TTL_S = 60 * 60 * 24 * 120
_REDIS_KEY_PREFIX = "rl:state:"
_MAX_HISTORY = 200 


def _redis_key(user_id: str) -> str:
    return f"{_REDIS_KEY_PREFIX}{user_id}"


def _make_redis() -> aioredis.Redis:
    return aioredis.from_url(settings.redis_url, decode_responses=True)


def _default_state() -> dict[str, Any]:
    return {"difficulty_index": 0, "reward_history": []}


# Public API
async def load_state(user_id: str, db: AsyncSession) -> dict[str, Any]:
    """Đọc trạng thái RL. Thử Redis trước, fallback PostgreSQL."""
    try:
        async with _make_redis() as r:
            raw = await r.get(_redis_key(user_id))
        if raw:
            return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RL state: Redis read failed, falling back to DB (%s)", exc)

    return await _db_load(user_id, db)


async def save_state(user_id: str, state: dict[str, Any], db: AsyncSession) -> None:
    """Ghi state vào Redis và PostgreSQL."""
    # Trim lịch sử trước khi lưu
    history = state.get("reward_history", [])
    if len(history) > _MAX_HISTORY:
        state = {**state, "reward_history": history[-_MAX_HISTORY:]}

    # L1: Redis
    try:
        async with _make_redis() as r:
            await r.set(_redis_key(user_id), json.dumps(state), ex=_REDIS_TTL_S)
    except Exception as exc:  # noqa: BLE001
        logger.warning("RL state: Redis write failed (%s)", exc)

    # L2: PostgreSQL (upsert)
    await _db_save(user_id, state, db)


# DB helpers
async def _db_load(user_id: str, db: AsyncSession) -> dict[str, Any]:
    try:
        uid = _uuid_mod.UUID(user_id)
        result = await db.execute(
            select(UserRLState).where(UserRLState.user_id == uid)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return _default_state()
        return {
            "difficulty_index": row.difficulty_index,
            "reward_history": json.loads(row.reward_history_json or "[]"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("RL state: DB read failed (%s)", exc)
        return _default_state()


async def _db_save(user_id: str, state: dict[str, Any], db: AsyncSession) -> None:
    try:
        uid = _uuid_mod.UUID(user_id)
        stmt = (
            pg_insert(UserRLState)
            .values(
                user_id=uid,
                difficulty_index=state["difficulty_index"],
                reward_history_json=json.dumps(state.get("reward_history", [])),
            )
            .on_conflict_do_update(
                index_elements=["user_id"],
                set_={
                    "difficulty_index": state["difficulty_index"],
                    "reward_history_json": json.dumps(state.get("reward_history", [])),
                },
            )
        )
        await db.execute(stmt)
        await db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.error("RL state: DB write failed (%s)", exc)
        await db.rollback()
