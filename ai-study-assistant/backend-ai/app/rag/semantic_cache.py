"""Semantic cache trên Redis: câu hỏi tương tự câu đã trả lời => trả lại câu trả lời, bỏ qua
tìm kiếm + re-rank + LLM.

Chạy trên Redis thường (``redis:7-alpine`` trong docker-compose) — không cần RediSearch/Redis Stack.
Mỗi phạm vi chỉ giữ vài trăm mục nên so cosine bằng numpy là đủ nhanh (vài ms).

An toàn dữ liệu (quan trọng hơn tốc độ)
---------------------------------------
* KHÓA PHẠM VI gồm ``owner_id`` => không bao giờ trả câu trả lời của người dùng này cho người khác.
* Phạm vi còn gồm tài liệu được chọn, intent (qa/explanation/summarization), ``top_k`` và "dấu vân tay"
  của model (LLM + embedding + re-ranker): đổi model là cache cũ tự không còn khớp.
* ``generation`` theo từng user: bump khi có tài liệu mới được nhúng xong => câu trả lời cũ (dựa trên
  tập tài liệu cũ) không bị dùng lại. Mục cũ tự hết hạn theo TTL.
* Hai câu hỏi chỉ khác nhau về SỐ ("slide 3" vs "slide 4", "chương 2" vs "chương 5") KHÔNG bao giờ
  được coi là trùng, dù embedding rất giống nhau.
* Chỉ lưu câu trả lời HOÀN CHỈNH, có nguồn trích dẫn. Không lưu lỗi, "không tìm thấy nội dung",
  hay câu trả lời bị ngắt giữa chừng.

Chịu lỗi (fail-open)
--------------------
Redis chậm / chết => bỏ qua cache (log cảnh báo) và chat vẫn chạy bình thường. Sau một lỗi, cache bị
tạm tắt ``_COOLDOWN_S`` giây để không phải chờ timeout ở mỗi request.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from starlette.concurrency import run_in_threadpool

from app.config import settings

logger = logging.getLogger(__name__)

PREFIX = "asa:sc"
_COOLDOWN_S = 30.0
_DIGITS = re.compile(r"\d+")


def _numbers(text: str) -> frozenset:
    return frozenset(_DIGITS.findall(text))


def _normalise(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


def generation_key(owner_id: str) -> str:
    return f"{PREFIX}:gen:{owner_id}"


@dataclass
class CacheHit:
    payload: Dict[str, Any]
    similarity: float
    matched_query: str


class CacheSession:
    """Một lượt hỏi: giữ embedding + khóa đã tính để lookup() rồi store() không phải tính lại."""

    def __init__(self, cache: "SemanticCache", embedding: np.ndarray, query: str, index_key: str):
        self._cache = cache
        self.embedding = embedding
        self.query = query
        self.index_key = index_key

    async def lookup(self) -> Optional[CacheHit]:
        return await self._cache._lookup(self)

    async def store(self, payload: Dict[str, Any]) -> bool:
        return await self._cache._store(self, payload)


class SemanticCache:
    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        fingerprint_fn: Optional[Callable[[], str]] = None,
        client_factory: Optional[Callable[[], Any]] = None,
    ):
        self._embed_fn = embed_fn
        self._fingerprint_fn = fingerprint_fn
        self._client_factory = client_factory
        self._client: Any = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._down_until = 0.0

    # ---- hạ tầng ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(settings.semantic_cache_enabled)

    def _available(self) -> bool:
        return self.enabled and time.monotonic() >= self._down_until

    def _trip(self, exc: BaseException) -> None:
        self._down_until = time.monotonic() + _COOLDOWN_S
        logger.warning(f"Semantic cache unavailable ({type(exc).__name__}: {exc}) — "
                       f"skipping it for {int(_COOLDOWN_S)}s; chat continues without cache.")

    def _redis(self):
        # Client asyncio gắn với event loop đã tạo ra nó -> tạo lại nếu loop đổi (vd. giữa các test).
        loop = asyncio.get_running_loop()
        if self._client is None or self._client_loop is not loop:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    settings.redis_url,
                    socket_timeout=settings.semantic_cache_timeout_s,
                    socket_connect_timeout=settings.semantic_cache_timeout_s,
                    health_check_interval=0,
                )
            self._client_loop = loop
        return self._client

    def _embed(self, text: str) -> np.ndarray:
        if self._embed_fn is not None:
            return self._embed_fn(text)
        from app.rag import vector_store as vs

        return vs.embed_query(text)

    def _fingerprint(self) -> str:
        if self._fingerprint_fn is not None:
            return self._fingerprint_fn()
        from app.rag import vector_store as vs

        llm = settings.ollama_model_name if settings.llm_backend == "ollama" else settings.llm_model_name
        return f"{settings.llm_backend}:{llm}|{vs.model_fingerprint()}"

    # ---- API ----------------------------------------------------------------------
    async def open(
        self, owner_id: str, document_id: Optional[str], intent: str, top_k: int, query: str
    ) -> Optional[CacheSession]:
        """Chuẩn bị một lượt tra cứu. Trả None nếu cache tắt/không dùng được."""
        if not self._available() or not query.strip():
            return None
        try:
            embedding, fingerprint = await run_in_threadpool(
                lambda: (_normalise(self._embed(query)), self._fingerprint())
            )
            generation = int(await self._redis().get(generation_key(owner_id)) or 0)
        except Exception as exc:  # noqa: BLE001 - cache KHÔNG được làm hỏng chat
            self._trip(exc)
            return None
        scope = hashlib.sha1(
            f"{document_id or '*'}|{intent}|{top_k}|{fingerprint}".encode("utf-8")
        ).hexdigest()[:16]
        return CacheSession(self, embedding, query, f"{PREFIX}:idx:{owner_id}:{generation}:{scope}")

    async def _lookup(self, session: CacheSession) -> Optional[CacheHit]:
        threshold = float(settings.semantic_cache_threshold)
        try:
            r = self._redis()
            ids: List[bytes] = await r.zrevrange(session.index_key, 0, settings.semantic_cache_max_entries - 1)
            if not ids:
                return None
            pipe = r.pipeline(transaction=False)
            for entry_id in ids:
                pipe.hmget(self._entry_key(entry_id), "emb", "q")
            rows = await pipe.execute()

            query_numbers = _numbers(session.query)
            best_id, best_sim, best_q = None, -1.0, ""
            stale: List[bytes] = []
            for entry_id, (emb, q) in zip(ids, rows):
                if emb is None:  # mục đã hết hạn theo TTL nhưng còn trong chỉ mục
                    stale.append(entry_id)
                    continue
                vec = np.frombuffer(emb, dtype=np.float32)
                if vec.shape != session.embedding.shape:
                    continue  # model embedding khác chiều -> bỏ qua
                sim = float(np.dot(vec, session.embedding))
                matched = q.decode("utf-8") if isinstance(q, (bytes, bytearray)) else str(q or "")
                if sim > best_sim and _numbers(matched) == query_numbers:
                    best_id, best_sim, best_q = entry_id, sim, matched
            if stale:
                await r.zrem(session.index_key, *stale)
            if best_id is None or best_sim < threshold:
                return None

            raw = await r.hget(self._entry_key(best_id), "payload")
            if raw is None:
                return None
            payload = json.loads(raw)
            # LRU: mục vừa được dùng đẩy lên đầu, và đếm số lần trúng cho dễ theo dõi.
            await r.zadd(session.index_key, {best_id: time.time()})
            await r.hincrby(self._entry_key(best_id), "hits", 1)
            return CacheHit(payload=payload, similarity=best_sim, matched_query=best_q)
        except Exception as exc:  # noqa: BLE001
            self._trip(exc)
            return None

    async def _store(self, session: CacheSession, payload: Dict[str, Any]) -> bool:
        try:
            r = self._redis()
            entry_id = uuid.uuid4().hex
            key = self._entry_key(entry_id)
            ttl = int(settings.semantic_cache_ttl_s)
            pipe = r.pipeline(transaction=False)
            pipe.hset(key, mapping={
                "q": session.query,
                "emb": session.embedding.astype(np.float32).tobytes(),
                "payload": json.dumps(payload, ensure_ascii=False),
                "hits": 0,
            })
            pipe.expire(key, ttl)
            pipe.zadd(session.index_key, {entry_id: time.time()})
            pipe.expire(session.index_key, ttl)
            await pipe.execute()

            # Giới hạn kích thước phạm vi: bỏ các mục cũ nhất (ít được dùng nhất).
            limit = int(settings.semantic_cache_max_entries)
            extra = int(await r.zcard(session.index_key)) - limit
            if extra > 0:
                old = await r.zrange(session.index_key, 0, extra - 1)
                if old:
                    await r.zrem(session.index_key, *old)
                    await r.delete(*[self._entry_key(o) for o in old])
            return True
        except Exception as exc:  # noqa: BLE001
            self._trip(exc)
            return False

    @staticmethod
    def _entry_key(entry_id) -> str:
        if isinstance(entry_id, (bytes, bytearray)):
            entry_id = entry_id.decode("ascii")
        return f"{PREFIX}:e:{entry_id}"

    async def invalidate_owner(self, owner_id: str) -> None:
        """Vô hiệu toàn bộ cache của một user (tăng generation; mục cũ tự hết hạn theo TTL)."""
        try:
            await self._redis().incr(generation_key(owner_id))
        except Exception as exc:  # noqa: BLE001
            self._trip(exc)


def invalidate_owner_sync(owner_id: str) -> None:
    """Bản đồng bộ cho Celery worker: gọi sau khi tài liệu mới được nhúng xong.

    Không bao giờ ném lỗi — nếu Redis không có thì cache cũng không hoạt động, và mục cũ tự hết hạn.
    """
    if not settings.semantic_cache_enabled or not owner_id:
        return
    try:
        import redis

        client = redis.Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.semantic_cache_timeout_s,
            socket_connect_timeout=settings.semantic_cache_timeout_s,
        )
        try:
            client.incr(generation_key(owner_id))
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not invalidate semantic cache for owner {owner_id}: {exc}")


semantic_cache = SemanticCache()
