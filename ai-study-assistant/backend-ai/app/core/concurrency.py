"""Giới hạn số việc nặng chạy ĐỒNG THỜI (khác với rate limit: không đếm theo thời gian mà đếm số đang chạy).

Vì sao cần: mỗi luồng sinh câu trả lời (stream) giữ một thread của pool (mặc định 40) gần như suốt thời gian
model đang nghĩ. Chỉ cần vài chục câu hỏi cùng lúc là pool cạn và MỌI endpoint — kể cả đăng nhập — bị treo.
Rate limit "30/phút" không cứu được vì nó cho phép xếp hàng hàng chục câu cùng lúc.

Bộ đếm nằm trong tiến trình (không dùng Redis): bảo vệ đúng thứ cần bảo vệ là thread pool CỦA TIẾN TRÌNH này,
không thêm độ trễ hay điểm lỗi. Chạy nhiều worker uvicorn thì giới hạn là theo từng worker.
"""
from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Dict, Iterator, Optional, Tuple

from fastapi import HTTPException, status

from app.config import settings


class Lease:
    """Một "chỗ" đã giữ. ``release()`` an toàn khi gọi nhiều lần."""

    __slots__ = ("_on_release", "_released")

    def __init__(self, on_release: Callable[[], None]):
        self._on_release = on_release
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._on_release()


class ConcurrencyGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._per_user: Dict[str, int] = {}
        self._llm = 0

    # --- per user -----------------------------------------------------------------
    def user_in_flight(self, key: str) -> int:
        with self._lock:
            return self._per_user.get(key, 0)

    def user_has_room(self, key: str) -> bool:
        limit = settings.max_concurrent_requests_per_user
        return limit <= 0 or self.user_in_flight(key) < limit

    def try_acquire_user(self, key: str) -> Optional[Lease]:
        limit = settings.max_concurrent_requests_per_user
        with self._lock:
            current = self._per_user.get(key, 0)
            if 0 < limit <= current:
                return None
            self._per_user[key] = current + 1
        return Lease(lambda: self._release_user(key))

    def _release_user(self, key: str) -> None:
        with self._lock:
            left = self._per_user.get(key, 0) - 1
            if left > 0:
                self._per_user[key] = left
            else:
                self._per_user.pop(key, None)

    # --- system-wide: number of LLM calls ----------------------------------------------
    def llm_in_flight(self) -> int:
        with self._lock:
            return self._llm

    def try_acquire_llm(self) -> Optional[Lease]:
        limit = settings.max_concurrent_generations
        with self._lock:
            if 0 < limit <= self._llm:
                return None
            self._llm += 1
        return Lease(self._release_llm)

    def _release_llm(self) -> None:
        with self._lock:
            self._llm = max(0, self._llm - 1)


class SlotTimeout(Exception):
    """Chờ lượt quá lâu."""


class AsyncSlots:
    """Semaphore bất đồng bộ có giới hạn đọc từ settings và thời gian chờ tối đa.

    Semaphore gắn với event loop nên được tạo lại khi loop hoặc giới hạn đổi (vd. giữa các test).
    """

    def __init__(self, limit_fn: Callable[[], int]):
        self._limit_fn = limit_fn
        self._state: Optional[Tuple[asyncio.AbstractEventLoop, int, asyncio.Semaphore]] = None

    def _semaphore(self) -> Optional[asyncio.Semaphore]:
        limit = self._limit_fn()
        if limit <= 0:
            return None  # 0 = unlimited.
        loop = asyncio.get_running_loop()
        if self._state is None or self._state[0] is not loop or self._state[1] != limit:
            self._state = (loop, limit, asyncio.Semaphore(limit))
        return self._state[2]

    @asynccontextmanager
    async def slot(self, wait_s: float):
        sem = self._semaphore()
        if sem is None:
            yield
            return
        try:
            await asyncio.wait_for(sem.acquire(), timeout=wait_s)
        except asyncio.TimeoutError as exc:
            raise SlotTimeout from exc
        try:
            yield
        finally:
            sem.release()


USER_BUSY = "You already have requests in progress. Please wait for them to finish before asking again."
LLM_BUSY = "The assistant is busy answering other questions. Please try again in a few seconds."


def too_busy(detail: str, retry_after: int = 5) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=detail, headers={"Retry-After": str(retry_after)}
    )


gate = ConcurrencyGate()


@contextmanager
def user_request(key: str) -> Iterator[None]:
    """Giữ một chỗ "yêu cầu đang chạy" của user trong khối ``with``; hết chỗ => 429."""
    lease = gate.try_acquire_user(key)
    if lease is None:
        raise too_busy(USER_BUSY)
    try:
        yield
    finally:
        lease.release()


@contextmanager
def llm_slot() -> Iterator[None]:
    """Giữ một chỗ gọi LLM toàn hệ thống trong khối ``with``; hết chỗ => 429 "đang bận"."""
    lease = gate.try_acquire_llm()
    if lease is None:
        raise too_busy(LLM_BUSY)
    try:
        yield
    finally:
        lease.release()


upload_slots = AsyncSlots(lambda: settings.max_concurrent_uploads)
