"""Làm chậm việc dò mật khẩu THEO TỪNG TÀI KHOẢN (rate limit theo IP không chặn được kẻ dùng nhiều IP).

Cơ chế "khóa mềm" (không khóa cứng, để kẻ xấu không dùng nó khóa người khác):

* Each FAILED login increments the account counter (which expires after ``login_fail_window_s``).
* When the counter reaches ``login_fail_threshold``, the account gets only ONE attempt per ``login_soft_lock_interval_s``
  giây, bất kể từ IP nào: dò mật khẩu bị chặn ở mức vài lần/phút, còn chủ tài khoản chỉ phải chờ vài giây.
* A blocked attempt does NOT check the password (so it does not become a "password correctness oracle" for attackers) and does not increment the counter.
* A successful login clears the counter.
* This also applies to NONEXISTENT accounts (keyed by the identifier string), so account existence is not revealed.

Lưu trong Redis (đúng với nhiều worker); Redis lỗi => dùng bộ nhớ trong tiến trình thay vì bỏ trống phòng thủ.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from typing import Any, Dict, Optional, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

PREFIX = "asa:login"
_COOLDOWN_S = 30.0
_MEM_MAX = 10_000


def _key(identifier: str) -> str:
    # Normalize so "Bob" / " bob " cannot evade the counter; hash the identifier so raw email is not stored in Redis.
    digest = hashlib.sha256(identifier.strip().lower().encode("utf-8")).hexdigest()[:32]
    return f"{PREFIX}:{digest}"


class _Memory:
    """Bản đồng bộ trong tiến trình, cùng ngữ nghĩa với phần Redis."""

    def __init__(self) -> None:
        self._fails: Dict[str, Tuple[int, float]] = {}   # key -> (failed-attempt count, expiration time)
        self._slots: Dict[str, float] = {}               # key -> next allowed attempt time

    def _sweep(self, now: float) -> None:
        if len(self._fails) + len(self._slots) > _MEM_MAX:
            self._fails = {k: v for k, v in self._fails.items() if v[1] > now}
            self._slots = {k: v for k, v in self._slots.items() if v > now}

    def failures(self, key: str) -> int:
        count, expires = self._fails.get(key, (0, 0.0))
        return count if expires > time.monotonic() else 0

    def add_failure(self, key: str, window: int) -> None:
        now = time.monotonic()
        self._sweep(now)
        self._fails[key] = (self.failures(key) + 1, now + window)

    def take_slot(self, key: str, interval: int) -> int:
        now = time.monotonic()
        ready = self._slots.get(key, 0.0)
        if ready > now:
            return max(1, math.ceil(ready - now))
        self._slots[key] = now + interval
        return 0

    def clear(self, key: str) -> None:
        self._fails.pop(key, None)
        self._slots.pop(key, None)


class LoginThrottle:
    def __init__(self, client_factory=None):
        self._client_factory = client_factory
        self._client: Any = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._down_until = 0.0
        self._memory = _Memory()

    def _redis(self):
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

    def _use_redis(self) -> bool:
        return time.monotonic() >= self._down_until

    def _trip(self, exc: BaseException) -> None:
        self._down_until = time.monotonic() + _COOLDOWN_S
        logger.warning("Login throttle: Redis unavailable (%s: %s) — using in-process counters for %ds.",
                       type(exc).__name__, exc, int(_COOLDOWN_S))

    @property
    def enabled(self) -> bool:
        return settings.login_fail_threshold > 0

    async def check(self, identifier: str) -> int:
        """0 = được thử; >0 = số giây phải chờ (tài khoản đang bị khóa mềm)."""
        if not self.enabled:
            return 0
        key = _key(identifier)
        interval = max(1, int(settings.login_soft_lock_interval_s))
        if self._use_redis():
            try:
                r = self._redis()
                failures = int(await r.get(f"{key}:fail") or 0)
                if failures < settings.login_fail_threshold:
                    return 0
                # SET NX: only ONE attempt can claim the slot in each `interval`-second window.
                if await r.set(f"{key}:slot", 1, nx=True, ex=interval):
                    return 0
                ttl = int(await r.ttl(f"{key}:slot"))
                return ttl if ttl > 0 else interval
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)
        if self._memory.failures(key) < settings.login_fail_threshold:
            return 0
        return self._memory.take_slot(key, interval)

    async def record_failure(self, identifier: str) -> None:
        if not self.enabled:
            return
        key = _key(identifier)
        window = max(1, int(settings.login_fail_window_s))
        if self._use_redis():
            try:
                r = self._redis()
                pipe = r.pipeline(transaction=False)
                pipe.incr(f"{key}:fail")
                pipe.expire(f"{key}:fail", window)
                await pipe.execute()
                return
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)
        self._memory.add_failure(key, window)

    async def reset(self, identifier: str) -> None:
        """Đăng nhập đúng: xóa bộ đếm và chỗ chờ."""
        if not self.enabled:
            return
        key = _key(identifier)
        self._memory.clear(key)
        if self._use_redis():
            try:
                await self._redis().delete(f"{key}:fail", f"{key}:slot")
            except Exception as exc:  # noqa: BLE001
                self._trip(exc)


login_throttle = LoginThrottle()
