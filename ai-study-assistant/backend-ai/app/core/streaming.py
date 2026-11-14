"""Tiện ích cho SSE streaming.

Hai vấn đề của cách làm cũ (``iterate_in_threadpool`` + ``run_in_threadpool`` trần):

1. **Ngắt kết nối không dừng được LLM.** Khi người dùng đóng tab / bấm "Dừng", Starlette hủy generator
   của response, nhưng generator ĐỒNG BỘ sinh token (đang giữ kết nối HTTP tới Ollama) không ai đóng =>
   Ollama cứ tiếp tục sinh hết câu trả lời, tốn GPU/CPU cho một câu không ai đọc.
   ``aiter_blocking`` đóng generator ngay khi vòng lặp bị hủy (sau khi lần ``next()`` đang chạy kết thúc).

2. **Im lặng quá lâu.** Lần đầu nạp model hoặc prompt dài có thể mất > 60 giây trước token đầu tiên;
   nginx / proxy / trình duyệt hay cắt kết nối rảnh. ``HEARTBEAT`` cho phép gửi comment SSE định kỳ.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Iterator, Optional

import anyio.to_thread
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# Dòng comment SSE: EventSource/parsers bỏ qua nó, nhưng nó giữ cho kết nối "sống".
SSE_KEEPALIVE = ": keep-alive\n\n"


class _Heartbeat:
    def __repr__(self) -> str:  # pragma: no cover
        return "HEARTBEAT"


HEARTBEAT = _Heartbeat()


@dataclass
class Done:
    """Kết quả cuối của ``run_with_heartbeat``."""

    value: Any


def _close_quietly(gen: Iterator) -> None:
    try:
        gen.close()
    except Exception:  # noqa: BLE001 - dọn dẹp, không được ném lỗi
        logger.debug("Closing blocking generator failed", exc_info=True)


async def aiter_blocking(gen: Iterator, heartbeat_s: Optional[float] = None) -> AsyncIterator[Any]:
    """Duyệt một generator ĐỒNG BỘ (có thể chặn lâu) từ code async.

    * Mỗi ``next()`` chạy trong thread pool nên không chặn event loop.
    * Nếu vòng lặp bị hủy / dừng sớm / có lỗi, generator được ``close()`` => ``with httpx.stream(...)``
      bên trong nó thoát và kết nối tới Ollama bị đóng.
    * Nếu ``heartbeat_s`` được đặt, yield ``HEARTBEAT`` mỗi ``heartbeat_s`` giây khi đang chờ phần tử kế.
    """
    end = object()
    pending: Optional[asyncio.Future] = None
    try:
        while True:
            pending = asyncio.ensure_future(anyio.to_thread.run_sync(next, gen, end))
            while True:
                done, _ = await asyncio.wait({pending}, timeout=heartbeat_s)
                if done:
                    break
                yield HEARTBEAT
            item = pending.result()
            pending = None
            if item is end:
                return
            yield item
    finally:
        if pending is not None and not pending.done():
            # Một next() vẫn đang chạy trong thread (không hủy được giữa chừng): đóng generator
            # ngay khi nó xong, vì close() lên generator đang chạy sẽ lỗi.
            pending.add_done_callback(lambda _fut: _close_quietly(gen))
        else:
            _close_quietly(gen)


async def run_with_heartbeat(
    func: Callable[..., Any], *args: Any, interval: Optional[float] = None, **kwargs: Any
) -> AsyncIterator[Any]:
    """Chạy ``func`` (đồng bộ, nặng) trong thread pool; trong lúc chờ yield ``HEARTBEAT`` định kỳ,
    cuối cùng yield ``Done(value)`` (hoặc ném lại exception của ``func``)."""
    task = asyncio.ensure_future(run_in_threadpool(func, *args, **kwargs))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                break
            yield HEARTBEAT
        yield Done(task.result())
    finally:
        if not task.done():
            task.cancel()
