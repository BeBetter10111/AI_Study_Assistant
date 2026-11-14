"""aiter_blocking / run_with_heartbeat: đóng generator khi dừng sớm, heartbeat khi chờ lâu."""
import asyncio
import threading
import time

from app.core.streaming import HEARTBEAT, Done, aiter_blocking, run_with_heartbeat


async def _wait_closed(src, timeout=3.0):
    """Chờ không chặn event loop (callback đóng generator chạy ngay trên loop)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if src.closed.is_set():
            return True
        await asyncio.sleep(0.02)
    return False


class _Source:
    """Generator đồng bộ giả lập luồng LLM: ghi lại việc nó có bị close() hay không."""

    def __init__(self, items, delay=0.0, boom_after=None):
        self.items, self.delay, self.boom_after = items, delay, boom_after
        self.closed = threading.Event()
        self.produced = 0

    def __iter__(self):
        return self._run()

    def _run(self):
        try:
            for i, item in enumerate(self.items):
                if self.boom_after is not None and i == self.boom_after:
                    raise RuntimeError("upstream broke")
                time.sleep(self.delay)
                self.produced += 1
                yield item
        finally:
            self.closed.set()


async def test_yields_everything_in_order_and_closes():
    src = _Source(["a", "b", "c"])
    got = [x async for x in aiter_blocking(iter(src))]
    assert got == ["a", "b", "c"]
    assert src.closed.is_set()


async def test_early_exit_closes_the_generator():
    src = _Source(list("abcdefghij"))
    stream = aiter_blocking(iter(src))
    async for item in stream:
        if item == "c":
            break
    await stream.aclose()
    assert await _wait_closed(src), "generator must be closed so the upstream (Ollama) connection is dropped"
    assert src.produced < 10


async def test_cancellation_closes_the_generator_even_mid_next():
    src = _Source(list("abcdefghij"), delay=0.2)

    async def consume():
        async for _ in aiter_blocking(iter(src)):
            pass

    task = asyncio.ensure_future(consume())
    await asyncio.sleep(0.5)  # đang chờ một next() chậm trong thread
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    assert await _wait_closed(src), "generator must be closed after the in-flight next() finishes"
    assert src.produced < 10


async def test_error_propagates_and_generator_is_closed():
    src = _Source(["a", "b", "c"], boom_after=1)
    seen = []
    try:
        async for item in aiter_blocking(iter(src)):
            seen.append(item)
        raise AssertionError("should have raised")
    except RuntimeError as exc:
        assert "upstream broke" in str(exc)
    assert seen == ["a"]
    assert src.closed.is_set()


async def test_heartbeat_while_waiting_for_a_slow_item():
    src = _Source(["only"], delay=0.35)
    got = [x async for x in aiter_blocking(iter(src), heartbeat_s=0.1)]
    assert got.count(HEARTBEAT) >= 2
    assert got[-1] == "only"


async def test_run_with_heartbeat_returns_value_after_beats():
    def slow():
        time.sleep(0.35)
        return 42

    ticks = [t async for t in run_with_heartbeat(slow, interval=0.1)]
    assert sum(t is HEARTBEAT for t in ticks) >= 2
    assert isinstance(ticks[-1], Done) and ticks[-1].value == 42


async def test_run_with_heartbeat_reraises():
    def bad():
        raise ValueError("nope")

    try:
        async for _ in run_with_heartbeat(bad, interval=0.05):
            pass
        raise AssertionError("should have raised")
    except ValueError as exc:
        assert "nope" in str(exc)
