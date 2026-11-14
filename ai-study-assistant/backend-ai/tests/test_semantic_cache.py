"""Semantic cache: tính đúng đắn (hit/miss), cô lập giữa user, vô hiệu hóa, giới hạn, chịu lỗi.

Chạy trên fakeredis. Muốn chạy trên Redis thật: ``TEST_REDIS_URL=redis://localhost:6379/15 pytest ...``.
"""
import os

import numpy as np
import pytest

fakeredis = pytest.importorskip("fakeredis")

from app.config import settings  # noqa: E402
from app.rag import semantic_cache as sc  # noqa: E402
from app.rag.semantic_cache import SemanticCache  # noqa: E402

REAL = os.environ.get("TEST_REDIS_URL")

# Embedding giả có kiểm soát: cosine(q, q_paraphrase) ≈ 0.97, các câu khác trực giao.
VECTORS = {
    "stack la gi": [1.0, 0.0, 0.0, 0.0],
    "stack nghia la gi": [0.97, 0.243, 0.0, 0.0],      # cosine ≈ 0.970
    "stack khac hoan toan": [0.8, 0.6, 0.0, 0.0],       # cosine = 0.80 (dưới ngưỡng)
    "queue la gi": [0.0, 0.0, 1.0, 0.0],
    "slide 3 noi gi": [0.0, 0.0, 0.0, 1.0],
    "slide 4 noi gi": [0.0, 0.0, 0.0, 1.0],             # embedding GIỐNG HỆT, chỉ khác số
    "x1": [0.0, 0.0, 0.7, 0.7],
    "x2": [0.0, 0.7, 0.7, 0.0],
    "x3": [0.7, 0.0, 0.0, 0.7],
    "x4": [0.5, 0.5, 0.5, 0.5],
    "x5": [0.1, 0.9, 0.1, 0.4],
}


def _embed(text):
    return np.array(VECTORS[text], dtype=np.float32)


@pytest.fixture
def server():
    return fakeredis.FakeServer()


@pytest.fixture
def cache(server, monkeypatch):
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.92)
    monkeypatch.setattr(settings, "semantic_cache_ttl_s", 3600)
    monkeypatch.setattr(settings, "semantic_cache_max_entries", 200)
    if REAL:
        import redis.asyncio as aioredis

        factory = lambda: aioredis.from_url(REAL)  # noqa: E731
    else:
        factory = lambda: fakeredis.FakeAsyncRedis(server=server)  # noqa: E731
    return SemanticCache(embed_fn=_embed, fingerprint_fn=lambda: "fp-1", client_factory=factory)


@pytest.fixture(autouse=True)
async def _flush_real():
    if REAL:
        import redis.asyncio as aioredis

        r = aioredis.from_url(REAL)
        await r.flushdb()
        await r.aclose()
    yield


async def _store(cache, query, answer="Stack là LIFO [1]", owner="u1", doc=None, intent="qa", top_k=3):
    session = await cache.open(owner, doc, intent, top_k, query)
    assert session is not None
    ok = await session.store({"answer": answer, "citations": [{"chunk_index": 1, "content": "c"}],
                              "standalone_query": query})
    assert ok
    return session


async def _lookup(cache, query, owner="u1", doc=None, intent="qa", top_k=3):
    session = await cache.open(owner, doc, intent, top_k, query)
    return await session.lookup() if session else None


async def test_similar_question_hits_and_returns_payload(cache):
    await _store(cache, "stack la gi")
    hit = await _lookup(cache, "stack nghia la gi")
    assert hit is not None
    assert hit.payload["answer"] == "Stack là LIFO [1]"
    assert hit.payload["citations"][0]["chunk_index"] == 1
    assert hit.similarity > 0.96
    assert hit.matched_query == "stack la gi"


async def test_identical_question_hits(cache):
    await _store(cache, "stack la gi")
    assert (await _lookup(cache, "stack la gi")).similarity == pytest.approx(1.0, abs=1e-4)


async def test_below_threshold_and_unrelated_miss(cache):
    await _store(cache, "stack la gi")
    assert await _lookup(cache, "stack khac hoan toan") is None   # cosine 0.80 < 0.92
    assert await _lookup(cache, "queue la gi") is None            # trực giao


async def test_threshold_is_configurable(cache, monkeypatch):
    await _store(cache, "stack la gi")
    monkeypatch.setattr(settings, "semantic_cache_threshold", 0.75)
    assert await _lookup(cache, "stack khac hoan toan") is not None


async def test_questions_differing_only_in_numbers_never_match(cache):
    await _store(cache, "slide 3 noi gi", answer="Slide 3: ...")
    assert await _lookup(cache, "slide 4 noi gi") is None
    hit = await _lookup(cache, "slide 3 noi gi")
    assert hit is not None and hit.payload["answer"] == "Slide 3: ..."


async def test_other_user_never_sees_my_answers(cache):
    await _store(cache, "stack la gi", owner="alice")
    assert await _lookup(cache, "stack la gi", owner="bob") is None
    assert await _lookup(cache, "stack la gi", owner="alice") is not None


async def test_scope_document_intent_topk_must_match(cache):
    await _store(cache, "stack la gi", doc="doc-A", intent="qa", top_k=3)
    assert await _lookup(cache, "stack la gi", doc="doc-A", intent="qa", top_k=3) is not None
    assert await _lookup(cache, "stack la gi", doc="doc-B") is None
    assert await _lookup(cache, "stack la gi", doc=None) is None           # "tất cả tài liệu" là phạm vi khác
    assert await _lookup(cache, "stack la gi", doc="doc-A", intent="summarization") is None
    assert await _lookup(cache, "stack la gi", doc="doc-A", top_k=5) is None


async def test_model_change_invalidates(cache):
    await _store(cache, "stack la gi")
    cache._fingerprint_fn = lambda: "fp-2-int8"
    assert await _lookup(cache, "stack la gi") is None


async def test_invalidate_owner_drops_old_answers_only_for_that_owner(cache):
    await _store(cache, "stack la gi", owner="alice")
    await _store(cache, "stack la gi", owner="bob")
    await cache.invalidate_owner("alice")
    assert await _lookup(cache, "stack la gi", owner="alice") is None
    assert await _lookup(cache, "stack la gi", owner="bob") is not None
    # câu trả lời mới sau khi vô hiệu hóa lại dùng được
    await _store(cache, "stack la gi", owner="alice", answer="mới")
    assert (await _lookup(cache, "stack la gi", owner="alice")).payload["answer"] == "mới"


async def test_generation_captured_at_open_so_stale_answers_stay_stale(cache):
    session = await cache.open("alice", None, "qa", 3, "stack la gi")   # trước khi tài liệu mới xong
    await cache.invalidate_owner("alice")                                # tài liệu mới được nhúng xong
    await session.store({"answer": "cũ", "citations": [], "standalone_query": "stack la gi"})
    assert await _lookup(cache, "stack la gi", owner="alice") is None    # câu trả lời cũ không sống lại


async def test_ttl_is_set_on_entry_and_index(cache, server):
    session = await _store(cache, "stack la gi")
    r = cache._redis()
    assert 0 < await r.ttl(session.index_key) <= 3600
    keys = [k async for k in r.scan_iter(f"{sc.PREFIX}:e:*")]
    assert len(keys) == 1 and 0 < await r.ttl(keys[0]) <= 3600


async def test_size_cap_evicts_oldest(cache, monkeypatch):
    monkeypatch.setattr(settings, "semantic_cache_max_entries", 3)
    for q in ["x1", "x2", "x3", "x4", "x5"]:
        await _store(cache, q, answer=q)
    session = await cache.open("u1", None, "qa", 3, "x1")
    r = cache._redis()
    assert await r.zcard(session.index_key) == 3
    assert len([k async for k in r.scan_iter(f"{sc.PREFIX}:e:*")]) == 3   # mục bị loại cũng bị xóa
    assert await _lookup(cache, "x1") is None                              # cũ nhất đã bị loại
    assert await _lookup(cache, "x5") is not None


async def test_expired_entries_are_skipped_and_cleaned_from_index(cache):
    session = await _store(cache, "stack la gi")
    r = cache._redis()
    for key in [k async for k in r.scan_iter(f"{sc.PREFIX}:e:*")]:
        await r.delete(key)                                                # mô phỏng hết TTL
    assert await _lookup(cache, "stack la gi") is None
    assert await r.zcard(session.index_key) == 0


async def test_hit_refreshes_lru_and_counts_hits(cache):
    await _store(cache, "stack la gi")
    await _lookup(cache, "stack la gi")
    await _lookup(cache, "stack la gi")
    r = cache._redis()
    key = [k async for k in r.scan_iter(f"{sc.PREFIX}:e:*")][0]
    assert int(await r.hget(key, "hits")) == 2


async def test_disabled_cache_is_a_noop(cache, monkeypatch):
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    assert await cache.open("u1", None, "qa", 3, "stack la gi") is None


async def test_redis_failure_is_fail_open_and_trips_breaker():
    calls = {"n": 0}

    class Broken:
        async def get(self, *_a, **_k):
            calls["n"] += 1
            raise ConnectionError("redis is down")

    cache = SemanticCache(embed_fn=_embed, fingerprint_fn=lambda: "fp", client_factory=Broken)
    assert await cache.open("u1", None, "qa", 3, "stack la gi") is None   # không ném lỗi
    assert calls["n"] == 1
    assert await cache.open("u1", None, "qa", 3, "stack la gi") is None   # breaker: không gọi Redis nữa
    assert calls["n"] == 1


async def test_failure_during_lookup_or_store_never_raises(cache):
    session = await cache.open("u1", None, "qa", 3, "stack la gi")

    class Broken:
        def pipeline(self, *a, **k):
            raise ConnectionError("gone")

        async def zrevrange(self, *a, **k):
            raise ConnectionError("gone")

    cache._client = Broken()
    assert await session.lookup() is None
    cache._down_until = 0.0
    assert await session.store({"answer": "x", "citations": [], "standalone_query": "q"}) is False


async def test_invalidate_owner_sync_bumps_the_same_generation_key(server, monkeypatch):
    if REAL:
        pytest.skip("sync path is covered on fakeredis")
    import redis

    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(redis.Redis, "from_url", classmethod(lambda cls, *a, **k: fakeredis.FakeRedis(server=server)))
    sc.invalidate_owner_sync("alice")
    sc.invalidate_owner_sync("alice")
    check = fakeredis.FakeAsyncRedis(server=server)
    assert int(await check.get(sc.generation_key("alice"))) == 2
    sc.invalidate_owner_sync("")            # owner rỗng: không làm gì, không lỗi
