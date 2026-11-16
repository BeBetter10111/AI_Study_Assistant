"""Giới hạn chống lạm dụng: số yêu cầu nặng đồng thời, upload (hạn mức + đồng thời + kích thước), khóa mềm đăng nhập,
dọn/giới hạn hội thoại."""
import asyncio
import json
import threading
import time
import types
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy import func, select
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

import app.api.endpoints.auth as auth_module
import app.api.endpoints.chat as chat_module
import app.api.endpoints.documents as docs_module
import app.api.endpoints.quiz as quiz_module
import app.core.database as coredb
from app.api.endpoints.chat import QueryRequest
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core import conversation_store
from app.core.concurrency import LLM_BUSY, USER_BUSY, ConcurrencyGate, gate
from app.core.login_throttle import LoginThrottle
from app.core.request_limits import UploadSizeLimitMiddleware
from app.models.models import ChatMessage, Conversation, Document

PW = "firstpassword1"
HIT = {
    "document_id": "d1", "chunk_index": 0, "score": 0.9, "content": "Classes are blueprints.",
    "page_number": 3, "slide_number": None, "slide_title": None, "section_title": None, "is_summary": False,
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    # The gate counter is global: each test must start and finish at 0.
    monkeypatch.setattr(chat_module, "gate", gate)
    yield
    assert gate.llm_in_flight() == 0, "an LLM slot leaked"


async def _login(client, name="alice"):
    res = await client.post("/api/auth/register",
                            json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    me = await client.get("/api/auth/me", headers=headers)
    return headers, me.json()["id"]


def _events(response):
    return [json.loads(line[5:]) for line in response.text.split("\n\n") if line.startswith("data:")]


def _fake_stream(tokens=("Answer ", "[1]"), error=None):
    def fake(prompt, system_prompt, *a, **k):
        def gen():
            if error:
                yield {"error": error}
                return
            for t in tokens:
                yield {"token": t}
            yield {"done": True}
        return gen()
    return fake


async def _ask(client, headers, question="What is a class?", **extra):
    return await client.post("/api/chat/query/stream", headers=headers, json={"question": question, **extra})


# =====================================================================================
# 1) Limit concurrent resource-intensive requests
# =====================================================================================
def test_gate_per_user_cap_and_idempotent_release(monkeypatch):
    g = ConcurrencyGate()
    monkeypatch.setattr(settings, "max_concurrent_requests_per_user", 2)
    a, b = g.try_acquire_user("u1"), g.try_acquire_user("u1")
    assert a and b and g.try_acquire_user("u1") is None            # At the limit of 2 => reject the third request.
    assert g.try_acquire_user("u2") is not None                    # Another user is unaffected.
    assert not g.user_has_room("u1")
    a.release()
    a.release()                                                    # Releasing again must not return an extra slot.
    assert g.user_in_flight("u1") == 1 and g.user_has_room("u1")
    b.release()
    assert g.user_in_flight("u1") == 0


def test_gate_global_llm_cap_and_zero_means_unlimited(monkeypatch):
    g = ConcurrencyGate()
    monkeypatch.setattr(settings, "max_concurrent_generations", 2)
    one, two = g.try_acquire_llm(), g.try_acquire_llm()
    assert g.try_acquire_llm() is None and g.llm_in_flight() == 2
    one.release()
    assert g.try_acquire_llm() is not None
    monkeypatch.setattr(settings, "max_concurrent_generations", 0)
    assert all(g.try_acquire_llm() is not None for _ in range(50))
    two.release()


async def test_stream_returns_429_with_retry_after_when_user_has_too_many_requests(app_client, monkeypatch):
    headers, uid = await _login(app_client)
    monkeypatch.setattr(settings, "max_concurrent_requests_per_user", 2)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    held = [gate.try_acquire_user(uid), gate.try_acquire_user(uid)]

    res = await _ask(app_client, headers)

    assert res.status_code == 429 and res.headers["retry-after"] == "5"
    assert USER_BUSY in res.text
    for lease in held:
        lease.release()
    assert _events(await _ask(app_client, headers))[-1]["type"] == "result"   # A request succeeds after the slot is released.
    assert gate.user_in_flight(uid) == 0


async def test_other_users_are_not_blocked_by_one_busy_user(app_client, monkeypatch):
    a, a_id = await _login(app_client, "alice")
    b, _ = await _login(app_client, "bobby")
    monkeypatch.setattr(settings, "max_concurrent_requests_per_user", 1)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    lease = gate.try_acquire_user(a_id)

    assert (await _ask(app_client, a)).status_code == 429
    assert _events(await _ask(app_client, b))[-1]["type"] == "result"
    lease.release()


async def test_llm_slots_exhausted_gives_busy_error_that_is_not_saved_or_cached(app_client, monkeypatch):
    headers, uid = await _login(app_client)
    conv = (await app_client.post("/api/conversations", headers=headers, json={})).json()
    monkeypatch.setattr(settings, "max_concurrent_generations", 1)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    taken = gate.try_acquire_llm()

    events = _events(await _ask(app_client, headers, conversation_id=conv["id"]))

    assert events[-1] == {"type": "error", "detail": LLM_BUSY}
    assert not any(e["type"] in ("token", "result") for e in events)
    detail = (await app_client.get(f"/api/conversations/{conv['id']}", headers=headers)).json()
    assert [m["role"] for m in detail["messages"]] == ["user"]            # No "answer" is saved.
    taken.release()
    assert _events(await _ask(app_client, headers, conversation_id=conv["id"]))[-1]["type"] == "result"


async def test_cached_answers_are_served_even_when_llm_slots_are_full(app_client, monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    from app.rag.semantic_cache import SemanticCache

    server = fakeredis.FakeServer()
    cache = SemanticCache(
        embed_fn=lambda t: np.array([1.0, 0.0], dtype=np.float32), fingerprint_fn=lambda: "fp",
        client_factory=lambda: fakeredis.FakeAsyncRedis(server=server))
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(chat_module, "semantic_cache", cache)
    headers, _ = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    assert _events(await _ask(app_client, headers))[-1]["data"]["cached"] is False

    monkeypatch.setattr(settings, "max_concurrent_generations", 1)
    taken = gate.try_acquire_llm()                                          # All LLM slots are occupied.
    events = _events(await _ask(app_client, headers))
    taken.release()

    assert events[-1]["type"] == "result" and events[-1]["data"]["cached"] is True


@pytest.mark.parametrize("how", ["finish", "llm_error", "client_disconnect", "search_error"])
async def test_leases_are_always_returned(monkeypatch, how):
    """Chỗ giữ cho user + LLM phải được trả lại ở MỌI kết cục, kể cả khi client ngắt giữa chừng."""
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks",
                        (lambda **k: (_ for _ in ()).throw(RuntimeError("qdrant down"))) if how == "search_error"
                        else (lambda **k: [HIT]))

    def llm(prompt, system_prompt, *a, **k):
        def gen():
            if how == "llm_error":
                yield {"error": "Ollama down"}
                return
            for i in range(1000):
                yield {"token": f"t{i} "}
        return gen()

    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", llm)
    uid = str(uuid.uuid4())
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b"",
                       "client": ("t", 1), "server": ("t", 80)})
    response = await chat_module.query_stream(request, QueryRequest(question="What is a class?"),
                                              types.SimpleNamespace(id=uuid.UUID(uid)))
    body = response.body_iterator
    if how == "client_disconnect":
        async for chunk in body:
            if '"token"' in chunk:
                break
        assert gate.llm_in_flight() == 1 and gate.user_in_flight(uid) == 1     # The slots remain occupied while streaming.
        await body.aclose()
    elif how == "finish":
        monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
        async for _ in body:
            pass
    else:
        async for _ in body:
            pass
    await asyncio.sleep(0.05)
    assert gate.user_in_flight(uid) == 0 and gate.llm_in_flight() == 0


async def test_unstarted_response_leaks_nothing(monkeypatch):
    """Client ngắt trước khi stream bắt đầu: events() không bao giờ chạy => chưa giữ chỗ nào."""
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    uid = str(uuid.uuid4())
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b"",
                       "client": ("t", 1), "server": ("t", 80)})
    await chat_module.query_stream(request, QueryRequest(question="What is a class?"),
                                   types.SimpleNamespace(id=uuid.UUID(uid)))
    assert gate.user_in_flight(uid) == 0 and gate.llm_in_flight() == 0


async def test_query_and_quiz_endpoints_also_respect_the_limits(app_client, monkeypatch):
    headers, uid = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "generate_rag_response", lambda **k: "Answer [1]")
    monkeypatch.setattr(quiz_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(quiz_module.llm_engine, "generate_structured_output",
                        lambda **k: {"questions": [{"question": "Q?", "options": ["a", "b"], "correct_index": 0,
                                                    "explanation": "e"}]})
    quiz_body = {"topic": "classes", "num_questions": 1}
    assert (await app_client.post("/api/chat/query", headers=headers, json={"question": "What is a class?"})
            ).status_code == 200
    assert (await app_client.post("/api/quiz/generate", headers=headers, json=quiz_body)).status_code == 200

    monkeypatch.setattr(settings, "max_concurrent_generations", 1)
    taken = gate.try_acquire_llm()
    for url, body in (("/api/chat/query", {"question": "What is a class?"}), ("/api/quiz/generate", quiz_body)):
        res = await app_client.post(url, headers=headers, json=body)
        assert res.status_code == 429 and res.headers["retry-after"] == "5", url
    taken.release()

    monkeypatch.setattr(settings, "max_concurrent_requests_per_user", 1)
    lease = gate.try_acquire_user(uid)
    for url, body in (("/api/chat/query", {"question": "What is a class?"}), ("/api/quiz/generate", quiz_body)):
        assert (await app_client.post(url, headers=headers, json=body)).status_code == 429, url
    lease.release()
    assert gate.user_in_flight(uid) == 0 and gate.llm_in_flight() == 0


# =====================================================================================
# 2) Upload: quotas, concurrency, and size
# =====================================================================================
PDF = b"%PDF-1.4\n" + b"x" * 1000


class FakeMinio:
    def __init__(self, delay=0.0):
        self.delay, self.calls, self.active, self.peak = delay, [], 0, 0
        self._lock = threading.Lock()

    def bucket_exists(self, *_a):
        return True

    def put_object(self, bucket, name, data, length, content_type=None):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(self.delay)
            self.calls.append({"name": name, "length": length, "data": data, "starts_at_0": data.tell() == 0,
                               "bytes": data.read()})
        finally:
            with self._lock:
                self.active -= 1


@pytest.fixture
def minio(monkeypatch):
    fake = FakeMinio()
    queued = []
    monkeypatch.setattr(docs_module, "minio_client", fake)
    monkeypatch.setattr(docs_module, "task_process_document",
                        types.SimpleNamespace(delay=lambda *a, **k: queued.append(a)))
    fake.queued = queued
    return fake


async def _upload(client, headers, name="a.pdf", content=PDF):
    return await client.post("/api/documents/upload", headers=headers,
                             files={"file": (name, content, "application/pdf")})


async def _add_doc(owner_id, size=None, name="x.pdf"):
    async with coredb.async_session() as s:
        s.add(Document(owner_id=uuid.UUID(owner_id), filename=name, mime_type="application/pdf",
                       status="embedded", size_bytes=size))
        await s.commit()


async def test_upload_streams_the_file_and_records_its_size(app_client, minio):
    headers, uid = await _login(app_client)

    res = await _upload(app_client, headers)

    assert res.status_code == 201
    call = minio.calls[0]
    assert call["length"] == len(PDF) and call["bytes"] == PDF and call["starts_at_0"]
    assert not isinstance(call["data"], (bytes, bytearray))        # File-like object, NOT the entire file loaded into memory.
    async with coredb.async_session() as s:
        doc = (await s.execute(select(Document))).scalars().one()
        assert doc.size_bytes == len(PDF) and doc.storage_path.endswith(".pdf")
    assert len(minio.queued) == 1


async def test_document_count_quota(app_client, minio, monkeypatch):
    headers, uid = await _login(app_client)
    monkeypatch.setattr(settings, "max_documents_per_user", 2)
    await _add_doc(uid)
    assert (await _upload(app_client, headers)).status_code == 201          # Second document: still within the quota.

    res = await _upload(app_client, headers, "c.pdf")

    assert res.status_code == 403 and "Document limit reached (2)" in res.json()["detail"]
    assert len(minio.calls) == 1                                            # No additional write to MinIO.


async def test_storage_quota_counts_bytes_and_treats_old_documents_as_zero(app_client, minio, monkeypatch):
    headers, uid = await _login(app_client)
    monkeypatch.setattr(settings, "max_storage_per_user_mb", 1)
    await _add_doc(uid, size=None)                                          # Older document without size_bytes.
    await _add_doc(uid, size=1024 * 1024 - 500)
    assert (await _upload(app_client, headers, content=b"%PDF-1.4\n" + b"y" * 100)).status_code == 201

    res = await _upload(app_client, headers, content=b"%PDF-1.4\n" + b"y" * 1000)

    assert res.status_code == 403 and "Storage limit reached (1MB)" in res.json()["detail"]


async def test_quotas_are_per_user_and_can_be_disabled(app_client, minio, monkeypatch):
    a, a_id = await _login(app_client, "alice")
    b, _ = await _login(app_client, "bobby")
    monkeypatch.setattr(settings, "max_documents_per_user", 1)
    await _add_doc(a_id)
    assert (await _upload(app_client, a)).status_code == 403
    assert (await _upload(app_client, b)).status_code == 201                # Alice's quota does not affect Bob.
    monkeypatch.setattr(settings, "max_documents_per_user", 0)
    monkeypatch.setattr(settings, "max_storage_per_user_mb", 0)
    assert (await _upload(app_client, a)).status_code == 201                # 0 = unlimited.


async def test_oversize_file_is_rejected_without_reading_it(app_client, minio, monkeypatch):
    headers, _ = await _login(app_client)
    monkeypatch.setattr(docs_module, "MAX_UPLOAD_SIZE_BYTES", 500)

    res = await _upload(app_client, headers)

    assert res.status_code == 413 and minio.calls == []
    async with coredb.async_session() as s:
        assert (await s.execute(select(func.count(Document.id)))).scalar_one() == 0


async def test_uploads_are_limited_to_a_few_at_a_time_but_all_succeed(app_client, minio, monkeypatch):
    headers, _ = await _login(app_client)
    minio.delay = 0.25
    monkeypatch.setattr(settings, "max_concurrent_uploads", 2)
    monkeypatch.setattr(settings, "upload_slot_wait_s", 10.0)

    results = await asyncio.gather(*[_upload(app_client, headers, f"f{i}.pdf") for i in range(5)])

    assert [r.status_code for r in results] == [201] * 5
    assert minio.peak <= 2 and len(minio.calls) == 5


async def test_upload_gets_503_with_retry_after_when_waiting_too_long(app_client, minio, monkeypatch):
    headers, _ = await _login(app_client)
    minio.delay = 0.5
    monkeypatch.setattr(settings, "max_concurrent_uploads", 1)
    monkeypatch.setattr(settings, "upload_slot_wait_s", 0.05)

    first, second = await asyncio.gather(_upload(app_client, headers, "a.pdf"), _upload(app_client, headers, "b.pdf"))

    codes = sorted([first.status_code, second.status_code])
    assert codes == [201, 503]
    busy = first if first.status_code == 503 else second
    assert busy.headers["retry-after"] == "5"
    async with coredb.async_session() as s:                                 # A rejected request leaves no junk record.
        assert (await s.execute(select(func.count(Document.id)))).scalar_one() == 1


# ---- Middleware rejects oversized bodies (before even writing a temporary file) --------------------
def _wrapped_app(limit):
    seen = {"calls": 0, "bytes": 0}

    async def endpoint(request):
        seen["bytes"] = len(await request.body())   # Like real Starlette: read/write the body BEFORE the endpoint completes.
        seen["calls"] += 1
        return JSONResponse({"ok": True})

    app = Starlette(routes=[Route("/api/documents/upload", endpoint, methods=["POST"]),
                            Route("/other", endpoint, methods=["POST"])])
    app.add_middleware(UploadSizeLimitMiddleware, path="/api/documents/upload", max_body_bytes=limit)
    return app, seen


async def _client(app):
    import httpx
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def test_middleware_rejects_by_content_length_before_the_endpoint_runs():
    app, seen = _wrapped_app(100)
    async with await _client(app) as c:
        res = await c.post("/api/documents/upload", content=b"x" * 500)
        assert res.status_code == 413 and "maximum upload size" in res.json()["detail"]
        assert seen["calls"] == 0
        assert (await c.post("/api/documents/upload", content=b"x" * 50)).status_code == 200   # Within the limit: allowed.
        assert (await c.post("/other", content=b"x" * 5000)).status_code == 200               # Different path: not affected.


async def test_middleware_cuts_chunked_bodies_without_content_length():
    app, seen = _wrapped_app(100)

    async def chunks():
        for _ in range(10):
            yield b"x" * 40

    async with await _client(app) as c:
        res = await c.post("/api/documents/upload", content=chunks())            # No Content-Length header.
        assert res.status_code == 413
        assert seen["calls"] == 0


# =====================================================================================
# 3) Per-account login soft lock
# =====================================================================================
@pytest.fixture
def throttle(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    server = fakeredis.FakeServer()
    t = LoginThrottle(client_factory=lambda: fakeredis.FakeAsyncRedis(server=server))
    monkeypatch.setattr(auth_module, "login_throttle", t)
    monkeypatch.setattr(settings, "login_fail_threshold", 3)
    monkeypatch.setattr(settings, "login_fail_window_s", 600)
    monkeypatch.setattr(settings, "login_soft_lock_interval_s", 2)
    return t


async def _try_login(client, identifier, password):
    return await client.post("/api/auth/login", json={"identifier": identifier, "password": password})


async def test_under_the_threshold_nothing_is_throttled(app_client, throttle):
    await _login(app_client)
    for _ in range(2):
        assert (await _try_login(app_client, "alice", "wrong-password")).status_code == 401
    assert (await _try_login(app_client, "alice", PW)).status_code == 200


async def test_soft_lock_allows_one_attempt_per_interval_and_never_checks_blocked_ones(app_client, throttle):
    await _login(app_client)
    for _ in range(3):                                                     # Reach the threshold.
        assert (await _try_login(app_client, "alice", "wrong-password")).status_code == 401

    assert (await _try_login(app_client, "alice", "wrong-again")).status_code == 401   # First allowed attempt.
    blocked = await _try_login(app_client, "alice", PW)                    # Correct password, BUT currently blocked.
    assert blocked.status_code == 429                                      # => No correctness oracle when blocked.
    assert 1 <= int(blocked.headers["retry-after"]) <= 2
    assert "Too many failed attempts" in blocked.json()["detail"]

    await asyncio.sleep(2.1)
    ok = await _try_login(app_client, "alice", PW)                         # After the wait period, the account owner can log in.
    assert ok.status_code == 200
    for _ in range(4):                                                     # A successful login clears the counter.
        assert (await _try_login(app_client, "alice", PW)).status_code == 200


async def test_throttle_is_per_account_and_ignores_case_and_spaces(app_client, throttle):
    await _login(app_client, "alice")
    await _login(app_client, "bobby")
    for ident in ("alice", "ALICE", "  Alice "):                            # Same account despite different casing/spacing.
        await _try_login(app_client, ident, "wrong-password")
    await _try_login(app_client, "alice", "wrong-password")                  # Allowed attempt.
    assert (await _try_login(app_client, "Alice", PW)).status_code == 429
    assert (await _try_login(app_client, "bobby", PW)).status_code == 200    # Another account is unaffected.


async def test_unknown_accounts_are_throttled_the_same_way(app_client, throttle):
    for _ in range(3):
        assert (await _try_login(app_client, "ghost", "whatever-pass")).status_code == 401
    await _try_login(app_client, "ghost", "whatever-pass")
    assert (await _try_login(app_client, "ghost", "whatever-pass")).status_code == 429   # Do not reveal which accounts exist.


async def test_throttle_works_without_redis_and_can_be_disabled(app_client, monkeypatch):
    class Broken:
        async def get(self, *a, **k):
            raise ConnectionError("redis is down")

        def pipeline(self, *a, **k):
            raise ConnectionError("redis is down")

    t = LoginThrottle(client_factory=Broken)
    monkeypatch.setattr(auth_module, "login_throttle", t)
    monkeypatch.setattr(settings, "login_fail_threshold", 2)
    monkeypatch.setattr(settings, "login_soft_lock_interval_s", 30)
    await _login(app_client)
    for _ in range(2):
        assert (await _try_login(app_client, "alice", "wrong-password")).status_code == 401
    await _try_login(app_client, "alice", "wrong-password")
    assert (await _try_login(app_client, "alice", PW)).status_code == 429     # Still protected when Redis is unavailable.

    monkeypatch.setattr(settings, "login_fail_threshold", 0)                 # 0 = disabled.
    assert (await _try_login(app_client, "alice", PW)).status_code == 200


# =====================================================================================
# 5) Conversations: reuse empty conversations, clean up, and enforce limits
# =====================================================================================
async def _create_conv(client, headers, document_id=None):
    return await client.post("/api/conversations", headers=headers, json={"document_id": document_id})


async def _count(model, **where):
    async with coredb.async_session() as s:
        stmt = select(func.count(model.id))
        for k, v in where.items():
            stmt = stmt.where(getattr(model, k) == v)
        return (await s.execute(stmt)).scalar_one()


async def test_creating_repeatedly_reuses_the_single_empty_conversation(app_client):
    headers, uid = await _login(app_client)
    first = (await _create_conv(app_client, headers)).json()
    for _ in range(5):
        again = (await _create_conv(app_client, headers)).json()
        assert again["id"] == first["id"]
    assert await _count(Conversation, owner_id=uuid.UUID(uid)) == 1        # Six calls still produce exactly one row.

    await conversation_store.append_message(first["id"], uid, "user", "hi", title_from="hi")
    second = (await _create_conv(app_client, headers)).json()
    assert second["id"] != first["id"]                                      # Once it has a message, create a new conversation.


async def test_reused_conversation_takes_the_new_document_scope(app_client):
    headers, uid = await _login(app_client)
    async with coredb.async_session() as s:
        doc = Document(owner_id=uuid.UUID(uid), filename="a.pdf", status="embedded")
        s.add(doc)
        await s.commit()
        doc_id = str(doc.id)
    first = (await _create_conv(app_client, headers)).json()
    assert first["document_id"] is None
    again = (await _create_conv(app_client, headers, doc_id)).json()
    assert again["id"] == first["id"] and again["document_id"] == doc_id


async def test_stale_empty_conversations_are_purged_but_real_ones_are_kept(app_client):
    headers, uid = await _login(app_client)
    other, other_id = await _login(app_client, "bobby")
    old = datetime.now(timezone.utc) - timedelta(days=3)
    async with coredb.async_session() as s:
        stale = Conversation(owner_id=uuid.UUID(uid), title="", created_at=old)
        kept = Conversation(owner_id=uuid.UUID(uid), title="Giữ lại", created_at=old)
        bobs_stale = Conversation(owner_id=uuid.UUID(other_id), title="", created_at=old)
        s.add_all([stale, kept, bobs_stale])
        await s.flush()
        s.add(ChatMessage(conversation_id=kept.id, role="user", content="q"))
        await s.commit()
        stale_id, kept_id, bobs_id = stale.id, kept.id, bobs_stale.id

    await _create_conv(app_client, headers)                                  # Clean up this user's conversations.

    async with coredb.async_session() as s:
        assert await s.get(Conversation, stale_id) is None
        assert await s.get(Conversation, kept_id) is not None                # Has messages => keep it.
        assert await s.get(Conversation, bobs_id) is not None                # Belongs to another user: do not touch.

    async with coredb.async_session() as s:                                  # Global cleanup (at startup).
        assert await conversation_store.purge_stale_empty(s, 86400) == 1
        await s.commit()
        assert await s.get(Conversation, bobs_id) is None
        assert await s.get(Conversation, kept_id) is not None


async def test_startup_sweep_runs_in_lifespan(app_client, monkeypatch):
    import app.main as m

    owner = uuid.uuid4()
    old = datetime.now(timezone.utc) - timedelta(days=3)
    async with coredb.async_session() as s:
        from app.models.models import User
        s.add(User(id=owner, username="sweeper", email="s@example.com", hashed_password="x"))
        await s.flush()
        c = Conversation(owner_id=owner, title="", created_at=old)
        s.add(c)
        await s.commit()
        cid = c.id
    monkeypatch.setattr(settings, "jwt_secret", "z" * 48)
    monkeypatch.setattr(settings, "auto_migrate_schema", False)
    async with m.lifespan(m.app):
        pass
    async with coredb.async_session() as s:
        assert await s.get(Conversation, cid) is None


async def test_conversation_count_cap(app_client, monkeypatch):
    headers, uid = await _login(app_client)
    monkeypatch.setattr(settings, "max_conversations_per_user", 2)
    for i in range(2):
        conv = (await _create_conv(app_client, headers)).json()
        await conversation_store.append_message(conv["id"], uid, "user", f"q{i}", title_from=f"q{i}")

    res = await _create_conv(app_client, headers)

    assert res.status_code == 403 and "Conversation limit reached (2)" in res.json()["detail"]
    first = (await app_client.get("/api/conversations", headers=headers)).json()[0]
    assert (await app_client.delete(f"/api/conversations/{first['id']}", headers=headers)).status_code == 204
    assert (await _create_conv(app_client, headers)).status_code == 201     # Can create one after deleting an existing conversation.


async def test_messages_per_conversation_cap(app_client, monkeypatch):
    headers, _ = await _login(app_client)
    conv = (await _create_conv(app_client, headers)).json()
    monkeypatch.setattr(settings, "max_messages_per_conversation", 2)
    searches = {"n": 0}

    def search(**k):
        searches["n"] += 1
        return [HIT]

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    assert _events(await _ask(app_client, headers, conversation_id=conv["id"]))[-1]["type"] == "result"

    res = await _ask(app_client, headers, "another one", conversation_id=conv["id"])   # Already has 2 messages.

    assert res.status_code == 409 and "reached its limit of 2 messages" in res.json()["detail"]
    assert searches["n"] == 1                                                # Do not spend resources on the rejected request.
