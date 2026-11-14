"""/api/chat/query/stream — bước xử lý, token streaming, semantic cache, lỗi, hủy khi client ngắt."""
import json
import threading
import types
import uuid

import numpy as np
import pytest
from starlette.requests import Request

import app.api.endpoints.chat as chat_module
from app.api.endpoints.chat import QueryRequest
from app.api.security.rate_limiter import limiter
from app.config import settings

_HIT = {
    "document_id": "d1", "chunk_index": 0, "score": 0.9, "content": "Classes are blueprints.",
    "page_number": 3, "slide_number": None, "slide_title": None,
    "section_title": None, "is_summary": False,
}
PW = "firstpassword1"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)                      # bộ đếm rate limit dùng chung giữa test
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)      # mặc định: không đụng Redis


def _events(response):
    return [json.loads(line[5:]) for line in response.text.split("\n\n") if line.startswith("data:")]


async def _login(client, name="alice"):
    res = await client.post("/api/auth/register",
                            json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _fake_stream(tokens=("Answer ", "[1]"), error=None, record=None):
    def fake(prompt, system_prompt, *a, **k):
        if record is not None:
            record.append(prompt)

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


async def test_requires_login(app_client):
    res = await app_client.post("/api/chat/query/stream", json={"question": "hi"})
    assert res.status_code == 401


async def test_stream_reports_real_steps_tokens_then_result(app_client, monkeypatch):
    headers = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    response = await _ask(app_client, headers)

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response)
    steps = [(e["id"], e["status"]) for e in events if e["type"] == "step"]
    assert steps == [
        ("intent", "running"), ("intent", "done"),
        ("search", "running"), ("search", "done"),
        ("generate", "running"), ("generate", "done"),
    ]
    search_done = next(e for e in events if e.get("id") == "search" and e["status"] == "done")
    assert "Page 3" in search_done["detail"]
    tokens = [e["text"] for e in events if e["type"] == "token"]
    assert tokens == ["Answer ", "[1]"]
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["answer"] == "Answer [1]"
    assert events[-1]["data"]["cached"] is False
    assert events[-1]["data"]["citations"][0]["page_number"] == 3
    # token đến TRƯỚC result (đó là điều làm chữ hiện dần trên giao diện)
    kinds = [e["type"] for e in events]
    assert kinds.index("token") < kinds.index("result")


async def test_no_hits_skips_generation(app_client, monkeypatch):
    headers = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [])

    events = _events(await _ask(app_client, headers, "anything?"))

    assert not any(e.get("id") == "generate" for e in events)
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["citations"] == []


async def test_search_failure_is_reported_in_stream(app_client, monkeypatch):
    headers = await _login(app_client)

    def boom(**_):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", boom)

    events = _events(await _ask(app_client, headers, "hi there"))

    assert events[-1]["type"] == "error"
    assert "qdrant down" in events[-1]["detail"]
    assert not any(e["type"] == "result" for e in events)


async def test_llm_error_is_an_error_event_not_an_answer(app_client, monkeypatch):
    headers = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response",
                        _fake_stream(error="Could not connect to Ollama"))

    events = _events(await _ask(app_client, headers))

    assert events[-1] == {"type": "error", "detail": "Could not connect to Ollama"}
    assert not any(e["type"] == "result" for e in events)


async def test_empty_model_output_is_an_error(app_client, monkeypatch):
    headers = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream(tokens=("  ",)))

    events = _events(await _ask(app_client, headers))

    assert events[-1]["type"] == "error"


async def test_history_triggers_rewrite_and_is_used_for_search(app_client, monkeypatch):
    headers = await _login(app_client)
    seen = {}
    monkeypatch.setattr(chat_module, "_rewrite_query",
                        lambda q, history: seen.update(history=history) or "What is polymorphism in OOP?")

    def search(**k):
        seen["query"] = k["query"]
        return [_HIT]

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    history = [{"role": "user", "content": "Tell me about OOP"}, {"role": "assistant", "content": "OOP is ..."}]
    events = _events(await _ask(app_client, headers, "and polymorphism?", history=history))

    assert any(e.get("id") == "rewrite" for e in events)
    assert seen["query"] == "What is polymorphism in OOP?"
    assert len(seen["history"]) == 2
    assert events[-1]["data"]["standalone_query"] == "What is polymorphism in OOP?"


# ---------------------------------------------------------------------------------------
# Semantic cache qua endpoint
# ---------------------------------------------------------------------------------------
@pytest.fixture
def cache_on(monkeypatch):
    fakeredis = pytest.importorskip("fakeredis")
    from app.rag.semantic_cache import SemanticCache

    vectors = {
        "What is a class?": [1.0, 0.0, 0.0],
        "What does a class mean?": [0.97, 0.243, 0.0],   # cosine ≈ 0.97 với câu trên, cùng intent (qa)
        "Explain what a class is": [0.97, 0.243, 0.0],   # cùng nghĩa nhưng intent = explanation
        "What is an interface?": [0.0, 1.0, 0.0],
    }
    server = fakeredis.FakeServer()
    cache = SemanticCache(
        embed_fn=lambda t: np.array(vectors.get(t, [0.0, 0.0, 1.0]), dtype=np.float32),
        fingerprint_fn=lambda: "test-fp",
        client_factory=lambda: fakeredis.FakeAsyncRedis(server=server),
    )
    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(chat_module, "semantic_cache", cache)
    return cache


async def test_similar_question_is_served_from_cache_without_search_or_llm(app_client, monkeypatch, cache_on):
    headers = await _login(app_client)
    calls = {"search": 0, "llm": 0}

    def search(**k):
        calls["search"] += 1
        return [_HIT]

    def llm(prompt, system_prompt, *a, **k):
        calls["llm"] += 1
        return _fake_stream()(prompt, system_prompt)

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", llm)

    first = _events(await _ask(app_client, headers, "What is a class?"))
    assert first[-1]["data"]["cached"] is False
    assert calls == {"search": 1, "llm": 1}

    second = _events(await _ask(app_client, headers, "What does a class mean?"))
    assert calls == {"search": 1, "llm": 1}                      # KHÔNG gọi lại search / LLM
    result = second[-1]
    assert result["type"] == "result" and result["data"]["cached"] is True
    assert result["data"]["answer"] == "Answer [1]"
    assert result["data"]["citations"][0]["page_number"] == 3     # nguồn trích dẫn được giữ nguyên
    cache_step = [e for e in second if e.get("id") == "cache" and e["status"] == "done"]
    assert cache_step and "similar" in cache_step[0]["detail"]
    assert not any(e["type"] == "token" for e in second)

    other = _events(await _ask(app_client, headers, "What is an interface?"))   # khác nghĩa => miss
    assert other[-1]["data"]["cached"] is False
    assert calls == {"search": 2, "llm": 2}

    # Cùng nghĩa nhưng khác intent (explanation dùng prompt khác) => phạm vi cache khác => miss.
    explained = _events(await _ask(app_client, headers, "Explain what a class is"))
    assert explained[-1]["data"]["cached"] is False
    assert calls == {"search": 3, "llm": 3}


async def test_cache_is_not_shared_between_users(app_client, monkeypatch, cache_on):
    alice = await _login(app_client, "alice")
    bob = await _login(app_client, "bobby")
    calls = {"llm": 0}

    def llm(prompt, system_prompt, *a, **k):
        calls["llm"] += 1
        return _fake_stream()(prompt, system_prompt)

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", llm)

    await _ask(app_client, alice, "What is a class?")
    res = _events(await _ask(app_client, bob, "What is a class?"))
    assert res[-1]["data"]["cached"] is False and calls["llm"] == 2


async def test_errors_and_empty_results_are_never_cached(app_client, monkeypatch, cache_on):
    headers = await _login(app_client)
    searches = {"n": 0}

    def search(**k):
        searches["n"] += 1
        return []

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    await _ask(app_client, headers, "What is a class?")
    await _ask(app_client, headers, "What is a class?")
    assert searches["n"] == 2                                      # "không tìm thấy" không được cache

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream(error="Ollama down"))
    assert _events(await _ask(app_client, headers, "What is a class?"))[-1]["type"] == "error"

    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())
    again = _events(await _ask(app_client, headers, "What is a class?"))
    assert again[-1]["data"]["cached"] is False                    # lỗi trước đó không để lại mục nào
    assert again[-1]["data"]["answer"] == "Answer [1]"


async def test_redis_down_does_not_break_chat(app_client, monkeypatch):
    from app.rag.semantic_cache import SemanticCache

    class Broken:
        async def get(self, *a, **k):
            raise ConnectionError("redis down")

    monkeypatch.setattr(settings, "semantic_cache_enabled", True)
    monkeypatch.setattr(chat_module, "semantic_cache", SemanticCache(
        embed_fn=lambda t: np.array([1.0, 0.0], dtype=np.float32),
        fingerprint_fn=lambda: "fp", client_factory=Broken))
    headers = await _login(app_client)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    events = _events(await _ask(app_client, headers))

    assert events[-1]["type"] == "result" and events[-1]["data"]["answer"] == "Answer [1]"


# ---------------------------------------------------------------------------------------
# Client ngắt kết nối => dừng LLM
# ---------------------------------------------------------------------------------------
async def test_closing_the_response_closes_the_llm_stream(monkeypatch):
    closed = threading.Event()

    def llm(prompt, system_prompt, *a, **k):
        def gen():
            try:
                for i in range(1000):
                    yield {"token": f"t{i} "}
            finally:
                closed.set()          # = with httpx.stream(...) thoát => Ollama ngừng sinh

        return gen()

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", llm)
    request = Request({"type": "http", "method": "POST", "path": "/api/chat/query/stream", "headers": [],
                       "query_string": b"", "client": ("test", 1), "server": ("test", 80)})
    user = types.SimpleNamespace(id=uuid.uuid4())

    response = await chat_module.query_stream(request, QueryRequest(question="What is a class?"), user)
    body = response.body_iterator
    seen_token = False
    async for chunk in body:
        if '"type": "token"' in chunk or '"type":"token"' in chunk:
            seen_token = True
            break                      # người dùng bấm "Dừng" / đóng tab giữa chừng
    assert seen_token
    await body.aclose()

    import asyncio
    import time

    deadline = time.monotonic() + 3
    while not closed.is_set() and time.monotonic() < deadline:
        await asyncio.sleep(0.02)
    assert closed.is_set(), "LLM generator must be closed when the client goes away"
