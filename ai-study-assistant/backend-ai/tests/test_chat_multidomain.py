"""Chat wiring for multi-domain retrieval: query expansion, legal article completion, tool routes."""
import json

import pytest

import app.api.endpoints.chat as chat_module
from app.agent.tools import math_tool
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.rag import query_expansion

PW = "firstpassword1"
HIT = {
    "document_id": "d1", "chunk_index": 0, "score": 0.9, "content": "Paracetamol 500 mg.",
    "page_number": 1, "slide_number": None, "slide_title": None, "section_title": None, "is_summary": False,
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    monkeypatch.setattr(settings, "query_expansion_enabled", True)
    monkeypatch.setattr(settings, "umls_api_key", "", raising=False)
    query_expansion.reset_linkers()


def _events(response):
    return [json.loads(line[5:]) for line in response.text.split("\n\n") if line.startswith("data:")]


async def _login(client, name="multi"):
    res = await client.post("/api/auth/register",
                            json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _fake_stream(prompt, system_prompt, *a, **k):
    def gen():
        yield {"token": "OK"}
        yield {"done": True}
    return gen()


async def test_stream_passes_expansion_terms_to_search(app_client, monkeypatch):
    headers = await _login(app_client)
    seen = {}

    def search(**kw):
        seen.update(kw)
        return [HIT]

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream)
    res = await app_client.post("/api/chat/query/stream", headers=headers,
                                json={"question": "Liều dùng Panadol cho trẻ em?"})
    assert _events(res)[-1]["type"] == "result"
    assert "paracetamol" in [t.lower() for t in seen["expansion_terms"]]


async def test_non_stream_query_also_expands(app_client, monkeypatch):
    headers = await _login(app_client, "multi2")
    seen = {}

    def search(**kw):
        seen.update(kw)
        return [HIT]

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)
    monkeypatch.setattr(chat_module.llm_engine, "generate_rag_response", lambda **k: "Answer [1]")
    res = await app_client.post("/api/chat/query", headers=headers, json={"question": "Tác dụng phụ của Advil?"})
    assert res.status_code == 200, res.text
    assert "ibuprofen" in [t.lower() for t in seen["expansion_terms"]]


def test_search_hits_completes_legal_articles(monkeypatch):
    legal_hit = {**HIT, "article_key": "2:5"}
    sibling = {**HIT, "chunk_index": 1, "article_key": "2:5", "expanded": True}
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [legal_hit])
    monkeypatch.setattr(chat_module.vector_store, "expand_legal_articles",
                        lambda hits, owner_id=None: [*hits, sibling])
    hits = chat_module._search_hits("Điều 5 quy định gì?", 4, None, "owner")
    assert [h["chunk_index"] for h in hits] == [0, 1]


def test_search_hits_skips_expansion_when_nothing_found(monkeypatch):
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [])

    def boom(*a, **k):
        raise AssertionError("must not be called without hits")

    monkeypatch.setattr(chat_module.vector_store, "expand_legal_articles", boom)
    assert chat_module._search_hits("q", 4, None, "owner") == []


def test_search_hits_survives_expansion_failure(monkeypatch):
    seen = {}

    def search(**kw):
        seen.update(kw)
        return [HIT]

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", search)

    def boom(_q):
        raise RuntimeError("linker down")

    monkeypatch.setattr(chat_module, "expand_query", boom)
    assert chat_module._search_hits("Panadol", 4, None, "owner") == [HIT]
    assert seen["expansion_terms"] == []


async def test_math_question_uses_sympy_tool_without_retrieval(app_client, monkeypatch):
    headers = await _login(app_client, "multi3")
    monkeypatch.setattr(math_tool.settings, "math_tool_enabled", True)

    def no_search(**kw):
        raise AssertionError("tool-only answers must not hit the vector store")

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", no_search)
    res = await app_client.post("/api/chat/query", headers=headers, json={"question": "Tính 12 * 12"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["intent"] == "math" and body["tool_used"] and "144" in body["answer"]
