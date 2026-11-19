"""Post-generation verification: groundedness score, numbers, citations, abstention, fail-open, API wiring."""
import json

import numpy as np
import pytest

import app.api.endpoints.chat as chat_module
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.rag import verification as v

HITS = [
    {"content": "Stack là cấu trúc dữ liệu LIFO. Phần tử được thêm vào và lấy ra từ đỉnh. "
                "Thao tác push có độ phức tạp O(1).", "document_id": "d", "chunk_index": 0},
    {"content": "Queue hoạt động theo nguyên tắc FIFO. Phần tử đầu tiên được thêm vào sẽ được lấy ra đầu tiên.",
     "document_id": "d", "chunk_index": 1},
]
GOOD = ("Stack là cấu trúc dữ liệu LIFO [1]. Phần tử được thêm và lấy ra từ đỉnh của stack [1]. "
        "Queue hoạt động theo nguyên tắc FIFO [2].")
BAD = ("Stack là cấu trúc dữ liệu LIFO [1]. Thao tác push có độ phức tạp O(n) và dùng 64 byte bộ nhớ [3]. "
       "Queue được phát minh năm 1950 bởi Dijkstra.")


def bow_embed(texts):
    """Deterministic bag-of-words embedding so semantic scoring can be tested without the real model."""
    vocab = {}
    rows = []
    for t in texts:
        row = {}
        for w in v._content_tokens(t):
            idx = vocab.setdefault(w, len(vocab))
            row[idx] = row.get(idx, 0) + 1
        rows.append(row)
    out = np.zeros((len(texts), 4096), dtype=np.float32)
    for i, row in enumerate(rows):
        for k, c in row.items():
            out[i, k % 4096] += c
    return out


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(settings, "verification_enabled", True)
    monkeypatch.setattr(settings, "verification_max_claims", 20)


def test_grounded_answer_gets_high_confidence_lexical_and_semantic():
    for kwargs in ({"use_embeddings": False}, {"embed_fn": bow_embed, "use_embeddings": True}):
        res = v.verify_answer(GOOD, HITS, **kwargs)
        assert res.label == "high" and res.confidence >= 0.75
        assert res.supported_ratio == 1.0 and not res.unsupported_claims


def test_hallucinated_answer_is_low_with_unsupported_claims_numbers_and_bad_citation():
    res = v.verify_answer(BAD, HITS, embed_fn=bow_embed)
    assert res.label == "low" and res.confidence < 0.5
    assert 3 in res.invalid_citations                          # [3] does not exist (only 2 passages)
    bad_numbers = {n for c in res.claims for n in c.missing_numbers}
    assert {"64", "1950"} <= bad_numbers                         # numbers absent from the context
    assert len(res.unsupported_claims) == 2


def test_invented_number_alone_makes_a_claim_unsupported():
    res = v.verify_answer("Thao tác push của stack có độ phức tạp 99 bước [1].", HITS, use_embeddings=False)
    assert not res.claims[0].supported and res.claims[0].missing_numbers == ["99"]


def test_abstention_is_not_penalised_and_trivial_answers_are_unverified():
    assert v.verify_answer("Tài liệu không đề cập đến nội dung này.", HITS).label == "abstain"
    assert v.verify_answer("The context does not contain this information.", HITS).confidence is None
    assert v.verify_answer("Có.", HITS).label == "unverified"


def test_markdown_bullets_and_citations_are_stripped_from_claims():
    claims = v.split_claims("- **Stack** là cấu trúc dữ liệu LIFO [1].\n1. Queue hoạt động theo nguyên tắc FIFO [2]")
    assert [c["cited"] for c in claims] == [[1], [2]]
    assert all("[" not in c["text"] and "*" not in c["text"] for c in claims)


def test_disabled_empty_and_no_hits_return_none(monkeypatch):
    assert v.verify_answer("", HITS) is None and v.verify_answer(GOOD, []) is None
    monkeypatch.setattr(settings, "verification_enabled", False)
    assert v.verify_answer(GOOD, HITS) is None


def test_embedding_failure_falls_back_to_lexical_and_errors_are_swallowed():
    def boom(_texts):
        raise RuntimeError("model not loaded")

    res = v.verify_answer(GOOD, HITS, embed_fn=boom)
    assert res is not None and res.label == "high" and res.used_embeddings is False


def test_public_payload_shape():
    pub = v.verify_answer(BAD, HITS, use_embeddings=False).to_public()
    assert set(pub) == {"confidence", "confidence_label", "supported_ratio", "unsupported_claims", "invalid_citations"}
    assert 0 <= pub["confidence"] <= 1 and pub["confidence_label"] == "low"


# ---- API wiring -----------------------------------------------------------------------------------------------
PW = "firstpassword1"


async def _login(client, name):
    res = await client.post("/api/auth/register", json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def _fake_stream(text):
    def fn(prompt, system_prompt, *a, **k):
        def gen():
            yield {"token": text}
            yield {"done": True}
        return gen()
    return fn


async def test_stream_result_carries_confidence_and_unsupported_claims(app_client, monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    monkeypatch.setattr(settings, "verification_use_embeddings", False)
    headers = await _login(app_client, "ver1")
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: HITS)
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream(BAD))
    res = await app_client.post("/api/chat/query/stream", headers=headers, json={"question": "Stack là gì?"})
    events = [json.loads(line[5:]) for line in res.text.split("\n\n") if line.startswith("data:")]
    data = events[-1]["data"]
    assert data["confidence_label"] == "low" and data["confidence"] < 0.5
    assert data["unsupported_claims"] and 3 in data["invalid_citations"]
    assert ("verify", "done") in [(e["id"], e["status"]) for e in events if e["type"] == "step"]


async def test_non_stream_query_includes_confidence(app_client, monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    monkeypatch.setattr(settings, "verification_use_embeddings", False)
    headers = await _login(app_client, "ver2")
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: HITS)
    monkeypatch.setattr(chat_module.llm_engine, "generate_rag_response", lambda **k: GOOD)
    res = await app_client.post("/api/chat/query", headers=headers, json={"question": "Stack là gì?"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["confidence_label"] == "high" and body["unsupported_claims"] == []


async def test_verification_can_be_switched_off(app_client, monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)
    monkeypatch.setattr(settings, "verification_enabled", False)
    headers = await _login(app_client, "ver3")
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: HITS)
    monkeypatch.setattr(chat_module.llm_engine, "generate_rag_response", lambda **k: GOOD)
    body = (await app_client.post("/api/chat/query", headers=headers, json={"question": "Stack là gì?"})).json()
    assert body["confidence"] is None and body["confidence_label"] is None
