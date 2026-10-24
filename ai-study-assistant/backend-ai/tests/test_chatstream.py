"""/api/chat/query/stream — the UI shows these steps instead of a bare 'Thinking…'."""
import json

import app.api.endpoints.chat as chat_module

_HIT = {
    "document_id": "d1", "chunk_index": 0, "score": 0.9, "content": "Classes are blueprints.",
    "page_number": 3, "slide_number": None, "slide_title": None,
    "section_title": None, "is_summary": False,
}


def _events(response):
    return [
        json.loads(line[5:])
        for line in response.text.split("\n\n")
        if line.startswith("data:")
    ]


async def test_stream_reports_real_steps_then_result(app_client, monkeypatch):
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [_HIT])
    monkeypatch.setattr(chat_module.llm_engine, "generate_rag_response", lambda **k: "Answer [1]")

    response = await app_client.post("/api/chat/query/stream", json={"question": "What is a class?"})

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
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["answer"] == "Answer [1]"
    assert events[-1]["data"]["citations"][0]["page_number"] == 3


async def test_stream_no_hits_skips_generation(app_client, monkeypatch):
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [])

    events = _events(await app_client.post("/api/chat/query/stream", json={"question": "anything?"}))

    assert not any(e.get("id") == "generate" for e in events)
    assert events[-1]["type"] == "result"
    assert events[-1]["data"]["citations"] == []


async def test_stream_failure_is_reported_in_stream(app_client, monkeypatch):
    def boom(**_):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", boom)

    events = _events(await app_client.post("/api/chat/query/stream", json={"question": "hi there"}))

    assert events[-1]["type"] == "error"
    assert "qdrant down" in events[-1]["detail"]


async def test_stream_rejects_blank_question(app_client):
    response = await app_client.post("/api/chat/query/stream", json={"question": "   "})
    assert response.status_code == 400
