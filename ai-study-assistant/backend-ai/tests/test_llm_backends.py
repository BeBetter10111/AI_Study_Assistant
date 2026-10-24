import httpx
import pytest
from pydantic import BaseModel

from app.config import settings
from app.core import llm_engine as llm_module
from app.core.llm_engine import LLMEngineManager


class _Schema(BaseModel):
    topic: str
    count: int


def _ollama_engine(monkeypatch) -> LLMEngineManager:
    monkeypatch.setattr(settings, "llm_backend", "ollama")
    monkeypatch.setattr(settings, "ollama_base_url", "http://ollama.test:11434/")
    monkeypatch.setattr(settings, "ollama_model_name", "qwen2.5:3b")
    return LLMEngineManager()


def _fake_post(monkeypatch, handler):
    def post(url, json=None, timeout=None):
        request = httpx.Request("POST", url)
        return handler(url, json, timeout, request)

    monkeypatch.setattr(llm_module.httpx, "post", post)


def test_invalid_backend_fails_fast(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "olama")
    with pytest.raises(ValueError):
        LLMEngineManager()


def test_ollama_backend_does_not_load_vllm(monkeypatch):
    engine = _ollama_engine(monkeypatch)
    assert engine.backend == "ollama"
    assert engine.llm is None and engine.outlines_model is None


def test_ollama_rag_response_sends_expected_payload(monkeypatch):
    engine = _ollama_engine(monkeypatch)
    captured = {}

    def handler(url, payload, timeout, request):
        captured.update(url=url, payload=payload, timeout=timeout)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "  Xin chào  "}}, request=request)

    _fake_post(monkeypatch, handler)
    answer = engine.generate_rag_response(prompt="Câu hỏi?", system_prompt="Hệ thống", max_tokens=123)

    assert answer == "Xin chào"
    assert captured["url"] == "http://ollama.test:11434/api/chat"
    payload = captured["payload"]
    assert payload["model"] == "qwen2.5:3b"
    assert payload["stream"] is False
    assert [m["role"] for m in payload["messages"]] == ["system", "user"]
    assert payload["options"]["num_predict"] == 123
    assert payload["options"]["num_ctx"] == settings.ollama_num_ctx
    assert "format" not in payload
    assert captured["timeout"] == settings.ollama_timeout_s


def test_ollama_unreachable_returns_friendly_message(monkeypatch):
    engine = _ollama_engine(monkeypatch)

    def handler(url, payload, timeout, request):
        raise httpx.ConnectError("refused", request=request)

    _fake_post(monkeypatch, handler)
    answer = engine.generate_rag_response(prompt="q", system_prompt="s")
    assert "Ollama" in answer and "http://ollama.test:11434" in answer


def test_ollama_timeout_returns_friendly_message(monkeypatch):
    engine = _ollama_engine(monkeypatch)

    def handler(url, payload, timeout, request):
        raise httpx.ReadTimeout("slow", request=request)

    _fake_post(monkeypatch, handler)
    assert "quá chậm" in engine.generate_rag_response(prompt="q", system_prompt="s")


def test_ollama_missing_model_tells_user_to_pull(monkeypatch):
    engine = _ollama_engine(monkeypatch)

    def handler(url, payload, timeout, request):
        return httpx.Response(404, json={"error": "model not found"}, request=request)

    _fake_post(monkeypatch, handler)
    assert "ollama pull qwen2.5:3b" in engine.generate_rag_response(prompt="q", system_prompt="s")


def test_ollama_structured_output_valid_json(monkeypatch):
    engine = _ollama_engine(monkeypatch)
    captured = {}

    def handler(url, payload, timeout, request):
        captured.update(payload=payload)
        return httpx.Response(
            200, json={"message": {"content": '{"topic": "OOP", "count": 3}'}}, request=request
        )

    _fake_post(monkeypatch, handler)
    result = engine.generate_structured_output(prompt="Tạo quiz", schema_class=_Schema)

    assert result == {"topic": "OOP", "count": 3}
    assert captured["payload"]["format"] == _Schema.model_json_schema()
    assert "Tạo quiz" in captured["payload"]["messages"][0]["content"]


@pytest.mark.parametrize("bad_content", ["không phải json", '{"topic": "OOP"}', '{"topic": "OOP", "count": "abc"}'])
def test_ollama_structured_output_rejects_bad_json(monkeypatch, bad_content):
    engine = _ollama_engine(monkeypatch)

    def handler(url, payload, timeout, request):
        return httpx.Response(200, json={"message": {"content": bad_content}}, request=request)

    _fake_post(monkeypatch, handler)
    with pytest.raises(ValueError):
        engine.generate_structured_output(prompt="x", schema_class=_Schema)


def test_ollama_structured_output_unreachable_raises_runtime_error(monkeypatch):
    engine = _ollama_engine(monkeypatch)

    def handler(url, payload, timeout, request):
        raise httpx.ConnectError("refused", request=request)

    _fake_post(monkeypatch, handler)
    with pytest.raises(RuntimeError):
        engine.generate_structured_output(prompt="x", schema_class=_Schema)


async def test_chat_query_success_path_runs_llm_in_threadpool(app_client, monkeypatch):
    from app.api.endpoints import chat

    hit = {"score": 0.9, "content": "Khóa chính là...", "document_id": "doc-1", "chunk_index": 2}
    monkeypatch.setattr(chat.vector_store, "search_similar_chunks", lambda **kw: [hit])
    monkeypatch.setattr(chat.llm_engine, "generate_rag_response", lambda prompt, system_prompt: "Trả lời mẫu")

    response = await app_client.post("/api/chat/query", json={"question": "Khóa chính là gì?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Trả lời mẫu"
    assert body["citations"][0]["chunk_index"] == 2


async def test_quiz_generate_filters_questions_with_invalid_correct_index(app_client, monkeypatch):
    from app.api.endpoints import quiz

    hit = {"score": 0.9, "content": "nội dung", "document_id": "d", "chunk_index": 0}
    monkeypatch.setattr(quiz.vector_store, "search_similar_chunks", lambda **kw: [hit])

    good = {"question": "Q1", "options": ["a", "b"], "correct_index": 1, "explanation": "e"}
    out_of_range = {"question": "Q2", "options": ["a", "b"], "correct_index": 5, "explanation": "e"}
    monkeypatch.setattr(
        quiz.llm_engine,
        "generate_structured_output",
        lambda prompt, schema_class: {"topic": "T", "questions": [good, out_of_range]},
    )

    response = await app_client.post("/api/quiz/generate", json={"topic": "T", "num_questions": 2})
    assert response.status_code == 200
    assert [q["question"] for q in response.json()["questions"]] == ["Q1"]


async def test_quiz_generate_returns_502_when_llm_fails(app_client, monkeypatch):
    from app.api.endpoints import quiz

    hit = {"score": 0.9, "content": "nội dung", "document_id": "d", "chunk_index": 0}
    monkeypatch.setattr(quiz.vector_store, "search_similar_chunks", lambda **kw: [hit])

    def boom(prompt, schema_class):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(quiz.llm_engine, "generate_structured_output", boom)
    response = await app_client.post("/api/quiz/generate", json={"topic": "T", "num_questions": 2})
    assert response.status_code == 502


def test_vllm_backend_still_works_after_lazy_import_refactor(monkeypatch):
    """Đường vLLM (môi trường chạy thật) không được hỏng khi dời import vào
    trong hàm: giả lập CUDA khả dụng + vLLM/Outlines giả rồi kiểm tra cả
    sinh text lẫn structured output.
    """
    import sys
    import types

    monkeypatch.setattr(settings, "llm_backend", "vllm")
    monkeypatch.setattr(sys.modules["torch"].cuda, "is_available", staticmethod(lambda: True))

    class FakeLLM:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, prompts, params):
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=" vllm trả lời ")])]

    class FakeGenerate:
        @staticmethod
        def json(model, schema_class):
            return lambda prompt, temperature: schema_class(topic="OOP", count=1)

    monkeypatch.setattr(sys.modules["vllm"], "LLM", FakeLLM)
    monkeypatch.setattr(sys.modules["outlines"], "generate", FakeGenerate)

    engine = LLMEngineManager()

    assert isinstance(engine.llm, FakeLLM)
    assert engine.llm.kwargs["tensor_parallel_size"] == settings.llm_tensor_parallel_size
    assert engine.generate_rag_response(prompt="q", system_prompt="s") == "vllm trả lời"
    assert engine.generate_structured_output(prompt="p", schema_class=_Schema) == {"topic": "OOP", "count": 1}


def test_vllm_backend_without_cuda_degrades_gracefully(monkeypatch):
    monkeypatch.setattr(settings, "llm_backend", "vllm")
    engine = LLMEngineManager()
    assert engine.llm is None
    assert "chưa sẵn sàng" in engine.generate_rag_response(prompt="q", system_prompt="s")
    with pytest.raises(RuntimeError):
        engine.generate_structured_output(prompt="p", schema_class=_Schema)
