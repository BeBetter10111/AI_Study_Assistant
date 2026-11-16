"""Liệt kê/xóa tài liệu, lịch sử hội thoại (CRUD + lưu qua streaming), kiểm tra JWT_SECRET."""
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import app.api.endpoints.chat as chat_module
import app.api.endpoints.documents as docs_module
import app.core.database as coredb
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core import conversation_store
from app.core.security_check import enforce_jwt_secret, insecure_secret_reason
from app.models.models import ChatMessage, Conversation, Document, QuizAttempt

PW = "firstpassword1"
HIT = {
    "document_id": "d1", "chunk_index": 0, "score": 0.9, "content": "Classes are blueprints.",
    "page_number": 3, "slide_number": None, "slide_title": None, "section_title": None, "is_summary": False,
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(limiter, "enabled", False)
    monkeypatch.setattr(settings, "semantic_cache_enabled", False)


async def _login(client, name="alice"):
    res = await client.post("/api/auth/register",
                            json={"username": name, "email": f"{name}@example.com", "password": PW})
    assert res.status_code == 201, res.text
    headers = {"Authorization": f"Bearer {res.json()['access_token']}"}
    me = await client.get("/api/auth/me", headers=headers)
    return headers, me.json()["id"]


async def _add_doc(owner_id, name="a.pdf", status="embedded", age_minutes=0, storage_path=None):
    async with coredb.async_session() as s:
        doc = Document(owner_id=uuid.UUID(owner_id), filename=name, mime_type="application/pdf", status=status,
                       storage_path=storage_path,
                       created_at=datetime.now(timezone.utc) - timedelta(minutes=age_minutes))
        s.add(doc)
        await s.commit()
        return str(doc.id)


# =====================================================================================
# Tài liệu: liệt kê
# =====================================================================================
async def test_document_list_requires_login_and_is_scoped_and_newest_first(app_client):
    assert (await app_client.get("/api/documents")).status_code == 401
    a, a_id = await _login(app_client, "alice")
    b, b_id = await _login(app_client, "bobby")
    old = await _add_doc(a_id, "old.pdf", age_minutes=60)
    new = await _add_doc(a_id, "new.pdf", status="pending", age_minutes=1)
    await _add_doc(b_id, "secret-of-bob.pdf")

    rows = (await app_client.get("/api/documents", headers=a)).json()

    assert [r["id"] for r in rows] == [new, old]
    assert [r["filename"] for r in rows] == ["new.pdf", "old.pdf"]
    assert [r["status"] for r in rows] == ["pending", "embedded"]
    assert all("storage_path" not in r for r in rows)


# =====================================================================================
# Tài liệu: xóa
# =====================================================================================
@pytest.fixture
def backends(monkeypatch):
    calls = {"vectors": [], "minio": [], "cache": [], "fail_vectors": False, "fail_minio": False}

    def delete_document(doc_id, owner_id=None):
        if calls["fail_vectors"]:
            raise RuntimeError("qdrant down")
        calls["vectors"].append((doc_id, owner_id))

    class Minio:
        def remove_object(self, bucket, path):
            if calls["fail_minio"]:
                raise RuntimeError("minio down")
            calls["minio"].append((bucket, path))

    async def invalidate(owner_id):
        calls["cache"].append(owner_id)

    monkeypatch.setattr(docs_module.vector_store, "delete_document", delete_document)
    monkeypatch.setattr(docs_module, "minio_client", Minio())
    monkeypatch.setattr(docs_module.semantic_cache, "invalidate_owner", invalidate)
    return calls


async def test_delete_removes_vectors_file_row_and_keeps_history(app_client, backends):
    headers, uid = await _login(app_client)
    doc = await _add_doc(uid, storage_path="abc.pdf")
    async with coredb.async_session() as s:
        conv = Conversation(owner_id=uuid.UUID(uid), title="t", document_id=uuid.UUID(doc))
        attempt = QuizAttempt(user_id=uuid.UUID(uid), document_id=uuid.UUID(doc), is_correct=True, response_time_s=1.0)
        s.add_all([conv, attempt])
        await s.commit()
        conv_id, attempt_id = conv.id, attempt.id

    res = await app_client.delete(f"/api/documents/{doc}", headers=headers)

    assert res.status_code == 204
    assert backends["vectors"] == [(doc, uid)]                       # chỉ xóa vector của đúng tài liệu + chủ sở hữu
    assert backends["minio"] == [(settings.minio_bucket, "abc.pdf")]
    assert backends["cache"] == [uid]                                # câu trả lời cache có thể trích dẫn tài liệu này
    assert (await app_client.get("/api/documents", headers=headers)).json() == []
    async with coredb.async_session() as s:
        assert (await s.get(Conversation, conv_id)).document_id is None   # hội thoại còn, chỉ gỡ liên kết
        assert (await s.get(QuizAttempt, attempt_id)).document_id is None  # lịch sử quiz còn


async def test_cannot_delete_someone_elses_document(app_client, backends):
    a, a_id = await _login(app_client, "alice")
    b, _ = await _login(app_client, "bobby")
    doc = await _add_doc(a_id, storage_path="x.pdf")

    res = await app_client.delete(f"/api/documents/{doc}", headers=b)

    assert res.status_code == 404
    assert backends["vectors"] == [] and backends["minio"] == []
    assert len((await app_client.get("/api/documents", headers=a)).json()) == 1


async def test_delete_validation_and_auth(app_client, backends):
    headers, _ = await _login(app_client)
    assert (await app_client.delete("/api/documents/not-a-uuid", headers=headers)).status_code == 400
    assert (await app_client.delete(f"/api/documents/{uuid.uuid4()}", headers=headers)).status_code == 404
    assert (await app_client.delete(f"/api/documents/{uuid.uuid4()}")).status_code == 401


async def test_document_still_processing_cannot_be_deleted_but_stuck_one_can(app_client, backends):
    headers, uid = await _login(app_client)
    busy = await _add_doc(uid, "busy.pdf", status="pending", age_minutes=2)
    stuck = await _add_doc(uid, "stuck.pdf", status="pending", age_minutes=180)

    res = await app_client.delete(f"/api/documents/{busy}", headers=headers)
    assert res.status_code == 409 and backends["vectors"] == []

    assert (await app_client.delete(f"/api/documents/{stuck}", headers=headers)).status_code == 204
    assert backends["vectors"] == [(stuck, uid)]


async def test_vector_failure_aborts_and_keeps_the_document(app_client, backends):
    headers, uid = await _login(app_client)
    doc = await _add_doc(uid, storage_path="x.pdf")
    backends["fail_vectors"] = True

    res = await app_client.delete(f"/api/documents/{doc}", headers=headers)

    assert res.status_code == 502
    assert len((await app_client.get("/api/documents", headers=headers)).json()) == 1   # còn nguyên để thử lại
    assert backends["minio"] == [] and backends["cache"] == []


async def test_minio_failure_does_not_block_deletion(app_client, backends):
    headers, uid = await _login(app_client)
    doc = await _add_doc(uid, storage_path="x.pdf")
    backends["fail_minio"] = True

    assert (await app_client.delete(f"/api/documents/{doc}", headers=headers)).status_code == 204
    assert (await app_client.get("/api/documents", headers=headers)).json() == []


def test_vector_store_delete_document_filters_by_document_and_owner(monkeypatch):
    from app.rag.vector_store import VectorStoreManager

    models = sys.modules["qdrant_client.http.models"]
    rec = {}
    monkeypatch.setattr(models, "FieldCondition", lambda key, match: ("cond", key, match), raising=False)
    monkeypatch.setattr(models, "MatchValue", lambda value: value, raising=False)
    monkeypatch.setattr(models, "Filter", lambda must: ("filter", must), raising=False)
    monkeypatch.setattr(models, "FilterSelector", lambda filter: ("selector", filter), raising=False)

    class Client:
        def delete(self, collection_name, points_selector):
            rec.update(collection=collection_name, selector=points_selector)

    store = VectorStoreManager(collection_name="c1")
    store._client, store._collection_ready = Client(), True
    store.delete_document("doc-1", "owner-1")

    assert rec["collection"] == "c1"
    assert rec["selector"] == ("selector", ("filter", [("cond", "document_id", "doc-1"),
                                                       ("cond", "owner_id", "owner-1")]))


# =====================================================================================
# Hội thoại: CRUD
# =====================================================================================
async def _new_conv(client, headers, document_id=None):
    res = await client.post("/api/conversations", headers=headers, json={"document_id": document_id})
    assert res.status_code == 201, res.text
    return res.json()


async def test_empty_conversation_is_hidden_until_it_has_a_message(app_client):
    headers, uid = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    assert (await app_client.get("/api/conversations", headers=headers)).json() == []

    assert await conversation_store.append_message(conv["id"], uid, "user", "Stack là gì?", title_from="Stack là gì?")

    listed = (await app_client.get("/api/conversations", headers=headers)).json()
    assert [c["id"] for c in listed] == [conv["id"]]
    assert listed[0]["title"] == "Stack là gì?"          # tiêu đề tự đặt từ câu hỏi đầu tiên


async def test_conversation_detail_messages_in_order_with_citations(app_client):
    headers, uid = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    await conversation_store.append_message(conv["id"], uid, "user", "Q1", title_from="Q1")
    await conversation_store.append_message(conv["id"], uid, "assistant", "A1", intent="qa", cached=True,
                                            citations=[{"chunk_index": 2, "content": "nguồn"}])
    await conversation_store.append_message(conv["id"], uid, "user", "Q2")

    detail = (await app_client.get(f"/api/conversations/{conv['id']}", headers=headers)).json()

    assert [m["content"] for m in detail["messages"]] == ["Q1", "A1", "Q2"]
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant", "user"]
    assert detail["messages"][1]["citations"] == [{"chunk_index": 2, "content": "nguồn"}]
    assert detail["messages"][1]["cached"] is True and detail["messages"][1]["intent"] == "qa"
    assert detail["title"] == "Q1"                       # tin nhắn sau không đổi tiêu đề


async def test_long_first_question_gets_a_short_title(app_client):
    headers, uid = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    question = "Giải thích " + "rất dài " * 40
    await conversation_store.append_message(conv["id"], uid, "user", question, title_from=question)
    title = (await app_client.get(f"/api/conversations/{conv['id']}", headers=headers)).json()["title"]
    assert len(title) <= conversation_store.TITLE_MAX and title.endswith("…")


async def test_rename_and_delete(app_client):
    headers, uid = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    await conversation_store.append_message(conv["id"], uid, "user", "Q", title_from="Q")

    res = await app_client.patch(f"/api/conversations/{conv['id']}", headers=headers, json={"title": "  Ôn   chương 3 "})
    assert res.status_code == 200 and res.json()["title"] == "Ôn chương 3"
    assert (await app_client.patch(f"/api/conversations/{conv['id']}", headers=headers, json={"title": ""})
            ).status_code == 422
    assert (await app_client.patch(f"/api/conversations/{conv['id']}", headers=headers, json={"title": "   "})
            ).status_code == 422                          # chỉ toàn khoảng trắng cũng không hợp lệ

    assert (await app_client.delete(f"/api/conversations/{conv['id']}", headers=headers)).status_code == 204
    assert (await app_client.get(f"/api/conversations/{conv['id']}", headers=headers)).status_code == 404
    async with coredb.async_session() as s:               # tin nhắn cũng bị xóa
        assert (await s.execute(select(ChatMessage))).scalars().all() == []


async def test_other_users_cannot_touch_my_conversations(app_client):
    a, a_id = await _login(app_client, "alice")
    b, _ = await _login(app_client, "bobby")
    conv = await _new_conv(app_client, a)
    await conversation_store.append_message(conv["id"], a_id, "user", "private", title_from="private")

    assert (await app_client.get("/api/conversations", headers=b)).json() == []
    assert (await app_client.get(f"/api/conversations/{conv['id']}", headers=b)).status_code == 404
    assert (await app_client.patch(f"/api/conversations/{conv['id']}", headers=b, json={"title": "x"})
            ).status_code == 404
    assert (await app_client.delete(f"/api/conversations/{conv['id']}", headers=b)).status_code == 404
    # store cũng không ghi/đọc hộ người khác
    assert await conversation_store.append_message(conv["id"], "00000000-0000-0000-0000-000000000000",
                                                   "user", "intruder") is False
    assert await conversation_store.recent_history(conv["id"], str(uuid.uuid4())) == []
    assert (await app_client.get(f"/api/conversations/{conv['id']}", headers=a)).json()["messages"][0]["content"] == "private"


async def test_conversation_ids_and_documents_are_validated(app_client):
    headers, uid = await _login(app_client)
    assert (await app_client.get("/api/conversations/not-a-uuid", headers=headers)).status_code == 404
    assert (await app_client.get("/api/conversations")).status_code == 401
    other, o_id = await _login(app_client, "bobby")
    foreign_doc = await _add_doc(o_id)
    res = await app_client.post("/api/conversations", headers=headers, json={"document_id": foreign_doc})
    assert res.status_code == 404                          # không gắn hội thoại vào tài liệu của người khác
    mine = await _add_doc(uid)
    assert (await _new_conv(app_client, headers, mine))["document_id"] == mine


# =====================================================================================
# Hội thoại: lưu qua /query/stream
# =====================================================================================
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


async def _stream(client, headers, question, conversation_id=None, **extra):
    body = {"question": question, **extra}
    if conversation_id:
        body["conversation_id"] = conversation_id
    return await client.post("/api/chat/query/stream", headers=headers, json=body)


async def _messages(client, headers, conv_id):
    return (await client.get(f"/api/conversations/{conv_id}", headers=headers)).json()


async def test_stream_saves_question_and_answer_with_citations(app_client, monkeypatch):
    headers, _ = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    events = _events(await _stream(app_client, headers, "What is a class?", conv["id"]))

    assert events[-1]["type"] == "result"
    detail = await _messages(app_client, headers, conv["id"])
    assert detail["title"] == "What is a class?"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "What is a class?"), ("assistant", "Answer [1]")]
    assert detail["messages"][1]["citations"][0]["page_number"] == 3
    assert detail["messages"][1]["cached"] is False


async def test_followup_uses_server_history_not_client_history(app_client, monkeypatch):
    headers, uid = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    seen = {}
    monkeypatch.setattr(chat_module, "_rewrite_query",
                        lambda q, history: seen.update(history=[(m.role, m.content) for m in history]) or q)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream(tokens=("OOP is X.",)))

    await _stream(app_client, headers, "Tell me about OOP", conv["id"])
    # client cố tình gửi history giả: phải bị bỏ qua khi đã có conversation_id
    fake = [{"role": "user", "content": "FAKE"}]
    await _stream(app_client, headers, "and polymorphism?", conv["id"], history=fake)

    assert seen["history"] == [("user", "Tell me about OOP"), ("assistant", "OOP is X.")]
    detail = await _messages(app_client, headers, conv["id"])
    assert [m["content"] for m in detail["messages"]] == [
        "Tell me about OOP", "OOP is X.", "and polymorphism?", "OOP is X."]


async def test_failed_answer_keeps_the_question_only(app_client, monkeypatch):
    headers, _ = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream(error="Ollama down"))

    events = _events(await _stream(app_client, headers, "What is a class?", conv["id"]))

    assert events[-1]["type"] == "error"
    detail = await _messages(app_client, headers, conv["id"])
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [("user", "What is a class?")]


async def test_no_result_and_fast_paths_are_saved_too(app_client, monkeypatch):
    headers, _ = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [])

    await _stream(app_client, headers, "What is a class?", conv["id"])

    detail = await _messages(app_client, headers, conv["id"])
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert "No relevant content" in detail["messages"][1]["content"]


async def test_stream_rejects_foreign_or_unknown_conversation(app_client, monkeypatch):
    a, a_id = await _login(app_client, "alice")
    b, _ = await _login(app_client, "bobby")
    conv = await _new_conv(app_client, a)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    assert (await _stream(app_client, b, "hi there", conv["id"])).status_code == 404
    assert (await _stream(app_client, a, "hi there", str(uuid.uuid4()))).status_code == 404
    assert (await _stream(app_client, a, "hi there", "garbage")).status_code == 404
    assert (await _messages(app_client, a, conv["id"]))["messages"] == []         # không ghi gì vào hội thoại của A

    # Không có conversation_id: hành vi cũ, không lưu gì
    assert _events(await _stream(app_client, a, "What is a class?"))[-1]["type"] == "result"
    assert (await app_client.get("/api/conversations", headers=a)).json() == []


async def test_saving_history_failure_never_breaks_chat(app_client, monkeypatch):
    headers, _ = await _login(app_client)
    conv = await _new_conv(app_client, headers)
    monkeypatch.setattr(chat_module.vector_store, "search_similar_chunks", lambda **k: [HIT])
    monkeypatch.setattr(chat_module.llm_engine, "stream_rag_response", _fake_stream())

    real = coredb.async_session
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] > 2:             # get_owned + recent_history chạy được, các lần GHI thì hỏng
            raise RuntimeError("db write failed")
        return real()

    monkeypatch.setattr(coredb, "async_session", flaky)
    events = _events(await _stream(app_client, headers, "What is a class?", conv["id"]))

    assert events[-1]["type"] == "result" and events[-1]["data"]["answer"] == "Answer [1]"


# =====================================================================================
# JWT_SECRET
# =====================================================================================
@pytest.mark.parametrize("secret", [None, "", "   ", "change-me", "CHANGE-ME-please-make-it-long-enough-123456",
                                    "short", "x" * 31, "my-super-password-that-is-long-enough-123"])
def test_weak_secrets_are_rejected(secret):
    assert insecure_secret_reason(secret) is not None


@pytest.mark.parametrize("secret", ["a" * 32, "037" + "f" * 61, "Zx9-" * 12])
def test_strong_secrets_are_accepted(secret):
    assert insecure_secret_reason(secret) is None


def test_enforce_raises_in_production_and_only_warns_in_development(caplog):
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        enforce_jwt_secret("change-me", "production")
    with pytest.raises(RuntimeError):
        enforce_jwt_secret("change-me", "")                      # rỗng => coi như production
    with caplog.at_level("ERROR"):
        enforce_jwt_secret("change-me", "development")           # không ném lỗi
    assert "JWT_SECRET" in caplog.text
    enforce_jwt_secret("a" * 40, "production")                   # secret tốt: im lặng


async def test_app_refuses_to_start_with_default_secret(monkeypatch):
    import app.main as m

    monkeypatch.setattr(settings, "jwt_secret", "change-me")
    monkeypatch.setattr(settings, "app_env", "production")
    with pytest.raises(RuntimeError, match="JWT_SECRET"):
        async with m.lifespan(m.app):
            pytest.fail("the application must not start")


