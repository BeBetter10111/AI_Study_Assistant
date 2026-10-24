"""Test tích hợp qua HTTP thật (httpx.AsyncClient + ASGITransport), DB SQLite
trong bộ nhớ. Đây chính là các trường hợp biên đã từng gây lỗi thật trong
quá trình phát triển — giữ lại làm regression test để không tái diễn.
"""
import uuid


async def test_health_check(app_client):
    response = await app_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


async def test_register_then_login(app_client):
    email = "frog@example.com"
    password = "hunter22"

    register_response = await app_client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )
    assert register_response.status_code == 201
    assert "access_token" in register_response.json()

    dup_response = await app_client.post(
        "/api/auth/register", json={"email": email, "password": password}
    )
    assert dup_response.status_code == 409

    bad_login = await app_client.post(
        "/api/auth/login", json={"email": email, "password": "wrong"}
    )
    assert bad_login.status_code == 401

    good_login = await app_client.post(
        "/api/auth/login", json={"email": email, "password": password}
    )
    assert good_login.status_code == 200
    assert "access_token" in good_login.json()


async def test_upload_requires_auth(app_client):
    response = await app_client.post(
        "/api/documents/upload",
        files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert response.status_code == 401


async def test_upload_rejects_unsupported_extension(app_client):
    register_response = await app_client.post(
        "/api/auth/register", json={"email": "ext@example.com", "password": "hunter22"}
    )
    token = register_response.json()["access_token"]

    response = await app_client.post(
        "/api/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


async def test_document_status_rejects_invalid_uuid(app_client):
    response = await app_client.get("/api/documents/not-a-uuid/status")
    assert response.status_code == 400


async def test_document_status_404_for_unknown_document(app_client):
    response = await app_client.get(f"/api/documents/{uuid.uuid4()}/status")
    assert response.status_code == 404


async def test_chat_query_rejects_empty_question(app_client):
    response = await app_client.post("/api/chat/query", json={"question": "   "})
    assert response.status_code == 400


async def test_chat_query_answers_gracefully_with_no_documents(app_client):
    response = await app_client.post(
        "/api/chat/query", json={"question": "What is in the syllabus?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []


async def test_quiz_generate_404_without_matching_context(app_client):
    response = await app_client.post(
        "/api/quiz/generate", json={"topic": "OOP", "num_questions": 3}
    )
    assert response.status_code == 404


async def test_quiz_submit_requires_auth(app_client):
    response = await app_client.post(
        f"/api/quiz/{uuid.uuid4()}/submit",
        json={"is_correct": True, "response_time_s": 5, "hints_used": 0},
    )
    assert response.status_code == 401


async def test_quiz_submit_with_auth_updates_difficulty(app_client):
    register_response = await app_client.post(
        "/api/auth/register", json={"email": "quiz@example.com", "password": "hunter22"}
    )
    token = register_response.json()["access_token"]

    response = await app_client.post(
        f"/api/quiz/{uuid.uuid4()}/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_correct": True, "response_time_s": 5, "hints_used": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert "attempt_id" in body
    assert body["next_difficulty"] in {"easy", "medium", "hard"}


async def test_quiz_submit_rejects_negative_response_time(app_client):
    register_response = await app_client.post(
        "/api/auth/register", json={"email": "neg@example.com", "password": "hunter22"}
    )
    token = register_response.json()["access_token"]

    response = await app_client.post(
        f"/api/quiz/{uuid.uuid4()}/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_correct": True, "response_time_s": -5, "hints_used": 0},
    )
    assert response.status_code == 422
