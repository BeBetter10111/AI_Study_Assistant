"""Test tích hợp qua HTTP thật (httpx.AsyncClient + ASGITransport), DB SQLite
trong bộ nhớ. Đây chính là các trường hợp biên đã từng gây lỗi thật trong
quá trình phát triển — giữ lại làm regression test để không tái diễn.
"""
import uuid

from jose import jwt as _jwt

from app.core.quiz_tokens import issue_quiz_id


async def _auth(client, name="flowuser"):
    """Register a user and return (headers, user_id)."""
    res = await client.post(
        "/api/auth/register",
        json={"username": name, "email": f"{name}@example.com", "password": "hunter22"},
    )
    assert res.status_code == 201, res.text
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, _jwt.get_unverified_claims(token)["sub"]


async def test_health_check(app_client):
    response = await app_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"] == "ok"


async def test_register_then_login(app_client):
    email = "frog@example.com"
    password = "hunter22"
    username = "froguser"

    register_response = await app_client.post(
        "/api/auth/register", json={"username": username, "email": email, "password": password}
    )
    assert register_response.status_code == 201
    assert "access_token" in register_response.json()

    dup_response = await app_client.post(
        "/api/auth/register", json={"username": username, "email": email, "password": password}
    )
    assert dup_response.status_code == 409

    bad_login = await app_client.post(
        "/api/auth/login", json={"identifier": email, "password": "wrong"}
    )
    assert bad_login.status_code == 401

    good_login = await app_client.post(
        "/api/auth/login", json={"identifier": email, "password": password}
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
        "/api/auth/register",
        json={"username": "extuser", "email": "ext@example.com", "password": "hunter22"},
    )
    token = register_response.json()["access_token"]

    response = await app_client.post(
        "/api/documents/upload",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 415


async def test_document_status_rejects_invalid_uuid(app_client):
    headers, _ = await _auth(app_client)
    response = await app_client.get("/api/documents/not-a-uuid/status", headers=headers)
    assert response.status_code == 400


async def test_document_status_404_for_unknown_document(app_client):
    headers, _ = await _auth(app_client)
    response = await app_client.get(f"/api/documents/{uuid.uuid4()}/status", headers=headers)
    assert response.status_code == 404


async def test_chat_query_rejects_empty_question(app_client):
    headers, _ = await _auth(app_client)
    response = await app_client.post("/api/chat/query", json={"question": "   "}, headers=headers)
    assert response.status_code == 400


async def test_chat_query_answers_gracefully_with_no_documents(app_client):
    headers, _ = await _auth(app_client)
    response = await app_client.post(
        "/api/chat/query", json={"question": "What is in the syllabus?"}, headers=headers
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []


async def test_quiz_generate_404_without_matching_context(app_client):
    headers, _ = await _auth(app_client)
    response = await app_client.post(
        "/api/quiz/generate", json={"topic": "OOP", "num_questions": 3}, headers=headers
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
        "/api/auth/register",
        json={"username": "quizuser", "email": "quiz@example.com", "password": "hunter22"},
    )
    token = register_response.json()["access_token"]
    user_id = _jwt.get_unverified_claims(token)["sub"]

    response = await app_client.post(
        f"/api/quiz/{issue_quiz_id(user_id, 3)}/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_correct": True, "response_time_s": 5, "hints_used": 0},
    )
    assert response.status_code == 200
    body = response.json()
    assert "attempt_id" in body
    assert body["next_difficulty"] in {"easy", "medium", "hard"}


async def test_quiz_submit_rejects_negative_response_time(app_client):
    register_response = await app_client.post(
        "/api/auth/register",
        json={"username": "neguser", "email": "neg@example.com", "password": "hunter22"},
    )
    token = register_response.json()["access_token"]
    user_id = _jwt.get_unverified_claims(token)["sub"]

    response = await app_client.post(
        f"/api/quiz/{issue_quiz_id(user_id, 3)}/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_correct": True, "response_time_s": -5, "hints_used": 0},
    )
    assert response.status_code == 422


async def test_register_conflict_message_does_not_reveal_which_field_exists(app_client):
    base = {"username": "dupuser", "email": "dup@example.com", "phone": "0901234567", "password": "hunter22"}
    assert (await app_client.post("/api/auth/register", json=base)).status_code == 201
    messages = set()
    for override in ({}, {"username": "other1"}, {"username": "other2", "email": "o2@example.com"}):
        res = await app_client.post("/api/auth/register", json={**base, **override})
        assert res.status_code == 409
        messages.add(res.json()["detail"])
    assert len(messages) == 1  # same answer whether username, email or phone collided
