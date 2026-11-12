"""Hồ sơ cá nhân (GET/PATCH /me) và đổi mật khẩu khi đang đăng nhập."""
import pytest

from app.api.security.rate_limiter import limiter

PW = "firstpassword1"


@pytest.fixture(autouse=True)
def _no_rate_limit(monkeypatch):
    # Bộ đếm rate limit dùng chung giữa các test (Redis không có trong CI -> đếm trong bộ nhớ).
    monkeypatch.setattr(limiter, "enabled", False)


async def _register(client, username="alice", email="alice@example.com", phone=None, full_name="Alice A"):
    body = {"username": username, "email": email, "password": PW, "full_name": full_name}
    if phone:
        body["phone"] = phone
    res = await client.post("/api/auth/register", json=body)
    assert res.status_code == 201, res.text
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


async def test_me_requires_login(app_client):
    assert (await app_client.get("/api/auth/me")).status_code == 401


async def test_me_returns_profile_without_secrets(app_client):
    headers = await _register(app_client)
    res = await app_client.get("/api/auth/me", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["username"] == "alice"
    assert data["email"] == "alice@example.com"
    assert data["full_name"] == "Alice A"
    assert data["role"] == "Student"
    assert "hashed_password" not in data


async def test_patch_updates_only_sent_fields_and_can_clear(app_client):
    headers = await _register(app_client, phone="0900000001")

    res = await app_client.patch("/api/auth/me", headers=headers, json={"full_name": "  Alice Nguyen  "})
    assert res.status_code == 200
    assert res.json()["full_name"] == "Alice Nguyen"
    assert res.json()["phone"] == "0900000001"  # không gửi -> giữ nguyên

    res = await app_client.patch("/api/auth/me", headers=headers, json={"phone": ""})
    assert res.status_code == 200
    assert res.json()["phone"] is None

    res = await app_client.patch("/api/auth/me", headers=headers, json={"username": "alice2"})
    assert res.json()["username"] == "alice2"
    assert (await app_client.get("/api/auth/me", headers=headers)).json()["username"] == "alice2"


async def test_patch_cannot_change_email_or_role(app_client):
    headers = await _register(app_client)
    res = await app_client.patch(
        "/api/auth/me", headers=headers, json={"email": "evil@example.com", "role": "Admin"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["email"] == "alice@example.com"
    assert data["role"] == "Student"


async def test_patch_rejects_taken_username_and_phone(app_client):
    await _register(app_client, username="bob", email="bob@example.com", phone="0911111111")
    headers = await _register(app_client)

    res = await app_client.patch("/api/auth/me", headers=headers, json={"username": "bob"})
    assert res.status_code == 409
    res = await app_client.patch("/api/auth/me", headers=headers, json={"phone": "0911111111"})
    assert res.status_code == 409

    # Đặt lại đúng username hiện tại không phải xung đột với chính mình.
    res = await app_client.patch("/api/auth/me", headers=headers, json={"username": "alice"})
    assert res.status_code == 200


async def test_patch_rejects_short_username(app_client):
    headers = await _register(app_client)
    res = await app_client.patch("/api/auth/me", headers=headers, json={"username": " a "})
    assert res.status_code == 422


async def test_change_password_flow(app_client):
    headers = await _register(app_client)

    wrong = await app_client.post(
        "/api/auth/change-password",
        headers=headers,
        json={"current_password": "nope-nope", "new_password": "secondpassword2", "confirm_password": "secondpassword2"},
    )
    assert wrong.status_code == 400

    mismatch = await app_client.post(
        "/api/auth/change-password",
        headers=headers,
        json={"current_password": PW, "new_password": "secondpassword2", "confirm_password": "different-one"},
    )
    assert mismatch.status_code == 400

    same = await app_client.post(
        "/api/auth/change-password",
        headers=headers,
        json={"current_password": PW, "new_password": PW, "confirm_password": PW},
    )
    assert same.status_code == 400

    ok = await app_client.post(
        "/api/auth/change-password",
        headers=headers,
        json={"current_password": PW, "new_password": "secondpassword2", "confirm_password": "secondpassword2"},
    )
    assert ok.status_code == 200
    new_headers = {"Authorization": f"Bearer {ok.json()['access_token']}"}

    # Token cũ bị thu hồi, token mới dùng được.
    assert (await app_client.get("/api/auth/me", headers=headers)).status_code == 401
    assert (await app_client.get("/api/auth/me", headers=new_headers)).status_code == 200

    assert (
        await app_client.post("/api/auth/login", json={"identifier": "alice", "password": PW})
    ).status_code == 401
    assert (
        await app_client.post("/api/auth/login", json={"identifier": "alice", "password": "secondpassword2"})
    ).status_code == 200
