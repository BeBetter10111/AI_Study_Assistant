"""Luồng quên mật khẩu: forgot -> verify code -> reset -> login lại."""
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import update

import app.api.dependencies as deps
from app.api.endpoints import auth as auth_module
from app.models.models import PasswordResetToken

OLD_PW = "oldpassword1"
NEW_PW = "brandnewpass2"
EMAIL = "reset@example.com"


@pytest.fixture
def sent(monkeypatch):
    box = []
    monkeypatch.setattr(auth_module, "send_password_reset_email", lambda to, code: box.append((to, code)))
    return box


async def _register(client, email=EMAIL, username="resetuser"):
    res = await client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": OLD_PW},
    )
    assert res.status_code == 201
    return res.json()["access_token"]


async def _protected_call(client, token):
    """Gọi một endpoint cần đăng nhập (quiz submit); 401 nếu token bị từ chối."""
    return await client.post(
        f"/api/quiz/{uuid.uuid4()}/submit",
        headers={"Authorization": f"Bearer {token}"},
        json={"is_correct": True, "response_time_s": 5, "hints_used": 0},
    )


async def _forgot(client, email=EMAIL):
    return await client.post("/api/auth/forgot-password", json={"email": email})


async def _verify(client, code, email=EMAIL):
    return await client.post("/api/auth/verify-reset-code", json={"email": email, "code": code})


async def _reset(client, token, pw=NEW_PW, confirm=None):
    return await client.post(
        "/api/auth/reset-password",
        json={"reset_token": token, "new_password": pw, "confirm_password": confirm or pw},
    )


async def test_full_flow_updates_password_and_revokes_old_sessions(app_client, sent):
    old_access = await _register(app_client)
    assert (await _forgot(app_client)).status_code == 200
    assert len(sent) == 1 and sent[0][0] == EMAIL
    code = sent[0][1]

    verified = await _verify(app_client, code)
    assert verified.status_code == 200
    reset_token = verified.json()["reset_token"]

    done = await _reset(app_client, reset_token)
    assert done.status_code == 200

    assert (await app_client.post("/api/auth/login", json={"identifier": EMAIL, "password": OLD_PW})).status_code == 401
    login = await app_client.post("/api/auth/login", json={"identifier": EMAIL, "password": NEW_PW})
    assert login.status_code == 200

    # Token cấp trước khi đổi mật khẩu bị từ chối, token mới dùng được.
    assert (await _protected_call(app_client, old_access)).status_code == 401
    assert (await _protected_call(app_client, login.json()["access_token"])).status_code == 200


async def test_unknown_email_gets_same_response_and_no_email(app_client, sent):
    res = await _forgot(app_client, "nobody@example.com")
    assert res.status_code == 200
    assert res.json()["message"] == auth_module.GENERIC_FORGOT_MESSAGE
    assert sent == []


async def test_wrong_code_is_rejected_and_burned_after_max_attempts(app_client, sent):
    await _register(app_client)
    await _forgot(app_client)
    good = sent[0][1]
    for _ in range(5):
        assert (await _verify(app_client, "00000000-0000-0000-0000-000000000000")).status_code == 400
    assert (await _verify(app_client, good)).status_code == 400  # mã đúng cũng bị khoá


async def test_code_is_single_use(app_client, sent):
    await _register(app_client)
    await _forgot(app_client)
    code = sent[0][1]
    token = (await _verify(app_client, code)).json()["reset_token"]
    assert (await _reset(app_client, token)).status_code == 200
    assert (await _reset(app_client, token, pw="anotherpass3")).status_code == 400
    assert (await _verify(app_client, code)).status_code == 400


async def test_new_code_invalidates_previous_one(app_client, sent, monkeypatch):
    await _register(app_client)
    monkeypatch.setattr(auth_module.settings, "password_reset_resend_cooldown_s", 0)
    await _forgot(app_client)
    await _forgot(app_client)
    first, second = sent[0][1], sent[1][1]
    assert (await _verify(app_client, first)).status_code == 400
    assert (await _verify(app_client, second)).status_code == 200


async def test_resend_cooldown_does_not_send_second_email(app_client, sent):
    await _register(app_client)
    await _forgot(app_client)
    await _forgot(app_client)
    assert len(sent) == 1


async def test_expired_code_rejected(app_client, sent):
    await _register(app_client)
    await _forgot(app_client)
    code = sent[0][1]
    async with deps.async_session() as db:
        await db.execute(
            update(PasswordResetToken).values(expires_at=auth_module._now() - timedelta(minutes=1))
        )
        await db.commit()
    assert (await _verify(app_client, code)).status_code == 400


async def test_reset_requires_verified_session_and_valid_password(app_client, sent):
    access = await _register(app_client)
    await _forgot(app_client)
    code = sent[0][1]

    # Access token thường không dùng được làm reset_token.
    assert (await _reset(app_client, access)).status_code == 400

    token = (await _verify(app_client, code)).json()["reset_token"]
    assert (await _reset(app_client, token, pw=NEW_PW, confirm="different123")).status_code == 400
    assert (await _reset(app_client, token, pw="short")).status_code == 422
    assert (await _reset(app_client, token, pw=OLD_PW)).status_code == 400  # trùng mật khẩu cũ
    assert (await _reset(app_client, token)).status_code == 200  # token vẫn dùng được sau các lỗi trên


async def test_reset_token_cannot_be_used_as_access_token(app_client, sent):
    await _register(app_client)
    await _forgot(app_client)
    token = (await _verify(app_client, sent[0][1])).json()["reset_token"]
    assert (await _protected_call(app_client, token)).status_code == 401
