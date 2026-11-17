"""Stateless, user-bound quiz ids.

``/quiz/generate`` used to return a random UUID that ``/quiz/{id}/submit`` accepted blindly, so a client could
submit unlimited fabricated results under any id and drive its own RL difficulty / flood ``quiz_attempts``.

A quiz id is still a UUID (so the frontend and DB are unchanged) but its 16 bytes are::

    6 random bytes | 1 byte question count | 9 bytes HMAC(secret, "quiz-id" | user_id | random | count)

``verify_quiz_id`` therefore proves that *this server issued this id to this user* and tells how many answers
may be submitted for it (one per question) — no table or migration needed.  It does NOT prove an individual
answer is correct (the server does not keep the answer key); that stays client-reported.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from typing import Optional

from app.config import settings

_NONCE_LEN = 6
_MAC_LEN = 9
MAX_QUESTIONS = 255


def _mac(user_id: str, nonce: bytes, count: int) -> bytes:
    key = hmac.new(settings.jwt_secret.encode("utf-8"), b"asa-quiz-id-v1", hashlib.sha256).digest()
    msg = b"|".join([str(user_id).encode("utf-8"), nonce, bytes([count])])
    return hmac.new(key, msg, hashlib.sha256).digest()[:_MAC_LEN]


def issue_quiz_id(user_id: str, num_questions: int) -> str:
    count = max(1, min(int(num_questions), MAX_QUESTIONS))
    nonce = os.urandom(_NONCE_LEN)
    return str(uuid.UUID(bytes=nonce + bytes([count]) + _mac(user_id, nonce, count)))


def verify_quiz_id(user_id: str, quiz_id: uuid.UUID) -> Optional[int]:
    """Number of answers allowed for ``quiz_id`` if it was issued to ``user_id``, else ``None``."""
    raw = quiz_id.bytes
    nonce, count, mac = raw[:_NONCE_LEN], raw[_NONCE_LEN], raw[_NONCE_LEN + 1:]
    if count == 0 or not hmac.compare_digest(mac, _mac(user_id, nonce, count)):
        return None
    return count
