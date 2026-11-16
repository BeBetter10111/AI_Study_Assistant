"""Kiểm tra cấu hình bảo mật lúc khởi động.

``JWT_SECRET`` mặc định là ``change-me``. Nếu quên đặt biến môi trường, ai cũng tự ký được token hợp lệ
cho BẤT KỲ user nào (kể cả đọc tài liệu của người khác) — nên backend từ chối khởi động thay vì chạy
âm thầm với secret ai cũng biết.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

MIN_SECRET_LENGTH = 32
_PLACEHOLDERS = ("change-me", "changeme", "change_me", "secret", "password", "your-secret", "default")

HOWTO = 'Tạo secret mạnh:  python -c "import secrets; print(secrets.token_urlsafe(48))"  rồi đặt vào JWT_SECRET trong .env'


def insecure_secret_reason(secret: Optional[str]) -> Optional[str]:
    """Trả về lý do nếu secret không an toàn, ngược lại None."""
    value = (secret or "").strip()
    if not value:
        return "JWT_SECRET đang trống"
    lowered = value.lower()
    if any(p in lowered for p in _PLACEHOLDERS):
        return "JWT_SECRET vẫn là giá trị mẫu/mặc định"
    if len(value) < MIN_SECRET_LENGTH:
        return f"JWT_SECRET quá ngắn ({len(value)} ký tự, cần ít nhất {MIN_SECRET_LENGTH})"
    return None


def enforce_jwt_secret(secret: Optional[str], app_env: str) -> None:
    """production (mặc định) => ném RuntimeError; môi trường khác => chỉ cảnh báo lớn."""
    reason = insecure_secret_reason(secret)
    if reason is None:
        return
    message = f"Cấu hình không an toàn: {reason}. {HOWTO}"
    if (app_env or "production").strip().lower() in ("development", "dev", "local", "test"):
        logger.error("%s  (APP_ENV=%s nên vẫn khởi động — KHÔNG dùng cấu hình này khi triển khai thật)",
                     message, app_env)
        return
    raise RuntimeError(message + "  (Chỉ để chạy thử trên máy bạn: đặt APP_ENV=development.)")
