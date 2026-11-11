"""Rate-limiting helpers — backed by Redis via slowapi.

Usage
-----
Import `limiter` and mount the exception handler in main.py, then decorate
individual endpoints:

    @router.post("/login")
    @limiter.limit("10/minute")
    async def login(request: Request, ...):
        ...

The Request object *must* appear in the endpoint signature (FastAPI injects it
automatically — just declare it as a parameter) so that slowapi can read the
client IP from it.

Limits defined here (conservative — tune in production):
- Auth endpoints (login, register, forgot-password, verify-code): 10 req/min per IP
- Chat / query endpoints: 30 req/min per authenticated user (keyed on JWT sub)
- Document upload: 20 req/min per authenticated user
- Quiz generate: 20 req/min per authenticated user
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings


def _get_user_or_ip(request) -> str:
    """Rate-limit key: JWT 'sub' claim when present, otherwise client IP.

    This avoids punishing innocent users who share the same NAT/proxy IP with
    an abuser while still protecting unauthenticated endpoints by IP.
    """
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        token = auth[7:]
        try:
            from jose import jwt as _jwt
            payload = _jwt.decode(
                token,
                settings.jwt_secret,
                algorithms=[settings.jwt_algorithm],
                options={"verify_exp": False},  # expiry checked by dependency
            )
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:
            pass
    return get_remote_address(request)


# A single Limiter instance is shared across the application.
# slowapi reads the storage_uri from the constructor — point it at the same
# Redis instance that Celery already uses so we don't need an extra container.
limiter = Limiter(
    key_func=_get_user_or_ip,
    storage_uri=settings.redis_url,
    default_limits=[],          # no blanket default; each endpoint sets its own
    # PHẢI để False. Với headers_enabled=True, slowapi bắt buộc MỌI endpoint có
    # @limiter.limit khai báo thêm tham số `response: Response` (khi endpoint trả
    # về model/dict như TokenResponse). Thiếu tham số đó, slowapi ném
    #   Exception: parameter `response` must be an instance of starlette...Response
    # SAU KHI endpoint đã chạy xong (user đã được commit vào DB) -> HTTP 500 ->
    # trình duyệt chỉ báo "Network Error". Khi bị giới hạn, header Retry-After
    # vẫn được trả về bởi handler RateLimitExceeded trong main.py.
    headers_enabled=False,
    # Nếu Redis tạm thời không kết nối được, KHÔNG để register/login nổ 500
    # (frontend sẽ chỉ thấy "Network Error"). Chuyển sang đếm trong bộ nhớ
    # của tiến trình — giới hạn vẫn còn hiệu lực — và tự quay lại Redis khi
    # Redis hoạt động trở lại. Lỗi vẫn được ghi log cảnh báo.
    in_memory_fallback_enabled=True,
)
