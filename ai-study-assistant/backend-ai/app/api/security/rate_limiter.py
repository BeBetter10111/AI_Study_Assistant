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
    # MUST remain False. With headers_enabled=True, slowapi requires EVERY endpoint
    # using @limiter.limit to declare an additional `response: Response` parameter
    # (when the endpoint returns a model/dict such as TokenResponse). Without it, slowapi raises
    #   Exception: parameter `response` must be an instance of starlette...Response
    # AFTER the endpoint has completed (the user has already been committed to the DB) -> HTTP 500 ->
    # the browser only reports "Network Error". When a request is rate-limited, the Retry-After header
    # is still returned by the RateLimitExceeded handler in main.py.
    headers_enabled=False,
    # If Redis is temporarily unavailable, do NOT let registration/login fail with HTTP 500
    # (the frontend would only show "Network Error"). Fall back to an in-process counter
    # so the limit remains active, and automatically return to Redis when it recovers.
    # The failure is still recorded as a warning in the log.
    in_memory_fallback_enabled=True,
)
