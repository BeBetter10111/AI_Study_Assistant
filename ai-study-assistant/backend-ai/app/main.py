import asyncio
import logging
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.api.endpoints import auth, chat, conversations, documents, quiz
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core import conversation_store
from app.core import database as coredb
from app.core.database import engine
from app.core.llm_engine import llm_engine
from app.core.request_limits import UploadSizeLimitMiddleware
from app.core.schema import ensure_schema
from app.core.security_check import enforce_jwt_secret
from app.rag import vector_store as vs

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = ("localhost", "127.0.0.1")


def _expand_local_origins(origins: list[str]) -> list[str]:
    """For each development origin (localhost / 127.0.0.1), add its counterpart origin.

    Browsers treat ``http://localhost:5173`` and ``http://127.0.0.1:5173`` as
    TWO different origins. If .env lists only localhost but the site is opened
    using 127.0.0.1, all requests are blocked by CORS and axios only reports "Network Error".
    Production origins (real domains) are not affected.
    """
    expanded = list(origins)
    for origin in origins:
        try:
            parts = urlsplit(origin)
            host, port = parts.hostname, parts.port
        except ValueError:
            continue
        if host not in _LOCAL_HOSTS:
            continue
        for other in _LOCAL_HOSTS:
            if other == host:
                continue
            alt = f"{parts.scheme}://{other}" + (f":{port}" if port else "")
            if alt not in expanded:
                expanded.append(alt)
    return expanded


# CORS origin list — parsed from the settings value at startup.
ALLOWED_ORIGINS = _expand_local_origins(
    [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
)


class UnhandledErrorMiddleware:
    """Biến exception chưa được xử lý thành JSON 500 *nằm bên trong* CORS.

    Starlette đặt ServerErrorMiddleware ở lớp NGOÀI CÙNG, bên ngoài
    CORSMiddleware, nên một lỗi 500 bất ngờ (DB/Redis lỗi, bug...) được trả về
    KHÔNG có header Access-Control-Allow-Origin. Trình duyệt chặn response đó
    và axios chỉ thấy "Network Error" — che mất lỗi thật. Middleware này bắt lỗi
    ở lớp trong để CORS vẫn gắn header, giúp frontend đọc được 500 và backend
    log đầy đủ traceback.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def tracking_send(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, tracking_send)
        except Exception as exc:
            if response_started:
                raise  # Headers have already been sent; the response can no longer be changed to 500.
            logger.exception(
                "Unhandled error on %s %s", scope.get("method"), scope.get("path")
            )
            # Include the error type (without exposing its content/details) for quick diagnosis; for example,
            # "(ProgrammingError)" may indicate a database schema mismatch. The full traceback is in the backend log.
            response = JSONResponse(
                {"detail": f"Internal server error ({type(exc).__name__}). Check the backend logs."},
                status_code=500,
            )
            await response(scope, receive, send)


@asynccontextmanager
async def lifespan(_app):
    # Fail fast: with a default/weak JWT_SECRET, anyone can forge tokens. This must run BEFORE anything else.
    enforce_jwt_secret(settings.jwt_secret, settings.app_env)

    # A pre-existing database may lack columns/tables required by the current code (init_db.sql only runs when
    # the volume is empty) -> user queries raise UndefinedColumnError and registration/login return HTTP 500.
    if settings.auto_migrate_schema:
        try:
            await ensure_schema(engine)
        except Exception:
            logger.exception("Automatic schema upgrade failed; the API may return 500 errors")

    # Clean up old empty conversations (created by opening "New chat" without sending a message).
    # Cleanup must not prevent startup.
    try:
        async with coredb.async_session() as session:
            removed = await conversation_store.purge_stale_empty(session, settings.empty_conversation_ttl_s)
            await session.commit()
        if removed:
            logger.info("Removed %d stale empty conversation(s).", removed)
    except Exception:  # noqa: BLE001
        logger.warning("Could not purge stale empty conversations", exc_info=True)

    def _warm():
        try:
            vs._get_embedding_model().encode("warmup")
            rr = vs._get_reranker()
            if rr:
                rr.predict([("warmup", "warmup")])
            llm_engine.generate_rag_response("hi", "Reply with one word.", max_tokens=1)
        except Exception:
            logger.warning("Warmup failed, continuing without it", exc_info=True)
    warmup_task = asyncio.create_task(run_in_threadpool(_warm))
    try:
        yield
    finally:
        if not warmup_task.done():
            warmup_task.cancel()
            await asyncio.gather(warmup_task, return_exceptions=True)

app = FastAPI(title="AI Study Assistant", version="0.1.0", lifespan=lifespan)

# Rate limiter — must be attached before routers are mounted.
# Returns HTTP 429 with Retry-After when a limit is exceeded.
app.state.limiter = limiter


def _retry_after_seconds(exc: RateLimitExceeded) -> int | None:
    """Độ dài cửa sổ giới hạn (giây). slowapi KHÔNG có thuộc tính ``retry_after``
    (code cũ dùng getattr nên header Retry-After không bao giờ được gửi)."""
    try:
        return int(exc.limit.limit.get_expiry())
    except Exception:  # noqa: BLE001 - Do not let a secondary error break the 429 response.
        return None


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    retry_after = _retry_after_seconds(exc)
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down and try again later."},
        headers=headers,
    )

# Order matters: add_middleware places each new layer on the OUTSIDE, so
# UnhandledErrorMiddleware must be added BEFORE CORS to remain INSIDE CORS.
# Reject oversized upload bodies BEFORE Starlette writes them to disk. Add this BEFORE
# UnhandledErrorMiddleware (so it stays INSIDE it) to avoid treating its internal exception as HTTP 500;
# CORS remains outside, so the 413 response still includes CORS headers.
app.add_middleware(
    UploadSizeLimitMiddleware,
    path="/api/documents/upload",
    max_body_bytes=documents.MAX_UPLOAD_SIZE_BYTES + 1024 * 1024,  # +1 MB for multipart overhead.
)
app.add_middleware(UnhandledErrorMiddleware)

# CORS — explicit allow-list instead of wildcard "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,       # allow cookies / Authorization headers
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(conversations.router, prefix="/api/conversations", tags=["conversations"])
app.include_router(quiz.router, prefix="/api/quiz", tags=["quiz"])

@app.get("/health")
async def health(response: Response, db: AsyncSession = Depends(get_db)):
    checks = {"database": "ok"}
    is_healthy = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:  
        checks["database"] = f"unreachable: {exc}"
        is_healthy = False

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {"status": "ok" if is_healthy else "degraded", "checks": checks}
