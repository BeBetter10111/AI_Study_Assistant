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
from app.api.endpoints import auth, chat, documents, quiz
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.llm_engine import llm_engine
from app.rag import vector_store as vs

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = ("localhost", "127.0.0.1")


def _expand_local_origins(origins: list[str]) -> list[str]:
    """Với mỗi origin dev (localhost / 127.0.0.1), thêm origin "anh em" của nó.

    Trình duyệt coi ``http://localhost:5173`` và ``http://127.0.0.1:5173`` là
    HAI origin khác nhau. Nếu .env chỉ khai báo localhost mà bạn mở web bằng
    127.0.0.1 thì mọi request bị CORS chặn và axios chỉ báo "Network Error".
    Origin production (domain thật) không bị ảnh hưởng.
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
        except Exception:
            if response_started:
                raise  # header đã gửi đi, không thể đổi thành 500 nữa
            logger.exception(
                "Unhandled error on %s %s", scope.get("method"), scope.get("path")
            )
            response = JSONResponse(
                {"detail": "Internal server error. Check the backend logs."},
                status_code=500,
            )
            await response(scope, receive, send)


@asynccontextmanager
async def lifespan(_app):
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
    except Exception:  # noqa: BLE001 - không để lỗi phụ làm hỏng response 429
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
