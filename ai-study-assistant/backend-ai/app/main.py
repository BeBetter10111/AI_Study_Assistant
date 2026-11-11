from fastapi import Depends, FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.concurrency import run_in_threadpool
from app.rag import vector_store as vs
from app.core.llm_engine import llm_engine

from app.api.dependencies import get_db
from app.api.endpoints import auth, chat, documents, quiz
from app.api.security.rate_limiter import limiter
from app.config import settings

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# CORS origin list — parsed from the settings value at startup.
ALLOWED_ORIGINS = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]


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


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    retry_after = getattr(exc, "retry_after", None)
    headers = {"Retry-After": str(retry_after)} if retry_after else {}
    return JSONResponse(
        status_code=429,
        content={"detail": "Too many requests. Please slow down and try again later."},
        headers=headers,
    )

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
