import logging
import mimetypes
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.concurrency import SlotTimeout, upload_slots
from app.core.database import minio_client
from app.models.models import Conversation, Document, QuizAttempt, User
from app.rag.semantic_cache import semantic_cache
from app.rag.vector_store import vector_store
from app.worker.tasks import task_process_document

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md"}
MAX_UPLOAD_SIZE_BYTES = 150 * 1024 * 1024  

# Magic bytes (file signatures) for allowed formats.
# Checked AFTER reading the file so we validate actual content, not just the
# client-supplied filename extension or Content-Type header.
_MAGIC_BYTES: dict[str, list[bytes]] = {
    ".pdf":  [b"%PDF"],
    ".docx": [b"PK\x03\x04"],   # ZIP-based Office Open XML
    ".pptx": [b"PK\x03\x04"],
    ".md":   [],                 # plain text — no magic bytes; extension check is enough
}


def _check_magic_bytes(extension: str, data: bytes) -> bool:
    """Return True if *data* starts with a known magic sequence for *extension*.

    For formats with no magic bytes (Markdown) this always returns True.
    """
    signatures = _MAGIC_BYTES.get(extension, [])
    if not signatures:
        return True  # no signature defined — accept
    return any(data[:len(sig)] == sig for sig in signatures)


async def _upload_size(file: UploadFile) -> int:
    """Dung lượng file mà KHÔNG đọc nội dung vào RAM (Starlette đã ghi sẵn ra file tạm)."""
    size = getattr(file, "size", None)
    if size is not None:
        return int(size)
    file.file.seek(0, 2)
    end = file.file.tell()
    file.file.seek(0)
    return int(end)


async def _enforce_quota(db: AsyncSession, owner_id, incoming_bytes: int) -> None:
    """Hạn mức mỗi user: số tài liệu và tổng dung lượng. Tài liệu cũ (size_bytes NULL) tính là 0 byte."""
    count, used = (await db.execute(
        select(func.count(Document.id), func.coalesce(func.sum(Document.size_bytes), 0))
        .where(Document.owner_id == owner_id)
    )).one()
    max_docs = settings.max_documents_per_user
    if max_docs > 0 and int(count) >= max_docs:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Document limit reached ({max_docs}). Delete some documents before uploading more.",
        )
    cap = settings.max_storage_per_user_mb * 1024 * 1024
    if cap > 0 and int(used) + incoming_bytes > cap:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Storage limit reached ({settings.max_storage_per_user_mb}MB). "
                   "Delete some documents before uploading more.",
        )


def _store_object(storage_path: str, fileobj, size: int, mime_type: str) -> None:
    """Ghi file lên MinIO (đồng bộ — gọi trong threadpool). Truyền thẳng file tạm, không nạp cả file vào RAM."""
    if not minio_client.bucket_exists(settings.minio_bucket):
        minio_client.make_bucket(settings.minio_bucket)
    fileobj.seek(0)
    minio_client.put_object(
        settings.minio_bucket, storage_path, fileobj, length=size, content_type=mime_type,
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
@limiter.limit("20/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    filename = file.filename or ""
    if not filename.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Filename is empty.")

    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f" '{extension or 'no extension'}' not supported. "
            f"Recommended: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"

    size = await _upload_size(file)
    if size <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")
    if size > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File {filename} exceeds the maximum size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    # Security: validate actual file content via magic bytes, not just the
    # client-supplied extension/Content-Type which can be trivially spoofed.
    # The signature is only in the first few bytes, so read only the beginning of the file.
    await file.seek(0)
    header = await file.read(16)
    await file.seek(0)
    if not _check_magic_bytes(extension, header):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File content does not match the expected format for '{extension}'. "
                "Please upload a valid file."
            ),
        )

    owner_id = current_user.id
    await _enforce_quota(db, owner_id, size)
    await db.commit()

    try:
        async with upload_slots.slot(settings.upload_slot_wait_s):
            document = Document(
                owner_id=owner_id, filename=filename, mime_type=mime_type, status="pending", size_bytes=size,
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)

            storage_path = f"{document.id}{extension}"
            try:
                await run_in_threadpool(_store_object, storage_path, file.file, size, mime_type)
            except Exception as exc:
                document.status = "failed"
                await db.commit()
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Can not upload file {filename}: {exc}",
                ) from exc

            document.storage_path = storage_path
            await db.commit()
    except SlotTimeout as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The server is busy handling other uploads. Please try again in a moment.",
            headers={"Retry-After": "5"},
        ) from exc

    # Pass owner_id to Celery so the worker can store it in every Qdrant point.
    # .delay() makes a synchronous network call to the Redis broker: run it in a thread pool, and
    # return a clear 503 if the broker fails (previously: 500 after the file was saved, leaving the document "pending").
    try:
        await run_in_threadpool(
            task_process_document.delay,
            str(document.id), storage_path, mime_type, str(owner_id),
        )
    except Exception as exc:
        logger.exception("Could not enqueue processing for document %s", document.id)
        document.status = "failed"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The document was saved but the processing queue is unavailable. Please try again later.",
        ) from exc

    return {"document_id": str(document.id), "status": document.status}


@router.get("/{document_id}/status")
@limiter.limit("120/minute")
async def get_document_status(
    request: Request,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return processing status for a document owned by the calling user.

    Security: ownership check prevents one user from polling the status of
    another user's document and inferring what files are being processed.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="document_id is not a valid UUID."
        ) from exc

    result = await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.owner_id == current_user.id)
    )
    document = result.scalar_one_or_none()
    if document is None:
        # Return 404 for both "not found" and "not yours" — avoids leaking existence
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
    }


class MindmapNode(BaseModel):
    id: str
    name: str
    type: str = "concept"
    chunk_indices: List[int] = []


class MindmapEdge(BaseModel):
    id: str
    source: str
    target: str
    relation: str
    weight: float = 1.0
    chunk_index: Optional[int] = None


class MindmapResponse(BaseModel):
    document_id: str
    enabled: bool
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]
    stats: dict


@router.get("/{document_id}/mindmap", response_model=MindmapResponse)
@limiter.limit("30/minute")
async def get_document_mindmap(
    request: Request,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Knowledge-graph mindmap for one owned document (GraphRAG). Empty when KG is off or not built yet."""
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="document_id is not a valid UUID."
        ) from exc

    document = (await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.owner_id == current_user.id)
    )).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if not settings.kg_enabled:
        return MindmapResponse(
            document_id=str(document.id), enabled=False, nodes=[], edges=[], stats={"nodes": 0, "edges": 0},
        )

    from app.rag import knowledge_graph

    graph = await run_in_threadpool(
        knowledge_graph.mindmap_for_document, str(document.id), str(current_user.id),
    )
    return MindmapResponse(
        document_id=str(document.id),
        enabled=True,
        nodes=[MindmapNode(**n) for n in graph.get("nodes") or []],
        edges=[MindmapEdge(**e) for e in graph.get("edges") or []],
        stats=graph.get("stats") or {"nodes": 0, "edges": 0},
    )


# List & delete documents
LIST_LIMIT = 200
# A document left "pending" longer than this is assumed to have been abandoned by a worker -> allow deletion to unblock it.
STUCK_PENDING_AFTER = timedelta(minutes=30)


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    created_at: Optional[datetime] = None


@router.get("", response_model=List[DocumentOut])
@limiter.limit("120/minute")
async def list_documents(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Tài liệu của người dùng hiện tại, mới nhất trước (để giao diện khôi phục danh sách sau khi tải lại)."""
    rows = (await db.execute(
        select(Document)
        .where(Document.owner_id == current_user.id)
        .order_by(Document.created_at.desc())
        .limit(LIST_LIMIT)
    )).scalars().all()
    return [
        DocumentOut(id=str(d.id), filename=d.filename, status=d.status or "pending", created_at=d.created_at)
        for d in rows
    ]


def _is_recent(moment: Optional[datetime]) -> bool:
    if moment is None:
        return False
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - moment < STUCK_PENDING_AFTER


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("30/minute")
async def delete_document(
    request: Request,
    document_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Xóa tài liệu và MỌI dấu vết của nó: vector (Qdrant), file gốc (MinIO), bản ghi (Postgres).

    Thứ tự được chọn để lỗi nào cũng không để lại thứ gì không dọn được:
    vector trước (lỗi => dừng, báo 502, thử lại được) -> bản ghi DB -> file MinIO (lỗi chỉ ghi log).
    Câu trả lời đã cache của user cũng bị vô hiệu vì có thể trích dẫn tài liệu này.
    """
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="document_id is not a valid UUID."
        ) from exc

    document = (await db.execute(
        select(Document).where(Document.id == doc_uuid, Document.owner_id == current_user.id)
    )).scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    if document.status == "pending" and _is_recent(document.created_at):
        # The worker is embedding this document: deleting it now would leave orphaned vectors written afterward.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document is still being processed. Please try again in a moment.",
        )

    owner_id = str(current_user.id)
    try:
        await run_in_threadpool(vector_store.delete_document, str(document.id), owner_id)
    except Exception as exc:
        logger.exception("Could not delete vectors of document %s", document.id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not remove the document from the search index. Please try again.",
        ) from exc

    storage_path = document.storage_path
    # Keep quiz/conversation history and only remove the document link (do not depend on database ON DELETE behavior).
    await db.execute(update(QuizAttempt).where(QuizAttempt.document_id == document.id).values(document_id=None))
    await db.execute(update(Conversation).where(Conversation.document_id == document.id).values(document_id=None))
    # Use a direct DELETE statement (not the ORM) so SQLAlchemy does not lazy-load Document relationships,
    # which is not allowed in an async session.
    await db.execute(delete(Document).where(Document.id == document.id))
    await db.commit()

    if storage_path:
        try:
            await run_in_threadpool(minio_client.remove_object, settings.minio_bucket, storage_path)
        except Exception:  # noqa: BLE001 - The record is already deleted; an orphaned file is harmless, just log it.
            logger.warning("Could not remove object %s from MinIO", storage_path, exc_info=True)

    await semantic_cache.invalidate_owner(owner_id)
    return None
