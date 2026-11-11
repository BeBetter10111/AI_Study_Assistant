import io
import logging
import mimetypes
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.api.security.rate_limiter import limiter
from app.config import settings
from app.core.database import minio_client
from app.models.models import Document, User
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

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is empty.")
    if len(contents) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File {filename} exceeds the maximum size of {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)}MB.",
        )

    # Security: validate actual file content via magic bytes, not just the
    # client-supplied extension/Content-Type which can be trivially spoofed.
    if not _check_magic_bytes(extension, contents):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File content does not match the expected format for '{extension}'. "
                "Please upload a valid file."
            ),
        )

    document = Document(
        owner_id=current_user.id,
        filename=filename,
        mime_type=mime_type,
        status="pending",
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    storage_path = f"{document.id}{extension}"

    try:
        if not minio_client.bucket_exists(settings.minio_bucket):
            minio_client.make_bucket(settings.minio_bucket)

        minio_client.put_object(
            settings.minio_bucket,
            storage_path,
            io.BytesIO(contents),
            length=len(contents),
            content_type=mime_type,
        )
    except Exception as exc:
        document.status = "failed"
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Can not upload file {filename}: {exc}",
        ) from exc

    document.storage_path = storage_path
    await db.commit()

    # Pass owner_id to Celery so the worker can store it in every Qdrant point.
    # .delay() là một lệnh gọi mạng đồng bộ tới Redis broker: chạy trong threadpool, và nếu
    # broker lỗi thì báo 503 rõ ràng (trước đây: 500 sau khi file đã lưu, document kẹt "pending").
    try:
        await run_in_threadpool(
            task_process_document.delay,
            str(document.id), storage_path, mime_type, str(current_user.id),
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
async def get_document_status(
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
