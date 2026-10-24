import io
import mimetypes
import os
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_current_user, get_db
from app.config import settings
from app.core.database import minio_client
from app.models.models import Document, User
from app.worker.tasks import task_process_document

router = APIRouter()
#If 
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md"}
MAX_UPLOAD_SIZE_BYTES = 400 * 1024 * 1024  


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
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

    task_process_document.delay(str(document.id), storage_path, mime_type)

    return {"document_id": str(document.id), "status": document.status}


@router.get("/{document_id}/status")
async def get_document_status(document_id: str, db: AsyncSession = Depends(get_db)):
    try:
        doc_uuid = uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="document_id is not a valid UUID."
        ) from exc

    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    return {
        "document_id": str(document.id),
        "filename": document.filename,
        "status": document.status,
    }
