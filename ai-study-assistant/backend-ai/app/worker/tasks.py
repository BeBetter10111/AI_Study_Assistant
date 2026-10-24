"""Celery worker tasks — document ingestion pipeline.

Uses the new structured ingestion path (extract_and_chunk) so that each chunk
carries rich metadata (page number, slide number/title, section heading) that
the vector store persists for precise citations.
"""
import logging
import os
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.core.database import minio_client
from app.models.models import Document
from app.rag import ingestion
from app.rag.vector_store import vector_store
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Celery worker is synchronous — use a separate sync SQLAlchemy engine.
# Lazy-init to avoid import-time side effects (no driver needed until first task).
_sync_engine = None


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = create_engine(settings.database_url.replace("+asyncpg", ""))
    return _sync_engine


def _update_document_status(document_id: str, status: str) -> None:
    with Session(_get_sync_engine()) as session:
        document = session.get(Document, document_id)
        if document is None:
            logger.warning(f"Document {document_id} not found to update status.")
            return
        document.status = status
        session.commit()


@celery_app.task(name="task_process_document", bind=True, max_retries=3)
def task_process_document(self, document_id: str, storage_path: str, mime_type: str):
    _update_document_status(document_id, "chunking")

    suffix = os.path.splitext(storage_path)[1]
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name

        minio_client.fget_object(settings.minio_bucket, storage_path, tmp_path)

        # Structured ingestion — returns List[ChunkData]
        chunks = ingestion.extract_and_chunk(tmp_path, mime_type)
        if not chunks:
            raise ValueError("No valid chunks created after processing the document.")

        vector_store.upsert_chunks(chunks, document_id)
        _update_document_status(document_id, "embedded")
        logger.info(
            f"Document {document_id} processed successfully: {len(chunks)} chunks "
            f"(includes per-segment summary chunks)."
        )

    except Exception as exc:
        logger.exception(f"Error processing document {document_id}: {exc}")
        _update_document_status(document_id, "failed")
        raise
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
