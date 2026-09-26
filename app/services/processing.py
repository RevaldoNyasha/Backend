"""Process uploaded documents: extract text, chunk it, and record the outcome.

A document is waiting for work while its status is ``processing``. The worker
(`app/services/processing_worker.py`) calls `process_document` for each one.
Audio and video are transcribed instead (`app/services/transcription.py`).
"""

import logging

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import (
    STATUS_FAILED,
    STATUS_PROCESSED,
    STATUS_PROCESSING,
    TRANSCRIBABLE_KINDS,
    Document,
)
from app.models.document_chunk import DocumentChunk
from app.models.project import Project
from app.models.result import RESULT_EXTRACTED_TEXT, Result
from app.services.document_processor import EXTRACTABLE_KINDS, ExtractionError, extract_text
from app.services.project import mark_project_updated
from app.services.storage import get_storage
from app.utils.chunking import chunk_text

logger = logging.getLogger(__name__)

MAX_ERROR_LENGTH = 500


class DocumentAlreadyProcessing(Exception):
    pass


class TranscriptionDisabled(Exception):
    """A recording cannot be retried: transcription is off on this server."""


def next_pending_document_id(db: Session) -> str | None:
    """Oldest document still waiting for text extraction."""
    return db.scalar(
        select(Document.id)
        .where(Document.status == STATUS_PROCESSING, Document.kind.in_(EXTRACTABLE_KINDS))
        .order_by(Document.uploaded_at)
        .limit(1)
    )


def process_document(db: Session, document_id: str) -> None:
    document = db.get(Document, document_id)
    if document is None or document.status != STATUS_PROCESSING:
        return
    if document.kind not in EXTRACTABLE_KINDS:
        return  # audio/video: handled by transcription later

    try:
        content = get_storage().read(document.storage_path)
        extracted = extract_text(document.kind, content)
    except FileNotFoundError:
        _mark_failed(db, document, "The uploaded file is missing. Please upload it again.")
        return
    except ExtractionError as error:
        _mark_failed(db, document, str(error))
        return
    except Exception:
        logger.exception("Unexpected error extracting document %s", document_id)
        _mark_failed(db, document, "Processing failed unexpectedly. Try again.")
        return

    # Replace any earlier output (e.g. when reprocessing).
    db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
    db.execute(
        delete(Result).where(
            Result.document_id == document.id, Result.result_type == RESULT_EXTRACTED_TEXT
        )
    )
    db.add(
        Result(
            project_id=document.project_id,
            document_id=document.id,
            result_type=RESULT_EXTRACTED_TEXT,
            content=extracted.full_text,
        )
    )
    chunk_index = 0
    for page in extracted.pages:
        for chunk in chunk_text(page.text):
            db.add(
                DocumentChunk(
                    project_id=document.project_id,
                    document_id=document.id,
                    chunk_index=chunk_index,
                    text=chunk.text,
                    page=page.number,
                    char_start=chunk.start,
                    char_end=chunk.end,
                )
            )
            chunk_index += 1

    document.status = STATUS_PROCESSED
    document.processing_error = None
    _touch_project(db, document)
    try:
        db.commit()
    except IntegrityError:
        # The document (or its project) was deleted while we worked: nothing to save.
        db.rollback()
        return
    logger.info("Processed document %s into %d chunks", document_id, chunk_index)


def reprocess_document(db: Session, project: Project, document: Document) -> Document:
    """Queue a document to be processed again (e.g. after a failure)."""
    if document.status == STATUS_PROCESSING:
        raise DocumentAlreadyProcessing
    if document.kind in TRANSCRIBABLE_KINDS and not settings.transcription_enabled:
        raise TranscriptionDisabled
    document.status = STATUS_PROCESSING
    document.processing_error = None
    document.processing_progress = None
    mark_project_updated(project)
    db.commit()
    db.refresh(document)
    return document


def extracted_text_for(db: Session, document: Document) -> str | None:
    return db.scalar(
        select(Result.content).where(
            Result.document_id == document.id, Result.result_type == RESULT_EXTRACTED_TEXT
        )
    )


def chunk_count_for(db: Session, document: Document) -> int:
    return len(
        db.scalars(select(DocumentChunk.id).where(DocumentChunk.document_id == document.id)).all()
    )


def _mark_failed(db: Session, document: Document, reason: str) -> None:
    db.rollback()
    document = db.get(Document, document.id)
    if document is None:
        return
    document.status = STATUS_FAILED
    document.processing_error = reason[:MAX_ERROR_LENGTH]
    _touch_project(db, document)
    db.commit()
    logger.info("Document %s failed processing: %s", document.id, reason)


def _touch_project(db: Session, document: Document) -> None:
    project = db.get(Project, document.project_id)
    if project is not None:
        mark_project_updated(project)
