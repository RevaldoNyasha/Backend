"""Uploading, listing and deleting a project's documents."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import BYTES_PER_MB
from app.models.document import STATUS_FAILED, STATUS_PROCESSING, Document
from app.models.project import Project
from app.services.processing_worker import notify_worker
from app.services.project import mark_project_updated
from app.services.storage import get_storage
from app.services.transcription import hold_reason, is_recording
from app.utils.file_validation import (
    max_size_mb_for_kind,
    normalize_extension,
    resolve_kind_and_type,
)

# HTTP-like status codes carried on rejections, so the route can answer with them.
REJECT_EMPTY = 400
REJECT_TOO_LARGE = 413
REJECT_UNSUPPORTED = 415


@dataclass(frozen=True)
class UploadRejection:
    filename: str
    status_code: int
    detail: str


@dataclass
class UploadResult:
    stored: list[Document] = field(default_factory=list)
    rejected: list[UploadRejection] = field(default_factory=list)


def upload_documents(
    db: Session,
    project: Project,
    files: Iterable[tuple[str | None, BinaryIO]],
) -> UploadResult:
    """Validate and store each ``(filename, stream)``; bad files are skipped, not fatal.

    Each accepted file is saved under a random name and gets a ``processing``
    document. Rejected files leave nothing behind.
    """
    result = UploadResult()
    for raw_name, stream in files:
        name = Path(raw_name or "").name  # drop any client-supplied directories
        outcome = _upload_one(db, project, name, stream)
        if isinstance(outcome, UploadRejection):
            result.rejected.append(outcome)
        else:
            result.stored.append(outcome)
    if result.stored:
        notify_worker()  # start extracting straight away
    return result


def _upload_one(
    db: Session,
    project: Project,
    name: str,
    stream: BinaryIO,
) -> Document | UploadRejection:
    kind_and_type = resolve_kind_and_type(name)
    if kind_and_type is None:
        return UploadRejection(name, REJECT_UNSUPPORTED, "Unsupported file type")

    kind, document_type = kind_and_type
    limit_mb = max_size_mb_for_kind(kind)
    limit_bytes = limit_mb * BYTES_PER_MB
    content = stream.read(limit_bytes + 1)  # one extra byte reveals "too large"

    if not content:
        return UploadRejection(name, REJECT_EMPTY, "Uploaded file is empty")
    if len(content) > limit_bytes:
        return UploadRejection(
            name, REJECT_TOO_LARGE, f"{kind.capitalize()} files are limited to {limit_mb} MB"
        )

    storage = get_storage()
    storage_key = f"{project.id}/{uuid4().hex}{Path(name).suffix.lower()}"
    storage.save(storage_key, content)

    document = Document(
        project_id=project.id,
        name=name,
        kind=kind,
        type=document_type,
        extension=normalize_extension(name),
        storage_path=storage_key,
        size_bytes=len(content),
        status=STATUS_PROCESSING,
    )
    if is_recording(document):
        # Transcription off (server) or automatic transcription off (Settings):
        # keep the file, and let "Retry" transcribe it when wanted.
        reason = hold_reason(db, document)
        if reason is not None:
            document.status = STATUS_FAILED
            document.processing_error = reason
    db.add(document)
    mark_project_updated(project)
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage.delete(storage_key)  # no orphaned file without a row
        raise
    db.refresh(document)
    return document


def list_documents(db: Session, project: Project) -> list[Document]:
    return list(
        db.scalars(
            select(Document)
            .where(Document.project_id == project.id)
            .order_by(Document.uploaded_at.desc())
        )
    )


def delete_document(db: Session, project: Project, document: Document) -> None:
    storage_key = document.storage_path
    db.delete(document)
    mark_project_updated(project)
    db.commit()
    get_storage().delete(storage_key)
