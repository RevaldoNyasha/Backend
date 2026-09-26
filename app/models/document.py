from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.utils.ids import new_id

KIND_INTERVIEW = "interview"
KIND_FOCUS_GROUP = "focus_group"
KIND_SURVEY = "survey"
KIND_NOTES = "notes"
KIND_AUDIO = "audio"
KIND_VIDEO = "video"
KIND_PDF = "pdf"
KIND_DOCX = "docx"
KIND_CSV = "csv"

# Recordings: turned into text by transcription, not text extraction.
TRANSCRIBABLE_KINDS = frozenset({KIND_AUDIO, KIND_VIDEO})

TYPE_INTERVIEW = "Interview"
TYPE_FOCUS_GROUP = "Focus Group"
TYPE_SURVEY = "Survey"
TYPE_RESEARCH_NOTES = "Research Notes"

STATUS_PROCESSING = "processing"
STATUS_PROCESSED = "processed"
STATUS_FAILED = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    extension: Mapped[str] = mapped_column(String(10), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=STATUS_PROCESSING,
    )
    participant_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Why processing failed, shown to the user; empty unless status is "failed".
    processing_error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # What slow processing is doing right now (e.g. "Transcribing: 43%"); empty otherwise.
    processing_progress: Mapped[str | None] = mapped_column(String(120), nullable=True)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
