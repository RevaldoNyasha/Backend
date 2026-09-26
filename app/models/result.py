from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.utils.ids import new_id

RESULT_EXTRACTED_TEXT = "extracted_text"
RESULT_TRANSCRIPTION = "transcription"
# An English translation of a non-English transcript.
RESULT_TRANSLATION = "translation"
RESULT_SUMMARY = "summary"
RESULT_REPORT = "report"
RESULT_ANALYSIS = "analysis"

# Generated results (summaries, reports, analyses) are produced in the
# background; extracted text is written directly as "ready".
RESULT_STATUS_PROCESSING = "processing"
RESULT_STATUS_READY = "ready"
RESULT_STATUS_FAILED = "failed"


class Result(Base):
    """A generated output: extracted text, transcript, summary, report or analysis.

    `document_id` is set for per-document outputs and empty for project-wide ones
    (e.g. a project analysis). `project_id` is always set so every lookup can be
    scoped to one project.
    """

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=True
    )
    result_type: Mapped[str] = mapped_column(String(32), nullable=False)
    # Empty while a generated result is still processing.
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=RESULT_STATUS_READY, server_default=RESULT_STATUS_READY
    )
    # Why generation failed; shown to the user.
    error: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # Which model produced the content (e.g. "qwen2.5:3b").
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    # Extra facts about a generated result, e.g. for an analysis the documents it
    # covered and its progress while running.
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
