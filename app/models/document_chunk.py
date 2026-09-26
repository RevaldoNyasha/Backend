from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.utils.ids import new_id


class DocumentChunk(Base):
    """A passage of a document's text, the unit the assistant searches over.

    `project_id` is stored on every chunk (not only via the document) so a
    search can filter by project directly and never touch another project.
    `page` is the 1-based PDF page, or empty for formats without pages.
    `char_start`/`char_end` locate the chunk within that page's text.
    `start_seconds`/`end_seconds` are set for transcribed audio and video.
    `embedding` is a normalised float32 vector (see `app/ai/embeddings.py`), filled
    by the processing worker after chunking; empty means "not embedded yet".
    """

    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # For transcribed recordings: where the passage is spoken, in seconds.
    start_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    # Which model produced `embedding`; chunks from another model get re-embedded.
    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        server_default=func.now(),
        nullable=False,
    )
