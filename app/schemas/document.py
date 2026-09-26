from pydantic import Field, computed_field

from app.schemas.base import Schema, UtcDateTime
from app.utils.file_validation import format_file_size


class DocumentResponse(Schema):
    id: str
    project_id: str
    name: str
    kind: str
    type: str
    extension: str
    status: str
    participant_count: int = 0
    language: str | None = None
    uploaded_at: UtcDateTime
    updated_at: UtcDateTime
    size_bytes: int = Field(alias="fileSizeBytes")
    # Why processing failed; null unless status is "failed".
    processing_error: str | None = None
    # What slow processing is doing now, e.g. "Transcribing: 43%"; null otherwise.
    processing_progress: str | None = None

    @computed_field(alias="fileSize", return_type=str)
    @property
    def file_size(self) -> str:
        return format_file_size(self.size_bytes)


class DocumentTextResponse(Schema):
    """The text extracted from a processed document."""

    document_id: str
    text: str
    character_count: int
    chunk_count: int
    # Chunks that have an embedding and are therefore searchable.
    embedded_chunk_count: int


class TranscriptSegment(Schema):
    start: float  # seconds
    end: float
    text: str


class TranscriptResponse(Schema):
    """A recording's transcript with timestamps, plus its English translation if made."""

    document_id: str
    language: str | None = None
    duration: float | None = None  # seconds
    model: str | None = None
    segments: list[TranscriptSegment]
    # Set when "Translate transcripts to English" produced one.
    translation: list[TranscriptSegment] | None = None
