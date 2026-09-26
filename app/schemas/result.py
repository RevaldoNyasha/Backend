from pydantic import Field

from app.schemas.base import Schema, UtcDateTime


class AnalysisDocument(Schema):
    id: str
    name: str


class AnalysisResponse(Schema):
    """A project's cross-document analysis and where it is in generation.

    `status` is `processing` (with `progress`, e.g. "Reading documents (2 of 5)"),
    then `ready` (Markdown `content`, and the `documents` it covered) or `failed`
    (with `error`).
    """

    project_id: str
    status: str
    content: str | None = None
    error: str | None = None
    model: str | None = None
    progress: str | None = None
    documents: list[AnalysisDocument] = Field(default_factory=list)
    created_at: UtcDateTime
    completed_at: UtcDateTime | None = None


class SummaryResponse(Schema):
    """A document's AI summary and where it is in generation.

    `status` is `processing` while the model works, then `ready` (with
    `content` as Markdown) or `failed` (with `error`).
    """

    document_id: str
    status: str
    content: str | None = None
    error: str | None = None
    model: str | None = None
    created_at: UtcDateTime
    completed_at: UtcDateTime | None = None
