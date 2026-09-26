from pydantic import ConfigDict, Field

from app.schemas.base import Schema, UtcDateTime
from app.schemas.result import AnalysisDocument


class ReportTemplateResponse(Schema):
    type: str
    label: str
    description: str
    # Fixed headings; empty for `custom`, whose headings the request supplies.
    sections: list[str]
    custom_sections: bool


class ReportCreate(Schema):
    """Contract §8.3. Everything is optional; `type` defaults to a thematic analysis."""

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    title: str | None = Field(default=None, max_length=200)
    type: str | None = None
    # Headings for a `custom` report (ignored by the other types).
    sections: list[str] | None = Field(default=None, max_length=20)
    # Limit the report to these processed documents (default: all of them).
    document_ids: list[str] | None = Field(default=None, max_length=200)


class ReportUpdate(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    title: str | None = Field(default=None, max_length=200)
    status: str | None = None  # 'draft' | 'final'


class ReportGeneration(Schema):
    state: str  # 'processing' | 'ready' | 'failed'
    progress: str | None = None
    error: str | None = None
    started_at: UtcDateTime
    completed_at: UtcDateTime | None = None


class ReportResponse(Schema):
    """Contract §8.1 `ResearchReport`, plus generation state and (on `GET` one) `content`."""

    id: str
    project_id: str
    title: str
    summary: str
    created_at: UtcDateTime
    sections: list[str]
    status: str  # 'draft' | 'final'
    type: str
    documents: list[AnalysisDocument] = Field(default_factory=list)
    model: str | None = None
    generation: ReportGeneration
    content: str | None = None
