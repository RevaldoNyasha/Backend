from pydantic import ConfigDict, Field

from app.schemas.base import Schema, UtcDateTime


class ProjectCreate(Schema):
    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    name: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)


class ProjectResponse(Schema):
    id: str
    name: str
    description: str
    document_count: int = 0
    participant_count: int = 0
    interview_count: int = 0
    focus_group_count: int = 0
    themes: list[str] = Field(default_factory=list)
    codes: list[str] = Field(default_factory=list)
    status: str
    created_at: UtcDateTime
    updated_at: UtcDateTime
