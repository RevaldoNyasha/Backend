from app.schemas.base import Schema


class SearchResult(Schema):
    """One matching passage from a project's documents."""

    document_id: str
    document_name: str
    chunk_index: int
    page: int | None = None
    text: str
    score: float
