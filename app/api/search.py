from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status

from app.ai.embeddings import get_embedder
from app.core.dependencies import CurrentProject, DbSession
from app.schemas.search import SearchResult
from app.services.retrieval import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    SearchUnavailable,
    search_project,
)

router = APIRouter(tags=["search"])


@router.get("/projects/{project_id}/search", response_model=list[SearchResult])
def search(
    project_id: str,
    current_project: CurrentProject,
    db: DbSession,
    q: Annotated[str, Query(min_length=1, max_length=1000, description="The question")],
    limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> list[SearchResult]:
    """Passages of this project's documents that best match `q`, best first.

    Only this project's processed, embedded documents are searched. An empty
    list means nothing was relevant enough.
    """
    query = q.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Query is empty"
        )
    try:
        results = search_project(db, current_project.id, query, get_embedder(), limit=limit)
    except SearchUnavailable:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is unavailable: the embedding model could not be loaded",
        ) from None
    return [
        SearchResult(
            document_id=result.document_id,
            document_name=result.document_name,
            chunk_index=result.chunk_index,
            page=result.page,
            text=result.text,
            score=result.score,
        )
        for result in results
    ]
