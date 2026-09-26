"""Find the passages of one project that best match a question.

This is the retrieval half of RAG. It only ever reads chunks whose
`project_id` is the requested project, so an answer for Project A can never
draw on Project B, even when both belong to the same user.

Search is a plain cosine similarity over the project's chunk vectors, done in
NumPy (`_nearest_chunks`). That is fast enough for development-sized projects;
after the move to Neon, pgvector replaces that one function.
"""

from dataclasses import dataclass

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.embeddings import Embedder, bytes_to_vector
from app.core.config import settings
from app.models.document import Document
from app.models.document_chunk import DocumentChunk

DEFAULT_LIMIT = 5
MAX_LIMIT = 20


class SearchUnavailable(Exception):
    """The embedding model could not be used to embed the question."""


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_name: str
    chunk_index: int
    page: int | None
    char_start: int
    char_end: int
    text: str
    score: float
    # Set for transcribed recordings: when the passage is spoken (seconds).
    start_seconds: float | None = None


def search_project(
    db: Session,
    project_id: str,
    query: str,
    embedder: Embedder,
    limit: int = DEFAULT_LIMIT,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Best-matching chunks of `project_id` for `query`, highest score first.

    Only chunks embedded by the current model are considered. Matches scoring
    below `min_score` (default `settings.retrieval_min_score`) are dropped, so an
    unrelated question returns nothing rather than weak passages.
    """
    threshold = settings.retrieval_min_score if min_score is None else min_score
    try:
        query_vector = embedder.embed_query(query)
    except Exception as error:
        raise SearchUnavailable(str(error)) from error

    ranked = _nearest_chunks(db, project_id, embedder.model_name, query_vector, limit)
    ranked = [(chunk_id, score) for chunk_id, score in ranked if score >= threshold]
    if not ranked:
        return []

    rows = db.execute(
        select(DocumentChunk, Document.name)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            DocumentChunk.project_id == project_id,  # belt and braces: same project only
            DocumentChunk.id.in_([chunk_id for chunk_id, _ in ranked]),
        )
    ).all()
    by_id = {chunk.id: (chunk, name) for chunk, name in rows}

    results = []
    for chunk_id, score in ranked:
        if chunk_id not in by_id:
            continue  # deleted between the two queries
        chunk, document_name = by_id[chunk_id]
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                document_name=document_name,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                text=chunk.text,
                score=round(score, 4),
                start_seconds=chunk.start_seconds,
            )
        )
    return results


def _nearest_chunks(
    db: Session,
    project_id: str,
    model_name: str,
    query_vector: np.ndarray,
    limit: int,
) -> list[tuple[str, float]]:
    """(chunk_id, cosine score) of the closest chunks in one project.

    Vectors are normalised, so the dot product is the cosine similarity.
    Replace this function with a pgvector query after the Neon migration.
    """
    rows = db.execute(
        select(DocumentChunk.id, DocumentChunk.embedding).where(
            DocumentChunk.project_id == project_id,
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.embedding_model == model_name,
        )
    ).all()
    if not rows:
        return []

    ids = [row.id for row in rows]
    matrix = np.vstack([bytes_to_vector(row.embedding) for row in rows])
    query = np.asarray(query_vector, dtype=np.float32)
    if matrix.shape[1] != query.shape[0]:
        return []  # vectors from an incompatible model; they will be re-embedded
    scores = matrix @ query

    count = min(limit, len(ids))
    top = np.argpartition(-scores, count - 1)[:count]
    top = top[np.argsort(-scores[top])]
    return [(ids[index], float(scores[index])) for index in top]
