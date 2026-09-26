"""Give document chunks their embeddings, a batch at a time.

This runs after text extraction as a separate stage: a document is `processed`
once its text and chunks exist, and its chunks become searchable as soon as they
are embedded. If the embedding model is unavailable, documents are not failed;
the chunks simply wait and are embedded once the model loads.
"""

import logging

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session
from sqlalchemy.orm.exc import StaleDataError

from app.ai.embeddings import Embedder, vector_to_bytes
from app.models.document_chunk import DocumentChunk

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


def _needs_embedding(model_name: str):
    return or_(DocumentChunk.embedding.is_(None), DocumentChunk.embedding_model != model_name)


def embed_pending_chunks(db: Session, embedder: Embedder, limit: int = BATCH_SIZE) -> int:
    """Embed up to `limit` chunks that have no vector (or one from another model).

    Returns how many chunks were handled; 0 means nothing is waiting.
    """
    chunks = db.scalars(
        select(DocumentChunk)
        .where(_needs_embedding(embedder.model_name))
        .order_by(DocumentChunk.created_at, DocumentChunk.document_id, DocumentChunk.chunk_index)
        .limit(limit)
    ).all()
    if not chunks:
        return 0

    vectors = embedder.embed_documents([chunk.text for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk.embedding = vector_to_bytes(vector)
        chunk.embedding_model = embedder.model_name
    try:
        db.commit()
    except StaleDataError:
        # Some chunks were deleted (document removed or reprocessed) meanwhile.
        db.rollback()
    return len(chunks)


def pending_embedding_count(db: Session, model_name: str) -> int:
    return db.scalar(
        select(func.count()).select_from(DocumentChunk).where(_needs_embedding(model_name))
    )


def embedded_chunk_count(db: Session, document_id: str, model_name: str) -> int:
    return db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.embedding.is_not(None),
            DocumentChunk.embedding_model == model_name,
        )
    )
