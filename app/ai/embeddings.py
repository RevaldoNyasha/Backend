"""Text → vector with BGE-M3 (sentence-transformers).

Only this module knows about the embedding model. The rest of the app uses the
`Embedder` interface and `get_embedder()`, so tests can swap in a small fake and
the model can change without touching services.

Vectors are normalised (length 1), so cosine similarity is a plain dot product.
They are stored as raw float32 bytes; see `vector_to_bytes` / `bytes_to_vector`.
"""

import logging
import threading
import time
from typing import TYPE_CHECKING, Protocol

import numpy as np

from app.core.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Our chunks are ~1,000 characters (~250 tokens); capping the sequence length
# keeps CPU time and memory predictable.
MAX_SEQUENCE_TOKENS = 1024
BATCH_SIZE = 8


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One normalised float32 row per text."""
        ...

    def embed_query(self, text: str) -> np.ndarray: ...


class EmbeddingStatus:
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


class SentenceTransformerEmbedder:
    """Loads the model on first use (once per process), not at startup."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()
        self.status = EmbeddingStatus.NOT_LOADED
        self.error: str | None = None
        self.load_seconds: float | None = None

    def _load(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                self.status = EmbeddingStatus.LOADING
                started = time.monotonic()
                try:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.model_name, device=self._device)
                    model.max_seq_length = min(model.max_seq_length, MAX_SEQUENCE_TOKENS)
                except Exception as error:
                    self.status = EmbeddingStatus.ERROR
                    self.error = f"{type(error).__name__}: {error}"[:300]
                    raise
                self._model = model
                self.load_seconds = time.monotonic() - started
                self.status = EmbeddingStatus.READY
                self.error = None
                logger.info("Loaded %s in %.1fs", self.model_name, self.load_seconds)
        return self._model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors = self._load().encode(
            texts, batch_size=BATCH_SIZE, normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # BGE-M3 needs no special query prefix.
        return self.embed_documents([text])[0]


_embedder: Embedder | None = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = SentenceTransformerEmbedder(settings.embedding_model)
    return _embedder


def set_embedder(embedder: Embedder | None) -> None:
    """Swap the embedder (tests use a fake)."""
    global _embedder
    _embedder = embedder


def embedder_status() -> dict[str, object]:
    embedder = get_embedder()
    return {
        "model": embedder.model_name,
        "status": getattr(embedder, "status", EmbeddingStatus.READY),
        "error": getattr(embedder, "error", None),
        "loadSeconds": getattr(embedder, "load_seconds", None),
    }


def vector_to_bytes(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def bytes_to_vector(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32)
