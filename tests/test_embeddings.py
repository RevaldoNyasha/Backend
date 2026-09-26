from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai import embeddings as embeddings_module
from app.ai import llm
from app.ai.embeddings import (
    EmbeddingStatus,
    SentenceTransformerEmbedder,
    bytes_to_vector,
    set_embedder,
    vector_to_bytes,
)
from app.core.config import Settings, settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Document, DocumentChunk
from app.services.processing_worker import ProcessingWorker
from app.services.storage import LocalStorage, set_storage
from tests.conftest import FAKE_DIMENSION, FakeEmbedder

# ---------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def uploads_root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "uploads"
    root.mkdir()
    set_storage(LocalStorage(root))
    yield root
    set_storage(None)


@pytest.fixture
def engine() -> Iterator[Engine]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def db(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    with session_factory() as session:
        yield session


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def worker(session_factory: sessionmaker[Session]) -> ProcessingWorker:
    return ProcessingWorker(session_factory, embeddings_enabled=True)


def upload_notes(client: TestClient, text: str, email: str = "ada@example.com") -> tuple:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": email, "password": "password123"},
    ).json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    project_id = client.post("/api/v1/projects", json={"name": "Study"}, headers=headers).json()[
        "id"
    ]
    document = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("notes.txt", text.encode(), "text/plain")},
        headers=headers,
    ).json()[0]
    return headers, project_id, document["id"]


LONG_TEXT = "\n\n".join(f"Paragraph {i}. Farmers described transport costs." for i in range(80))


def chunks_of(db: Session, document_id: str) -> list[DocumentChunk]:
    db.expire_all()
    return list(
        db.scalars(select(DocumentChunk).where(DocumentChunk.document_id == document_id)).all()
    )


# ---------------------------------------------------------------- vectors


def test_vectors_round_trip_through_bytes() -> None:
    vector = np.array([0.25, -0.5, 1.0], dtype=np.float32)

    assert np.array_equal(bytes_to_vector(vector_to_bytes(vector)), vector)
    assert len(vector_to_bytes(np.zeros(1024))) == 4096  # float32


# ---------------------------------------------------------------- worker


def test_worker_embeds_every_chunk_after_processing(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers, project_id, document_id = upload_notes(client, LONG_TEXT)

    worker.run_pending()

    chunks = chunks_of(db, document_id)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.embedding_model == settings.embedding_model
        vector = bytes_to_vector(chunk.embedding)
        assert vector.shape == (FAKE_DIMENSION,)
        assert np.isclose(np.linalg.norm(vector), 1.0)

    text = client.get(
        f"/api/v1/projects/{project_id}/documents/{document_id}/text", headers=headers
    ).json()
    assert text["embeddedChunkCount"] == text["chunkCount"] == len(chunks)


def test_embedding_failure_keeps_the_document_processed_and_retries_later(
    client: TestClient, db: Session, worker: ProcessingWorker, fake_embedder: FakeEmbedder
) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            self.calls.append(list(texts))
            raise RuntimeError("model download failed")

    broken = BrokenEmbedder()
    set_embedder(broken)
    headers, project_id, document_id = upload_notes(client, "Some field notes.")

    worker.run_pending()

    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "processed"
    assert all(chunk.embedding is None for chunk in chunks_of(db, document_id))

    # Within the back-off window the model is not tried again.
    worker.run_pending()
    assert len(broken.calls) == 1

    # Once the model works and the back-off has passed, the chunks catch up.
    set_embedder(fake_embedder)
    worker._embedding_retry_at = 0.0
    worker.run_pending()
    assert all(chunk.embedding is not None for chunk in chunks_of(db, document_id))


def test_chunks_from_another_model_are_re_embedded(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    _, _, document_id = upload_notes(client, "Some field notes.")
    set_embedder(FakeEmbedder(model_name="old-model"))
    worker.run_pending()
    assert {chunk.embedding_model for chunk in chunks_of(db, document_id)} == {"old-model"}

    set_embedder(FakeEmbedder())
    worker.run_pending()

    assert {chunk.embedding_model for chunk in chunks_of(db, document_id)} == {
        settings.embedding_model
    }


def test_chunks_deleted_during_embedding_do_not_break_the_worker(
    client: TestClient, session_factory: sessionmaker[Session], worker: ProcessingWorker
) -> None:
    _, _, document_id = upload_notes(client, "Some field notes.")
    worker.process_documents()

    class DeletingEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            with session_factory() as other:  # the document is removed mid-embedding
                other.execute(delete(Document).where(Document.id == document_id))
                other.commit()
            return super().embed_documents(texts)

    set_embedder(DeletingEmbedder())

    worker.embed_chunks()  # must not raise

    with session_factory() as session:
        assert session.scalars(select(DocumentChunk)).all() == []


def test_embeddings_can_be_switched_off(
    client: TestClient, db: Session, session_factory: sessionmaker[Session]
) -> None:
    _, _, document_id = upload_notes(client, "Some field notes.")

    ProcessingWorker(session_factory, embeddings_enabled=False).run_pending()

    assert all(chunk.embedding is None for chunk in chunks_of(db, document_id))


# ---------------------------------------------------------------- embedder


def test_embedder_loads_lazily_and_reports_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    def failing_model(*args: object, **kwargs: object) -> None:
        raise OSError("no internet")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", failing_model)
    embedder = SentenceTransformerEmbedder("BAAI/bge-m3")
    assert embedder.status == EmbeddingStatus.NOT_LOADED  # nothing loaded at construction

    with pytest.raises(OSError):
        embedder.embed_query("hello")

    assert embedder.status == EmbeddingStatus.ERROR
    assert "no internet" in embedder.error


def test_default_embedder_is_created_on_demand(monkeypatch: pytest.MonkeyPatch) -> None:
    set_embedder(None)

    embedder = embeddings_module.get_embedder()

    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert embedder.status == EmbeddingStatus.NOT_LOADED


# ---------------------------------------------------------------- settings and health


def test_blank_model_settings_fall_back_to_defaults() -> None:
    configured = Settings(_env_file=None, ollama_model="  ", embedding_model="")

    assert configured.ollama_model == "qwen2.5:3b"
    assert configured.embedding_model == "BAAI/bge-m3"


def test_ollama_status_reports_unreachable_server(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(llm.httpx, "get", refuse)

    status = llm.ollama_status()

    assert status["reachable"] is False
    assert status["modelAvailable"] is False


@pytest.mark.parametrize(
    ("pulled", "available"),
    [(["qwen2.5:3b"], True), (["qwen2.5:3b-instruct"], False), ([], False)],
)
def test_ollama_status_checks_the_model_is_pulled(
    monkeypatch: pytest.MonkeyPatch, pulled: list[str], available: bool
) -> None:
    monkeypatch.setattr(settings, "ollama_model", "qwen2.5:3b")
    monkeypatch.setattr(
        llm.httpx,
        "get",
        lambda *args, **kwargs: httpx.Response(
            200,
            json={"models": [{"name": name} for name in pulled]},
            request=httpx.Request("GET", "http://ollama"),
        ),
    )

    status = llm.ollama_status()

    assert status["reachable"] is True
    assert status["modelAvailable"] is available


def test_ai_health_endpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(
        "app.api.health.ollama_status",
        lambda: {"model": "qwen2.5:3b", "reachable": True, "modelAvailable": True, "error": None},
    )
    upload_notes(client, "Waiting to be embedded.")

    body = client.get("/health/ai").json()

    assert body["status"] == "ok"
    assert body["llm"]["modelAvailable"] is True
    assert body["embeddings"]["model"] == settings.embedding_model
    assert body["embeddings"]["pendingChunks"] == 0  # not processed yet, so no chunks


def test_ai_health_with_groq_reports_configuration_without_calling_it(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "gsk_secret_value")

    body = client.get("/health/ai").json()

    assert body["llm"]["provider"] == "groq"
    assert body["llm"]["configured"] is True
    assert "gsk_secret_value" not in str(body)

    monkeypatch.setattr(settings, "groq_api_key", "")
    body = client.get("/health/ai").json()
    assert body["status"] == "degraded"
    assert "GROQ_API_KEY" in body["llm"]["error"]


def test_plain_health_is_unchanged(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
