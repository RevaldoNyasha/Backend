from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.embeddings import set_embedder
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import DocumentChunk
from app.services.processing_worker import ProcessingWorker
from app.services.retrieval import search_project
from app.services.storage import LocalStorage, set_storage
from tests.conftest import FakeEmbedder

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


def register(client: TestClient, email: str = "ada@example.com") -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": email, "password": "password123"},
    ).json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def make_project(client: TestClient, headers: dict[str, str], name: str) -> str:
    return client.post("/api/v1/projects", json={"name": name}, headers=headers).json()["id"]


def add_notes(
    client: TestClient, headers: dict[str, str], project_id: str, name: str, text: str
) -> str:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": (name, text.encode(), "text/plain")},
        headers=headers,
    )
    assert response.status_code == 202
    return response.json()[0]["id"]


def search(client: TestClient, headers: dict[str, str], project_id: str, q: str, **params):
    return client.get(
        f"/api/v1/projects/{project_id}/search", params={"q": q, **params}, headers=headers
    )


TRANSPORT = "Farmers said transport costs to the market doubled."
FINANCE = "Smallholders explained that bank loans and financing are hard to get."
WEATHER = "Rain fell heavily during the planting season."


# ---------------------------------------------------------------- ranking


def test_search_returns_the_best_matching_passage_first(
    client: TestClient, worker: ProcessingWorker
) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    add_notes(client, headers, project, "transport.txt", TRANSPORT)
    add_notes(client, headers, project, "finance.txt", FINANCE)
    add_notes(client, headers, project, "weather.txt", WEATHER)
    worker.run_pending()

    response = search(client, headers, project, "transport costs to the market")

    assert response.status_code == 200
    results = response.json()
    assert results[0]["documentName"] == "transport.txt"
    assert results[0]["text"] == TRANSPORT
    assert set(results[0]) == {"documentId", "documentName", "chunkIndex", "page", "text", "score"}
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_unrelated_questions_return_nothing(client: TestClient, worker: ProcessingWorker) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    add_notes(client, headers, project, "transport.txt", TRANSPORT)
    worker.run_pending()

    assert search(client, headers, project, "chocolate cake recipe").json() == []


def test_limit_is_respected_and_bounded(client: TestClient, worker: ProcessingWorker) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    for index in range(4):
        add_notes(client, headers, project, f"t{index}.txt", f"{TRANSPORT} Visit {index}.")
    worker.run_pending()

    assert len(search(client, headers, project, "transport costs market", limit=2).json()) == 2
    assert search(client, headers, project, "transport", limit=0).status_code == 422
    assert search(client, headers, project, "transport", limit=21).status_code == 422
    assert search(client, headers, project, "   ").status_code == 422


def test_pdf_results_carry_their_page(
    client: TestClient, worker: ProcessingWorker, db: Session
) -> None:
    import pymupdf

    headers = register(client)
    project = make_project(client, headers, "Farming")
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), WEATHER)
    pdf.new_page().insert_text((72, 72), TRANSPORT)
    client.post(
        f"/api/v1/projects/{project}/documents",
        files={"files": ("paper.pdf", pdf.tobytes(), "application/pdf")},
        headers=headers,
    )
    worker.run_pending()

    top = search(client, headers, project, "transport costs market").json()[0]

    assert top["page"] == 2


# ---------------------------------------------------------------- the project boundary


def test_search_never_returns_another_projects_chunks(
    client: TestClient, worker: ProcessingWorker
) -> None:
    headers = register(client)
    project_a = make_project(client, headers, "Project A")
    project_b = make_project(client, headers, "Project B")
    add_notes(client, headers, project_b, "b-transport.txt", TRANSPORT)
    worker.run_pending()

    # Same user, but only Project B has the matching text.
    assert search(client, headers, project_a, "transport costs to the market").json() == []


def test_identical_text_in_two_projects_only_matches_the_searched_one(
    client: TestClient, worker: ProcessingWorker
) -> None:
    headers = register(client)
    project_a = make_project(client, headers, "Project A")
    project_b = make_project(client, headers, "Project B")
    document_a = add_notes(client, headers, project_a, "a.txt", TRANSPORT)
    document_b = add_notes(client, headers, project_b, "b.txt", TRANSPORT)
    worker.run_pending()

    results_a = search(client, headers, project_a, "transport costs market").json()
    results_b = search(client, headers, project_b, "transport costs market").json()

    assert {result["documentId"] for result in results_a} == {document_a}
    assert {result["documentId"] for result in results_b} == {document_b}


def test_other_users_cannot_search_a_project(client: TestClient, worker: ProcessingWorker) -> None:
    owner = register(client)
    project = make_project(client, owner, "Private")
    add_notes(client, owner, project, "notes.txt", TRANSPORT)
    worker.run_pending()
    stranger = register(client, email="eve@example.com")

    assert search(client, stranger, project, "transport").status_code == 404
    assert client.get(f"/api/v1/projects/{project}/search", params={"q": "x"}).status_code == 401


# ---------------------------------------------------------------- what is searchable


def test_unembedded_chunks_are_not_searched(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    add_notes(client, headers, project, "notes.txt", TRANSPORT)
    ProcessingWorker(session_factory, embeddings_enabled=False).run_pending()

    assert search(client, headers, project, "transport costs market").json() == []


def test_chunks_from_another_model_are_not_searched(
    client: TestClient, worker: ProcessingWorker, db: Session
) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    add_notes(client, headers, project, "notes.txt", TRANSPORT)
    worker.run_pending()
    db.execute(update(DocumentChunk).values(embedding_model="some-older-model"))
    db.commit()

    assert search(client, headers, project, "transport costs market").json() == []


def test_empty_project_returns_nothing(client: TestClient) -> None:
    headers = register(client)
    project = make_project(client, headers, "Empty")

    assert search(client, headers, project, "anything").json() == []


def test_search_reports_when_the_model_is_unavailable(client: TestClient) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            raise RuntimeError("model failed to load")

    headers = register(client)
    project = make_project(client, headers, "Farming")
    set_embedder(BrokenEmbedder())

    response = search(client, headers, project, "transport")

    assert response.status_code == 503


def test_service_min_score_can_be_overridden(
    client: TestClient, worker: ProcessingWorker, db: Session, fake_embedder: FakeEmbedder
) -> None:
    headers = register(client)
    project = make_project(client, headers, "Farming")
    add_notes(client, headers, project, "notes.txt", TRANSPORT)
    worker.run_pending()

    strict = search_project(db, project, "transport", fake_embedder, min_score=0.99)
    loose = search_project(db, project, "transport", fake_embedder, min_score=-1.0)

    assert strict == []
    assert [result.document_name for result in loose] == ["notes.txt"]
    assert db.scalars(select(DocumentChunk.project_id)).all() == [project]
