from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.llm import LLMUnavailable, set_llm
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Result
from app.models.result import RESULT_ANALYSIS
from app.services import analyzer
from app.services.processing_worker import GenerationWorker, ProcessingWorker
from app.services.storage import LocalStorage, set_storage
from tests.conftest import FakeLLM

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
def processor(session_factory: sessionmaker[Session]) -> ProcessingWorker:
    return ProcessingWorker(session_factory, embeddings_enabled=False)


@pytest.fixture
def generator(session_factory: sessionmaker[Session]) -> GenerationWorker:
    return GenerationWorker(session_factory)


def register(client: TestClient, email: str = "ada@example.com") -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": email, "password": "password123"},
    ).json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def make_project(client: TestClient, headers: dict, documents: dict[str, str]) -> str:
    project_id = client.post("/api/v1/projects", json={"name": "Farming"}, headers=headers).json()[
        "id"
    ]
    for name, text in documents.items():
        client.post(
            f"/api/v1/projects/{project_id}/documents",
            files={"files": (name, text.encode(), "text/plain")},
            headers=headers,
        )
    return project_id


DOCUMENTS = {
    "Interview_03.txt": "P03 said transport costs doubled and loans need title deeds.",
    "Interview_07.txt": "P07 said the cooperative shares one tractor and young people leave.",
}


def analysis_url(project_id: str) -> str:
    return f"/api/v1/projects/{project_id}/analysis"


def analyze_url(project_id: str) -> str:
    return f"/api/v1/projects/{project_id}/analyze"


# ---------------------------------------------------------------- the endpoints


def test_analysis_runs_in_the_background_and_covers_every_document(
    client: TestClient,
    processor: ProcessingWorker,
    generator: GenerationWorker,
    fake_llm: FakeLLM,
) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()

    started = client.post(analyze_url(project_id), headers=headers)
    assert started.status_code == 202
    assert started.json()["status"] == "processing"
    assert started.json()["progress"] == "Waiting to start"

    assert generator.run_pending() == 1

    body = client.get(analysis_url(project_id), headers=headers).json()
    assert body["status"] == "ready"
    assert body["content"]
    assert body["progress"] is None
    assert body["model"] == "fake-llm"
    assert [document["name"] for document in body["documents"]] == list(DOCUMENTS)

    # One note per document (no summaries existed), then the analysis itself.
    assert len(fake_llm.calls) == 3
    final = fake_llm.calls[-1]
    assert "## Major themes" in final["prompt"] and "## Overall conclusions" in final["prompt"]
    assert '<document name="Interview 03">' in final["prompt"]
    assert '<document name="Interview 07">' in final["prompt"]
    assert "not instructions" in final["system"]


def test_existing_summaries_are_reused(
    client: TestClient,
    processor: ProcessingWorker,
    generator: GenerationWorker,
    fake_llm: FakeLLM,
) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()
    documents = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()
    for document in documents:
        client.post(
            f"/api/v1/projects/{project_id}/documents/{document['id']}/summary", headers=headers
        )
    generator.run_pending()
    calls_before = len(fake_llm.calls)

    client.post(analyze_url(project_id), headers=headers)
    generator.run_pending()

    assert len(fake_llm.calls) - calls_before == 1  # only the final analysis call


def test_no_analysis_yet_is_a_404(client: TestClient) -> None:
    headers = register(client)
    project_id = make_project(client, headers, {})

    assert client.get(analysis_url(project_id), headers=headers).status_code == 404


def test_cannot_analyse_without_processed_documents(client: TestClient) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)  # uploaded, not processed yet

    response = client.post(analyze_url(project_id), headers=headers)

    assert response.status_code == 409
    assert "no processed documents" in response.json()["detail"]


def test_only_one_analysis_runs_at_a_time_and_a_rerun_replaces_it(
    client: TestClient,
    db: Session,
    processor: ProcessingWorker,
    generator: GenerationWorker,
) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()
    client.post(analyze_url(project_id), headers=headers)
    assert client.post(analyze_url(project_id), headers=headers).status_code == 409

    generator.run_pending()
    assert client.post(analyze_url(project_id), headers=headers).status_code == 202
    generator.run_pending()

    count = db.scalar(
        select(func.count()).select_from(Result).where(Result.result_type == RESULT_ANALYSIS)
    )
    assert count == 1


def test_llm_unavailable_fails_with_a_reason(
    client: TestClient, processor: ProcessingWorker, generator: GenerationWorker
) -> None:
    class DownLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            raise LLMUnavailable("Ollama is not running at http://localhost:11434.")

    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()
    set_llm(DownLLM())
    client.post(analyze_url(project_id), headers=headers)

    generator.run_pending()

    body = client.get(analysis_url(project_id), headers=headers).json()
    assert body["status"] == "failed"
    assert "Ollama is not running" in body["error"]
    assert body["content"] is None


def test_progress_is_reported_while_reading_documents(
    client: TestClient,
    session_factory: sessionmaker[Session],
    processor: ProcessingWorker,
    generator: GenerationWorker,
) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()
    client.post(analyze_url(project_id), headers=headers)
    seen: list[str] = []

    class WatchingLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            seen.append(client.get(analysis_url(project_id), headers=headers).json()["progress"])
            return super().generate(system, prompt, **kwargs)

    set_llm(WatchingLLM())
    generator.run_pending()

    assert seen == [
        "Reading documents (1 of 2): Interview_03.txt",
        "Reading documents (2 of 2): Interview_07.txt",
        "Writing the analysis",
    ]


def test_deleting_the_project_during_analysis_is_harmless(
    client: TestClient,
    session_factory: sessionmaker[Session],
    processor: ProcessingWorker,
    generator: GenerationWorker,
) -> None:
    headers = register(client)
    project_id = make_project(client, headers, DOCUMENTS)
    processor.run_pending()
    client.post(analyze_url(project_id), headers=headers)

    class DeletingLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            with session_factory() as other:
                other.execute(delete(Result).where(Result.result_type == RESULT_ANALYSIS))
                other.commit()
            return "notes"

    set_llm(DeletingLLM())
    generator.run_pending()  # must not raise

    assert client.get(analysis_url(project_id), headers=headers).status_code == 404


def test_other_users_cannot_run_or_read_the_analysis(
    client: TestClient, processor: ProcessingWorker
) -> None:
    owner = register(client)
    project_id = make_project(client, owner, DOCUMENTS)
    processor.run_pending()
    client.post(analyze_url(project_id), headers=owner)
    stranger = register(client, email="eve@example.com")

    assert client.post(analyze_url(project_id), headers=stranger).status_code == 404
    assert client.get(analysis_url(project_id), headers=stranger).status_code == 404


def test_the_analysis_only_uses_its_own_projects_documents(
    client: TestClient,
    processor: ProcessingWorker,
    generator: GenerationWorker,
    fake_llm: FakeLLM,
) -> None:
    headers = register(client)
    project_a = make_project(client, headers, {"a.txt": "Alpha project notes about farming."})
    project_b = client.post("/api/v1/projects", json={"name": "B"}, headers=headers).json()["id"]
    client.post(
        f"/api/v1/projects/{project_b}/documents",
        files={"files": ("secret.txt", b"Beta project secret clinic notes.", "text/plain")},
        headers=headers,
    )
    processor.run_pending()

    client.post(analyze_url(project_a), headers=headers)
    generator.run_pending()

    prompts = " ".join(call["prompt"] for call in fake_llm.calls)
    assert "Alpha project notes" in prompts
    assert "Beta project secret" not in prompts and "secret" not in prompts
    documents = client.get(analysis_url(project_a), headers=headers).json()["documents"]
    assert [document["name"] for document in documents] == ["a.txt"]


# ---------------------------------------------------------------- the prompt


def test_too_many_notes_are_condensed_first(
    fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(analyzer, "MAX_NOTES_CHARS", 200)
    notes = [(f"Interview {i}", "Transport and loans. " * 10) for i in range(8)]

    analyzer.write_analysis(fake_llm, notes)

    assert any("Merge them" in call["prompt"] for call in fake_llm.calls)
    assert "Analyse them together" in fake_llm.calls[-1]["prompt"]
    assert "8 documents" in fake_llm.calls[-1]["prompt"]


def test_single_document_prompt_mentions_differences_rule(fake_llm: FakeLLM) -> None:
    analyzer.write_analysis(fake_llm, [("Interview 03", "Transport is expensive.")])

    prompt = fake_llm.calls[-1]["prompt"]
    assert "1 document " in prompt
    assert "differences cannot be assessed" in prompt
