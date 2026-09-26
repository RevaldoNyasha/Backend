import io
from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.llm import LLMUnavailable, set_llm
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.services.processing_worker import GenerationWorker, ProcessingWorker
from app.services.report_templates import TEMPLATES
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
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def generator(session_factory: sessionmaker[Session]) -> GenerationWorker:
    return GenerationWorker(session_factory)


DOCUMENTS = {
    "Interview_03.txt": "P03 said transport costs doubled and loans need title deeds.",
    "Interview_07.txt": "P07 said the cooperative shares one tractor and young people leave.",
}

REPORT_BODY = (
    "## Executive Summary\n\nTransport and equipment are the main barriers. They affect "
    "every farm.\n\n## Methodology\n\nTwo interviews were read.\n\n## Major Themes\n\n"
    "- **Transport costs**: costs doubled (Interview 03).\n- **Shared equipment**: one "
    "tractor (Interview 07).\n\n## Evidence\n\n- *transport costs doubled* (Interview 03)\n\n"
    "## Conclusions\n\nCosts and equipment limit farmers."
)


@pytest.fixture
def project(
    client: TestClient, session_factory: sessionmaker[Session]
) -> tuple[dict[str, str], str, list[str]]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "password123"},
    ).json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    project_id = client.post("/api/v1/projects", json={"name": "Farming"}, headers=headers).json()[
        "id"
    ]
    ids = [
        client.post(
            f"/api/v1/projects/{project_id}/documents",
            files={"files": (name, text.encode(), "text/plain")},
            headers=headers,
        ).json()[0]["id"]
        for name, text in DOCUMENTS.items()
    ]
    ProcessingWorker(session_factory, embeddings_enabled=False).run_pending()
    return headers, project_id, ids


def reports_url(project_id: str) -> str:
    return f"/api/v1/projects/{project_id}/reports"


def generate(client: TestClient, headers: dict, project_id: str, **body) -> dict:
    response = client.post(reports_url(project_id), json=body, headers=headers)
    assert response.status_code == 202, response.text
    return response.json()


# ---------------------------------------------------------------- templates


def test_templates_are_listed_with_their_sections(client: TestClient, project) -> None:
    headers, _, _ = project

    templates = client.get("/api/v1/report-templates", headers=headers).json()

    assert [t["type"] for t in templates] == [
        "thematic_analysis",
        "evidence_summary",
        "comparison",
        "custom",
    ]
    assert templates[0]["sections"] == [
        "Executive Summary",
        "Methodology",
        "Major Themes",
        "Evidence",
        "Conclusions",
    ]
    assert templates[3]["customSections"] is True and templates[3]["sections"] == []


# ---------------------------------------------------------------- generating


def test_report_is_generated_in_the_background_from_the_template(
    client: TestClient, project, generator: GenerationWorker, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = project
    fake_llm.replies = ["P03 notes.", "P07 notes.", REPORT_BODY]

    draft = generate(client, headers, project_id, type="thematic_analysis")

    assert draft["status"] == "draft"
    assert draft["generation"]["state"] == "processing"
    assert draft["title"] == "Farming — Thematic Analysis"
    assert draft["sections"][0] == "Executive Summary"
    assert [d["id"] for d in draft["documents"]] == ids

    generator.run_pending()

    report = client.get(f"{reports_url(project_id)}/{draft['id']}", headers=headers).json()
    assert report["generation"]["state"] == "ready"
    assert report["content"] == REPORT_BODY
    # The summary states the real document count, then the report's first sentence.
    assert report["summary"] == (
        "A thematic analysis of 2 documents: Transport and equipment are the main barriers."
    )
    prompt = fake_llm.calls[-1]["prompt"]
    for heading in TEMPLATES["thematic_analysis"].headings():
        assert f"## {heading}\n" in prompt
    assert "Interview 03" in prompt and "Interview 07" in prompt
    assert "[1]" in prompt  # only as the instruction not to use citation markers

    listed = client.get(reports_url(project_id), headers=headers).json()
    assert [r["id"] for r in listed] == [draft["id"]]
    assert "content" not in listed[0] or listed[0]["content"] is None


def test_custom_report_uses_the_given_sections_and_documents(
    client: TestClient, project, generator: GenerationWorker, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = project

    draft = generate(
        client,
        headers,
        project_id,
        type="custom",
        title="Transport brief",
        sections=["## Costs", "Roads", "Costs"],
        documentIds=[ids[0]],
    )
    generator.run_pending()

    assert draft["title"] == "Transport brief"
    assert draft["sections"] == ["Costs", "Roads"]  # cleaned and de-duplicated
    assert [d["id"] for d in draft["documents"]] == [ids[0]]
    prompt = fake_llm.calls[-1]["prompt"]
    assert "## Costs\n" in prompt and "## Roads\n" in prompt
    assert "Interview 07" not in prompt


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"type": "novel"}, "Unknown report type"),
        ({"type": "custom"}, "at least one section"),
        ({"documentIds": ["not-a-document"]}, "not processed documents"),
    ],
)
def test_invalid_requests_are_refused(client: TestClient, project, body, message) -> None:
    headers, project_id, _ = project

    response = client.post(reports_url(project_id), json=body, headers=headers)

    assert response.status_code == 409
    assert message in response.json()["detail"]


def test_a_project_without_processed_documents_cannot_have_a_report(client: TestClient) -> None:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Eve", "email": "eve@example.com", "password": "password123"},
    ).json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    project_id = client.post("/api/v1/projects", json={"name": "Empty"}, headers=headers).json()[
        "id"
    ]

    response = client.post(reports_url(project_id), json={}, headers=headers)

    assert response.status_code == 409
    assert "no processed documents" in response.json()["detail"]


def test_llm_failure_marks_the_report_failed(
    client: TestClient, project, generator: GenerationWorker
) -> None:
    class DownLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            raise LLMUnavailable("Groq is not configured.")

    headers, project_id, _ = project
    set_llm(DownLLM())
    draft = generate(client, headers, project_id)

    generator.run_pending()

    report = client.get(f"{reports_url(project_id)}/{draft['id']}", headers=headers).json()
    assert report["generation"]["state"] == "failed"
    assert report["generation"]["error"] == "Groq is not configured."


# ---------------------------------------------------------------- managing


def test_mark_final_rename_and_delete(
    client: TestClient, project, generator: GenerationWorker, fake_llm: FakeLLM
) -> None:
    headers, project_id, _ = project
    draft = generate(client, headers, project_id)
    url = f"{reports_url(project_id)}/{draft['id']}"

    too_early = client.patch(url, json={"status": "final"}, headers=headers)
    assert too_early.status_code == 422  # still being written

    generator.run_pending()
    updated = client.patch(url, json={"status": "final", "title": "Final"}, headers=headers).json()
    assert (updated["status"], updated["title"]) == ("final", "Final")

    assert client.delete(url, headers=headers).status_code == 204
    assert client.get(url, headers=headers).status_code == 404


def test_reports_are_private_to_the_project_owner(client: TestClient, project) -> None:
    headers, project_id, _ = project
    draft = generate(client, headers, project_id)
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Eve", "email": "eve@example.com", "password": "password123"},
    ).json()["accessToken"]
    stranger = {"Authorization": f"Bearer {token}"}

    assert client.get(reports_url(project_id), headers=stranger).status_code == 404
    assert (
        client.get(f"{reports_url(project_id)}/{draft['id']}", headers=stranger).status_code
        == 404
    )


# ---------------------------------------------------------------- export


def test_export_pdf_and_docx(
    client: TestClient, project, generator: GenerationWorker, fake_llm: FakeLLM
) -> None:
    headers, project_id, _ = project
    fake_llm.replies = ["P03 notes.", "P07 notes.", REPORT_BODY]
    draft = generate(client, headers, project_id, title="Farm barriers")
    url = f"{reports_url(project_id)}/{draft['id']}/export"

    not_ready = client.get(url, params={"format": "pdf"}, headers=headers)
    assert not_ready.status_code == 409

    generator.run_pending()

    pdf = client.get(url, params={"format": "pdf"}, headers=headers)
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert 'filename="Farm_barriers.pdf"' in pdf.headers["content-disposition"]
    text = "".join(page.get_text() for page in pymupdf.open(stream=pdf.content, filetype="pdf"))
    assert "Farm barriers" in text
    assert "Executive Summary" in text and "Transport costs" in text
    assert "**" not in text and "##" not in text  # Markdown was rendered, not printed

    docx = client.get(url, params={"format": "docx"}, headers=headers)
    assert docx.status_code == 200
    assert "wordprocessingml" in docx.headers["content-type"]
    paragraphs = DocxDocument(io.BytesIO(docx.content)).paragraphs
    styles = {p.text: p.style.name for p in paragraphs}
    assert styles["Farm barriers"] == "Title"
    assert styles["Executive Summary"].startswith("Heading")
    bullet = next(p for p in paragraphs if p.text.startswith("Transport costs"))
    assert bullet.style.name == "List Bullet"
    assert bullet.runs[0].bold is True  # **Transport costs**
    assert all("**" not in p.text for p in paragraphs)

    assert client.get(url, params={"format": "odt"}, headers=headers).status_code == 422
