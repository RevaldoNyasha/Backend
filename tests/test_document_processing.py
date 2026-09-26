import io
import threading
import time
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Document, DocumentChunk, Result
from app.services.document_processor import ExtractionError, clean_text, extract_text
from app.services.processing_worker import ProcessingWorker
from app.services.storage import LocalStorage, set_storage
from app.utils.chunking import chunk_text

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
    return ProcessingWorker(session_factory)


def auth_headers(client: TestClient, email: str = "ada@example.com") -> dict[str, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": email, "password": "password123"},
    )
    return {"Authorization": f"Bearer {response.json()['accessToken']}"}


def make_project(client: TestClient, headers: dict[str, str], name: str = "Study") -> str:
    return client.post("/api/v1/projects", json={"name": name}, headers=headers).json()["id"]


def upload(
    client: TestClient, project_id: str, headers: dict[str, str], name: str, content: bytes
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": (name, content, "application/octet-stream")},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    return response.json()[0]


def make_pdf(*pages: str) -> bytes:
    pdf = pymupdf.open()
    for text in pages:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)
    return pdf.tobytes()


def make_docx(paragraphs: list[str], table: list[list[str]] | None = None) -> bytes:
    docx = DocxDocument()
    for paragraph in paragraphs:
        docx.add_paragraph(paragraph)
    if table:
        grid = docx.add_table(rows=len(table), cols=len(table[0]))
        for row_cells, values in zip(grid.rows, table, strict=True):
            for cell, value in zip(row_cells.cells, values, strict=True):
                cell.text = value
    buffer = io.BytesIO()
    docx.save(buffer)
    return buffer.getvalue()


# ---------------------------------------------------------------- extraction


def test_pdf_text_is_extracted_per_page() -> None:
    extracted = extract_text("pdf", make_pdf("Transport is expensive.", "Clinics are far."))

    assert [page.number for page in extracted.pages] == [1, 2]
    assert "Transport is expensive." in extracted.pages[0].text
    assert "Clinics are far." in extracted.pages[1].text


def test_pdf_without_text_is_rejected_with_a_clear_reason() -> None:
    with pytest.raises(ExtractionError, match="Scanned"):
        extract_text("pdf", make_pdf(""))


def test_unreadable_pdf_is_rejected() -> None:
    with pytest.raises(ExtractionError, match="not a readable PDF"):
        extract_text("pdf", b"definitely not a pdf")


def test_docx_paragraphs_and_tables_are_extracted_in_order() -> None:
    content = make_docx(
        ["Interview with P01.", "She described the clinic queue."],
        table=[["Question", "Answer"], ["Distance", "12 km"]],
    )

    text = extract_text("docx", content).full_text

    assert text.index("Interview with P01.") < text.index("Question | Answer")
    assert "Distance | 12 km" in text


def test_unreadable_docx_is_rejected() -> None:
    with pytest.raises(ExtractionError, match="not a readable DOCX"):
        extract_text("docx", b"PK not really a zip")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "cp1252", "utf-16"])
def test_plain_text_is_decoded(encoding: str) -> None:
    text = extract_text("notes", "Café visit — notes".encode(encoding)).full_text

    assert text == "Café visit — notes"


def test_empty_text_file_is_rejected() -> None:
    with pytest.raises(ExtractionError, match="No text"):
        extract_text("notes", b"   \n\n  ")


def test_csv_rows_become_header_value_lines() -> None:
    content = b"participant,age,answer\nP01,19,Too expensive\nP02,,Too far\n"

    text = extract_text("csv", content).full_text

    assert "participant: P01 | age: 19 | answer: Too expensive" in text
    assert "participant: P02 | answer: Too far" in text


def test_audio_is_not_extracted_here() -> None:
    with pytest.raises(ExtractionError, match="not supported"):
        extract_text("audio", b"ID3...")


def test_clean_text_normalises_whitespace_and_hyphenation() -> None:
    raw = "Access to  transpor-\ntation\r\n\r\n\r\n\r\nis   limited.\x00  "

    assert clean_text(raw) == "Access to transportation\n\nis limited."


# ---------------------------------------------------------------- chunking


def test_chunks_are_exact_slices_that_cover_the_text() -> None:
    text = " ".join(f"Sentence number {i} talks about transport costs." for i in range(80))

    chunks = chunk_text(text, target_chars=300, overlap_chars=60)

    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text
        assert len(chunk.text) <= 300
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    for previous, current in pairwise(chunks):
        assert current.start > previous.start  # always moves forward
        assert current.start <= previous.end  # overlapping or touching: nothing skipped


def test_chunks_prefer_paragraph_and_sentence_boundaries() -> None:
    text = "First paragraph sentence one. Sentence two.\n\nSecond paragraph here."

    chunks = chunk_text(text, target_chars=45, overlap_chars=0)

    assert [chunk.text for chunk in chunks] == [
        "First paragraph sentence one. Sentence two.",
        "Second paragraph here.",
    ]


def test_very_long_sentences_are_split() -> None:
    text = "word " * 2000

    chunks = chunk_text(text, target_chars=500, overlap_chars=0)

    assert all(len(chunk.text) <= 500 for chunk in chunks)
    assert chunks[-1].end == len(text.rstrip())


def test_empty_text_has_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("  \n\n ") == []


# ---------------------------------------------------------------- processing pipeline


def test_upload_is_processed_into_text_and_chunks(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    body = "\n\n".join(f"Paragraph {i}. Participants mentioned transport costs." for i in range(60))
    document = upload(client, project_id, headers, "notes.txt", body.encode())
    assert document["status"] == "processing"

    assert worker.run_pending() == 1

    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "processed"
    assert listed["processingError"] is None

    chunks = db.scalars(
        select(DocumentChunk).where(DocumentChunk.document_id == document["id"])
    ).all()
    assert len(chunks) > 1
    assert all(chunk.project_id == project_id for chunk in chunks)
    assert [chunk.chunk_index for chunk in sorted(chunks, key=lambda c: c.chunk_index)] == list(
        range(len(chunks))
    )

    text = client.get(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/text", headers=headers
    )
    assert text.status_code == 200
    assert text.json()["chunkCount"] == len(chunks)
    assert "Participants mentioned transport costs." in text.json()["text"]


def test_pdf_chunks_remember_their_page(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(
        client, project_id, headers, "paper.pdf", make_pdf("Page one text.", "Page two text.")
    )

    worker.run_pending()

    pages = db.scalars(
        select(DocumentChunk.page)
        .where(DocumentChunk.document_id == document["id"])
        .order_by(DocumentChunk.chunk_index)
    ).all()
    assert pages == [1, 2]


def test_text_is_unavailable_before_processing(client: TestClient) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(client, project_id, headers, "notes.txt", b"Some notes.")

    response = client.get(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/text", headers=headers
    )

    assert response.status_code == 409


def test_unreadable_file_fails_with_a_reason(client: TestClient, worker: ProcessingWorker) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    upload(client, project_id, headers, "broken.pdf", b"not a pdf")

    worker.run_pending()

    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "failed"
    assert listed["processingError"] == "This file is not a readable PDF"


def test_missing_stored_file_fails_with_a_reason(
    client: TestClient, worker: ProcessingWorker, uploads_root: Path
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    upload(client, project_id, headers, "notes.txt", b"Some notes.")
    for stored in (uploads_root / project_id).iterdir():
        stored.unlink()

    worker.run_pending()

    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "failed"
    assert "missing" in listed["processingError"]


def test_audio_is_left_for_the_transcription_worker(
    client: TestClient, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    upload(client, project_id, headers, "interview.mp3", b"ID3 audio bytes")

    assert worker.run_pending() == 0

    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "processing"


def test_reprocess_queues_a_failed_document_and_it_can_succeed(
    client: TestClient, worker: ProcessingWorker, uploads_root: Path
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(client, project_id, headers, "notes.txt", b"   ")
    worker.run_pending()
    assert (
        client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]["status"]
        == "failed"
    )

    # Fix the stored file, then retry.
    stored = next((uploads_root / project_id).iterdir())
    stored.write_bytes(b"Now there is real text.")
    response = client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/reprocess", headers=headers
    )
    assert response.status_code == 202
    assert response.json()["status"] == "processing"
    assert response.json()["processingError"] is None

    worker.run_pending()
    listed = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[0]
    assert listed["status"] == "processed"


def test_reprocess_replaces_previous_output(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(client, project_id, headers, "notes.txt", b"Short text.")
    worker.run_pending()
    client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/reprocess", headers=headers
    )
    worker.run_pending()

    results = db.scalar(
        select(func.count()).select_from(Result).where(Result.document_id == document["id"])
    )
    chunks = db.scalar(
        select(func.count())
        .select_from(DocumentChunk)
        .where(DocumentChunk.document_id == document["id"])
    )
    assert (results, chunks) == (1, 1)


def test_reprocess_rejects_a_document_that_is_already_processing(client: TestClient) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(client, project_id, headers, "notes.txt", b"Text.")

    response = client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/reprocess", headers=headers
    )

    assert response.status_code == 409


def test_other_users_cannot_reprocess_or_read_text(
    client: TestClient, worker: ProcessingWorker
) -> None:
    owner = auth_headers(client)
    project_id = make_project(client, owner)
    document = upload(client, project_id, owner, "notes.txt", b"Private notes.")
    worker.run_pending()
    stranger = auth_headers(client, email="eve@example.com")
    base = f"/api/v1/projects/{project_id}/documents/{document['id']}"

    assert client.post(f"{base}/reprocess", headers=stranger).status_code == 404
    assert client.get(f"{base}/text", headers=stranger).status_code == 404


def test_deleting_a_document_removes_its_chunks_and_results(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    document = upload(client, project_id, headers, "notes.txt", b"Some text to keep.")
    worker.run_pending()

    client.delete(f"/api/v1/projects/{project_id}/documents/{document['id']}", headers=headers)

    assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 0
    assert db.scalar(select(func.count()).select_from(Result)) == 0


def test_deleting_a_project_removes_its_chunks_and_results(
    client: TestClient, db: Session, worker: ProcessingWorker
) -> None:
    headers = auth_headers(client)
    project_id = make_project(client, headers)
    upload(client, project_id, headers, "notes.txt", b"Some text to keep.")
    worker.run_pending()

    client.delete(f"/api/v1/projects/{project_id}", headers=headers)

    assert db.scalar(select(func.count()).select_from(Document)) == 0
    assert db.scalar(select(func.count()).select_from(DocumentChunk)) == 0
    assert db.scalar(select(func.count()).select_from(Result)) == 0


def test_background_thread_processes_uploads(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    worker = ProcessingWorker(session_factory, poll_seconds=0.05)
    worker.start()
    try:
        headers = auth_headers(client)
        project_id = make_project(client, headers)
        upload(client, project_id, headers, "notes.txt", b"Processed in the background.")

        deadline = time.monotonic() + 10
        status = "processing"
        while status == "processing" and time.monotonic() < deadline:
            time.sleep(0.05)
            status = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()[
                0
            ]["status"]
        assert status == "processed"
    finally:
        worker.stop()
    assert not any(t.name == "document-processing" and t.is_alive() for t in threading.enumerate())
