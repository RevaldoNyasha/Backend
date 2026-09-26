from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Document
from app.services.storage import LocalStorage, set_storage

MB = 1024 * 1024


@pytest.fixture(autouse=True)
def uploads_root(tmp_path: Path) -> Iterator[Path]:
    root = tmp_path / "uploads"
    root.mkdir()
    set_storage(LocalStorage(root))
    yield root
    set_storage(None)


@pytest.fixture
def db_session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
    )
    session = session_factory()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


@pytest.fixture
def client(db_session: Session) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def auth_headers(client: TestClient, email: str = "john@example.com") -> dict[str, str]:
    client.post(
        "/api/v1/auth/register",
        json={"name": "John Doe", "email": email, "password": "password123"},
    )
    login_response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert login_response.status_code == 200
    return {"Authorization": f"Bearer {login_response.json()['accessToken']}"}


def make_project(
    client: TestClient,
    email: str = "john@example.com",
    name: str = "Healthcare Access Study",
) -> tuple[str, dict]:
    headers = auth_headers(client, email=email)
    response = client.post(
        "/api/v1/projects",
        json={"name": name, "description": "d"},
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()["id"], headers


def upload(
    client: TestClient,
    project_id: str,
    headers: dict[str, str],
    *files: tuple[str, int],
) -> object:
    parts = [("files", (name, b"x" * size, "application/octet-stream")) for name, size in files]
    return client.post(
        f"/api/v1/projects/{project_id}/documents",
        files=parts,
        headers=headers,
    )


def test_upload_document_succeeds(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("Interview_01.docx", 64))

    assert response.status_code == 202
    body = response.json()
    assert isinstance(body, list)
    assert len(body) == 1
    document = body[0]
    assert document["projectId"] == project_id
    assert document["name"] == "Interview_01.docx"
    assert document["kind"] == "docx"
    assert document["type"] == "Interview"
    assert document["extension"] == "DOCX"
    assert document["fileSize"] == "64 B"
    assert document["fileSizeBytes"] == 64
    assert document["status"] == "processing"
    assert document["participantCount"] == 0
    assert document["language"] is None
    assert document["uploadedAt"].endswith("Z")
    assert document["updatedAt"].endswith("Z")
    assert "storagePath" not in document


def test_upload_maps_every_accepted_extension(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(
        client,
        project_id,
        headers,
        ("a.pdf", 8),
        ("b.docx", 8),
        ("c.csv", 8),
        ("d.mp3", 8),
        ("e.wav", 8),
        ("f.mp4", 8),
        ("g.m4a", 8),
        ("h.mov", 8),
        ("i.wma", 8),
    )

    assert response.status_code == 202
    actual = [(item["extension"], item["kind"], item["type"]) for item in response.json()]
    assert actual == [
        ("PDF", "pdf", "Research Notes"),
        ("DOCX", "docx", "Interview"),
        ("CSV", "csv", "Research Notes"),
        ("MP3", "audio", "Interview"),
        ("WAV", "audio", "Interview"),
        ("MP4", "video", "Interview"),
        ("M4A", "audio", "Interview"),
        ("MOV", "video", "Interview"),
        ("WMA", "audio", "Interview"),
    ]


def test_upload_returns_a_record_per_file(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(
        client,
        project_id,
        headers,
        ("a.pdf", 8),
        ("b.mp3", 8),
        ("c.csv", 8),
    )

    assert response.status_code == 202
    assert [item["name"] for item in response.json()] == ["a.pdf", "b.mp3", "c.csv"]
    assert len({item["id"] for item in response.json()}) == 3


def test_upload_formats_file_size_for_display(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(
        client,
        project_id,
        headers,
        ("small.pdf", 128 * 1024),
        ("medium.mp3", 1_300_000),
        ("large.wav", 10 * MB),
    )

    assert [item["fileSize"] for item in response.json()] == ["128 KB", "1.2 MB", "10 MB"]


def test_upload_writes_content_to_disk(
    client: TestClient,
    db_session: Session,
    uploads_root: Path,
) -> None:
    project_id, headers = make_project(client)
    content = b"research notes"

    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("notes.pdf", content, "application/pdf")},
        headers=headers,
    )

    assert response.status_code == 202
    record = db_session.get(Document, response.json()[0]["id"])
    assert record is not None
    stored = uploads_root / record.storage_path
    assert stored.read_bytes() == content
    assert stored.parent.name == project_id


def test_upload_uses_random_storage_names(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("notes.pdf", 8), ("notes.pdf", 8))

    assert response.status_code == 202
    body = response.json()
    assert [item["name"] for item in body] == ["notes.pdf", "notes.pdf"]
    assert body[0]["id"] != body[1]["id"]


def test_upload_strips_directory_components(
    client: TestClient,
    db_session: Session,
    uploads_root: Path,
) -> None:
    project_id, headers = make_project(client)

    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("../../etc/passwd.pdf", b"nope", "application/pdf")},
        headers=headers,
    )

    assert response.status_code == 202
    assert response.json()[0]["name"] == "passwd.pdf"
    record = db_session.get(Document, response.json()[0]["id"])
    assert record is not None
    assert ".." not in record.storage_path
    assert uploads_root in (uploads_root / record.storage_path).resolve().parents


def test_upload_accepts_plain_text(client: TestClient, uploads_root: Path) -> None:
    project_id, headers = make_project(client)

    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("field_notes.txt", b"Day 1: spoke to three farmers.", "text/plain")},
        headers=headers,
    )

    assert response.status_code == 202
    document = response.json()[0]
    assert document["kind"] == "notes"
    assert document["type"] == "Research Notes"
    assert document["extension"] == "TXT"
    assert len(list((uploads_root / project_id).iterdir())) == 1


def test_plain_text_uses_the_document_size_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("huge.txt", 5 * MB + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == "Notes files are limited to 5 MB"


def test_failed_save_leaves_no_file_behind(
    client: TestClient,
    db_session: Session,
    uploads_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_id, headers = make_project(client)

    def broken_commit() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(db_session, "commit", broken_commit)

    with pytest.raises(RuntimeError):
        upload(client, project_id, headers, ("notes.pdf", 8))

    assert not (uploads_root / project_id).exists() or not any(
        (uploads_root / project_id).iterdir()
    )


def test_upload_rejects_unsupported_extension(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("malware.exe", 8))

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported file type"


def test_upload_rejects_empty_file(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("empty.pdf", 0))

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty"


def test_upload_rejects_document_over_the_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("big.pdf", 5 * MB + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == "Pdf files are limited to 5 MB"


def test_upload_accepts_document_at_the_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("max.pdf", 5 * MB))

    assert response.status_code == 202
    assert response.json()[0]["fileSizeBytes"] == 5 * MB


def test_upload_rejects_audio_over_the_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    limit = settings.upload_max_audio_size_mb
    response = upload(client, project_id, headers, ("long.mp3", limit * MB + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == f"Audio files are limited to {limit} MB"


def test_upload_accepts_audio_at_the_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    limit = settings.upload_max_audio_size_mb
    response = upload(client, project_id, headers, ("max.mp3", limit * MB))

    assert response.status_code == 202
    assert response.json()[0]["fileSizeBytes"] == limit * MB


def test_upload_rejects_video_over_the_limit(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("long.mp4", 100 * MB + 1))

    assert response.status_code == 413
    assert response.json()["detail"] == "Video files are limited to 100 MB"


def test_audio_video_and_document_limits_are_independent(client: TestClient) -> None:
    project_id, headers = make_project(client)
    six_mb = 6 * MB

    accepted = upload(
        client,
        project_id,
        headers,
        ("interview.mp3", six_mb),
        ("session.mp4", six_mb),
    )
    rejected = upload(client, project_id, headers, ("paper.pdf", six_mb))

    assert [item["name"] for item in accepted.json()] == ["interview.mp3", "session.mp4"]
    assert rejected.status_code == 413


def test_upload_succeeds_for_accepted_files_in_a_mixed_batch(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(
        client,
        project_id,
        headers,
        ("good.pdf", 8),
        ("bad.exe", 8),
        ("also-good.mp3", 8),
    )

    assert response.status_code == 202
    assert [item["name"] for item in response.json()] == ["good.pdf", "also-good.mp3"]


def test_upload_rejects_the_batch_when_every_file_fails(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = upload(client, project_id, headers, ("a.exe", 8), ("b.exe", 8))

    assert response.status_code == 415


def test_upload_stores_nothing_when_rejected(client: TestClient, db_session: Session) -> None:
    project_id, headers = make_project(client)

    upload(client, project_id, headers, ("big.pdf", 5 * MB + 1))

    assert list(db_session.scalars(select(Document))) == []


def test_upload_requires_token(client: TestClient) -> None:
    project_id, _ = make_project(client)

    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("notes.pdf", b"x", "application/pdf")},
    )

    assert response.status_code == 401


def test_upload_to_unknown_project_returns_not_found(client: TestClient) -> None:
    _, headers = make_project(client)

    response = upload(
        client,
        "00000000-0000-0000-0000-000000000000",
        headers,
        ("notes.pdf", 8),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_upload_to_other_users_project_returns_not_found(client: TestClient) -> None:
    project_id, _ = make_project(client, email="first@example.com")
    _, second_headers = make_project(client, email="second@example.com")

    response = upload(client, project_id, second_headers, ("notes.pdf", 8))

    assert response.status_code == 404


def test_upload_bumps_project_updated_at(client: TestClient) -> None:
    project_id, headers = make_project(client)
    before = client.get(f"/api/v1/projects/{project_id}", headers=headers).json()["updatedAt"]

    upload(client, project_id, headers, ("notes.pdf", 8))

    after = client.get(f"/api/v1/projects/{project_id}", headers=headers).json()["updatedAt"]
    assert after >= before


def test_project_counts_reflect_uploaded_documents(client: TestClient) -> None:
    project_id, headers = make_project(client)
    upload(client, project_id, headers, ("a.pdf", 8), ("b.mp3", 8), ("c.docx", 8))

    project = client.get(f"/api/v1/projects/{project_id}", headers=headers).json()

    assert project["documentCount"] == 3
    assert project["interviewCount"] == 0
    assert project["focusGroupCount"] == 0


def test_list_documents_returns_uploads_newest_first(client: TestClient) -> None:
    project_id, headers = make_project(client)
    upload(client, project_id, headers, ("first.pdf", 8), ("second.pdf", 8))

    response = client.get(f"/api/v1/projects/{project_id}/documents", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert [item["name"] for item in body] == ["second.pdf", "first.pdf"]
    assert all("storagePath" not in item for item in body)


def test_list_documents_is_empty_for_new_project(client: TestClient) -> None:
    project_id, headers = make_project(client)

    assert client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json() == []


def test_list_documents_requires_token(client: TestClient) -> None:
    project_id, _ = make_project(client)

    response = client.get(f"/api/v1/projects/{project_id}/documents")

    assert response.status_code == 401


def test_list_documents_of_other_users_project_returns_not_found(client: TestClient) -> None:
    project_id, _ = make_project(client, email="first@example.com")
    _, second_headers = make_project(client, email="second@example.com")

    response = client.get(f"/api/v1/projects/{project_id}/documents", headers=second_headers)

    assert response.status_code == 404


def test_delete_document_removes_row_and_disk_copy(
    client: TestClient,
    db_session: Session,
    uploads_root: Path,
) -> None:
    project_id, headers = make_project(client)
    document_id = upload(client, project_id, headers, ("notes.pdf", 8)).json()[0]["id"]
    record = db_session.get(Document, document_id)
    assert record is not None
    stored = uploads_root / record.storage_path
    assert stored.exists()

    response = client.delete(
        f"/api/v1/projects/{project_id}/documents/{document_id}",
        headers=headers,
    )

    assert response.status_code == 204
    assert response.content == b""
    assert not stored.exists()
    assert db_session.get(Document, document_id) is None


def test_delete_project_removes_its_documents_and_files(
    client: TestClient,
    db_session: Session,
    uploads_root: Path,
) -> None:
    project_id, headers = make_project(client)
    other_project_id, _ = make_project(client, name="Other Study")
    ids = [
        document["id"]
        for document in upload(client, project_id, headers, ("a.pdf", 8), ("b.mp3", 8)).json()
    ]
    kept_id = upload(client, other_project_id, headers, ("c.pdf", 8)).json()[0]["id"]
    assert (uploads_root / project_id).is_dir()

    response = client.delete(f"/api/v1/projects/{project_id}", headers=headers)

    assert response.status_code == 204
    db_session.expire_all()
    assert all(db_session.get(Document, document_id) is None for document_id in ids)
    assert not (uploads_root / project_id).exists()
    assert db_session.get(Document, kept_id) is not None
    assert (uploads_root / other_project_id).is_dir()


def test_delete_document_decrements_project_count(client: TestClient) -> None:
    project_id, headers = make_project(client)
    document_id = upload(client, project_id, headers, ("a.pdf", 8)).json()[0]["id"]

    client.delete(f"/api/v1/projects/{project_id}/documents/{document_id}", headers=headers)

    project = client.get(f"/api/v1/projects/{project_id}", headers=headers).json()
    assert project["documentCount"] == 0


def test_delete_document_returns_not_found_for_unknown_id(client: TestClient) -> None:
    project_id, headers = make_project(client)

    response = client.delete(
        f"/api/v1/projects/{project_id}/documents/00000000-0000-0000-0000-000000000000",
        headers=headers,
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found"


def test_delete_document_cannot_target_other_users_document(
    client: TestClient,
    db_session: Session,
) -> None:
    project_id, first_headers = make_project(client, email="first@example.com")
    _, second_headers = make_project(client, email="second@example.com")
    document_id = upload(client, project_id, first_headers, ("a.pdf", 8)).json()[0]["id"]

    response = client.delete(
        f"/api/v1/projects/{project_id}/documents/{document_id}",
        headers=second_headers,
    )

    assert response.status_code == 404
    assert db_session.get(Document, document_id) is not None


def test_delete_document_cannot_target_a_document_in_another_project(client: TestClient) -> None:
    first_project, headers = make_project(client, name="First Study")
    second_project, _ = make_project(client, name="Second Study")
    document_id = upload(client, first_project, headers, ("a.pdf", 8)).json()[0]["id"]

    response = client.delete(
        f"/api/v1/projects/{second_project}/documents/{document_id}",
        headers=headers,
    )

    assert response.status_code == 404


def test_delete_document_requires_token(client: TestClient) -> None:
    project_id, headers = make_project(client)
    document_id = upload(client, project_id, headers, ("a.pdf", 8)).json()[0]["id"]

    response = client.delete(f"/api/v1/projects/{project_id}/documents/{document_id}")

    assert response.status_code == 401
