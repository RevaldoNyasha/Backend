from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.whisper import Segment, WhisperTranscriber
from app.core.config import settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import DocumentChunk, Result
from app.services.processing_worker import ProcessingWorker, TranscriptionWorker
from app.services.storage import LocalStorage, set_storage
from app.services.transcription import format_timestamp, timed_chunks
from tests.conftest import FakeLLM, FakeTranscriber

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
def transcriber_worker(session_factory: sessionmaker[Session]) -> TranscriptionWorker:
    return TranscriptionWorker(session_factory)


@pytest.fixture
def embedder_worker(session_factory: sessionmaker[Session]) -> ProcessingWorker:
    return ProcessingWorker(session_factory, embeddings_enabled=True)


@pytest.fixture
def account(client: TestClient) -> tuple[dict[str, str], str]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "password123"},
    ).json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    project_id = client.post("/api/v1/projects", json={"name": "Farming"}, headers=headers).json()[
        "id"
    ]
    return headers, project_id


def upload(client: TestClient, headers: dict, project_id: str, name: str) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": (name, b"fake audio bytes", "application/octet-stream")},
        headers=headers,
    )
    assert response.status_code == 202, response.text
    return response.json()[0]


def documents(client: TestClient, headers: dict, project_id: str) -> list[dict]:
    return client.get(f"/api/v1/projects/{project_id}/documents", headers=headers).json()


# ---------------------------------------------------------------- the pipeline


@pytest.mark.parametrize(
    "name", ["interview.mp3", "focus group.m4a", "recorder.wma", "session.mp4"]
)
def test_recording_is_transcribed_into_timed_chunks(
    client: TestClient,
    account,
    transcriber_worker: TranscriptionWorker,
    fake_transcriber: FakeTranscriber,
    session_factory: sessionmaker[Session],
    name: str,
) -> None:
    headers, project_id = account
    document = upload(client, headers, project_id, name)
    assert document["status"] == "processing"

    assert transcriber_worker.run_pending() == 1

    [listed] = documents(client, headers, project_id)
    assert listed["status"] == "processed"
    assert listed["language"] == "en"
    assert listed["processingProgress"] is None
    assert fake_transcriber.calls[0]["bytes"] == len(b"fake audio bytes")

    with session_factory() as db:
        chunks = db.scalars(select(DocumentChunk).order_by(DocumentChunk.chunk_index)).all()
        results = {r.result_type: r for r in db.scalars(select(Result))}
    assert chunks[0].start_seconds == 0.0 and chunks[-1].end_seconds == 9.5
    assert "transport costs" in chunks[0].text
    assert results["extracted_text"].content.startswith("P03 said transport costs")
    transcript = results["transcription"]
    assert transcript.details["segments"][1] == {
        "start": 4.0,
        "end": 9.5,
        "text": "The road floods in January, so trucks cannot reach the farms.",
    }
    assert transcript.model_name == "faster-whisper fake-whisper"


def test_transcript_endpoint(
    client: TestClient, account, transcriber_worker: TranscriptionWorker
) -> None:
    headers, project_id = account
    document = upload(client, headers, project_id, "interview.mp3")
    url = f"/api/v1/projects/{project_id}/documents/{document['id']}/transcript"

    assert client.get(url, headers=headers).status_code == 409  # not transcribed yet
    transcriber_worker.run_pending()

    body = client.get(url, headers=headers).json()
    assert body["language"] == "en"
    assert body["duration"] == 9.5
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][0]["text"].startswith("P03 said")
    assert body["translation"] is None


def test_assistant_evidence_from_a_recording_has_a_timestamp(
    client: TestClient,
    account,
    transcriber_worker: TranscriptionWorker,
    embedder_worker: ProcessingWorker,
    fake_transcriber: FakeTranscriber,
    fake_llm: FakeLLM,
) -> None:
    headers, project_id = account
    fake_transcriber.segments = [
        Segment(0.0, 60.0, "We talked about the weather and the harvest festival."),
        Segment(272.0, 280.0, "P03 said transport costs to the market doubled this year."),
    ]
    upload(client, headers, project_id, "interview.mp3")
    transcriber_worker.run_pending()
    embedder_worker.run_pending()
    fake_llm.replies = ["Transport costs doubled."]

    answer = client.post(
        f"/api/v1/projects/{project_id}/chat",
        json={"question": "What did P03 say about transport costs to the market?"},
        headers=headers,
    ).json()

    evidence = answer["evidence"][0]
    assert evidence["participant"] == "P03"
    assert "timestamp" in evidence
    assert ", at 0" in fake_llm.calls[-1]["prompt"]  # the source location is a time, not a page


def test_translation_is_used_for_non_english_recordings(
    client: TestClient,
    account,
    transcriber_worker: TranscriptionWorker,
    fake_transcriber: FakeTranscriber,
    session_factory: sessionmaker[Session],
) -> None:
    headers, project_id = account
    client.put("/api/v1/users/me/preferences", json={"autoTranslate": True}, headers=headers)
    fake_transcriber.language = "sn"
    fake_transcriber.segments = [Segment(0.0, 5.0, "Mari yekufambisa yakapetwa kaviri.")]
    fake_transcriber.translation = [Segment(0.0, 5.0, "Transport money doubled.")]

    upload(client, headers, project_id, "hurukuro.mp3")
    transcriber_worker.run_pending()

    assert [call["task"] for call in fake_transcriber.calls] == ["transcribe", "translate"]
    assert fake_transcriber.calls[1]["language"] == "sn"
    with session_factory() as db:
        results = {r.result_type: r.content for r in db.scalars(select(Result))}
        chunk = db.scalar(select(DocumentChunk))
    assert results["transcription"] == "Mari yekufambisa yakapetwa kaviri."  # original kept
    assert results["translation"] == "Transport money doubled."
    assert results["extracted_text"] == "Transport money doubled."  # English is the working text
    assert chunk.text == "Transport money doubled."
    assert documents(client, headers, project_id)[0]["language"] == "sn"


def test_english_recordings_are_not_translated(
    client: TestClient, account, transcriber_worker, fake_transcriber: FakeTranscriber
) -> None:
    headers, project_id = account
    client.put("/api/v1/users/me/preferences", json={"autoTranslate": True}, headers=headers)

    upload(client, headers, project_id, "interview.mp3")
    transcriber_worker.run_pending()

    assert [call["task"] for call in fake_transcriber.calls] == ["transcribe"]


# ---------------------------------------------------------------- when not to transcribe


def test_auto_transcribe_off_holds_the_recording_until_retry(
    client: TestClient, account, transcriber_worker, fake_transcriber: FakeTranscriber
) -> None:
    headers, project_id = account
    client.put("/api/v1/users/me/preferences", json={"autoTranscribe": False}, headers=headers)

    document = upload(client, headers, project_id, "interview.mp3")

    assert document["status"] == "failed"
    assert "Automatic transcription is off" in document["processingError"]
    assert transcriber_worker.run_pending() == 0
    assert fake_transcriber.calls == []

    retried = client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/reprocess", headers=headers
    )
    assert retried.status_code == 202
    transcriber_worker.run_pending()
    assert documents(client, headers, project_id)[0]["status"] == "processed"


def test_transcription_disabled_on_the_server(
    client: TestClient, account, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers, project_id = account
    monkeypatch.setattr(settings, "transcription_enabled", False)

    document = upload(client, headers, project_id, "interview.mp3")

    assert document["status"] == "failed"
    assert "turned off on this server" in document["processingError"]
    retry = client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/reprocess", headers=headers
    )
    assert retry.status_code == 409


@pytest.mark.parametrize(
    ("error", "segments", "message"),
    [
        ("The recording could not be read. Check that the file plays and has sound.", None,
         "could not be read"),
        (None, [], "No speech was found"),
    ],
)  # fmt: skip
def test_unreadable_or_silent_recordings_fail_with_a_reason(
    client: TestClient,
    account,
    transcriber_worker,
    fake_transcriber: FakeTranscriber,
    error,
    segments,
    message,
) -> None:
    headers, project_id = account
    fake_transcriber.error = error
    if segments is not None:
        fake_transcriber.segments = segments

    upload(client, headers, project_id, "broken.mp3")
    transcriber_worker.run_pending()

    [listed] = documents(client, headers, project_id)
    assert listed["status"] == "failed"
    assert message in listed["processingError"]


def test_ai_health_reports_transcription(client: TestClient, account) -> None:
    headers, project_id = account
    upload(client, headers, project_id, "interview.mp3")

    transcription = client.get("/health/ai").json()["transcription"]

    assert transcription["enabled"] is True
    assert transcription["model"] == settings.whisper_model
    assert transcription["pendingRecordings"] == 1


# ---------------------------------------------------------------- helpers


def test_timed_chunks_end_on_segment_boundaries_with_overlap() -> None:
    # ~60-character segments: short enough that the ~150-character overlap repeats some.
    segments = [Segment(i * 5.0, i * 5.0 + 4.0, f"Sentence {i:02d} " + "x" * 48) for i in range(60)]

    text, chunks = timed_chunks(segments)

    assert len(chunks) > 1
    for chunk in chunks:
        assert text[chunk["char_start"] : chunk["char_end"]] == chunk["text"]
        assert len(chunk["text"]) <= 1000
        assert chunk["text"].startswith("Sentence")
    assert chunks[0]["start_seconds"] == 0.0
    assert chunks[-1]["end_seconds"] == 299.0
    # Consecutive chunks share their boundary segments, as documents overlap.
    assert chunks[1]["start_seconds"] < chunks[0]["end_seconds"]


def test_a_segment_longer_than_a_chunk_is_kept_whole() -> None:
    _, chunks = timed_chunks([Segment(0.0, 90.0, "y" * 1500), Segment(90.0, 95.0, "End.")])

    assert [len(chunk["text"]) for chunk in chunks] == [1500, 4]


def test_format_timestamp() -> None:
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(272.9) == "04:32"
    assert format_timestamp(4327) == "1:12:07"


def test_whisper_falls_back_to_the_cpu_when_the_gpu_cannot_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tried: list[str] = []

    class FakeModel:
        def __init__(self, name: str, device: str, compute_type: str) -> None:
            tried.append(f"{device}/{compute_type}")
            if device == "cuda":
                raise RuntimeError("Library cublas64_12.dll is not found")

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeModel)
    transcriber = WhisperTranscriber("small", device="auto")

    transcriber._load()

    assert tried == ["cuda/int8_float16", "cpu/int8"]
    assert transcriber.device == "cpu"
    assert transcriber.status == "ready"
