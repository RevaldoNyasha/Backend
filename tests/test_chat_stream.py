import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.llm import LLMError, LLMUnavailable, OllamaChatModel, set_llm
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Message
from app.services.processing_worker import ProcessingWorker
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


TRANSPORT = "P03 said transport costs to the market doubled this year. The road floods in January."


@pytest.fixture
def project(client: TestClient, session_factory: sessionmaker[Session]) -> tuple[dict, str]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": "ada@example.com", "password": "password123"},
    ).json()["accessToken"]
    headers = {"Authorization": f"Bearer {token}"}
    project_id = client.post("/api/v1/projects", json={"name": "Farming"}, headers=headers).json()[
        "id"
    ]
    client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": ("Interview_03.txt", TRANSPORT.encode(), "text/plain")},
        headers=headers,
    )
    ProcessingWorker(session_factory, embeddings_enabled=True).run_pending()
    return headers, project_id


def stream(client: TestClient, headers: dict, project_id: str, question: str, **extra):
    """POST to the stream endpoint; returns (status, [(event, data), ...])."""
    with client.stream(
        "POST",
        f"/api/v1/projects/{project_id}/chat/stream",
        json={"question": question, **extra},
        headers=headers,
    ) as response:
        if response.status_code != 200:
            response.read()
            return response.status_code, response.json()
        assert response.headers["content-type"].startswith("text/event-stream")
        return 200, parse_sse(response.read().decode())


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


# ---------------------------------------------------------------- streaming


def test_answer_streams_token_by_token_then_saves(
    client: TestClient, project, fake_llm: FakeLLM, db: Session
) -> None:
    headers, project_id = project
    fake_llm.replies = ["Transport costs doubled for P03 [1]."]

    status, events = stream(client, headers, project_id, "transport costs to the market")

    assert status == 200
    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert names[:-1] == ["token"] * (len(events) - 1) and len(events) > 3
    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert streamed == "Transport costs doubled for P03."  # no citation markers

    message = events[-1][1]["message"]
    assert message["role"] == "assistant"
    assert message["content"] == streamed
    assert message["conversationId"]
    assert message["evidence"][0]["participant"] == "P03"
    assert message["evidence"][0]["quote"] in TRANSPORT

    saved = client.get(
        f"/api/v1/projects/{project_id}/conversations/{message['conversationId']}/messages",
        headers=headers,
    ).json()
    assert [m["role"] for m in saved] == ["user", "assistant"]
    assert saved[1]["content"] == streamed


def test_follow_up_streams_into_the_same_conversation(client: TestClient, project) -> None:
    headers, project_id = project
    _, first = stream(client, headers, project_id, "transport costs to the market")
    conversation_id = first[-1][1]["message"]["conversationId"]

    _, second = stream(
        client, headers, project_id, "and the road flooding?", conversationId=conversation_id
    )

    assert second[-1][1]["message"]["conversationId"] == conversation_id


def test_nothing_found_is_streamed_without_calling_the_llm(
    client: TestClient, project, fake_llm: FakeLLM
) -> None:
    headers, project_id = project

    _, events = stream(client, headers, project_id, "chocolate cake recipe")

    assert "couldn't find anything" in events[0][1]["text"]
    assert events[-1][1]["message"]["evidence"] == []
    assert fake_llm.calls == []


def test_small_talk_streams_without_search_or_evidence(
    client: TestClient, project, fake_llm: FakeLLM, fake_embedder
) -> None:
    headers, project_id = project
    fake_embedder.calls.clear()

    _, events = stream(client, headers, project_id, "hello")

    streamed = "".join(data["text"] for name, data in events if name == "token")
    assert streamed == "Hello! How can I help you with your research?"
    message = events[-1][1]["message"]
    assert message["content"] == streamed
    assert message["evidence"] == []
    assert fake_embedder.calls == []
    assert fake_llm.calls == []  # no Groq request


# ---------------------------------------------------------------- failures


def test_ollama_down_is_an_http_error_before_streaming(
    client: TestClient, project, db: Session
) -> None:
    class DownLLM(FakeLLM):
        def stream(self, system: str, prompt: str, **kwargs: object) -> Iterator[str]:
            raise LLMUnavailable("Ollama is not running at http://localhost:11434.")
            yield  # pragma: no cover - makes this a generator

    headers, project_id = project
    set_llm(DownLLM())

    status, body = stream(client, headers, project_id, "transport costs to the market")

    assert status == 503
    assert "Ollama is not running" in body["detail"]
    assert db.scalar(select(func.count()).select_from(Message)) == 0


def test_failure_midway_sends_an_error_event_and_saves_nothing(
    client: TestClient, project, db: Session
) -> None:
    class DyingLLM(FakeLLM):
        def stream(self, system: str, prompt: str, **kwargs: object) -> Iterator[str]:
            yield "Transport "
            yield "costs "
            raise LLMError("The language model failed: out of memory")

    headers, project_id = project
    set_llm(DyingLLM())

    status, events = stream(client, headers, project_id, "transport costs to the market")

    assert status == 200
    assert [name for name, _ in events] == ["token", "token", "error"]
    assert "out of memory" in events[-1][1]["detail"]
    db.expire_all()
    assert db.scalar(select(func.count()).select_from(Message)) == 0


def test_unknown_conversation_and_other_users_are_rejected(client: TestClient, project) -> None:
    headers, project_id = project
    status, _ = stream(client, headers, project_id, "transport", conversationId="nope")
    assert status == 404

    other = client.post(
        "/api/v1/auth/register",
        json={"name": "Eve", "email": "eve@example.com", "password": "password123"},
    ).json()["accessToken"]
    status, _ = stream(client, {"Authorization": f"Bearer {other}"}, project_id, "transport")
    assert status == 404


# ---------------------------------------------------------------- the Ollama client


def ollama(handler) -> OllamaChatModel:
    return OllamaChatModel("http://ollama.test", "qwen2.5:3b", transport=httpx.MockTransport(handler))


def ndjson(*chunks: dict) -> bytes:
    return b"".join(json.dumps(chunk).encode() + b"\n" for chunk in chunks)


def test_ollama_stream_yields_pieces_until_done() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(
            200,
            content=ndjson(
                {"message": {"content": "one"}, "done": False},
                {"message": {"content": " two"}, "done": False},
                {"message": {"content": ""}, "done": True},
            ),
        )

    assert list(ollama(handler).stream("s", "p")) == ["one", " two"]


def test_ollama_stream_errors() -> None:
    with pytest.raises(LLMError, match="out of memory"):
        list(ollama(lambda r: httpx.Response(200, content=ndjson({"error": "out of memory"})))
             .stream("s", "p"))
    with pytest.raises(LLMUnavailable, match="ollama pull"):
        list(ollama(lambda r: httpx.Response(404, json={"error": "not found"})).stream("s", "p"))
    with pytest.raises(LLMError, match="empty"):
        list(ollama(lambda r: httpx.Response(200, content=ndjson({"done": True}))).stream("s", "p"))

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMUnavailable, match="not running"):
        list(ollama(refuse).stream("s", "p"))
