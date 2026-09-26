"""Groq provider and the development quota guard, against a fake Groq (no network)."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.groq import GroqChatModel, set_quota_guard
from app.ai.llm import LLMUnavailable, OllamaChatModel, get_llm, set_llm
from app.ai.quota import (
    QuotaExceeded,
    QuotaGuard,
    QuotaLimits,
    allow_quota_wait,
    estimate_chat_tokens,
    estimate_tokens,
)
from app.core.config import settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import LLMUsage, Message
from app.services.processing_worker import ProcessingWorker
from app.services.storage import LocalStorage, set_storage

API_KEY = "gsk_test_secret_key_123"
MODEL = settings.groq_model
LIMITS = QuotaLimits(
    minute_requests=3,
    minute_tokens=3000,
    daily_requests=5,
    daily_tokens=6000,
    minute_warning_percent=80,
    daily_warning_percent=80,
)

# ---------------------------------------------------------------- fixtures


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


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 26, 12, 0, 0, tzinfo=UTC)
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.advance(seconds)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def guard(session_factory: sessionmaker[Session], clock: Clock) -> Iterator[QuotaGuard]:
    guard = QuotaGuard(lambda: session_factory, "groq", MODEL, LIMITS, clock=clock, sleep=clock.sleep)
    set_quota_guard(guard)
    yield guard
    set_quota_guard(None)


class FakeGroq:
    """Answers like Groq's chat-completions endpoint and records every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.reply = "Transport costs doubled."
        self.status = 200
        self.headers: dict[str, str] = {
            "x-ratelimit-remaining-requests": "990",
            "x-ratelimit-remaining-tokens": "7500",
        }
        self.usage = {"prompt_tokens": 400, "completion_tokens": 50, "total_tokens": 450}

    @property
    def payloads(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert request.url.path == "/openai/v1/chat/completions"
        if self.status != 200:
            return httpx.Response(
                self.status,
                json={"error": {"message": "Rate limit reached", "type": "tokens"}},
                headers=self.headers,
            )
        if json.loads(request.content)["stream"]:
            words = self.reply.split(" ")
            chunks = [
                {"choices": [{"delta": {"content": word + (" " if i < len(words) - 1 else "")}}]}
                for i, word in enumerate(words)
            ]
            chunks.append({"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "x_groq": {"usage": self.usage}})  # fmt: skip
            body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(200, content=body + "data: [DONE]\n\n", headers=self.headers)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": self.reply}}], "usage": self.usage},
            headers=self.headers,
        )


@pytest.fixture
def groq() -> FakeGroq:
    return FakeGroq()


@pytest.fixture
def model(guard: QuotaGuard, groq: FakeGroq) -> GroqChatModel:
    return GroqChatModel(
        API_KEY,
        MODEL,
        guard,
        base_url="https://api.groq.com/openai/v1",
        max_output_tokens=700,
        transport=httpx.MockTransport(groq),
    )


def rows(session_factory: sessionmaker[Session]) -> list[LLMUsage]:
    with session_factory() as db:
        return list(db.scalars(select(LLMUsage).order_by(LLMUsage.created_at)))


# ---------------------------------------------------------------- requests


def test_generate_sends_one_request_and_records_reported_usage(
    model: GroqChatModel, groq: FakeGroq, session_factory
) -> None:
    answer = model.generate("You are helpful.", "What about transport?", max_tokens=2000)

    assert answer == "Transport costs doubled."
    [request] = groq.requests
    assert request.headers["Authorization"] == f"Bearer {API_KEY}"
    payload = groq.payloads[0]
    assert payload["model"] == MODEL
    assert payload["reasoning_effort"] == "none"  # no thinking tokens
    assert payload["max_completion_tokens"] == 700  # capped by GROQ_MAX_OUTPUT_TOKENS
    assert payload["messages"][1]["content"] == "What about transport?"

    [row] = rows(session_factory)
    assert (row.status, row.input_tokens, row.output_tokens, row.total_tokens) == (
        "success", 400, 50, 450,
    )  # fmt: skip
    assert row.estimated_input_tokens == estimate_chat_tokens(
        "You are helpful.", "What about transport?"
    )
    assert row.max_output_tokens == 700


def test_usage_counters_grow_with_each_request(
    model: GroqChatModel, guard: QuotaGuard, groq: FakeGroq
) -> None:
    model.generate("s", "one")
    model.generate("s", "two")

    report = guard.snapshot().to_dict()
    assert report["requests"] == {"minute": 2, "minuteLimit": 3, "today": 2, "dailyLimit": 5}
    assert report["tokens"]["today"] == 900
    assert report["groqReported"]["dailyRequestsRemaining"] == "990"


def test_warning_is_reported_but_does_not_block(
    model: GroqChatModel, guard: QuotaGuard, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(4):  # 4 of 5 daily requests = 80%
        model.generate("s", "q")
        clock.advance(61)

    report = guard.snapshot().to_dict()
    assert report["warning"] is True
    assert "approximately 80% of the configured daily allowance" in report["warningMessage"]
    assert report["blocked"] is False
    assert model.generate("s", "q")  # still allowed


def test_daily_request_limit_blocks_without_calling_groq(
    model: GroqChatModel, guard: QuotaGuard, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(5):
        model.generate("s", "q")
        clock.advance(61)
    sent = len(groq.requests)

    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "q")

    assert len(groq.requests) == sent  # Groq was not called
    assert caught.value.reason == "daily_request_limit"
    assert "wait until the daily quota refreshes" in str(caught.value)
    # The first request (made at 12:00) leaves our rolling day at 12:00 tomorrow.
    assert caught.value.to_dict()["retryAt"] == "2026-09-27T12:00:00Z"
    assert guard.snapshot().to_dict()["blocked"] is True


def test_daily_token_limit_blocks_without_calling_groq(
    model: GroqChatModel, groq: FakeGroq, clock: Clock
) -> None:
    groq.usage = {"prompt_tokens": 2500, "completion_tokens": 400}
    model.generate("s", "q")
    clock.advance(61)
    model.generate("s", "q")  # 5,800 of 6,000 tokens today
    clock.advance(61)

    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "q")

    assert caught.value.reason == "daily_token_limit"
    assert len(groq.requests) == 2


def test_minute_limit_blocks_until_the_window_moves_on(
    model: GroqChatModel, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(3):
        model.generate("s", "q")
        clock.advance(5)

    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "q")
    assert caught.value.reason == "minute_request_limit"
    assert "wait until the minute window resets" in str(caught.value)
    assert caught.value.retry_after_seconds() == 45  # the first request leaves at 12:01:00
    assert len(groq.requests) == 3

    clock.advance(46)
    assert model.generate("s", "q")


def test_minute_token_limit_counts_the_reserved_output(
    model: GroqChatModel, groq: FakeGroq
) -> None:
    groq.usage = {"prompt_tokens": 1000, "completion_tokens": 700}
    model.generate("s", "q", max_tokens=700)  # 1,700 of 3,000 this minute

    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "x" * 2400, max_tokens=700)  # ~800 input + 700 output reserved

    assert caught.value.reason == "minute_token_limit"
    assert len(groq.requests) == 1


def test_a_request_bigger_than_any_window_is_refused(model: GroqChatModel, groq: FakeGroq) -> None:
    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "x" * 20_000)

    assert caught.value.reason == "request_too_large"
    assert groq.requests == []


def test_groq_429_is_not_retried_and_pauses_requests(
    model: GroqChatModel, groq: FakeGroq, clock: Clock, session_factory
) -> None:
    groq.status = 429
    groq.headers = {"retry-after": "20"}

    with pytest.raises(QuotaExceeded) as caught:
        model.generate("s", "q")

    assert len(groq.requests) == 1  # exactly one attempt
    assert caught.value.reason == "groq_rate_limited"
    assert "paused requests to protect your development quota" in str(caught.value)
    assert caught.value.retry_after_seconds() == 20
    assert rows(session_factory)[0].status == "rate_limited"

    groq.status = 200
    with pytest.raises(QuotaExceeded):  # still paused: Groq is not asked again
        model.generate("s", "q")
    assert len(groq.requests) == 1

    clock.advance(21)
    assert model.generate("s", "q")


def test_background_jobs_wait_out_a_minute_limit(
    model: GroqChatModel, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(3):
        model.generate("s", "q")

    with allow_quota_wait(65):
        assert model.generate("s", "q")

    assert clock.slept == [60]
    assert len(groq.requests) == 4


def test_background_jobs_never_wait_for_the_daily_limit(
    model: GroqChatModel, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(5):
        model.generate("s", "q")
        clock.advance(61)

    with allow_quota_wait(65), pytest.raises(QuotaExceeded):
        model.generate("s", "q")
    assert clock.slept == []


# ---------------------------------------------------------------- streaming


def test_stream_yields_pieces_and_records_usage(
    model: GroqChatModel, groq: FakeGroq, session_factory
) -> None:
    groq.reply = "Costs doubled for farmers."

    pieces = list(model.stream("s", "q"))

    assert "".join(pieces) == "Costs doubled for farmers."
    assert len(pieces) == 4
    assert groq.payloads[0]["stream"] is True
    [row] = rows(session_factory)
    assert (row.status, row.streamed, row.total_tokens) == ("success", True, 450)


def test_stream_checks_the_quota_before_opening(
    model: GroqChatModel, groq: FakeGroq, clock: Clock
) -> None:
    for _ in range(3):
        model.generate("s", "q")

    pieces = model.stream("s", "q")
    with pytest.raises(QuotaExceeded):
        next(pieces)
    assert len(groq.requests) == 3


def test_stream_stopped_early_still_counts(model: GroqChatModel, session_factory) -> None:
    pieces = model.stream("s", "q")
    next(pieces)
    pieces.close()  # e.g. the browser went away

    [row] = rows(session_factory)
    assert row.status == "failed"
    assert row.total_tokens > 0


# ---------------------------------------------------------------- configuration and secrecy


def test_missing_api_key_sends_nothing(guard: QuotaGuard, groq: FakeGroq, session_factory) -> None:
    model = GroqChatModel("", MODEL, guard, transport=httpx.MockTransport(groq))

    with pytest.raises(LLMUnavailable, match="GROQ_API_KEY"):
        model.generate("s", "q")

    assert groq.requests == []
    assert rows(session_factory) == []


def test_api_key_never_appears_in_errors_or_reports(
    model: GroqChatModel, guard: QuotaGuard, groq: FakeGroq
) -> None:
    groq.status = 401
    with pytest.raises(LLMUnavailable) as caught:
        model.generate("s", "q")

    assert API_KEY not in str(caught.value)
    assert API_KEY not in repr(model)
    assert API_KEY not in json.dumps(guard.snapshot().to_dict())


def test_provider_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    set_llm(None)
    try:
        assert isinstance(get_llm(), OllamaChatModel)
        set_llm(None)
        monkeypatch.setattr(settings, "llm_provider", "groq")
        assert isinstance(get_llm(), GroqChatModel)
    finally:
        set_llm(None)


def test_token_estimate_is_conservative() -> None:
    text = "Participants described transport costs doubling after the floods. " * 20
    assert estimate_tokens(text) >= len(text) / 4  # real tokenizers average ~4 chars/token
    # "and" is one token (3 ASCII characters); "ñ" and "ú" count one token each.
    assert estimate_tokens("ñandú") == 1 + 2


# ---------------------------------------------------------------- through the API


@pytest.fixture
def client(session_factory: sessionmaker[Session], tmp_path: Path) -> Iterator[TestClient]:
    set_storage(LocalStorage(tmp_path))

    def override_get_db() -> Iterator[Session]:
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    set_storage(None)


@pytest.fixture
def project(client: TestClient, session_factory, model: GroqChatModel) -> tuple[dict, str]:
    set_llm(model)
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
        files={"files": ("notes.txt", b"P03 said transport costs doubled.", "text/plain")},
        headers=headers,
    )
    ProcessingWorker(session_factory, embeddings_enabled=True).run_pending()
    return headers, project_id


def ask(client: TestClient, headers: dict, project_id: str, question: str):
    return client.post(
        f"/api/v1/projects/{project_id}/chat", json={"question": question}, headers=headers
    )


def test_chat_answer_carries_evidence_and_usage(client: TestClient, project, groq: FakeGroq) -> None:
    headers, project_id = project
    groq.reply = "Transport costs doubled [1]."

    body = ask(client, headers, project_id, "What did P03 say about transport costs?").json()

    assert body["content"] == "Transport costs doubled."
    assert body["evidence"][0]["participant"] == "P03"
    assert body["usage"]["requests"]["today"] == 1
    assert body["usage"]["blocked"] is False
    assert API_KEY not in json.dumps(body)


def test_hello_spends_no_groq_request(client: TestClient, project, groq: FakeGroq) -> None:
    headers, project_id = project

    body = ask(client, headers, project_id, "hello").json()

    assert body["content"] == "Hello! How can I help you with your research?"
    assert groq.requests == []


def test_chat_over_the_limit_is_a_structured_429_and_saves_nothing(
    client: TestClient, project, groq: FakeGroq, session_factory
) -> None:
    headers, project_id = project
    for _ in range(3):
        ask(client, headers, project_id, "What did P03 say about transport costs?")
    with session_factory() as db:
        saved = db.scalar(select(func.count()).select_from(Message))

    response = ask(client, headers, project_id, "What did P03 say about transport costs?")

    assert response.status_code == 429
    detail = response.json()["detail"]
    assert detail["blocked"] is True
    assert detail["reason"] == "minute_request_limit"
    assert detail["retryAfter"] > 0
    assert response.headers["Retry-After"] == str(detail["retryAfter"])
    assert len(groq.requests) == 3
    with session_factory() as db:
        assert db.scalar(select(func.count()).select_from(Message)) == saved


def test_stream_over_the_limit_is_refused_before_streaming(
    client: TestClient, project, groq: FakeGroq
) -> None:
    headers, project_id = project
    for _ in range(3):
        ask(client, headers, project_id, "What did P03 say about transport costs?")

    response = client.post(
        f"/api/v1/projects/{project_id}/chat/stream",
        json={"question": "What did P03 say about transport costs?"},
        headers=headers,
    )

    assert response.status_code == 429
    assert response.json()["detail"]["reason"] == "minute_request_limit"
    assert len(groq.requests) == 3


def test_stream_through_the_api_reports_usage(client: TestClient, project, groq: FakeGroq) -> None:
    headers, project_id = project

    with client.stream(
        "POST",
        f"/api/v1/projects/{project_id}/chat/stream",
        json={"question": "What did P03 say about transport costs?"},
        headers=headers,
    ) as response:
        body = response.read().decode()

    done = json.loads(body.strip().split("\n\n")[-1].split("data: ", 1)[1])
    assert done["message"]["content"] == "Transport costs doubled."
    assert done["message"]["evidence"]
    assert done["usage"]["requests"]["today"] == 1


def test_usage_endpoint(client: TestClient, project, groq: FakeGroq) -> None:
    headers, project_id = project
    ask(client, headers, project_id, "What did P03 say about transport costs?")

    report = client.get("/api/v1/usage/groq", headers=headers).json()

    assert report["provider"] == "groq"
    assert report["model"] == MODEL
    assert report["requests"]["today"] == 1
    assert report["tokens"]["dailyLimit"] == 6000
    assert API_KEY not in json.dumps(report)
    assert client.get("/api/v1/usage/groq").status_code == 401
