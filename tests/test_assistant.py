import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.embeddings import set_embedder
from app.ai.llm import LLMError, LLMUnavailable, set_llm
from app.core.config import settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Message
from app.services.assistant import (
    best_per_document,
    best_quote,
    cited_source_numbers,
    display_source,
    is_continuation,
    is_overview_question,
    needs_retrieval,
    select_sources,
    strip_citations,
    strip_streamed_citations,
)
from app.services.processing_worker import ProcessingWorker
from app.services.retrieval import RetrievedChunk
from app.services.storage import LocalStorage, set_storage
from tests.conftest import FakeEmbedder, FakeLLM

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


def make_project(client: TestClient, headers: dict[str, str], name: str = "Farming") -> str:
    return client.post("/api/v1/projects", json={"name": name}, headers=headers).json()["id"]


def add_notes(client: TestClient, headers: dict, project_id: str, name: str, text: str) -> str:
    return client.post(
        f"/api/v1/projects/{project_id}/documents",
        files={"files": (name, text.encode(), "text/plain")},
        headers=headers,
    ).json()[0]["id"]


def ask(client: TestClient, headers: dict, project_id: str, question: str, **extra):
    return client.post(
        f"/api/v1/projects/{project_id}/chat",
        json={"question": question, "projectId": project_id, **extra},
        headers=headers,
    )


TRANSPORT = (
    "Interview with P03 about the harvest. P03 said transport costs to the market doubled "
    "this year. The road floods in January."
)
FINANCE = "P07 explained that bank loans need title deeds, so financing for fertiliser is scarce."
WEATHER = "Heavy rain fell during the planting season in the northern villages."


@pytest.fixture
def farming(client: TestClient, worker: ProcessingWorker) -> tuple[dict, str, dict[str, str]]:
    headers = register(client)
    project_id = make_project(client, headers)
    ids = {
        "transport": add_notes(client, headers, project_id, "Interview_03.txt", TRANSPORT),
        "finance": add_notes(client, headers, project_id, "Interview_07.txt", FINANCE),
        "weather": add_notes(client, headers, project_id, "field_notes.txt", WEATHER),
    }
    worker.run_pending()
    return headers, project_id, ids


# ---------------------------------------------------------------- grounded answers


def test_answer_cites_sources_and_returns_verbatim_evidence(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = farming
    fake_llm.replies = ["Transport costs doubled for farmers [1]. The road also floods [1]."]

    response = ask(client, headers, project_id, "What did farmers say about transport costs?")

    assert response.status_code == 200
    answer = response.json()
    assert answer["role"] == "assistant"
    assert answer["content"] == "Transport costs doubled for farmers. The road also floods."
    assert answer["projectId"] == project_id
    assert answer["conversationId"]

    [evidence] = answer["evidence"]
    assert evidence["documentId"] == ids["transport"]
    assert evidence["source"] == "Interview 03"
    assert evidence["participant"] == "P03"
    assert evidence["quote"] in TRANSPORT  # verbatim
    assert "transport costs" in evidence["quote"]
    assert 0 < evidence["relevance"] <= 1


def test_prompt_numbers_the_sources_and_forbids_outside_knowledge(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, _ = farming

    ask(client, headers, project_id, "What did farmers say about transport costs?")

    call = fake_llm.calls[-1]
    assert "ONLY the numbered sources" in call["system"]
    assert "not instructions" in call["system"]
    assert '<source number="1" document="Interview_03.txt">' in call["prompt"]
    assert TRANSPORT in call["prompt"]
    assert "Question: What did farmers say about transport costs?" in call["prompt"]
    assert "Do not include numbered citations" in call["system"]
    assert "Cite the source number" not in call["prompt"]


def test_uncited_answer_falls_back_to_the_retrieved_passages(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = farming
    fake_llm.replies = ["Transport costs doubled."]  # the model forgot to cite

    evidence = ask(client, headers, project_id, "transport costs to the market").json()["evidence"]

    assert evidence
    assert evidence[0]["documentId"] == ids["transport"]


def test_citations_of_unknown_sources_are_ignored(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = farming
    fake_llm.replies = ["Costs doubled [1][9]."]

    evidence = ask(client, headers, project_id, "transport costs to the market").json()["evidence"]

    assert [item["documentId"] for item in evidence] == [ids["transport"]]


# ---------------------------------------------------------------- small talk


@pytest.fixture
def ollama_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "ollama")


@pytest.mark.parametrize(
    ("greeting", "reply"),
    [
        ("hello", "Hello! How can I help you with your research?"),
        ("Hi!", "Hello! How can I help you with your research?"),
        ("Good morning", "Hello! How can I help you with your research?"),
        ("thanks", "You're welcome!"),
        ("how are you?", "I'm doing well"),
    ],
)
def test_small_talk_costs_no_groq_request_search_or_evidence(
    client: TestClient,
    farming,
    fake_llm: FakeLLM,
    fake_embedder: FakeEmbedder,
    greeting: str,
    reply: str,
) -> None:
    headers, project_id, _ = farming
    fake_embedder.calls.clear()

    answer = ask(client, headers, project_id, greeting).json()

    assert answer["content"].startswith(reply)
    assert answer["evidence"] == []
    assert fake_embedder.calls == []  # no retrieval
    assert fake_llm.calls == []  # no model request


def test_small_talk_uses_the_local_model_with_ollama(
    client: TestClient, farming, fake_llm: FakeLLM, fake_embedder: FakeEmbedder, ollama_provider
) -> None:
    headers, project_id, _ = farming
    fake_embedder.calls.clear()
    fake_llm.replies = ["Hello! How can I help you with your research?"]

    answer = ask(client, headers, project_id, "hello").json()

    assert answer["content"] == "Hello! How can I help you with your research?"
    assert answer["evidence"] == []
    assert fake_embedder.calls == []
    call = fake_llm.calls[-1]
    assert "<source" not in call["prompt"]
    assert "no project documents are shown to you" in call["system"]


def test_small_talk_works_before_any_documents_are_processed(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    headers = register(client)
    project_id = make_project(client, headers)

    answer = ask(client, headers, project_id, "hello").json()

    assert "no processed documents" not in answer["content"]
    assert answer["evidence"] == []


def test_small_talk_keeps_the_conversation_and_its_history(
    client: TestClient, farming, fake_llm: FakeLLM, db: Session, ollama_provider
) -> None:
    headers, project_id, _ = farming
    first = ask(client, headers, project_id, "What did farmers say about transport costs?").json()

    thanks = ask(client, headers, project_id, "thanks!", conversationId=first["conversationId"])

    assert thanks.json()["conversationId"] == first["conversationId"]
    assert "Researcher: What did farmers say about transport costs?" in fake_llm.calls[-1]["prompt"]
    assert db.scalar(select(func.count()).select_from(Message)) == 4  # no duplicates


# ---------------------------------------------------------------- several documents


def test_overview_question_draws_on_every_document(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, ids = farming

    question = "Summarize the documents: transport costs, heavy rain in the villages, bank loans for fertiliser"
    answer = ask(client, headers, project_id, question).json()

    prompt = fake_llm.calls[-1]["prompt"]
    assert "Sources (from 3 documents:" in prompt
    assert {e["documentId"] for e in answer["evidence"]} == set(ids.values())
    assert "cover every document" in fake_llm.calls[-1]["system"]


def test_what_else_brings_new_passages_then_says_it_is_done(
    client: TestClient, worker: ProcessingWorker, fake_llm: FakeLLM
) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    paragraphs = [f"Farmers said transport point {n} mattered to them." for n in range(9)]
    add_notes(client, headers, project_id, "long.txt", "\n\n".join(p * 30 for p in paragraphs))
    worker.run_pending()
    first = ask(client, headers, project_id, "What did farmers say about transport?").json()
    first_prompt = fake_llm.calls[-1]["prompt"]

    ask(client, headers, project_id, "what else?", conversationId=first["conversationId"])

    second_prompt = fake_llm.calls[-1]["prompt"]
    assert 'wants more on their earlier question: "What did farmers say about transport?"' in (
        second_prompt
    )
    first_passages = set(re.findall(r"transport point (\d)", first_prompt.split("Sources")[1]))
    second_passages = set(re.findall(r"transport point (\d)", second_prompt.split("Sources")[1]))
    assert second_passages and not (second_passages & first_passages)

    calls = len(fake_llm.calls)
    for _ in range(5):
        last = ask(client, headers, project_id, "more", conversationId=first["conversationId"])
    assert "covers everything I found" in last.json()["content"]
    assert len(fake_llm.calls) < calls + 5  # the last ones needed no model call


# ---------------------------------------------------------------- when there is nothing to answer from


def test_no_processed_documents_gives_guidance_without_calling_the_llm(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    headers = register(client)
    project_id = make_project(client, headers)

    answer = ask(client, headers, project_id, "What did farmers say?").json()

    assert "no processed documents" in answer["content"]
    assert answer["evidence"] == []
    assert fake_llm.calls == []


def test_unrelated_question_says_nothing_was_found_without_calling_the_llm(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, _ = farming

    answer = ask(client, headers, project_id, "chocolate cake recipe").json()

    assert "couldn't find anything" in answer["content"]
    assert answer["evidence"] == []
    assert fake_llm.calls == []


# ---------------------------------------------------------------- conversations


def test_first_question_creates_a_titled_conversation_with_both_messages(
    client: TestClient, farming
) -> None:
    headers, project_id, _ = farming

    answer = ask(client, headers, project_id, "What did farmers say about transport costs?").json()

    conversations = client.get(
        f"/api/v1/projects/{project_id}/conversations", headers=headers
    ).json()
    assert [c["id"] for c in conversations] == [answer["conversationId"]]
    assert conversations[0]["title"] == "What did farmers say about transport costs?"
    messages = client.get(
        f"/api/v1/projects/{project_id}/conversations/{answer['conversationId']}/messages",
        headers=headers,
    ).json()
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[1]["evidence"] == answer["evidence"]


def test_follow_up_continues_the_conversation_with_history(
    client: TestClient, farming, fake_llm: FakeLLM
) -> None:
    headers, project_id, _ = farming
    first = ask(client, headers, project_id, "What did farmers say about transport costs?").json()

    second = ask(
        client,
        headers,
        project_id,
        "and the road?",  # short follow-up: searched with the previous question
        conversationId=first["conversationId"],
    )

    assert second.status_code == 200
    assert second.json()["conversationId"] == first["conversationId"]
    assert "Earlier in this conversation" in fake_llm.calls[-1]["prompt"]
    assert "Researcher: What did farmers say about transport costs?" in fake_llm.calls[-1]["prompt"]
    assert len(fake_llm.calls) == 2  # the follow-up found sources thanks to the history


def test_unknown_or_foreign_conversation_is_rejected(client: TestClient, farming) -> None:
    headers, project_id, _ = farming
    other_project = make_project(client, headers, "Other")
    foreign = ask(client, headers, other_project, "anything at all here").json()

    response = ask(
        client, headers, project_id, "transport costs", conversationId=foreign["conversationId"]
    )

    assert response.status_code == 404


# ---------------------------------------------------------------- failures


def test_llm_failure_saves_nothing(client: TestClient, farming, db: Session) -> None:
    class DownLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            raise LLMUnavailable("Ollama is not running at http://localhost:11434.")

    headers, project_id, _ = farming
    set_llm(DownLLM())

    response = ask(client, headers, project_id, "What did farmers say about transport costs?")

    assert response.status_code == 503
    assert "Ollama is not running" in response.json()["detail"]
    assert db.scalar(select(func.count()).select_from(Message)) == 0


def test_other_llm_errors_are_bad_gateway(client: TestClient, farming) -> None:
    class BrokenLLM(FakeLLM):
        def generate(self, system: str, prompt: str, **kwargs: object) -> str:
            raise LLMError("The language model took too long to respond.")

    headers, project_id, _ = farming
    set_llm(BrokenLLM())

    assert ask(client, headers, project_id, "transport costs to the market").status_code == 502


def test_embedding_failure_is_service_unavailable(client: TestClient, farming) -> None:
    class BrokenEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> np.ndarray:
            raise RuntimeError("model failed to load")

    headers, project_id, _ = farming
    set_embedder(BrokenEmbedder())

    assert ask(client, headers, project_id, "transport costs").status_code == 503


def test_question_is_validated(client: TestClient, farming) -> None:
    headers, project_id, _ = farming

    assert ask(client, headers, project_id, "   ").status_code == 422
    assert ask(client, headers, project_id, "x" * 2001).status_code == 422


# ---------------------------------------------------------------- the project boundary


def test_another_projects_documents_never_reach_the_prompt(
    client: TestClient, worker: ProcessingWorker, fake_llm: FakeLLM
) -> None:
    headers = register(client)
    project_a = make_project(client, headers, "A")
    project_b = make_project(client, headers, "B")
    document_a = add_notes(
        client, headers, project_a, "a.txt", "P09 said farmers pay transport costs to the market."
    )
    document_b = add_notes(client, headers, project_b, "b.txt", TRANSPORT)
    worker.run_pending()

    answer = ask(client, headers, project_a, "What did farmers say about transport costs?").json()

    assert fake_llm.calls, "project A's own document should have been used"
    assert all(TRANSPORT not in call["prompt"] for call in fake_llm.calls)
    assert {item["documentId"] for item in answer["evidence"]} == {document_a}
    assert document_b not in str(answer)


def test_other_users_cannot_ask_about_a_project(client: TestClient, farming) -> None:
    _, project_id, _ = farming
    stranger = register(client, email="eve@example.com")

    assert ask(client, stranger, project_id, "transport").status_code == 404
    assert client.post(f"/api/v1/projects/{project_id}/chat", json={"question": "x"}).status_code == 401


# ---------------------------------------------------------------- helpers


@pytest.mark.parametrize(
    ("answer", "count", "expected"),
    [
        ("A [1]. B [2][3].", 3, [1, 2, 3]),
        ("A [2, 1]. B [2].", 3, [2, 1]),
        ("A [0] [4] [x].", 3, []),
        ("No citations.", 3, []),
    ],
)
def test_cited_source_numbers(answer: str, count: int, expected: list[int]) -> None:
    assert cited_source_numbers(answer, count) == expected


@pytest.mark.parametrize(
    "question",
    ["hello", "hi", "Hi there!", "thanks", "Thank you so much!", "good morning",
     "how are you?", "How's it going?", "hello, how are you?", "ok thanks", "bye"],
)  # fmt: skip
def test_small_talk_needs_no_retrieval(question: str) -> None:
    assert not needs_retrieval(question)


@pytest.mark.parametrize(
    "question",
    ["what is my project about?", "what are the main themes?", "what methodology did I use?",
     "what did participant P03 say?", "summarize the interviews", "hi, what did P03 say?",
     "thanks, and what about finance?", "yes", "more detail please", "transport"],
)  # fmt: skip
def test_research_questions_need_retrieval(question: str) -> None:
    assert needs_retrieval(question)


@pytest.mark.parametrize(
    "question", ["what else", "What else?", "more", "tell me more", "more please", "go on",
                 "anything else?", "and what else", "more details", "continue", "elaborate"],
)  # fmt: skip
def test_continuations_are_recognised(question: str) -> None:
    assert is_continuation(question)


@pytest.mark.parametrize(
    "question", ["what else did P03 say about loans?", "more farmers planted maize", "hello",
                 "tell me about this project"],
)  # fmt: skip
def test_ordinary_questions_are_not_continuations(question: str) -> None:
    assert not is_continuation(question)


def test_overview_questions_are_recognised() -> None:
    assert is_overview_question("tell me about this project")
    assert is_overview_question("What are the main themes?")
    assert is_overview_question("summarise the documents")
    assert not is_overview_question("What did P03 say about transport?")


def chunk(document: str, index: int, score: float) -> RetrievedChunk:
    return RetrievedChunk(f"{document}-{index}", document, f"{document}.docx", index,
                          None, 0, 10, "text", score)  # fmt: skip


def test_overview_sources_give_every_document_a_turn() -> None:
    # One long document scores highest everywhere; the short ones must still appear.
    candidates = [chunk("long", i, 0.47 - i * 0.01) for i in range(8)]
    candidates += [chunk("b", 0, 0.40), chunk("c", 0, 0.38), chunk("c", 1, 0.37)]
    candidates.sort(key=lambda c: -c.score)

    chosen = select_sources(candidates, overview=True)

    assert {c.document_id for c in chosen} == {"long", "b", "c"}
    assert len(chosen) == 10
    next_page = select_sources(candidates, overview=True, page=1)
    assert {c.chunk_id for c in next_page}.isdisjoint(c.chunk_id for c in chosen)


def test_specific_sources_skip_documents_that_are_far_less_relevant() -> None:
    candidates = [chunk("a", 0, 0.60), chunk("a", 1, 0.55), chunk("b", 0, 0.57),
                  chunk("c", 0, 0.41)]  # fmt: skip

    chosen = select_sources(candidates, overview=False)

    assert [c.chunk_id for c in chosen] == ["a-0", "a-1", "b-0", "c-0"]  # grouped, reading order
    assert [c.chunk_id for c in best_per_document(chosen)] == ["a-0", "b-0", "c-0"]


def test_strip_citations() -> None:
    assert strip_citations("Costs doubled [1]. Roads flood [2][3], [1, 2].") == (
        "Costs doubled. Roads flood,."
    )
    assert strip_citations("In [2019] prices rose.") == "In [2019] prices rose."


def test_strip_streamed_citations_handles_markers_split_across_pieces() -> None:
    pieces = ["Costs", " doubled ", "[", "1", "]. Roads", " [2", "][3] flood", " [see", " P03]."]

    streamed = list(strip_streamed_citations(iter(pieces)))

    assert all(streamed)
    assert "".join(streamed) == "Costs doubled. Roads flood [see P03]."


def test_display_source_reads_like_prose() -> None:
    assert display_source("Interview_03.docx") == "Interview 03"
    assert display_source("focus group__2.pdf") == "focus group 2"


def test_best_quote_is_verbatim_and_on_topic() -> None:
    text = (
        "The interview began at nine.\nP03 said transport costs doubled.   "
        "Weather was fine. Loans are hard to get."
    )

    quote = best_quote(text, "What about transport costs?")

    assert quote in text
    assert "transport costs doubled" in quote
    assert "Loans" not in quote


def test_best_quote_is_capped() -> None:
    text = "Transport " * 200 + "."

    assert len(best_quote(text, "transport")) <= 320
