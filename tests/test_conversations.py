from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Conversation, Message
from app.models.conversation import ROLE_ASSISTANT, ROLE_USER
from app.services.conversations import add_message

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


def register(client: TestClient, email: str = "ada@example.com") -> dict[str, str]:
    token = client.post(
        "/api/v1/auth/register",
        json={"name": "Ada", "email": email, "password": "password123"},
    ).json()["accessToken"]
    return {"Authorization": f"Bearer {token}"}


def make_project(client: TestClient, headers: dict[str, str], name: str = "Study") -> str:
    return client.post("/api/v1/projects", json={"name": name}, headers=headers).json()["id"]


def base(project_id: str) -> str:
    return f"/api/v1/projects/{project_id}/conversations"


def new_conversation(client: TestClient, headers: dict[str, str], project_id: str, **body) -> dict:
    response = client.post(base(project_id), json=body or None, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()


def say(db: Session, conversation_id: str, role: str, content: str, evidence=None) -> Message:
    return add_message(db, db.get(Conversation, conversation_id), role, content, evidence)


EVIDENCE = [
    {
        "id": "ev-1",
        "source": "Interview 03",
        "participant": "P03",
        "quote": "Transport costs doubled.",
        "documentId": "doc-1",
        "relevance": 0.64,
    }
]


# ---------------------------------------------------------------- creating and listing


def test_new_conversation_starts_empty_and_untitled(client: TestClient) -> None:
    headers = register(client)
    project_id = make_project(client, headers)

    conversation = new_conversation(client, headers, project_id)

    assert conversation["title"] == "New chat"
    assert conversation["projectId"] == project_id
    assert conversation["preview"] == ""
    assert conversation["messageCount"] == 0
    assert conversation["updatedAt"].endswith("Z")


def test_conversation_can_be_created_with_a_title(client: TestClient) -> None:
    headers = register(client)
    project_id = make_project(client, headers)

    assert new_conversation(client, headers, project_id, title="  Barriers  ")["title"] == (
        "Barriers"
    )
    too_long = client.post(base(project_id), json={"title": "x" * 201}, headers=headers)
    assert too_long.status_code == 422


def test_list_shows_newest_activity_first_with_preview_and_count(
    client: TestClient, db: Session
) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    older = new_conversation(client, headers, project_id)
    newer = new_conversation(client, headers, project_id)
    say(db, older["id"], ROLE_USER, "What did farmers say about transport?")
    say(db, older["id"], ROLE_ASSISTANT, "They said costs doubled.", EVIDENCE)

    listed = client.get(base(project_id), headers=headers).json()

    assert [item["id"] for item in listed] == [older["id"], newer["id"]]
    assert listed[0]["preview"] == "What did farmers say about transport?"
    assert listed[0]["messageCount"] == 2
    assert listed[1]["messageCount"] == 0


def test_first_question_names_an_untitled_conversation(client: TestClient, db: Session) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    untitled = new_conversation(client, headers, project_id)
    titled = new_conversation(client, headers, project_id, title="My title")

    say(db, untitled["id"], ROLE_USER, "What   are the main\nbarriers to access? " + "x" * 100)
    say(db, untitled["id"], ROLE_USER, "A second question")
    say(db, titled["id"], ROLE_USER, "Does not rename")

    listed = {item["id"]: item for item in client.get(base(project_id), headers=headers).json()}
    assert listed[untitled["id"]]["title"].startswith("What are the main barriers to access?")
    assert len(listed[untitled["id"]]["title"]) <= 80
    assert listed[titled["id"]]["title"] == "My title"


def test_empty_project_has_no_conversations(client: TestClient) -> None:
    headers = register(client)

    assert client.get(base(make_project(client, headers)), headers=headers).json() == []


# ---------------------------------------------------------------- messages


def test_messages_come_back_oldest_first_with_evidence(client: TestClient, db: Session) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    conversation = new_conversation(client, headers, project_id)
    say(db, conversation["id"], ROLE_USER, "Question one")
    say(db, conversation["id"], ROLE_ASSISTANT, "Answer one", EVIDENCE)

    messages = client.get(
        f"{base(project_id)}/{conversation['id']}/messages", headers=headers
    ).json()

    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[0]["evidence"] is None
    assert messages[1]["evidence"] == EVIDENCE
    assert messages[1]["conversationId"] == conversation["id"]
    assert messages[1]["projectId"] == project_id


def test_flat_messages_route_returns_the_latest_conversation(
    client: TestClient, db: Session
) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    assert client.get(f"/api/v1/projects/{project_id}/messages", headers=headers).json() == []

    first = new_conversation(client, headers, project_id)
    second = new_conversation(client, headers, project_id)
    say(db, second["id"], ROLE_USER, "In the second")
    say(db, first["id"], ROLE_USER, "In the first, most recently")

    messages = client.get(f"/api/v1/projects/{project_id}/messages", headers=headers).json()

    assert [message["content"] for message in messages] == ["In the first, most recently"]


# ---------------------------------------------------------------- renaming and deleting


def test_rename_conversation(client: TestClient) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    conversation = new_conversation(client, headers, project_id)
    url = f"{base(project_id)}/{conversation['id']}"

    renamed = client.patch(url, json={"title": "Transport questions"}, headers=headers)

    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Transport questions"
    assert client.patch(url, json={"title": "   "}, headers=headers).status_code == 422


def test_delete_conversation_removes_its_messages(client: TestClient, db: Session) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    conversation = new_conversation(client, headers, project_id)
    say(db, conversation["id"], ROLE_USER, "Question")

    response = client.delete(f"{base(project_id)}/{conversation['id']}", headers=headers)

    assert response.status_code == 204
    assert client.get(base(project_id), headers=headers).json() == []
    assert db.scalar(select(func.count()).select_from(Message)) == 0


def test_deleting_the_project_removes_its_conversations(client: TestClient, db: Session) -> None:
    headers = register(client)
    project_id = make_project(client, headers)
    conversation = new_conversation(client, headers, project_id)
    say(db, conversation["id"], ROLE_USER, "Question")

    client.delete(f"/api/v1/projects/{project_id}", headers=headers)

    assert db.scalar(select(func.count()).select_from(Conversation)) == 0
    assert db.scalar(select(func.count()).select_from(Message)) == 0


# ---------------------------------------------------------------- the project boundary


def test_a_conversation_is_only_reachable_through_its_own_project(
    client: TestClient, db: Session
) -> None:
    headers = register(client)
    project_a = make_project(client, headers, "A")
    project_b = make_project(client, headers, "B")
    conversation = new_conversation(client, headers, project_b)
    say(db, conversation["id"], ROLE_USER, "Private to B")
    wrong = f"{base(project_a)}/{conversation['id']}"

    assert client.get(f"{wrong}/messages", headers=headers).status_code == 404
    assert client.patch(wrong, json={"title": "x"}, headers=headers).status_code == 404
    assert client.delete(wrong, headers=headers).status_code == 404
    assert client.get(base(project_a), headers=headers).json() == []
    assert client.get(f"/api/v1/projects/{project_a}/messages", headers=headers).json() == []


def test_other_users_cannot_see_or_change_conversations(client: TestClient) -> None:
    owner = register(client)
    project_id = make_project(client, owner)
    conversation = new_conversation(client, owner, project_id)
    stranger = register(client, email="eve@example.com")
    url = f"{base(project_id)}/{conversation['id']}"

    assert client.get(base(project_id), headers=stranger).status_code == 404
    assert client.post(base(project_id), headers=stranger).status_code == 404
    assert client.get(f"{url}/messages", headers=stranger).status_code == 404
    assert client.delete(url, headers=stranger).status_code == 404
    assert client.get(base(project_id)).status_code == 401
