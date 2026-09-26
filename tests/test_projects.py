from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import Project


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
    register_response = client.post(
        "/api/v1/auth/register",
        json={"name": "John Doe", "email": email, "password": "password123"},
    )
    assert register_response.status_code == 201
    return {"Authorization": f"Bearer {register_response.json()['accessToken']}"}


def create_project_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "Healthcare Access Study",
        "description": "Barriers to primary healthcare access",
    }
    payload.update(overrides)
    return payload


def test_create_project_succeeds(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json=create_project_payload(),
        headers=auth_headers(client),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Healthcare Access Study"
    assert body["description"] == "Barriers to primary healthcare access"
    assert isinstance(body["id"], str)
    assert body["status"] == "active"
    assert body["documentCount"] == 0
    assert body["participantCount"] == 0
    assert body["interviewCount"] == 0
    assert body["focusGroupCount"] == 0
    assert body["themes"] == []
    assert body["codes"] == []
    assert body["createdAt"].endswith("Z")
    assert body["updatedAt"].endswith("Z")
    assert "user_id" not in body


def test_create_project_persists_row(client: TestClient, db_session: Session) -> None:
    headers = auth_headers(client)
    user_id = client.get("/api/v1/auth/me", headers=headers).json()["id"]

    response = client.post("/api/v1/projects", json=create_project_payload(), headers=headers)

    assert response.status_code == 201
    project = db_session.get(Project, response.json()["id"])
    assert project is not None
    assert project.user_id == user_id
    assert project.name == "Healthcare Access Study"


def test_create_project_defaults_empty_description(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json={"name": "Healthcare Access Study"},
        headers=auth_headers(client),
    )

    assert response.status_code == 201
    assert response.json()["description"] == ""


def test_create_project_trims_whitespace(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json=create_project_payload(name="  Healthcare Access  ", description="  notes  "),
        headers=auth_headers(client),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Healthcare Access"
    assert body["description"] == "notes"


def test_create_project_rejects_duplicate_name(client: TestClient) -> None:
    headers = auth_headers(client)
    client.post("/api/v1/projects", json=create_project_payload(), headers=headers)

    response = client.post("/api/v1/projects", json=create_project_payload(), headers=headers)

    assert response.status_code == 409
    assert response.json()["detail"] == "Project name already in use"


def test_duplicate_name_is_allowed_for_a_different_user(client: TestClient) -> None:
    first_headers = auth_headers(client, email="first@example.com")
    second_headers = auth_headers(client, email="second@example.com")

    assert client.post(
        "/api/v1/projects", json=create_project_payload(), headers=first_headers
    ).status_code == 201
    assert client.post(
        "/api/v1/projects", json=create_project_payload(), headers=second_headers
    ).status_code == 201


def test_create_project_rejects_blank_name(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json=create_project_payload(name="   "),
        headers=auth_headers(client),
    )

    assert response.status_code == 422


def test_create_project_accepts_name_at_the_limit(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json=create_project_payload(name="a" * 200),
        headers=auth_headers(client),
    )

    assert response.status_code == 201


def test_create_project_rejects_long_name(client: TestClient) -> None:
    response = client.post(
        "/api/v1/projects",
        json=create_project_payload(name="a" * 201),
        headers=auth_headers(client),
    )

    assert response.status_code == 422


def test_create_project_requires_token(client: TestClient) -> None:
    response = client.post("/api/v1/projects", json=create_project_payload())

    assert response.status_code == 401


def test_list_projects_is_empty_for_new_user(client: TestClient) -> None:
    response = client.get("/api/v1/projects", headers=auth_headers(client))

    assert response.status_code == 200
    assert response.json() == []


def test_list_projects_returns_newest_updated_first(client: TestClient) -> None:
    headers = auth_headers(client)
    first = client.post(
        "/api/v1/projects", json=create_project_payload(name="First"), headers=headers
    ).json()
    second = client.post(
        "/api/v1/projects", json=create_project_payload(name="Second"), headers=headers
    ).json()

    response = client.get("/api/v1/projects", headers=headers)

    assert response.status_code == 200
    assert [project["id"] for project in response.json()] == [second["id"], first["id"]]


def test_list_projects_is_scoped_to_the_owning_user(client: TestClient) -> None:
    first_headers = auth_headers(client, email="first@example.com")
    second_headers = auth_headers(client, email="second@example.com")
    first_id = client.post(
        "/api/v1/projects", json=create_project_payload(name="First"), headers=first_headers
    ).json()["id"]

    first_list = client.get("/api/v1/projects", headers=first_headers)
    second_list = client.get("/api/v1/projects", headers=second_headers)

    assert [project["id"] for project in first_list.json()] == [first_id]
    assert second_list.json() == []


def test_list_projects_requires_token(client: TestClient) -> None:
    response = client.get("/api/v1/projects")

    assert response.status_code == 401


def test_get_project_succeeds(client: TestClient) -> None:
    headers = auth_headers(client)
    created = client.post(
        "/api/v1/projects", json=create_project_payload(), headers=headers
    ).json()

    response = client.get(f"/api/v1/projects/{created['id']}", headers=headers)

    assert response.status_code == 200
    assert response.json() == created


def test_get_project_returns_not_found_for_unknown_id(client: TestClient) -> None:
    response = client.get("/api/v1/projects/00000000-0000-0000-0000-000000000000", headers=auth_headers(client))

    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_get_project_returns_not_found_for_other_users_project(client: TestClient) -> None:
    first_headers = auth_headers(client, email="first@example.com")
    second_headers = auth_headers(client, email="second@example.com")
    created = client.post(
        "/api/v1/projects", json=create_project_payload(), headers=first_headers
    ).json()

    response = client.get(f"/api/v1/projects/{created['id']}", headers=second_headers)

    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_get_project_requires_token(client: TestClient) -> None:
    response = client.get("/api/v1/projects/some-id")

    assert response.status_code == 401


def test_delete_project_succeeds(client: TestClient) -> None:
    headers = auth_headers(client)
    project_id = client.post(
        "/api/v1/projects", json=create_project_payload(), headers=headers
    ).json()["id"]

    response = client.delete(f"/api/v1/projects/{project_id}", headers=headers)

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/api/v1/projects/{project_id}", headers=headers).status_code == 404


def test_delete_project_returns_not_found_for_unknown_id(client: TestClient) -> None:
    response = client.delete(
        "/api/v1/projects/00000000-0000-0000-0000-000000000000",
        headers=auth_headers(client),
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Project not found"


def test_delete_project_cannot_target_other_users_project(client: TestClient) -> None:
    first_headers = auth_headers(client, email="first@example.com")
    second_headers = auth_headers(client, email="second@example.com")
    project_id = client.post(
        "/api/v1/projects", json=create_project_payload(), headers=first_headers
    ).json()["id"]

    response = client.delete(f"/api/v1/projects/{project_id}", headers=second_headers)

    assert response.status_code == 404
    assert client.get(f"/api/v1/projects/{project_id}", headers=first_headers).status_code == 200


def test_delete_project_requires_token(client: TestClient) -> None:
    headers = auth_headers(client)
    project_id = client.post(
        "/api/v1/projects", json=create_project_payload(), headers=headers
    ).json()["id"]

    assert client.delete(f"/api/v1/projects/{project_id}").status_code == 401
