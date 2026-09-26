from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import OAuthAccount, User


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    engine.dispose()


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


# ---------------------------------------------------------------- profile


def test_edit_profile_changes_only_what_was_sent(client: TestClient) -> None:
    headers = register(client)

    first = client.patch(
        "/api/v1/users/me",
        json={"name": "Ada Lovelace", "title": "Lead Researcher", "organization": "NUST"},
        headers=headers,
    ).json()
    second = client.patch("/api/v1/users/me", json={"title": ""}, headers=headers).json()

    assert (first["name"], first["title"], first["organization"]) == (
        "Ada Lovelace",
        "Lead Researcher",
        "NUST",
    )
    assert second["title"] is None  # cleared
    assert second["organization"] == "NUST"  # untouched
    assert client.get("/api/v1/auth/me", headers=headers).json()["name"] == "Ada Lovelace"


def test_profile_rejects_a_blank_name_and_needs_sign_in(client: TestClient) -> None:
    headers = register(client)

    assert client.patch("/api/v1/users/me", json={"name": "  "}, headers=headers).status_code == 422
    assert client.patch("/api/v1/users/me", json={"name": "X"}).status_code == 401


# ---------------------------------------------------------------- preferences


def test_preferences_default_then_persist(client: TestClient) -> None:
    headers = register(client)

    assert client.get("/api/v1/users/me/preferences", headers=headers).json() == {
        "autoTranscribe": True,
        "autoTranslate": False,
        "showEvidence": True,
    }

    saved = client.put(
        "/api/v1/users/me/preferences", json={"showEvidence": False}, headers=headers
    ).json()

    assert saved["showEvidence"] is False
    assert saved["autoTranscribe"] is True
    assert client.get("/api/v1/users/me/preferences", headers=headers).json() == saved


def test_preferences_are_per_user(client: TestClient) -> None:
    ada = register(client)
    eve = register(client, "eve@example.com")

    client.put("/api/v1/users/me/preferences", json={"autoTranslate": True}, headers=ada)

    assert client.get("/api/v1/users/me/preferences", headers=eve).json()["autoTranslate"] is False


# ---------------------------------------------------------------- password and sign-in methods


def test_change_password(client: TestClient) -> None:
    headers = register(client)

    wrong = client.patch(
        "/api/v1/auth/password",
        json={"currentPassword": "nope", "newPassword": "newpassword1"},
        headers=headers,
    )
    assert wrong.status_code == 400  # not 401: a typo must not sign the user out
    assert "incorrect" in wrong.json()["detail"]

    ok = client.patch(
        "/api/v1/auth/password",
        json={"currentPassword": "password123", "newPassword": "newpassword1"},
        headers=headers,
    )
    assert ok.status_code == 204

    login = {"email": "ada@example.com"}
    assert (
        client.post("/api/v1/auth/login", json={**login, "password": "password123"}).status_code
        == 401
    )
    assert (
        client.post("/api/v1/auth/login", json={**login, "password": "newpassword1"}).status_code
        == 200
    )


def test_short_new_password_is_rejected(client: TestClient) -> None:
    headers = register(client)

    response = client.patch(
        "/api/v1/auth/password",
        json={"currentPassword": "password123", "newPassword": "short"},
        headers=headers,
    )

    assert response.status_code == 422


def test_google_account_can_add_a_password_and_lists_its_sign_in_methods(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    headers = register(client)
    with session_factory() as db:
        user = db.query(User).one()
        user.password_hash = ""  # as if created by Google sign-in
        db.add(
            OAuthAccount(user_id=user.id, provider="google", subject="g-1", email="ada@example.com")
        )
        db.commit()

    security = client.get("/api/v1/users/me/security", headers=headers).json()
    assert security["hasPassword"] is False
    assert [(p["provider"], p["email"]) for p in security["providers"]] == [
        ("google", "ada@example.com")
    ]

    added = client.patch(
        "/api/v1/auth/password", json={"newPassword": "newpassword1"}, headers=headers
    )
    assert added.status_code == 204
    assert client.get("/api/v1/users/me/security", headers=headers).json()["hasPassword"] is True
    assert "passwordHash" not in str(security)
