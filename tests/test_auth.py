from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.dependencies import get_db
from app.core.security import create_access_token, decode_access_token, verify_password
from app.db.base import Base
from app.main import app
from app.models import RevokedToken, User


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


def register_user(client: TestClient, email: str = "john@example.com") -> dict[str, Any]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "John Doe",
            "email": email,
            "password": "password123",
        },
    )
    assert response.status_code == 201
    return response.json()["user"]


def test_registration_succeeds(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "John Doe",
            "email": "john@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 201
    body = response.json()
    user = body["user"]
    assert user["name"] == "John Doe"
    assert user["email"] == "john@example.com"
    assert user["role"] == "researcher"
    assert user["organization"] is None
    assert user["avatarUrl"] is None
    assert user["createdAt"].endswith("Z")
    assert isinstance(user["id"], str)
    assert body["accessToken"]
    assert body["expiresIn"] == settings.access_token_expire_minutes * 60
    assert "password" not in body["user"]
    assert "password_hash" not in body["user"]


def test_registration_accepts_organization(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Dr. T. Moyo",
            "email": "t.moyo@university.ac.zw",
            "password": "password123",
            "organization": "University of Zimbabwe",
        },
    )

    assert response.status_code == 201
    assert response.json()["user"]["organization"] == "University of Zimbabwe"


def test_duplicate_email_is_rejected(client: TestClient) -> None:
    register_user(client)

    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Another User",
            "email": "john@example.com",
            "password": "different-password",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Email already registered"


def test_password_is_not_stored_in_plaintext(client: TestClient, db_session: Session) -> None:
    register_user(client)

    user = db_session.scalar(select(User).where(User.email == "john@example.com"))

    assert user is not None
    assert user.password_hash != "password123"
    assert verify_password("password123", user.password_hash)


def test_login_succeeds_and_returns_jwt(client: TestClient) -> None:
    user_data = register_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "john@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user"] == user_data
    assert isinstance(body["accessToken"], str)
    assert body["accessToken"]
    assert body["expiresIn"] == settings.access_token_expire_minutes * 60
    assert "token_type" not in body


def test_login_rejects_invalid_password(client: TestClient) -> None:
    register_user(client)

    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "john@example.com",
            "password": "wrong-password",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_login_rejects_unknown_user(client: TestClient) -> None:
    response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "unknown@example.com",
            "password": "password123",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_current_user_returns_authenticated_user(client: TestClient) -> None:
    user_data = register_user(client)
    login_response = client.post(
        "/api/v1/auth/login",
        json={
            "email": "john@example.com",
            "password": "password123",
        },
    )
    token = login_response.json()["accessToken"]

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    assert response.json() == user_data
    assert "password_hash" not in response.json()


def test_current_user_rejects_missing_token(client: TestClient) -> None:
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_current_user_rejects_invalid_token(client: TestClient) -> None:
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid-token"},
    )

    assert response.status_code == 401


def login_token(client: TestClient, email: str = "john@example.com") -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "password123"},
    )
    assert response.status_code == 200
    return response.json()["accessToken"]


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_logout_revokes_the_token(client: TestClient) -> None:
    register_user(client)
    token = login_token(client)

    response = client.post("/api/v1/auth/logout", headers=bearer(token))

    assert response.status_code == 204
    assert response.content == b""
    after = client.get("/api/v1/auth/me", headers=bearer(token))
    assert after.status_code == 401
    assert after.json()["detail"] == "Token has been revoked"


def test_revoked_token_is_rejected_on_protected_routes(client: TestClient) -> None:
    register_user(client)
    token = login_token(client)
    client.post("/api/v1/auth/logout", headers=bearer(token))

    response = client.get("/api/v1/projects", headers=bearer(token))

    assert response.status_code == 401


def test_logout_leaves_other_sessions_signed_in(client: TestClient) -> None:
    register_user(client)
    first = login_token(client)
    second = login_token(client)

    client.post("/api/v1/auth/logout", headers=bearer(first))

    assert client.get("/api/v1/auth/me", headers=bearer(second)).status_code == 200


def test_logging_out_twice_is_rejected(client: TestClient) -> None:
    register_user(client)
    token = login_token(client)
    client.post("/api/v1/auth/logout", headers=bearer(token))

    response = client.post("/api/v1/auth/logout", headers=bearer(token))

    assert response.status_code == 401


def test_logout_requires_token(client: TestClient) -> None:
    response = client.post("/api/v1/auth/logout")

    assert response.status_code == 401


def test_logout_purges_expired_revocations(client: TestClient, db_session: Session) -> None:
    user = register_user(client)
    db_session.add(
        RevokedToken(
            jti="old-token",
            user_id=user["id"],
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
    )
    db_session.commit()

    client.post("/api/v1/auth/logout", headers=bearer(login_token(client)))

    assert db_session.get(RevokedToken, "old-token") is None
    assert db_session.scalar(select(func.count()).select_from(RevokedToken)) == 1


def test_tokens_have_unique_ids(client: TestClient) -> None:
    register_user(client)

    first = decode_access_token(login_token(client))
    second = decode_access_token(login_token(client))

    assert first["jti"] != second["jti"]


def test_token_without_jti_is_rejected(client: TestClient) -> None:
    user = register_user(client)
    legacy = jwt.encode(
        {"sub": user["id"], "exp": datetime.now(UTC) + timedelta(minutes=5)},
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )

    response = client.get("/api/v1/auth/me", headers=bearer(legacy))

    assert response.status_code == 401


def test_current_user_rejects_expired_token(client: TestClient) -> None:
    user_data = register_user(client)
    expired_token = create_access_token(
        user_data["id"],
        expires_delta=timedelta(minutes=-1),
    )

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )

    assert response.status_code == 401
