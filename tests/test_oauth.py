import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.dependencies import get_db
from app.db.base import Base
from app.main import app
from app.models import OAuthAccount, User
from app.services.oauth import _b64url, set_oauth_transport

# ---------------------------------------------------------------- fixtures


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)() as session:
        yield session
    engine.dispose()


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def google_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "google_client_id", "client-123.apps.googleusercontent.com")
    monkeypatch.setattr(settings, "google_client_secret", "shh")
    monkeypatch.setattr(settings, "frontend_url", "http://localhost:5173")


class FakeGoogle:
    """Stands in for Google's token and userinfo endpoints."""

    def __init__(self) -> None:
        self.profile = {
            "sub": "google-1",
            "email": "Ada@Example.com",
            "email_verified": True,
            "name": "Ada Lovelace",
            "picture": "https://example.com/ada.png",
        }
        self.token_status = 200
        self.token_requests: list[dict[str, list[str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "oauth2.googleapis.com":
            self.token_requests.append(parse_qs(request.content.decode()))
            if self.token_status != 200:
                return httpx.Response(self.token_status, json={"error": "invalid_grant"})
            return httpx.Response(200, json={"access_token": "google-access", "id_token": "x"})
        if request.url.host == "openidconnect.googleapis.com":
            assert request.headers["Authorization"] == "Bearer google-access"
            return httpx.Response(200, json=self.profile)
        raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture
def google() -> Iterator[FakeGoogle]:
    fake = FakeGoogle()
    set_oauth_transport(httpx.MockTransport(fake))
    yield fake
    set_oauth_transport(None)


def start(client: TestClient) -> tuple[dict[str, list[str]], str]:
    response = client.post("/api/v1/auth/oauth/google")
    assert response.status_code == 200
    body = response.json()
    url = urlparse(body["authorizationUrl"])
    assert (
        f"{url.scheme}://{url.netloc}{url.path}" == "https://accounts.google.com/o/oauth2/v2/auth"
    )
    return parse_qs(url.query), body["state"]


def callback(client: TestClient, state: str, code: str = "auth-code") -> httpx.Response:
    return client.get("/api/v1/auth/callback/google", params={"code": code, "state": state})


# ---------------------------------------------------------------- starting


def test_start_returns_googles_consent_url_with_state_and_pkce(client: TestClient) -> None:
    params, state = start(client)

    assert params["client_id"] == ["client-123.apps.googleusercontent.com"]
    assert params["redirect_uri"] == ["http://localhost:5173/auth/callback/google"]
    assert params["response_type"] == ["code"]
    assert params["scope"] == ["openid email profile"]
    assert params["state"] == [state]
    assert params["code_challenge_method"] == ["S256"]
    assert "shh" not in str(params)  # the secret never goes to the browser


def test_start_is_503_when_google_is_not_configured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "google_client_secret", "")

    response = client.post("/api/v1/auth/oauth/google")

    assert response.status_code == 503
    assert "not set up" in response.json()["detail"]


def test_unknown_provider_is_404(client: TestClient) -> None:
    assert client.post("/api/v1/auth/oauth/myspace").status_code == 404


# ---------------------------------------------------------------- finishing


def test_first_sign_in_creates_the_account_and_returns_a_session(
    client: TestClient, google: FakeGoogle, db: Session
) -> None:
    params, state = start(client)

    response = callback(client, state)

    assert response.status_code == 200
    body = response.json()
    assert body["user"]["email"] == "ada@example.com"
    assert body["user"]["name"] == "Ada Lovelace"
    assert body["user"]["avatarUrl"] == "https://example.com/ada.png"
    assert body["expiresIn"] == settings.access_token_expire_minutes * 60

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {body['accessToken']}"})
    assert me.json()["id"] == body["user"]["id"]

    # The code exchange used the secret, the same redirect URI and a matching PKCE verifier.
    [token_request] = google.token_requests
    assert token_request["client_secret"] == ["shh"]
    assert token_request["redirect_uri"] == ["http://localhost:5173/auth/callback/google"]
    verifier = token_request["code_verifier"][0]
    assert _b64url(hashlib.sha256(verifier.encode()).digest()) == params["code_challenge"][0]

    linked = db.scalar(select(OAuthAccount))
    assert (linked.provider, linked.subject, linked.user_id) == (
        "google",
        "google-1",
        body["user"]["id"],
    )


def test_returning_user_signs_in_to_the_same_account(
    client: TestClient, google: FakeGoogle, db: Session
) -> None:
    first = callback(client, start(client)[1]).json()
    google.profile["email"] = "ada@newmail.com"  # the Google account's email changed

    second = callback(client, start(client)[1]).json()

    assert second["user"]["id"] == first["user"]["id"]
    assert db.scalar(select(func.count()).select_from(User)) == 1


def test_oauth_account_cannot_sign_in_with_a_password(
    client: TestClient, google: FakeGoogle
) -> None:
    callback(client, start(client)[1])

    response = client.post("/api/v1/auth/login", json={"email": "ada@example.com", "password": ""})

    assert response.status_code in (401, 422)
    response = client.post(
        "/api/v1/auth/login", json={"email": "ada@example.com", "password": "anything"}
    )
    assert response.status_code == 401


def test_existing_password_account_is_not_taken_over(
    client: TestClient, google: FakeGoogle, db: Session
) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"name": "Someone", "email": "ada@example.com", "password": "password123"},
    )

    response = callback(client, start(client)[1])

    assert response.status_code == 409
    assert "Sign in with your email and password" in response.json()["detail"]
    assert db.scalar(select(func.count()).select_from(OAuthAccount)) == 0


def test_unverified_google_email_is_refused(client: TestClient, google: FakeGoogle) -> None:
    google.profile["email_verified"] = False

    response = callback(client, start(client)[1])

    assert response.status_code == 400
    assert "verified email" in response.json()["detail"]


@pytest.mark.parametrize(
    "state",
    [
        "not-a-token",
        jwt.encode({"purpose": "oauth-state", "provider": "google", "nonce": "n",
                    "exp": datetime.now(UTC) + timedelta(minutes=5)}, "wrong-secret"),
        jwt.encode({"purpose": "oauth-state", "provider": "github", "nonce": "n",
                    "exp": datetime.now(UTC) + timedelta(minutes=5)}, settings.jwt_secret_key),
    ],
)  # fmt: skip
def test_forged_or_foreign_state_is_rejected_before_calling_google(
    client: TestClient, google: FakeGoogle, state: str
) -> None:
    response = callback(client, state)

    assert response.status_code == 400
    assert google.token_requests == []


def test_expired_state_is_rejected(client: TestClient, google: FakeGoogle) -> None:
    expired = jwt.encode(
        {"purpose": "oauth-state", "provider": "google", "nonce": "n",
         "exp": datetime.now(UTC) - timedelta(seconds=1)},
        settings.jwt_secret_key,
    )  # fmt: skip

    response = callback(client, expired)

    assert response.status_code == 400
    assert "took too long" in response.json()["detail"]


def test_an_access_token_is_not_a_valid_state(client: TestClient, google: FakeGoogle) -> None:
    session = callback(client, start(client)[1]).json()

    assert callback(client, session["accessToken"]).status_code == 400


def test_rejected_code_is_a_400(client: TestClient, google: FakeGoogle) -> None:
    google.token_status = 400

    response = callback(client, start(client)[1])

    assert response.status_code == 400
    assert "did not accept" in response.json()["detail"]


def test_google_unreachable_is_a_502(client: TestClient) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    set_oauth_transport(httpx.MockTransport(refuse))
    try:
        response = callback(client, start(client)[1])
    finally:
        set_oauth_transport(None)

    assert response.status_code == 502
    assert "Could not reach Google" in response.json()["detail"]


# ---------------------------------------------------------------- GitHub


class FakeGitHub:
    """Stands in for GitHub's token, user and emails endpoints."""

    def __init__(self) -> None:
        self.user = {
            "id": 583231,
            "login": "octocat",
            "name": None,
            "avatar_url": "https://a/o.png",
        }
        self.emails = [
            {"email": "old@example.com", "primary": False, "verified": True},
            {"email": "Octo@Example.com", "primary": True, "verified": True},
        ]
        self.token_body: dict = {"access_token": "gho_abc", "token_type": "bearer"}
        self.token_requests: list[dict[str, list[str]]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.host == "github.com":
            assert request.url.path == "/login/oauth/access_token"
            self.token_requests.append(parse_qs(request.content.decode()))
            return httpx.Response(200, json=self.token_body)
        assert request.url.host == "api.github.com"
        assert request.headers["Authorization"] == "Bearer gho_abc"
        if request.url.path == "/user":
            return httpx.Response(200, json=self.user)
        if request.url.path == "/user/emails":
            return httpx.Response(200, json=self.emails)
        raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture
def github(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeGitHub]:
    monkeypatch.setattr(settings, "github_client_id", "Iv1.gh")
    monkeypatch.setattr(settings, "github_client_secret", "gh-shh")
    fake = FakeGitHub()
    set_oauth_transport(httpx.MockTransport(fake))
    yield fake
    set_oauth_transport(None)


def github_sign_in(client: TestClient) -> httpx.Response:
    response = client.post("/api/v1/auth/oauth/github")
    assert response.status_code == 200
    url = urlparse(response.json()["authorizationUrl"])
    params = parse_qs(url.query)
    assert f"{url.netloc}{url.path}" == "github.com/login/oauth/authorize"
    assert params["redirect_uri"] == ["http://localhost:5173/auth/callback/github"]
    assert params["scope"] == ["read:user user:email"]
    return client.get(
        "/api/v1/auth/callback/github",
        params={"code": "gh-code", "state": response.json()["state"]},
    )


def test_github_sign_in_uses_the_primary_verified_email(
    client: TestClient, github: FakeGitHub, db: Session
) -> None:
    response = github_sign_in(client)

    assert response.status_code == 200
    user = response.json()["user"]
    assert user["email"] == "octo@example.com"
    assert user["name"] == "octocat"  # no display name: the login is used
    assert user["avatarUrl"] == "https://a/o.png"
    assert github.token_requests[0]["client_secret"] == ["gh-shh"]
    linked = db.scalar(select(OAuthAccount))
    assert (linked.provider, linked.subject) == ("github", "583231")


def test_github_returning_user_keeps_the_account(client: TestClient, github: FakeGitHub) -> None:
    first = github_sign_in(client).json()
    github.emails = [{"email": "new@example.com", "primary": True, "verified": True}]

    assert github_sign_in(client).json()["user"]["id"] == first["user"]["id"]


def test_github_without_a_verified_email_is_refused(client: TestClient, github: FakeGitHub) -> None:
    github.emails = [{"email": "octo@example.com", "primary": True, "verified": False}]

    response = github_sign_in(client)

    assert response.status_code == 400
    assert "verified email" in response.json()["detail"]


def test_github_rejected_code_is_a_400(client: TestClient, github: FakeGitHub) -> None:
    github.token_body = {"error": "bad_verification_code", "error_description": "expired"}

    response = github_sign_in(client)

    assert response.status_code == 400
    assert "did not accept" in response.json()["detail"]


def test_same_verified_email_on_google_and_github_is_one_account(
    client: TestClient, google: FakeGoogle, monkeypatch: pytest.MonkeyPatch, db: Session
) -> None:
    from_google = callback(client, start(client)[1]).json()  # creates ada@example.com
    monkeypatch.setattr(settings, "github_client_id", "Iv1.gh")
    monkeypatch.setattr(settings, "github_client_secret", "gh-shh")
    fake = FakeGitHub()
    fake.emails = [{"email": "ada@example.com", "primary": True, "verified": True}]
    set_oauth_transport(httpx.MockTransport(fake))

    from_github = github_sign_in(client)

    assert from_github.status_code == 200
    assert from_github.json()["user"]["id"] == from_google["user"]["id"]
    assert db.scalar(select(func.count()).select_from(User)) == 1
    assert db.scalar(select(func.count()).select_from(OAuthAccount)) == 2


def test_github_does_not_take_over_a_password_account(
    client: TestClient, github: FakeGitHub
) -> None:
    client.post(
        "/api/v1/auth/register",
        json={"name": "Someone", "email": "octo@example.com", "password": "password123"},
    )

    assert github_sign_in(client).status_code == 409


def test_callback_requires_code_and_state(client: TestClient) -> None:
    assert client.get("/api/v1/auth/callback/google").status_code == 422
    assert client.get("/api/v1/auth/callback/google", params={"code": "c"}).status_code == 422


# ---------------------------------------------------------------- local and hosted frontends

HOSTED = "https://researchmind.pages.dev"


@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "hosted_frontend_url", HOSTED + "/")  # trailing slash is fine


def start_from(client: TestClient, provider: str, origin: str | None) -> dict[str, list[str]]:
    headers = {"Origin": origin} if origin else {}
    response = client.post(f"/api/v1/auth/oauth/{provider}", headers=headers)
    assert response.status_code == 200, response.text
    return parse_qs(urlparse(response.json()["authorizationUrl"]).query)


def test_google_returns_to_the_frontend_that_started_the_sign_in(
    client: TestClient, google: FakeGoogle, hosted
) -> None:
    local = start_from(client, "google", "http://localhost:5173")
    remote = start_from(client, "google", HOSTED)

    assert local["redirect_uri"] == ["http://localhost:5173/auth/callback/google"]
    assert remote["redirect_uri"] == [f"{HOSTED}/auth/callback/google"]
    assert local["client_id"] == remote["client_id"]  # one Google client serves both

    # The callback exchanges the code with the same redirect URI the sign-in used.
    assert callback(client, remote["state"][0]).status_code == 200
    assert google.token_requests[-1]["redirect_uri"] == [f"{HOSTED}/auth/callback/google"]


def test_unknown_origins_fall_back_to_the_local_frontend(client: TestClient, hosted) -> None:
    params = start_from(client, "google", "https://evil.example.com")

    assert params["redirect_uri"] == ["http://localhost:5173/auth/callback/google"]


def test_github_uses_its_hosted_app_for_the_hosted_frontend(
    client: TestClient, github: FakeGitHub, hosted, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "github_hosted_client_id", "Iv1.hosted")
    monkeypatch.setattr(settings, "github_hosted_client_secret", "hosted-shh")

    local = start_from(client, "github", "http://localhost:5173")
    remote = start_from(client, "github", HOSTED)

    assert local["client_id"] == ["Iv1.gh"]
    assert remote["client_id"] == ["Iv1.hosted"]
    response = client.get(
        "/api/v1/auth/callback/github", params={"code": "c", "state": remote["state"][0]}
    )
    assert response.status_code == 200
    assert github.token_requests[-1]["client_secret"] == ["hosted-shh"]


def test_github_without_a_hosted_app_says_so(
    client: TestClient, github: FakeGitHub, hosted
) -> None:
    response = client.post("/api/v1/auth/oauth/github", headers={"Origin": HOSTED})

    assert response.status_code == 503
    assert f"not set up for {HOSTED}" in response.json()["detail"]


def test_state_for_a_frontend_that_is_not_ours_is_rejected(
    client: TestClient, google: FakeGoogle
) -> None:
    forged = jwt.encode(
        {"purpose": "oauth-state", "provider": "google", "nonce": "n",
         "frontend": "https://evil.example.com",
         "exp": datetime.now(UTC) + timedelta(minutes=5)},
        settings.jwt_secret_key,
    )  # fmt: skip

    assert callback(client, forged).status_code == 400
    assert google.token_requests == []


def test_cors_allows_both_frontends(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.main import allowed_origins

    monkeypatch.setattr(settings, "cors_origins", ["http://127.0.0.1:5173"])
    monkeypatch.setattr(settings, "hosted_frontend_url", HOSTED + "/")

    assert allowed_origins() == ["http://127.0.0.1:5173", "http://localhost:5173", HOSTED]
