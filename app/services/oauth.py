"""Sign in with Google or GitHub using the OAuth authorization-code flow.

1. `start_sign_in` builds the provider's consent-page URL. The browser goes
   there; the provider sends it back to the frontend at
   `{FRONTEND_URL}/auth/callback/{provider}?code=...&state=...`.
2. The frontend passes `code` and `state` to `finish_sign_in`, which checks the
   state, exchanges the code for the user's profile (with the client secret,
   which never reaches the browser) and returns our own user.

`state` is a short-lived token signed with `JWT_SECRET_KEY`, so no server-side
session is needed; the frontend also checks it came back unchanged. PKCE is
used on top: the code verifier is derived from the state's nonce with the same
secret, so it never leaves the server.
"""

import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from urllib.parse import urlencode

import httpx
import jwt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.oauth_account import OAuthAccount
from app.models.user import User
from app.services.auth import get_user_by_email, normalize_email

STATE_LIFETIME = timedelta(minutes=10)
STATE_PURPOSE = "oauth-state"
HTTP_TIMEOUT = 10.0


class OAuthError(Exception):
    """The sign-in cannot complete (bad or expired state, rejected code...)."""


class OAuthNotConfigured(OAuthError):
    pass


class OAuthProviderError(OAuthError):
    """The provider could not be reached or answered unexpectedly."""


class OAuthAccountConflict(OAuthError):
    """The email already belongs to a password account."""


@dataclass(frozen=True)
class Profile:
    subject: str
    email: str | None
    email_verified: bool
    name: str
    avatar_url: str | None


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    authorize_url: str
    token_url: str
    scope: str
    extra_params: dict[str, str]
    # GitHub OAuth Apps accept a single callback URL, so each frontend needs its own app.
    one_callback_per_app: bool = False

    def client_for(self, frontend: str) -> "OAuthClient | None":
        """The credentials and redirect URI to use when `frontend` started the sign-in.

        The hosted frontend may have its own credentials (`<name>_hosted_client_id`):
        a GitHub OAuth App allows only one callback URL, so the hosted site needs a
        second app. Google allows several redirect URIs, so one client serves both.
        None when not configured for that frontend.
        """
        prefix = self.name
        hosted = frontend == _normalize(settings.hosted_frontend_url)
        if hosted and (
            self.one_callback_per_app or getattr(settings, f"{self.name}_hosted_client_id", "")
        ):
            prefix = f"{self.name}_hosted"
        client_id = getattr(settings, f"{prefix}_client_id", "").strip()
        client_secret = getattr(settings, f"{prefix}_client_secret", "").strip()
        if not (client_id and client_secret):
            return None
        return OAuthClient(client_id, client_secret, f"{frontend}/auth/callback/{self.name}")


@dataclass(frozen=True)
class OAuthClient:
    client_id: str
    client_secret: str = field(repr=False)
    redirect_uri: str = ""


PROVIDERS = {
    "google": Provider(
        name="google",
        label="Google",
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scope="openid email profile",
        extra_params={"prompt": "select_account"},
    ),
    "github": Provider(
        name="github",
        label="GitHub",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        # `user:email` lets us read private emails and whether they are verified.
        scope="read:user user:email",
        extra_params={"prompt": "select_account"},
        one_callback_per_app=True,
    ),
}
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GITHUB_API = "https://api.github.com"
GITHUB_HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


def get_provider(name: str) -> Provider | None:
    return PROVIDERS.get(name)


def frontend_urls() -> list[str]:
    """Where the app is served: `FRONTEND_URL` (local) and `HOSTED_FRONTEND_URL`."""
    urls = [_normalize(settings.frontend_url), _normalize(settings.hosted_frontend_url)]
    return [url for url in dict.fromkeys(urls) if url]


def frontend_for(origin: str | None) -> str:
    """The frontend a request came from, if it is one of ours; else the local one."""
    candidate = _normalize(origin or "")
    urls = frontend_urls()
    return candidate if candidate in urls else urls[0]


def _normalize(url: str) -> str:
    return url.strip().rstrip("/")


# ---------------------------------------------------------------- the flow


def start_sign_in(provider: Provider, origin: str | None = None) -> tuple[str, str]:
    """The provider's consent-page URL, and the `state` it will send back.

    `origin` is the frontend the request came from (local or hosted); the
    provider sends the browser back there.
    """
    frontend = frontend_for(origin)
    client = _client(provider, frontend)
    nonce = secrets.token_urlsafe(24)
    state = _encode_state(provider.name, nonce, frontend)
    params = {
        "client_id": client.client_id,
        "redirect_uri": client.redirect_uri,
        "response_type": "code",
        "scope": provider.scope,
        "state": state,
        "code_challenge": _code_challenge(_code_verifier(nonce)),
        "code_challenge_method": "S256",
        **provider.extra_params,
    }
    return f"{provider.authorize_url}?{urlencode(params)}", state


def finish_sign_in(db: Session, provider: Provider, code: str, state: str) -> User:
    """Check `state`, exchange `code` for the profile, and find or create the user."""
    nonce, frontend = _decode_state(state, provider.name)
    client = _client(provider, frontend)
    with _http_client() as http:
        access_token = _exchange_code(http, provider, client, code, _code_verifier(nonce))
        profile = _fetch_profile(http, provider, access_token)
    return _user_for(db, provider, profile)


def _client(provider: Provider, frontend: str) -> OAuthClient:
    client = provider.client_for(frontend)
    if client is None:
        where = "" if frontend == frontend_urls()[0] else f" for {frontend}"
        raise OAuthNotConfigured(
            f"Sign in with {provider.label} is not set up{where} on this server."
        )
    return client


# ---------------------------------------------------------------- state and PKCE


def _encode_state(provider: str, nonce: str, frontend: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "purpose": STATE_PURPOSE,
        "provider": provider,
        "nonce": nonce,
        # Where the sign-in started, so the callback uses the same redirect URI.
        "frontend": frontend,
        "iat": now,
        "exp": now + STATE_LIFETIME,
    }
    return jwt.encode(payload, _secret(), algorithm=settings.jwt_algorithm)


def _decode_state(state: str, provider: str) -> tuple[str, str]:
    """The state's nonce, and the frontend that started the sign-in."""
    try:
        payload = jwt.decode(
            state,
            _secret(),
            algorithms=[settings.jwt_algorithm],
            options={"require": ["exp", "nonce", "purpose", "provider"]},
        )
    except jwt.ExpiredSignatureError:
        raise OAuthError("The sign-in took too long. Please try again.") from None
    except jwt.InvalidTokenError:
        raise OAuthError("The sign-in link is invalid. Please try again.") from None
    frontend = str(payload.get("frontend") or frontend_urls()[0])
    if (
        payload["purpose"] != STATE_PURPOSE
        or payload["provider"] != provider
        or frontend not in frontend_urls()
    ):
        raise OAuthError("The sign-in link is invalid. Please try again.")
    return str(payload["nonce"]), frontend


def _code_verifier(nonce: str) -> str:
    digest = hmac.new(_secret().encode(), f"pkce:{nonce}".encode(), hashlib.sha256).digest()
    return _b64url(digest)  # 43 characters, as PKCE requires


def _code_challenge(verifier: str) -> str:
    return _b64url(hashlib.sha256(verifier.encode()).digest())


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _secret() -> str:
    if not settings.jwt_secret_key:
        raise RuntimeError("JWT_SECRET_KEY must be configured before signing in with OAuth")
    return settings.jwt_secret_key


# ---------------------------------------------------------------- talking to the provider

_transport: httpx.BaseTransport | None = None


def set_oauth_transport(transport: httpx.BaseTransport | None) -> None:
    """Route provider requests through `transport` (tests); None restores the network."""
    global _transport
    _transport = transport


def _http_client() -> httpx.Client:
    return httpx.Client(transport=_transport, timeout=HTTP_TIMEOUT)


def _exchange_code(
    http: httpx.Client, provider: Provider, client: OAuthClient, code: str, verifier: str
) -> str:
    response = _call(
        http,
        provider,
        "POST",
        provider.token_url,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": client.redirect_uri,
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "code_verifier": verifier,
        },
        headers={"Accept": "application/json"},
    )
    if response.status_code in (400, 401):
        # An expired, reused or foreign code; or a wrong client secret.
        raise OAuthError(f"{provider.label} did not accept the sign-in. Please try again.")
    data = _json(response, provider)
    if data.get("error"):
        # GitHub reports a bad or expired code as 200 with an `error` field.
        raise OAuthError(f"{provider.label} did not accept the sign-in. Please try again.")
    token = data.get("access_token")
    if not isinstance(token, str) or not token:
        raise OAuthProviderError(f"{provider.label} did not return an access token.")
    return token


def _fetch_profile(client: httpx.Client, provider: Provider, access_token: str) -> Profile:
    if provider.name == "github":
        return _github_profile(client, provider, access_token)
    return _google_profile(client, provider, access_token)


def _google_profile(client: httpx.Client, provider: Provider, access_token: str) -> Profile:
    response = _call(
        client,
        provider,
        "GET",
        GOOGLE_USERINFO_URL,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    data = _json(response, provider)
    subject = data.get("sub")
    if not subject:
        raise OAuthProviderError(f"{provider.label} did not return an account id.")
    email = data.get("email")
    return Profile(
        subject=str(subject),
        email=normalize_email(email) if isinstance(email, str) and email else None,
        email_verified=data.get("email_verified") is True,
        name=str(data.get("name") or "").strip() or (email or "").split("@")[0] or "Researcher",
        avatar_url=data.get("picture") or None,
    )


def _github_profile(client: httpx.Client, provider: Provider, access_token: str) -> Profile:
    headers = {**GITHUB_HEADERS, "Authorization": f"Bearer {access_token}"}
    data = _json(_call(client, provider, "GET", f"{GITHUB_API}/user", headers=headers), provider)
    subject = data.get("id")
    if not subject:
        raise OAuthProviderError(f"{provider.label} did not return an account id.")

    # The profile email is optional and says nothing about verification, so use
    # the primary verified address from the emails list instead.
    response = _call(client, provider, "GET", f"{GITHUB_API}/user/emails", headers=headers)
    try:
        emails = response.json() if response.status_code == 200 else []
    except ValueError:
        emails = []
    if not isinstance(emails, list):
        emails = []
    verified = [e for e in emails if isinstance(e, dict) and e.get("verified") and e.get("email")]
    chosen = next((e for e in verified if e.get("primary")), verified[0] if verified else None)
    email = normalize_email(chosen["email"]) if chosen else None

    login = str(data.get("login") or "")
    return Profile(
        subject=str(subject),
        email=email,
        email_verified=chosen is not None,
        name=str(data.get("name") or "").strip() or login or "Researcher",
        avatar_url=data.get("avatar_url") or None,
    )


def _call(
    client: httpx.Client, provider: Provider, method: str, url: str, **kwargs
) -> httpx.Response:
    try:
        return client.request(method, url, **kwargs)
    except httpx.HTTPError:
        raise OAuthProviderError(f"Could not reach {provider.label}. Please try again.") from None


def _json(response: httpx.Response, provider: Provider) -> dict:
    if response.status_code >= 400:
        raise OAuthProviderError(f"{provider.label} returned an error ({response.status_code}).")
    try:
        data = response.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise OAuthProviderError(f"{provider.label} returned an unexpected response.")
    return data


# ---------------------------------------------------------------- our user


def _user_for(db: Session, provider: Provider, profile: Profile) -> User:
    """The user this identity signs in as; created on first sign-in."""
    linked = db.scalar(
        select(OAuthAccount).where(
            OAuthAccount.provider == provider.name, OAuthAccount.subject == profile.subject
        )
    )
    if linked is not None:
        user = db.get(User, linked.user_id)
        if user is not None:
            return user

    if not profile.email or not profile.email_verified:
        raise OAuthError(
            f"Your {provider.label} account has no verified email address, so it cannot be "
            "used to sign in."
        )

    existing = get_user_by_email(db, profile.email)
    if existing is not None and not existing.password_hash:
        # Created by another provider that verified this same email (e.g. Google,
        # now GitHub): the same person, so this identity joins that account.
        return _link(db, existing, provider, profile)
    if existing is not None:
        # Registration does not verify emails, so anyone could have created a
        # password account with this address. Linking it would let them in too
        # (account pre-hijacking), so the owner signs in with their password.
        raise OAuthAccountConflict(
            f"An account with {profile.email} already exists. Sign in with your email and "
            "password instead."
        )

    user = User(
        name=profile.name[:255],
        email=profile.email,
        password_hash="",  # no password: this account signs in through the provider
        avatar_url=(profile.avatar_url or "")[:512] or None,
    )
    db.add(user)
    db.flush()
    return _link(db, user, provider, profile)


def _link(db: Session, user: User, provider: Provider, profile: Profile) -> User:
    db.add(
        OAuthAccount(
            user_id=user.id,
            provider=provider.name,
            subject=profile.subject,
            email=profile.email,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        # The same person finished two sign-ins at once; use the one that won.
        db.rollback()
        return _user_for(db, provider, profile)
    db.refresh(user)
    return user
