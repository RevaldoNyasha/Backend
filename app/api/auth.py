from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.core.dependencies import CurrentToken, CurrentUser, DbSession
from app.core.security import create_access_token
from app.models.user import User
from app.schemas.auth import (
    OAuthStartResponse,
    PasswordChange,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.services.auth import (
    WrongPassword,
    authenticate_user,
    change_password,
    create_user,
    get_user_by_email,
    normalize_email,
    revoke_token,
)
from app.services.oauth import (
    OAuthAccountConflict,
    OAuthError,
    OAuthNotConfigured,
    OAuthProviderError,
    Provider,
    finish_sign_in,
    get_provider,
    start_sign_in,
)

router = APIRouter(tags=["auth"])


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        user=UserResponse.model_validate(user),
        access_token=create_access_token(user.id),
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserCreate, db: DbSession) -> TokenResponse:
    email = normalize_email(payload.email)
    if get_user_by_email(db, email) is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    try:
        user = create_user(
            db,
            payload.name.strip(),
            email,
            payload.password,
            organization=payload.organization,
        )
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        ) from None

    return _token_response(user)


@router.post("/auth/login", response_model=TokenResponse)
def login(payload: UserLogin, db: DbSession) -> TokenResponse:
    user = authenticate_user(db, normalize_email(payload.email), payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _token_response(user)


@router.post("/auth/oauth/{provider}", response_model=OAuthStartResponse)
def oauth_start(provider: str, request: Request) -> OAuthStartResponse:
    """Start signing in with Google (contract §2.3): where to send the browser.

    The provider returns the user to `<that frontend>/auth/callback/{provider}`,
    which hands `code` and `state` to `GET /auth/callback/{provider}`.
    """
    try:
        # The browser says which frontend it is on (local or hosted).
        url, state = start_sign_in(_provider(provider), request.headers.get("origin"))
    except OAuthError as error:
        raise _oauth_http_error(error) from None
    return OAuthStartResponse(authorization_url=url, state=state)


@router.get("/auth/callback/{provider}", response_model=TokenResponse)
def oauth_callback(
    provider: str,
    code: Annotated[str, Query(min_length=1, max_length=2048)],
    state: Annotated[str, Query(min_length=1, max_length=2048)],
    db: DbSession,
) -> TokenResponse:
    """Finish signing in: same body as `/auth/login`. The first sign-in creates the account."""
    try:
        user = finish_sign_in(db, _provider(provider), code, state)
    except OAuthError as error:
        raise _oauth_http_error(error) from None
    return _token_response(user)


def _provider(name: str) -> Provider:
    provider = get_provider(name)
    if provider is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown sign-in provider")
    return provider


def _oauth_http_error(error: OAuthError) -> HTTPException:
    if isinstance(error, OAuthNotConfigured):
        code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(error, OAuthProviderError):
        code = status.HTTP_502_BAD_GATEWAY
    elif isinstance(error, OAuthAccountConflict):
        code = status.HTTP_409_CONFLICT
    else:
        code = status.HTTP_400_BAD_REQUEST
    return HTTPException(status_code=code, detail=str(error))


@router.get("/auth/me", response_model=UserResponse)
def me(current_user: CurrentUser) -> User:
    return current_user


@router.patch("/auth/password", status_code=status.HTTP_204_NO_CONTENT)
def update_password(payload: PasswordChange, current_user: CurrentUser, db: DbSession) -> None:
    """Change the password (contract §2.6), or add one to a Google/GitHub account.

    A wrong current password is `400`, not `401`, so the app does not sign the
    user out for a typo.
    """
    try:
        change_password(db, current_user, payload.current_password, payload.new_password)
    except WrongPassword:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Your current password is incorrect"
        ) from None


@router.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(token: CurrentToken, current_user: CurrentUser, db: DbSession) -> None:
    """Revoke the bearer token used for this request; other sessions stay signed in."""
    revoke_token(db, token.jti, current_user.id, token.expires_at)
